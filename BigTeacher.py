"""
teacher_big_lm.py – VRAM-efficient large teacher for distillation.

Three strategies, one interface.  Pick via `strategy=`:

  "quantized"   – 4-bit (bitsandbytes NF4) on GPU.  ~4 GB for 8B.
                  Fastest.  Needs `bitsandbytes` installed.

  "offload"     – FP16 weights pinned in CPU RAM, streamed to GPU
                  one layer at a time during forward/generate.
                  ~0 GB idle, ~0.5 GB peak per layer.  Slower, but
                  leaves almost all VRAM for the student.

  "hybrid"      – FP16 via device_map="auto" (accelerate).  Fits as
                  many layers on GPU as possible, offloads rest to CPU.
                  Faster than full offload, uses partial VRAM.  Requires
                  `pip install accelerate`.

All three release VRAM after each call (evict_from_gpu / _cleanup),
so you can alternate teacher → student without OOM.

Usage:
    teacher = BigCausalTeacher(
        "Qwen/Qwen3-8B",
        tok=my_fast_tokenizer,      # your trained Qwen2 fast tokenizer
        strategy="quantized",       # or "offload" or "hybrid"
    )
    texts, ids = teacher.draft(["Solve x^2=4"])
    lp = teacher.seq_logprob(ctx_ids, tgt_ids)
    teacher.evict_from_gpu()        # free VRAM for student step
"""

from __future__ import annotations


import logging
from contextlib import contextmanager
from typing import Any, List, Optional, Tuple

import torch
import torch.nn.functional as F

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------




# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class BigCausalTeacher:
    """
    VRAM-efficient large teacher.  Three strategies, one interface.
    """

    STRATEGIES = ("quantized", "offload", "hybrid")

    def __init__(
        self,
        model_or_name: Optional[Any] = None,
        device: str = "cuda",
        tok: Optional[Any] = None,
        strategy: str = "quantized",
        max_seq_len: int = 512,
        hf_token: Optional[str] = None,
    ):
        assert strategy in self.STRATEGIES, (
            f"strategy must be one of {self.STRATEGIES}, got {strategy!r}"
        )

        self.device = torch.device(device)
        self.strategy = strategy
        self.enabled = False
        self._tok = tok
        self._lm = None
        self._name = None
        self._hf_token = hf_token
        self._max_seq_len = max_seq_len
        self._on_gpu = False       # tracks whether weights are on GPU

        if model_or_name is None:
            return

        self._name = (
            model_or_name if isinstance(model_or_name, str)
            else "Qwen/Qwen3-8B"
        )

        try:
            if strategy == "quantized":
                self._init_quantized()
            elif strategy == "offload":
                self._init_offload()
            elif strategy == "hybrid":
                self._init_hybrid()
            self.enabled = True
            log.info("BigCausalTeacher [%s] ready – model: %s",
                     strategy, self._name)
        except Exception:
            log.exception("BigCausalTeacher init failed; disabling.")
            self.enabled = False

    # ==================================================================
    # Strategy-specific init
    # ==================================================================

    def _init_quantized(self):
        """Load model in 4-bit NF4 directly onto GPU (~4 GB for 8B)."""
        from transformers import AutoModelForCausalLM, BitsAndBytesConfig

        bnb_cfg = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )
        self._lm = AutoModelForCausalLM.from_pretrained(
            self._name,
            quantization_config=bnb_cfg,
            device_map={"": self.device},
            trust_remote_code=True,
            token=self._hf_token,
        )
        self._lm.eval()
        self._on_gpu = True
        self._load_tok_if_needed()

    def _init_offload(self):
        """Load FP16 weights into pinned CPU RAM. Stream per-call."""
        from transformers import AutoModelForCausalLM

        self._lm = AutoModelForCausalLM.from_pretrained(
            self._name,
            torch_dtype=torch.float16,
            device_map="cpu",
            trust_remote_code=True,
            token=self._hf_token,
        )
        self._lm.eval()
        # Pin memory for faster CPU→GPU transfers
        self._pin_model_memory()
        self._on_gpu = False
        self._load_tok_if_needed()

    def _init_hybrid(self):
        """
        Use transformers device_map="auto" with accelerate.  This lets
        accelerate place as many layers as fit on GPU, overflow to CPU,
        and handle the offloading automatically during forward passes.
        Faster than manual streaming (keeps hot layers on GPU) while
        using less VRAM than full quantized loading.

        Requires: `pip install accelerate`
        """
        from transformers import AutoModelForCausalLM

        self._lm = AutoModelForCausalLM.from_pretrained(
            self._name,
            torch_dtype=torch.float16,
            device_map="auto",           # accelerate splits layers
            trust_remote_code=True,
            token=self._hf_token,
            low_cpu_mem_usage=True,
            offload_folder="/tmp/teacher_offload",
        )
        self._lm.eval()
        self._on_gpu = True   # managed by accelerate, always callable
        self._load_tok_if_needed()

    def _load_tok_if_needed(self):
        if self._tok is None:
            from transformers import AutoTokenizer
            self._tok = AutoTokenizer.from_pretrained(
                self._name,
                trust_remote_code=True,
                token=self._hf_token,
            )

    def _pin_model_memory(self):
        """Pin CPU tensors for async CPU→GPU transfer."""
        for p in self._lm.parameters():
            if p.device.type == "cpu" and not p.data.is_pinned():
                try:
                    p.data = p.data.pin_memory()
                except Exception:
                    pass  # some tensor types can't be pinned, that's fine

    # ==================================================================
    # GPU residency management
    # ==================================================================

    def _move_to_gpu(self):
        """Move full model to GPU (for offload/hybrid strategies)."""
        self._maybe_reload()   # handles quantized model after eviction
        if self._on_gpu or self._lm is None:
            return
        self._lm.to(self.device)
        self._on_gpu = True

    def evict_from_gpu(self):
        """
        Release VRAM.  Call this between teacher and student steps.

        - quantized: bnb 4-bit models can't be moved to CPU.  We delete
                     the model and set a flag to reload on next use.
                     For repeated use, consider 'hybrid' instead.
        - offload: already on CPU, just flushes cache
        - hybrid: accelerate-managed, can't move — just flush cache.
        """
        if self._lm is None:
            return
        if self.strategy == "quantized":
            # bnb 4-bit tensors are device-locked — can't .to("cpu")
            del self._lm
            self._lm = None
            self._on_gpu = False
            self.enabled = False      # will re-enable on reload
            self._needs_reload = True
        torch.cuda.empty_cache()

    def _maybe_reload(self):
        """Re-initialise quantized model if it was evicted."""
        if getattr(self, "_needs_reload", False) and self._name is not None:
            self._init_quantized()
            self.enabled = True
            self._needs_reload = False

    @contextmanager
    def on_gpu(self):
        """Context manager: move to GPU on enter, evict on exit."""
        try:
            self._move_to_gpu()
            yield
        finally:
            self.evict_from_gpu()

    # ==================================================================
    # Layer-streaming forward (offload / hybrid)
    # ==================================================================

    @torch.inference_mode()
    def _forward_streamed(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Manual layer-by-layer forward through the model, keeping only
        one layer on GPU at a time.  Returns logits on GPU.

        Works for Qwen2/3, Llama, and anything with the standard
        {embed_tokens, layers[], norm, lm_head} layout.

        NOTE on attention masking: We don't pass an explicit causal mask.
        Modern transformers decoder layers default to causal via SDPA when
        no mask is given.  Sliding-window attention (if configured in Qwen3)
        is handled internally by each layer's self_attn based on its
        layer_type attribute — we don't need to supply a mask for that.
        For teacher inference on sequences ≤ max_seq_len this is correct.
        """
        model = self._lm.model   # the inner decoder stack
        device = self.device

        # --- Embed ---
        embed = model.embed_tokens
        embed.to(device)
        hidden = embed(input_ids.to(device))
        embed.to("cpu")

        # --- Position ids + cache_position (needed by newer transformers) ---
        seq_len = hidden.size(1)
        position_ids = torch.arange(
            seq_len, device=device, dtype=torch.long
        ).unsqueeze(0)
        cache_position = torch.arange(
            seq_len, device=device, dtype=torch.long
        )

        # --- Rotary embeddings (Qwen2/3, Llama expose model.rotary_emb) ---
        position_embeddings = None
        if hasattr(model, "rotary_emb"):
            rotary = model.rotary_emb
            rotary.to(device)
            position_embeddings = rotary(hidden, position_ids)
            rotary.to("cpu")

        # --- Transformer layers ---
        for layer in model.layers:
            layer.to(device)

            # Build kwargs compatible with both Qwen2 and Qwen3 decoders.
            # position_embeddings carries (cos, sin) from RoPE.
            # cache_position is used by newer transformers to index into
            # static caches / sliding window offsets.
            layer_kwargs = {
                "position_ids": position_ids,
                "use_cache": False,
                "cache_position": cache_position,
            }
            if position_embeddings is not None:
                layer_kwargs["position_embeddings"] = position_embeddings

            out = layer(hidden, **layer_kwargs)
            # Decoder layers return (hidden_states, ...) — grab first element
            hidden = out[0] if isinstance(out, (tuple, list)) else out

            layer.to("cpu")
            torch.cuda.empty_cache()

        # --- Norm ---
        model.norm.to(device)
        hidden = model.norm(hidden)
        model.norm.to("cpu")

        # --- LM head ---
        lm_head = self._lm.lm_head
        lm_head.to(device)
        logits = lm_head(hidden).float()
        lm_head.to("cpu")

        torch.cuda.empty_cache()
        return logits

    # ==================================================================
    # Unified forward dispatch
    # ==================================================================

    @torch.inference_mode()
    def _forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Returns logits (1, T, V) on GPU."""
        if self.strategy in ("quantized", "hybrid"):
            # quantized: model fully on GPU
            # hybrid: accelerate manages layer placement, forward works normally
            self._move_to_gpu()
            return self._lm(input_ids.to(self.device)).logits.float()
        else:
            # offload: manual layer streaming
            return self._forward_streamed(input_ids)

    # ==================================================================
    # Public interface (mirrors SmallCausalTeacher)
    # ==================================================================

    @property
    def hf_tok(self):
        return self._tok

    @torch.inference_mode()
    def _generate_ids(
        self,
        prompt: str,
        max_new_tokens: int = 128,
        temperature: float = 0.6,
        top_p: float = 0.95,
    ) -> torch.Tensor:
        if not self.enabled:
            return torch.zeros(0, dtype=torch.long, device=self.device)

        toks = self._tok(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=self._max_seq_len,
        )
        input_ids = toks["input_ids"]

        if self.strategy in ("quantized", "hybrid"):
            # Standard HF generate – model on GPU (or accelerate-managed)
            self._move_to_gpu()
            out = self._lm.generate(
                input_ids.to(self.device),
                do_sample=(temperature > 0.0),
                temperature=max(1e-5, temperature),
                top_p=top_p,
                max_new_tokens=max_new_tokens,
                pad_token_id=self._tok.eos_token_id or self._tok.pad_token_id,
            )
            gen = out[0, input_ids.size(1):]
            return gen
        else:
            # For offload/hybrid: greedy auto-regressive via streamed forward
            # (HF generate won't work with weights on CPU)
            return self._generate_streamed(
                input_ids, max_new_tokens, temperature, top_p,
            )

    @torch.inference_mode()
    def _generate_streamed(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int,
        temperature: float,
        top_p: float,
    ) -> torch.Tensor:
        """
        Auto-regressive generation using _forward_streamed.

        Each step does a full layer sweep, so this is slow but
        VRAM-minimal.  Fine for drafting short sequences.
        """
        ids = input_ids.to(self.device)
        generated = []

        for _ in range(max_new_tokens):
            logits = self._forward_streamed(ids)
            next_logits = logits[:, -1, :]  # (1, V)

            if temperature > 0.0:
                next_logits = next_logits / max(1e-5, temperature)
                probs = F.softmax(next_logits, dim=-1)
                # top-p filtering
                sorted_probs, sorted_idx = probs.sort(descending=True)
                cumsum = sorted_probs.cumsum(dim=-1)
                mask = cumsum - sorted_probs > top_p
                sorted_probs[mask] = 0.0
                sorted_probs.div_(sorted_probs.sum(dim=-1, keepdim=True))
                next_id = sorted_idx.gather(
                    -1, torch.multinomial(sorted_probs, 1)
                )
            else:
                next_id = next_logits.argmax(dim=-1, keepdim=True)

            generated.append(next_id.squeeze(-1))
            ids = torch.cat([ids, next_id], dim=1)

            # Stop on EOS
            eos = self._tok.eos_token_id
            if eos is not None and next_id.item() == eos:
                break

        if not generated:
            return torch.zeros(0, dtype=torch.long, device=self.device)
        return torch.stack(generated)

    @torch.inference_mode()
    def draft(
        self,
        prompts: List[str],
        max_new_tokens: int = 128,
        temperature: float = 0.6,
        top_p: float = 0.95,
    ) -> Tuple[List[str], List[torch.Tensor]]:
        texts, ids = [], []
        for p in prompts:
            y_ids = self._generate_ids(
                p, max_new_tokens=max_new_tokens,
                temperature=temperature, top_p=top_p,
            )
            ids.append(y_ids)
            texts.append(
                self._tok.decode(y_ids, skip_special_tokens=True)
                if self.enabled else ""
            )
        return texts, ids

    @torch.inference_mode()
    def seq_logprob(
        self,
        ctx_ids: torch.Tensor,
        tgt_ids: torch.Tensor,
    ) -> torch.Tensor:
        """
        Log P(tgt | ctx) under the teacher.  Returns scalar tensor.
        """
        if not self.enabled or ctx_ids.numel() == 0 or tgt_ids.numel() == 0:
            return torch.tensor(0.0, device=self.device)

        inp = torch.cat([ctx_ids, tgt_ids], dim=0).unsqueeze(0)
        logits = self._forward(inp)           # (1, T, V)
        logits = logits[:, :-1, :]            # shift: predict next token
        tgt = inp[:, 1:].to(self.device)

        lp = F.log_softmax(logits, dim=-1)
        lp = lp.gather(-1, tgt.unsqueeze(-1)).squeeze(-1)

        T_ctx = ctx_ids.numel()
        lp_tgt = lp[:, T_ctx:].sum(dim=-1)
        return lp_tgt.squeeze(0)

    # ==================================================================
    # Benchmarking helper
    # ==================================================================

    def benchmark(self, prompt: str = "What is 2+2?", max_new_tokens: int = 32):
        """
        Quick timing test.  Prints VRAM usage and wall-clock time
        for a forward pass, a short generation, and a seq_logprob call.
        """
        import time

        def _vram_mb():
            if torch.cuda.is_available():
                return torch.cuda.memory_allocated(self.device) / 1e6
            return 0.0

        print(f"\n{'='*60}")
        print(f"BigCausalTeacher benchmark  –  strategy={self.strategy}")
        print(f"{'='*60}")

        toks = self._tok(prompt, return_tensors="pt")
        ids = toks["input_ids"]

        # Forward pass
        torch.cuda.empty_cache()
        v0 = _vram_mb()
        t0 = time.perf_counter()
        logits = self._forward(ids)
        t1 = time.perf_counter()
        v1 = _vram_mb()
        print(f"forward     : {t1 - t0:7.3f}s   VRAM: {v0:.0f} → {v1:.0f} MB")

        # Generation
        torch.cuda.empty_cache()
        v0 = _vram_mb()
        t0 = time.perf_counter()
        self._generate_ids(prompt, max_new_tokens=max_new_tokens)
        t1 = time.perf_counter()
        v1 = _vram_mb()
        print(f"generate({max_new_tokens:>2}): {t1 - t0:7.3f}s   VRAM: {v0:.0f} → {v1:.0f} MB")

        # seq_logprob
        ctx = ids.squeeze(0).to(self.device)
        tgt = torch.randint(0, 1000, (16,), device=self.device)
        torch.cuda.empty_cache()
        v0 = _vram_mb()
        t0 = time.perf_counter()
        lp = self.seq_logprob(ctx, tgt)
        t1 = time.perf_counter()
        v1 = _vram_mb()
        print(f"seq_logprob : {t1 - t0:7.3f}s   VRAM: {v0:.0f} → {v1:.0f} MB   lp={lp.item():.2f}")

        # Evict and show idle VRAM
        self.evict_from_gpu()
        v2 = _vram_mb()
        print(f"after evict :                VRAM: {v2:.0f} MB")
        print(f"{'='*60}\n")