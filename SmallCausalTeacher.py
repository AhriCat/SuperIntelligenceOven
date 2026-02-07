# teacher_small_lm.pymodel_name="Qwen/Qwen3-0.6B
class SmallCausalTeacher:
    """
    Tiny teacher wrapper (HF causal LM). Greedy by default; configurable.
    If transformers absent, teacher silently disables.
    """
    def __init__(self, model_or_name: Optional[Any], device: str = "cuda"):
        self.device = torch.device(device)
        self.enabled = False
        self._tok = tok if tok else None
        self._lm = None
        if model_or_name is None:
            return
        try:
            from transformers import AutoTokenizer, AutoModelForCausalLM
            name = (model_or_name if isinstance(model_or_name, str)
                    else "Qwen/Qwen3-4B-Thinking-2507")
            self._tok = tok
            self._lm  = AutoModelForCausalLM.from_pretrained(
                name, trust_remote_code=True,
            ).to(self.device).eval()
            self.enabled = True
        except Exception:
            self.enabled = False

    @property
    def hf_tok(self):
        return self._tok

    @torch.inference_mode()
    def _generate_ids(self, prompt: str, max_new_tokens=128, temperature=0.0, top_p=0.95) -> torch.Tensor:
        if not self.enabled:
            return torch.zeros(0, dtype=torch.long, device=self.device)
        toks = self._tok(prompt, return_tensors="pt").to(self.device)
        out = self._lm.generate(
            **toks,
            do_sample=(temperature > 0.0),
            temperature=max(1e-5, temperature),
            top_p=top_p,
            max_new_tokens=max_new_tokens,
            pad_token_id=self._tok.eos_token_id or self._tok.pad_token_id
        )
        # return only the generated continuation slice
        gen = out[0, toks["input_ids"].size(1):]
        return gen

    @torch.inference_mode()
    def draft(self, prompts: List[str], max_new_tokens=128, temperature=0.0, top_p=0.95) -> Tuple[List[str], List[torch.Tensor]]:
        texts, ids = [], []
        for p in prompts:
            y_ids = self._generate_ids(p, max_new_tokens=max_new_tokens, temperature=temperature, top_p=top_p)
            ids.append(y_ids)
            if self.enabled:
                texts.append(self._tok.decode(y_ids, skip_special_tokens=True))
            else:
                texts.append("")
        return texts, ids

    @torch.inference_mode()
    def seq_logprob(self, ctx_ids: torch.Tensor, tgt_ids: torch.Tensor) -> torch.Tensor:
        """
        Log P(tgt | ctx) on teacher, scalar.
        """
        if not self.enabled or ctx_ids.numel() == 0 or tgt_ids.numel() == 0:
            return torch.tensor(0.0, device=self.device)
        inp = torch.cat([ctx_ids, tgt_ids], dim=0).unsqueeze(0)
        out = self._lm(inp)
        logits = out.logits[:, :-1, :]
        tgt = inp[:, 1:]
        lp = F.log_softmax(logits, dim=-1).gather(-1, tgt.unsqueeze(-1)).squeeze(-1)
        # sum only over target region
        Tctx = ctx_ids.numel()
        lp_tgt = lp[:, Tctx:].sum(dim=-1)
        return lp_tgt.squeeze(0)
