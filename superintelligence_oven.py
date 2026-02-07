"""
SUPERINTELLIGENCE OVEN  v0.1
=============================
GRPO training loop with:
  - Local policy model (trainable, Qwen-family, shared tokenizer)
  - Hot-swappable local teachers (SmallCausalTeacher / BigCausalTeacher)
  - QwenEmbedVerifier as semantic reward signal
  - Remote agent swarm (1-5 agents, each with independently configurable frontier model)
  - Optional frontier logprob calibration (DeepSeek / Grok top-k)

Integrates with existing model.teacher / model.verifier pattern.
All Qwen models share Qwen2TokenizerFast.

Usage:
    config = OvenConfig()
    oven = SuperintelligenceOven(config, model=your_model)
    asyncio.run(oven.train(prompt_source))
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass, field
from typing import (
    Any, Callable, Dict, List, Optional, Tuple, Union,
)
from collections import defaultdict
from contextlib import contextmanager
import asyncio
import aiohttp
import json
import time
import logging
import random

log = logging.getLogger("oven")
log.setLevel(logging.INFO)
if not log.handlers:
    log.addHandler(logging.StreamHandler())


# =============================================================================
# CONFIG
# =============================================================================

@dataclass
class AgentConfig:
    """Per-agent configuration — each agent can hit a different frontier model."""
    name: str
    role: str
    weight: float = 0.20
    provider: str = "claude"          # "claude" | "deepseek" | "grok" | "openai" | "local"
    model: str = "claude-sonnet-4-20250514"
    endpoint: str = ""                # auto-resolved from provider if blank
    api_key_env: str = ""             # env var name holding the key
    temperature: float = 0.1
    logprobs: bool = False            # request logprobs (DeepSeek/Grok only)
    top_logprobs: int = 20
    is_curriculum: bool = False       # curriculum agents don't score; they steer


# Sensible defaults — 5 agents, each swappable independently
DEFAULT_AGENTS = [
    AgentConfig(
        name="critic",
        role="Score correctness, coherence, and instruction-following",
        weight=0.30,
        provider="claude",
        model="claude-sonnet-4-20250514",
    ),
    AgentConfig(
        name="adversary",
        role="Probe for failure modes, hallucinations, and reward-hackable outputs",
        weight=0.20,
        provider="claude",
        model="claude-sonnet-4-20250514",
    ),
    AgentConfig(
        name="specialist",
        role="Score domain-specific accuracy and depth",
        weight=0.20,
        provider="deepseek",
        model="deepseek-chat",
        logprobs=True,
    ),
    AgentConfig(
        name="style",
        role="Score clarity, tone, and formatting quality",
        weight=0.15,
        provider="grok",
        model="grok-3-mini",
        logprobs=True,
    ),
    AgentConfig(
        name="curriculum",
        role="Analyze training dynamics and suggest next training prompts",
        weight=0.15,
        provider="claude",
        model="claude-sonnet-4-20250514",
        is_curriculum=True,
    ),
]


@dataclass
class TeacherConfig:
    """Which local teachers are available for hot-swap."""
    name: str
    kind: str               # "small" | "big"
    model_name: str
    strategy: str = "quantized"    # big teacher only: "quantized" | "offload" | "hybrid"
    priority: str = "default"
    hf_token: Optional[str] = None


DEFAULT_TEACHERS = [
    TeacherConfig(
        name="small_qwen3_4b",
        kind="small",
        model_name="Qwen/Qwen3-4B-Thinking-2507",
        priority="fast_iteration",
    ),
    TeacherConfig(
        name="big_qwen3_8b_q4",
        kind="big",
        model_name="Qwen/Qwen3-8B",
        strategy="quantized",
        priority="deep_reasoning",
    ),
    TeacherConfig(
        name="big_qwen3_8b_hybrid",
        kind="big",
        model_name="Qwen/Qwen3-8B",
        strategy="hybrid",
        priority="balanced",
    ),
]


@dataclass
class OvenConfig:
    # GRPO
    group_size: int = 8
    max_seq_len: int = 2048
    grpo_clip_eps: float = 0.2
    advantage_norm: bool = True

    # KL
    kl_beta: float = 0.1
    kl_anneal: bool = False          # linearly anneal kl_beta over training
    kl_beta_min: float = 0.01

    # Verifier
    verifier_weight: float = 0.1     # blend verifier score into reward
    verifier_model: str = "Qwen/Qwen3-Embedding-0.6B"

    # Teachers
    teachers: List[TeacherConfig] = field(default_factory=lambda: DEFAULT_TEACHERS)
    initial_teacher: str = "small_qwen3_4b"
    teacher_swap_interval: int = 500
    teacher_kl_collapse_threshold: float = 0.01

    # Agent swarm
    agents: List[AgentConfig] = field(default_factory=lambda: DEFAULT_AGENTS)
    agent_timeout: float = 30.0      # seconds per agent call
    agent_retry: int = 2

    # Frontier calibration
    frontier_calibration: bool = True
    calibration_interval: int = 50
    calibration_agent: str = "specialist"   # which agent to use for top-k comparison

    # Training
    lr: float = 1e-6
    batch_size: int = 4
    total_steps: int = 10000
    grad_clip: float = 1.0
    curriculum_interval: int = 100

    # Shared tokenizer
    tokenizer_name: str = "Qwen/Qwen2.5-0.5B"   # Qwen2TokenizerFast, shared

    # Device
    device: str = "cuda"


# =============================================================================
# ENDPOINT REGISTRY
# =============================================================================

PROVIDER_ENDPOINTS = {
    "claude":   "https://api.anthropic.com/v1/messages",
    "deepseek": "https://api.deepseek.com/v1/chat/completions",
    "grok":     "https://api.x.ai/v1/chat/completions",
    "openai":   "https://api.openai.com/v1/chat/completions",
}

PROVIDER_KEY_ENVS = {
    "claude":   "ANTHROPIC_API_KEY",
    "deepseek": "DEEPSEEK_API_KEY",
    "grok":     "XAI_API_KEY",
    "openai":   "OPENAI_API_KEY",
}


def resolve_endpoint(agent: AgentConfig) -> str:
    if agent.endpoint:
        return agent.endpoint
    return PROVIDER_ENDPOINTS.get(agent.provider, agent.endpoint)


def resolve_api_key(agent: AgentConfig) -> str:
    import os
    env = agent.api_key_env or PROVIDER_KEY_ENVS.get(agent.provider, "")
    return os.environ.get(env, "")


# =============================================================================
# TEACHER MANAGER
# =============================================================================

class TeacherManager:
    """
    Hot-swappable teacher pool.
    Wraps SmallCausalTeacher and BigCausalTeacher behind a unified interface.
    The active teacher is always accessible as `.current` and provides:
        - seq_logprob(ctx_ids, tgt_ids) -> scalar
        - draft(prompts, ...) -> (texts, ids)
        - full forward for KL computation
    """

    def __init__(self, config: OvenConfig, shared_tok):
        self.config = config
        self.shared_tok = shared_tok
        self.device = torch.device(config.device)
        self._pool: Dict[str, Any] = {}      # name -> teacher instance
        self._configs: Dict[str, TeacherConfig] = {
            t.name: t for t in config.teachers
        }
        self.current: Optional[Any] = None
        self.current_name: Optional[str] = None

    # ---- lazy loading ----

    def _load(self, name: str) -> Any:
        """Instantiate a teacher by name. Cached in pool."""
        if name in self._pool:
            return self._pool[name]

        tc = self._configs[name]
        if tc.kind == "small":
            teacher = self._make_small(tc)
        elif tc.kind == "big":
            teacher = self._make_big(tc)
        else:
            raise ValueError(f"Unknown teacher kind: {tc.kind!r}")

        self._pool[name] = teacher
        return teacher

    def _make_small(self, tc: TeacherConfig):
        """Import and build SmallCausalTeacher."""
        return _SmallCausalTeacher(
            model_or_name=tc.model_name,
            device=str(self.device),
            tok=self.shared_tok,
        )

    def _make_big(self, tc: TeacherConfig):
        """Import and build BigCausalTeacher."""
        return _BigCausalTeacher(
            model_or_name=tc.model_name,
            device=str(self.device),
            tok=self.shared_tok,
            strategy=tc.strategy,
            max_seq_len=self.config.max_seq_len,
            hf_token=tc.hf_token,
        )

    # ---- swap interface ----

    def swap_teacher(self, name: str):
        """
        Hot-swap the active teacher.
        Evicts the previous big teacher from GPU if applicable.
        """
        if name not in self._configs:
            raise KeyError(
                f"Unknown teacher {name!r}. "
                f"Available: {list(self._configs.keys())}"
            )

        # evict current big teacher to free VRAM
        if self.current is not None and hasattr(self.current, "evict_from_gpu"):
            self.current.evict_from_gpu()

        self.current = self._load(name)
        self.current_name = name
        log.info("[OVEN] Teacher swapped -> %s (%s)", name, self._configs[name].kind)

    def register_teacher(self, tc: TeacherConfig):
        """Add a new teacher config at runtime (e.g. from agent vote)."""
        self._configs[tc.name] = tc
        log.info("[OVEN] Registered new teacher config: %s", tc.name)

    # ---- KL computation ----

    @torch.inference_mode()
    def compute_kl(
        self,
        policy_logits: torch.Tensor,    # (T, V) or (1, T, V)
        token_ids: torch.Tensor,         # (T,) or (1, T)
    ) -> torch.Tensor:
        """
        KL(policy || teacher) using FULL local distributions.
        This is the real anchor — not an approximation from top-k.
        """
        if self.current is None or not self.current.enabled:
            return torch.tensor(0.0, device=self.device)

        # normalise shapes
        if policy_logits.dim() == 2:
            policy_logits = policy_logits.unsqueeze(0)
        if token_ids.dim() == 1:
            token_ids = token_ids.unsqueeze(0)

        # teacher forward
        teacher_logits = self.current._forward(token_ids)   # (1, T, V)

        # KL over full vocab, mean over sequence
        policy_lp = F.log_softmax(policy_logits, dim=-1)
        teacher_lp = F.log_softmax(teacher_logits, dim=-1)
        teacher_p = teacher_lp.exp()

        kl = (teacher_p * (teacher_lp - policy_lp)).sum(dim=-1).mean()
        return kl

    # ---- auto-swap heuristics ----

    def maybe_auto_swap(self, step: int, metrics: Dict) -> bool:
        """Check if teacher should be swapped based on training dynamics."""
        if step % self.config.teacher_swap_interval != 0:
            return False

        kl_mean = metrics.get("kl_mean", 1.0)
        current_kind = self._configs[self.current_name].kind

        # small teacher absorbed -> escalate to big
        if (
            current_kind == "small"
            and kl_mean < self.config.teacher_kl_collapse_threshold
        ):
            big_teachers = [
                n for n, c in self._configs.items() if c.kind == "big"
            ]
            if big_teachers:
                self.swap_teacher(big_teachers[0])
                return True

        # agent-driven swap
        if metrics.get("agent_swap_vote"):
            target = metrics.get("agent_swap_target")
            if target and target in self._configs:
                self.swap_teacher(target)
                return True

        return False


# =============================================================================
# AGENT SWARM
# =============================================================================

class AgentSwarm:
    """
    Remote reward committee.  Each agent can be a different frontier model.
    No gradients pass through here — just scalar rewards.
    """

    def __init__(self, config: OvenConfig):
        self.config = config
        self.agents: Dict[str, AgentConfig] = {a.name: a for a in config.agents}
        self.scoring_agents = [a for a in config.agents if not a.is_curriculum]
        self.curriculum_agents = [a for a in config.agents if a.is_curriculum]
        self.score_history: Dict[str, List] = defaultdict(list)

    # ---- per-agent model swap ----

    def swap_agent_model(
        self,
        agent_name: str,
        provider: Optional[str] = None,
        model: Optional[str] = None,
        endpoint: Optional[str] = None,
        logprobs: Optional[bool] = None,
        top_logprobs: Optional[int] = None,
    ):
        """
        Hot-swap the frontier model for a specific agent.

            swarm.swap_agent_model("critic", provider="deepseek", model="deepseek-reasoner")
            swarm.swap_agent_model("adversary", provider="grok", model="grok-3", logprobs=True)
            swarm.swap_agent_model("specialist", model="deepseek-chat-v3")
        """
        if agent_name not in self.agents:
            raise KeyError(
                f"Unknown agent {agent_name!r}. "
                f"Available: {list(self.agents.keys())}"
            )
        a = self.agents[agent_name]
        if provider is not None:
            a.provider = provider
        if model is not None:
            a.model = model
        if endpoint is not None:
            a.endpoint = endpoint
        if logprobs is not None:
            a.logprobs = logprobs
        if top_logprobs is not None:
            a.top_logprobs = top_logprobs

        log.info(
            "[OVEN] Agent '%s' model swapped -> %s/%s (logprobs=%s)",
            agent_name, a.provider, a.model, a.logprobs,
        )

    def swap_all_agents_model(self, provider: str, model: str, **kw):
        """Swap every scoring agent to the same frontier model."""
        for a in self.scoring_agents:
            self.swap_agent_model(a.name, provider=provider, model=model, **kw)

    # ---- scoring ----

    async def score_completions(
        self,
        prompt: str,
        completions: List[str],
        context: Optional[Dict] = None,
    ) -> Tuple[List[float], Dict]:
        """
        Fan out to all scoring agents, return (composite_scores, diagnostics).
        diagnostics includes per-agent scores, any logprobs returned, etc.
        """
        tasks = [
            self._query_agent(agent, prompt, completions, context)
            for agent in self.scoring_agents
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        composite = [0.0] * len(completions)
        total_weight = 0.0
        diagnostics: Dict[str, Any] = {"per_agent": {}, "frontier_logprobs": {}}

        for agent, result in zip(self.scoring_agents, results):
            if isinstance(result, Exception):
                log.warning("[OVEN] Agent '%s' failed: %s", agent.name, result)
                continue

            scores, meta = result
            w = agent.weight
            total_weight += w
            for i, s in enumerate(scores):
                composite[i] += w * s

            diagnostics["per_agent"][agent.name] = scores
            self.score_history[agent.name].append(scores)

            # stash any frontier logprobs for calibration
            if meta.get("logprobs"):
                diagnostics["frontier_logprobs"][agent.name] = meta["logprobs"]

        # normalise by actual weight (in case an agent failed)
        if total_weight > 0:
            composite = [s / total_weight for s in composite]

        return composite, diagnostics

    async def _query_agent(
        self,
        agent: AgentConfig,
        prompt: str,
        completions: List[str],
        context: Optional[Dict],
    ) -> Tuple[List[float], Dict]:
        """
        Call a single agent's frontier model.
        Returns (scores, metadata).
        Handles Claude vs OpenAI-compatible API shapes.
        """
        system_prompt = (
            f"You are a training reward agent.\n"
            f"Role: {agent.role}\n"
            f"Score each completion 0.0 to 1.0. Be precise — spread scores.\n"
            f"Return ONLY a JSON array of floats, nothing else."
        )
        user_msg = self._build_scoring_prompt(prompt, completions, context)

        endpoint = resolve_endpoint(agent)
        api_key = resolve_api_key(agent)

        if agent.provider == "claude":
            body = {
                "model": agent.model,
                "max_tokens": 1024,
                "temperature": agent.temperature,
                "system": system_prompt,
                "messages": [{"role": "user", "content": user_msg}],
            }
            headers = {
                "x-api-key": api_key,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json",
            }
        else:
            # OpenAI-compatible (DeepSeek, Grok, OpenAI)
            body = {
                "model": agent.model,
                "temperature": agent.temperature,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_msg},
                ],
            }
            if agent.logprobs:
                body["logprobs"] = True
                body["top_logprobs"] = agent.top_logprobs
            headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            }

        meta: Dict[str, Any] = {}

        for attempt in range(self.config.agent_retry + 1):
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.post(
                        endpoint,
                        json=body,
                        headers=headers,
                        timeout=aiohttp.ClientTimeout(total=self.config.agent_timeout),
                    ) as resp:
                        resp.raise_for_status()
                        data = await resp.json()

                scores = self._parse_response(agent, data, len(completions))

                # extract logprobs if returned
                if agent.logprobs and agent.provider != "claude":
                    meta["logprobs"] = self._extract_logprobs(data)

                return scores, meta

            except Exception as e:
                if attempt == self.config.agent_retry:
                    raise
                log.warning(
                    "[OVEN] Agent '%s' attempt %d failed: %s — retrying",
                    agent.name, attempt + 1, e,
                )
                await asyncio.sleep(1.0 * (attempt + 1))

        # unreachable but satisfies typing
        return [0.5] * len(completions), meta

    def _parse_response(
        self, agent: AgentConfig, data: Dict, expected: int
    ) -> List[float]:
        """Extract scores from API response, handling Claude vs OpenAI shapes."""
        if agent.provider == "claude":
            text = data.get("content", [{}])[0].get("text", "[]")
        else:
            text = data["choices"][0]["message"]["content"]

        # clean and parse
        text = text.strip()
        if text.startswith("```"):
            text = text.split("\n", 1)[-1].rsplit("```", 1)[0]

        try:
            scores = json.loads(text)
        except json.JSONDecodeError:
            log.warning("[OVEN] Agent '%s' returned non-JSON: %s", agent.name, text[:200])
            return [0.5] * expected

        if not isinstance(scores, list) or len(scores) != expected:
            log.warning(
                "[OVEN] Agent '%s' score count mismatch: got %d, expected %d",
                agent.name, len(scores) if isinstance(scores, list) else -1, expected,
            )
            if isinstance(scores, list):
                scores = (scores + [0.5] * expected)[:expected]
            else:
                scores = [0.5] * expected

        return [max(0.0, min(1.0, float(s))) for s in scores]

    def _extract_logprobs(self, data: Dict) -> Optional[Dict]:
        """Pull top-k logprobs from OpenAI-compatible response."""
        try:
            choices = data.get("choices", [{}])
            lp_content = choices[0].get("logprobs", {}).get("content", [])
            if not lp_content:
                return None
            return {
                str(i): {
                    entry["token"]: entry["logprob"]
                    for entry in tok_info.get("top_logprobs", [])
                }
                for i, tok_info in enumerate(lp_content)
            }
        except Exception:
            return None

    def _build_scoring_prompt(
        self,
        prompt: str,
        completions: List[str],
        context: Optional[Dict],
    ) -> str:
        parts = [f"PROMPT:\n{prompt}\n"]
        for i, c in enumerate(completions):
            parts.append(f"--- Completion {i} ---\n{c}\n")
        if context:
            parts.append(f"\nTRAINING CONTEXT:\n{json.dumps(context, default=str)}")
        return "\n".join(parts)

    # ---- curriculum ----

    async def get_curriculum_suggestions(
        self,
        recent_prompts: List[str],
        recent_scores: List[float],
        metrics: Dict,
    ) -> List[str]:
        """Ask curriculum agent(s) for new training prompts."""
        if not self.curriculum_agents:
            return []

        agent = self.curriculum_agents[0]
        system_prompt = (
            f"You are a curriculum designer for model training.\n"
            f"Role: {agent.role}\n"
            f"Return ONLY a JSON array of 10 prompt strings."
        )
        user_msg = (
            f"Recent prompts and mean scores:\n"
            f"{json.dumps(list(zip(recent_prompts, recent_scores)), indent=2)}\n\n"
            f"Metrics:\n{json.dumps(metrics, default=str, indent=2)}\n\n"
            f"The model struggles where scores are low. "
            f"Generate 10 new prompts targeting weaknesses."
        )

        endpoint = resolve_endpoint(agent)
        api_key = resolve_api_key(agent)

        if agent.provider == "claude":
            body = {
                "model": agent.model,
                "max_tokens": 2048,
                "system": system_prompt,
                "messages": [{"role": "user", "content": user_msg}],
            }
            headers = {
                "x-api-key": api_key,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json",
            }
        else:
            body = {
                "model": agent.model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_msg},
                ],
            }
            headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            }

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    endpoint, json=body, headers=headers,
                    timeout=aiohttp.ClientTimeout(total=60.0),
                ) as resp:
                    data = await resp.json()

            if agent.provider == "claude":
                text = data.get("content", [{}])[0].get("text", "[]")
            else:
                text = data["choices"][0]["message"]["content"]

            text = text.strip()
            if text.startswith("```"):
                text = text.split("\n", 1)[-1].rsplit("```", 1)[0]
            return json.loads(text)
        except Exception as e:
            log.warning("[OVEN] Curriculum agent failed: %s", e)
            return []


# =============================================================================
# FRONTIER CALIBRATOR (diagnostic, not loss)
# =============================================================================

class FrontierCalibrator:
    """
    Uses top-k logprobs returned by scoring agents (DeepSeek/Grok)
    as a diagnostic drift signal. NOT the KL anchor.
    """

    def __init__(self, config: OvenConfig):
        self.enabled = config.frontier_calibration
        self.interval = config.calibration_interval
        self.source_agent = config.calibration_agent

    def compute_drift_diagnostic(
        self,
        policy_log_probs: torch.Tensor,       # (T, V) log-softmax'd
        frontier_top_k: Dict[str, Dict],       # {"position": {token_str: logprob}}
        tokenizer,
    ) -> Dict:
        """
        Partial KL estimate from top-k. Returns metrics dict, NOT a loss term.
        """
        if not frontier_top_k:
            return {"frontier_drift": False}

        divergences = []
        for pos_str, top_k_map in frontier_top_k.items():
            pos = int(pos_str)
            if pos >= policy_log_probs.size(0):
                continue
            for token_str, frontier_lp in top_k_map.items():
                tok_ids = tokenizer.encode(token_str, add_special_tokens=False)
                if not tok_ids:
                    continue
                tid = tok_ids[0]
                if tid >= policy_log_probs.size(-1):
                    continue
                policy_lp = policy_log_probs[pos, tid].item()
                p_frontier = torch.exp(torch.tensor(frontier_lp))
                kl_contrib = (p_frontier * (frontier_lp - policy_lp)).item()
                divergences.append(kl_contrib)

        if not divergences:
            return {"frontier_drift": False}

        mean_kl = sum(divergences) / len(divergences)
        return {
            "partial_kl_mean": mean_kl,
            "partial_kl_max": max(divergences),
            "tokens_compared": len(divergences),
            "frontier_drift": mean_kl > 0.5,
        }


# =============================================================================
# INLINE TEACHER IMPLEMENTATIONS
# In production, replace with:
#   from teacher_small_lm import SmallCausalTeacher
#   from big_causal_teacher import BigCausalTeacher
# =============================================================================

class _SmallCausalTeacher:
    """SmallCausalTeacher — mirrors your existing implementation."""

    def __init__(
        self,
        model_or_name: Optional[Any] = None,
        device: str = "cuda",
        tok: Optional[Any] = None,
    ):
        self.device = torch.device(device)
        self.enabled = False
        self._tok = tok
        self._lm = None

        if model_or_name is None:
            return
        try:
            from transformers import AutoModelForCausalLM
            name = (
                model_or_name if isinstance(model_or_name, str)
                else "Qwen/Qwen3-4B-Thinking-2507"
            )
            self._lm = AutoModelForCausalLM.from_pretrained(
                name, trust_remote_code=True,
            ).to(self.device).eval()
            self.enabled = True
        except Exception:
            log.exception("SmallCausalTeacher init failed")
            self.enabled = False

    @property
    def hf_tok(self):
        return self._tok

    @torch.inference_mode()
    def _forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Returns logits (1, T, V) on device."""
        if input_ids.dim() == 1:
            input_ids = input_ids.unsqueeze(0)
        return self._lm(input_ids.to(self.device)).logits.float()

    @torch.inference_mode()
    def _generate_ids(
        self, prompt: str, max_new_tokens=128, temperature=0.0, top_p=0.95
    ) -> torch.Tensor:
        if not self.enabled:
            return torch.zeros(0, dtype=torch.long, device=self.device)
        toks = self._tok(prompt, return_tensors="pt").to(self.device)
        out = self._lm.generate(
            **toks,
            do_sample=(temperature > 0.0),
            temperature=max(1e-5, temperature),
            top_p=top_p,
            max_new_tokens=max_new_tokens,
            pad_token_id=self._tok.eos_token_id or self._tok.pad_token_id,
        )
        return out[0, toks["input_ids"].size(1):]

    @torch.inference_mode()
    def draft(
        self, prompts: List[str], max_new_tokens=128,
        temperature=0.0, top_p=0.95,
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
        self, ctx_ids: torch.Tensor, tgt_ids: torch.Tensor
    ) -> torch.Tensor:
        if not self.enabled or ctx_ids.numel() == 0 or tgt_ids.numel() == 0:
            return torch.tensor(0.0, device=self.device)
        inp = torch.cat([ctx_ids, tgt_ids], dim=0).unsqueeze(0)
        logits = self._lm(inp.to(self.device)).logits[:, :-1, :]
        tgt = inp[:, 1:].to(self.device)
        lp = F.log_softmax(logits, dim=-1)
        lp = lp.gather(-1, tgt.unsqueeze(-1)).squeeze(-1)
        Tctx = ctx_ids.numel()
        return lp[:, Tctx:].sum(dim=-1).squeeze(0)

    def evict_from_gpu(self):
        """No-op for small teacher (stays resident)."""
        pass


class _BigCausalTeacher:
    """
    Wrapper that tries to import your actual BigCausalTeacher,
    falls back to simplified inline version.
    """

    def __init__(self, model_or_name=None, device="cuda", tok=None,
                 strategy="quantized", max_seq_len=512, hf_token=None):
        self.device = torch.device(device)
        self.enabled = False
        self._tok = tok
        self._lm = None
        self.strategy = strategy
        self._name = model_or_name

        if model_or_name is None:
            return

        try:
            # --- Try importing your actual BigCausalTeacher ---
            from big_causal_teacher import BigCausalTeacher
            _impl = BigCausalTeacher(
                model_or_name=model_or_name,
                device=device,
                tok=tok,
                strategy=strategy,
                max_seq_len=max_seq_len,
                hf_token=hf_token,
            )
            # proxy: steal the impl's internals
            self.__dict__.update(_impl.__dict__)
            self._external_impl = _impl
            return
        except ImportError:
            log.info("BigCausalTeacher module not found, using inline fallback")

        # --- Fallback: load directly ---
        try:
            from transformers import AutoModelForCausalLM, BitsAndBytesConfig
            if strategy == "quantized":
                bnb_cfg = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_use_double_quant=True,
                )
                self._lm = AutoModelForCausalLM.from_pretrained(
                    model_or_name,
                    quantization_config=bnb_cfg,
                    device_map={"": self.device},
                    trust_remote_code=True,
                    token=hf_token,
                ).eval()
            else:
                self._lm = AutoModelForCausalLM.from_pretrained(
                    model_or_name,
                    torch_dtype=torch.float16,
                    device_map="auto" if strategy == "hybrid" else "cpu",
                    trust_remote_code=True,
                    token=hf_token,
                ).eval()
            self.enabled = True
        except Exception:
            log.exception("BigCausalTeacher fallback init failed")
            self.enabled = False

    @torch.inference_mode()
    def _forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        # delegate to external impl if available
        if hasattr(self, "_external_impl"):
            return self._external_impl._forward(input_ids)
        if input_ids.dim() == 1:
            input_ids = input_ids.unsqueeze(0)
        return self._lm(input_ids.to(self.device)).logits.float()

    @torch.inference_mode()
    def draft(self, prompts, max_new_tokens=128, temperature=0.6, top_p=0.95):
        if hasattr(self, "_external_impl"):
            return self._external_impl.draft(
                prompts, max_new_tokens=max_new_tokens,
                temperature=temperature, top_p=top_p,
            )
        # simplified fallback
        texts, ids = [], []
        for p in prompts:
            if not self.enabled:
                texts.append("")
                ids.append(torch.zeros(0, dtype=torch.long, device=self.device))
                continue
            toks = self._tok(p, return_tensors="pt").to(self.device)
            out = self._lm.generate(
                **toks, do_sample=(temperature > 0),
                temperature=max(1e-5, temperature), top_p=top_p,
                max_new_tokens=max_new_tokens,
                pad_token_id=self._tok.eos_token_id or self._tok.pad_token_id,
            )
            gen = out[0, toks["input_ids"].size(1):]
            ids.append(gen)
            texts.append(self._tok.decode(gen, skip_special_tokens=True))
        return texts, ids

    @torch.inference_mode()
    def seq_logprob(self, ctx_ids, tgt_ids):
        if hasattr(self, "_external_impl"):
            return self._external_impl.seq_logprob(ctx_ids, tgt_ids)
        if not self.enabled or ctx_ids.numel() == 0 or tgt_ids.numel() == 0:
            return torch.tensor(0.0, device=self.device)
        inp = torch.cat([ctx_ids, tgt_ids], dim=0).unsqueeze(0)
        logits = self._forward(inp)[:, :-1, :]
        tgt = inp[:, 1:].to(self.device)
        lp = F.log_softmax(logits, dim=-1).gather(-1, tgt.unsqueeze(-1)).squeeze(-1)
        Tctx = ctx_ids.numel()
        return lp[:, Tctx:].sum(dim=-1).squeeze(0)

    def evict_from_gpu(self):
        if hasattr(self, "_external_impl"):
            self._external_impl.evict_from_gpu()
            return
        if self._lm is not None and self.strategy == "quantized":
            del self._lm
            self._lm = None
            self.enabled = False
        torch.cuda.empty_cache()


# =============================================================================
# PROMPT SOURCE
# =============================================================================

class PromptSource:
    """Prompt pool with injection from curriculum agent."""

    def __init__(self, seed_prompts: List[str]):
        self.prompts = list(seed_prompts)
        self._injected = 0

    def sample(self, n: int) -> List[str]:
        return random.sample(self.prompts, min(n, len(self.prompts)))

    def inject(self, new_prompts: List[str]):
        self.prompts.extend(new_prompts)
        self._injected += len(new_prompts)
        log.info(
            "[OVEN] Injected %d curriculum prompts (total pool: %d)",
            len(new_prompts), len(self.prompts),
        )


# =============================================================================
# SUPERINTELLIGENCE OVEN  — main trainer
# =============================================================================

class SuperintelligenceOven:
    """
    Main training loop.

    Heat sources:
      1. Policy model logits (generation + gradient)
      2. Local teacher KL anchor (full distribution, hot-swappable)
      3. Agent swarm reward (black-box, per-agent frontier model)
      4. QwenEmbedVerifier semantic reward (local, no gradient)
      5. Frontier logprob calibration (diagnostic top-k)

    Integration pattern:
        model.teacher = oven.teacher_mgr.current
        model.verifier = oven.verifier
    """

    def __init__(
        self,
        config: OvenConfig,
        policy_model: nn.Module,
        tokenizer=None,
        verifier=None,
        prompt_source: Optional[PromptSource] = None,
    ):
        self.config = config
        self.device = torch.device(config.device)

        # ---- shared tokenizer (Qwen2TokenizerFast) ----
        if tokenizer is not None:
            self.tokenizer = tokenizer
        else:
            from transformers import AutoTokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                config.tokenizer_name, trust_remote_code=True,
            )

        # ---- policy ----
        self.policy = policy_model.to(self.device)
        self.optimizer = torch.optim.AdamW(
            self.policy.parameters(), lr=config.lr,
        )

        # ---- teacher manager ----
        self.teacher_mgr = TeacherManager(config, shared_tok=self.tokenizer)
        self.teacher_mgr.swap_teacher(config.initial_teacher)

        # ---- verifier ----
        if verifier is not None:
            self.verifier = verifier
        else:
            try:
                from verifier_qwen_embed import QwenEmbedVerifier
                self.verifier = QwenEmbedVerifier(
                    model_name=config.verifier_model,
                    device=config.device,
                )
            except ImportError:
                self.verifier = None
                log.warning("[OVEN] QwenEmbedVerifier not available — skipping")

        # ---- agent swarm ----
        self.swarm = AgentSwarm(config)

        # ---- frontier calibrator ----
        self.calibrator = FrontierCalibrator(config)

        # ---- prompt source ----
        self.prompt_source = prompt_source

        # ---- metrics ----
        self.metrics_history: List[Dict] = []

    # ================================================================
    # Wire into model.teacher / model.verifier
    # ================================================================

    def attach_to_model(self, model):
        """Sets model.teacher and model.verifier to oven-managed instances."""
        model.teacher = self.teacher_mgr.current
        if self.verifier is not None:
            model.verifier = self.verifier

    # ================================================================
    # PUBLIC SWAP API
    # ================================================================

    def swap_teacher(self, name: str):
        """Swap local KL-anchor teacher by name."""
        self.teacher_mgr.swap_teacher(name)

    def swap_agent(
        self,
        agent_name: str,
        provider: Optional[str] = None,
        model: Optional[str] = None,
        **kw,
    ):
        """Swap the frontier model for a specific agent (1-5)."""
        self.swarm.swap_agent_model(agent_name, provider=provider, model=model, **kw)

    def swap_all_agents(self, provider: str, model: str, **kw):
        """Swap all scoring agents to the same frontier model."""
        self.swarm.swap_all_agents_model(provider, model, **kw)

    # ================================================================
    # GENERATION
    # ================================================================

    def _generate_group(self, prompt: str, n: int) -> List[Dict]:
        """
        Generate n completions from the policy model.
        Returns list of dicts: {text, full_ids, gen_ids, gen_logits, gen_log_probs, prompt_len}
        """
        group = []
        input_ids = self.tokenizer(
            prompt, return_tensors="pt", truncation=True,
            max_length=self.config.max_seq_len,
        )["input_ids"].to(self.device)
        prompt_len = input_ids.size(1)

        for _ in range(n):
            with torch.no_grad():
                output = self.policy.generate(
                    input_ids,
                    do_sample=True,
                    temperature=0.8,
                    top_p=0.95,
                    max_new_tokens=self.config.max_seq_len // 2,
                    return_dict_in_generate=True,
                    output_scores=True,
                    pad_token_id=(
                        self.tokenizer.eos_token_id
                        or self.tokenizer.pad_token_id
                    ),
                )

            gen_ids = output.sequences[0, prompt_len:]

            if len(output.scores) == 0:
                # model generated nothing
                continue

            scores = torch.stack(output.scores, dim=0)   # (gen_len, V)

            # per-token log-probs under generation policy
            gen_log_probs = F.log_softmax(scores, dim=-1)
            # clamp gen_ids length to match scores length
            effective_len = min(gen_ids.size(0), scores.size(0))
            gen_ids_eff = gen_ids[:effective_len]
            token_log_probs = gen_log_probs[:effective_len].gather(
                -1, gen_ids_eff.unsqueeze(-1)
            ).squeeze(-1)

            group.append({
                "text": self.tokenizer.decode(gen_ids_eff, skip_special_tokens=True),
                "full_ids": output.sequences[0, :prompt_len + effective_len],
                "gen_ids": gen_ids_eff,
                "gen_logits": scores[:effective_len],
                "gen_log_probs": token_log_probs,
                "prompt_len": prompt_len,
            })

        return group

    # ================================================================
    # SINGLE GRPO STEP
    # ================================================================

    async def _step(self, step_num: int, prompts: List[str]) -> Dict:
        """One GRPO training step."""
        step_metrics = defaultdict(list)
        total_loss = torch.tensor(0.0, device=self.device, requires_grad=True)
        n_items = 0
        last_diag = {}

        for prompt in prompts:
            # ---- generate group ----
            group = self._generate_group(prompt, self.config.group_size)
            if not group:
                continue
            texts = [c["text"] for c in group]

            # ---- agent swarm scoring (async, remote) ----
            rewards, diag = await self.swarm.score_completions(
                prompt, texts,
                context=self._training_context(step_num),
            )
            last_diag = diag

            # ---- optional: blend verifier semantic score ----
            if self.verifier is not None and self.config.verifier_weight > 0:
                teacher = self.teacher_mgr.current
                gold = ""
                if teacher and teacher.enabled:
                    gold_texts, _ = teacher.draft([prompt], max_new_tokens=128)
                    gold = gold_texts[0] if gold_texts else ""

                if gold:
                    vw = self.config.verifier_weight
                    for i, text in enumerate(texts):
                        v_score = self.verifier.support(
                            gold_cont=gold, teacher_cont=text,
                        )
                        rewards[i] = (1.0 - vw) * rewards[i] + vw * v_score

            rewards_t = torch.tensor(rewards, device=self.device)

            # ---- group-relative advantages ----
            if self.config.advantage_norm and rewards_t.std() > 1e-8:
                advantages = (rewards_t - rewards_t.mean()) / (rewards_t.std() + 1e-8)
            else:
                advantages = rewards_t - rewards_t.mean()

            # ---- per-completion policy gradient + KL ----
            for j, (comp, adv) in enumerate(zip(group, advantages)):
                with torch.enable_grad():
                    cur_logits = self.policy(
                        comp["full_ids"].unsqueeze(0)
                    ).logits[0, comp["prompt_len"] - 1:-1, :]

                    # match lengths (safety)
                    eff_len = min(cur_logits.size(0), comp["gen_ids"].size(0))
                    cur_logits_eff = cur_logits[:eff_len]
                    gen_ids_eff = comp["gen_ids"][:eff_len]
                    old_lp_eff = comp["gen_log_probs"][:eff_len]

                    cur_lp = F.log_softmax(cur_logits_eff, dim=-1)
                    cur_token_lp = cur_lp.gather(
                        -1, gen_ids_eff.unsqueeze(-1)
                    ).squeeze(-1)

                    seq_lp = cur_token_lp.sum()
                    old_seq_lp = old_lp_eff.sum().detach()

                    # PPO-style clipped ratio
                    ratio = torch.exp(seq_lp - old_seq_lp)
                    clipped = torch.clamp(
                        ratio,
                        1.0 - self.config.grpo_clip_eps,
                        1.0 + self.config.grpo_clip_eps,
                    )
                    policy_loss = -torch.min(ratio * adv, clipped * adv)

                    # KL anchor (local teacher, full distribution)
                    kl_loss = self.teacher_mgr.compute_kl(
                        cur_logits_eff, gen_ids_eff,
                    )

                    # effective kl_beta (with optional annealing)
                    if self.config.kl_anneal:
                        progress = step_num / max(self.config.total_steps, 1)
                        beta = (
                            self.config.kl_beta * (1.0 - progress)
                            + self.config.kl_beta_min * progress
                        )
                    else:
                        beta = self.config.kl_beta

                    comp_loss = policy_loss + beta * kl_loss
                    total_loss = total_loss + comp_loss
                    n_items += 1

                step_metrics["policy_loss"].append(policy_loss.item())
                step_metrics["kl_loss"].append(kl_loss.item())
                step_metrics["reward"].append(rewards[j])
                step_metrics["advantage"].append(adv.item())

        # ---- gradient update ----
        if n_items > 0:
            mean_loss = total_loss / n_items
            self.optimizer.zero_grad()
            mean_loss.backward()
            torch.nn.utils.clip_grad_norm_(
                self.policy.parameters(), self.config.grad_clip,
            )
            self.optimizer.step()

        # ---- frontier calibration (diagnostic) ----
        if (
            self.calibrator.enabled
            and step_num % self.calibrator.interval == 0
            and step_num > 0
        ):
            frontier_lps = last_diag.get("frontier_logprobs", {}).get(
                self.config.calibration_agent
            )
            if frontier_lps and group:
                sample_logits = group[0]["gen_logits"]
                sample_lp = F.log_softmax(sample_logits, dim=-1)
                cal = self.calibrator.compute_drift_diagnostic(
                    sample_lp, frontier_lps, self.tokenizer,
                )
                step_metrics.update({k: [v] for k, v in cal.items()})
                if cal.get("frontier_drift"):
                    log.warning("[OVEN] ⚠️  Frontier drift at step %d", step_num)

        # ---- aggregate ----
        agg = {}
        for k, v in step_metrics.items():
            if isinstance(v, list) and v and isinstance(v[0], (int, float)):
                agg[k] = sum(v) / len(v)
            else:
                agg[k] = v
        agg["total_loss"] = total_loss.item() / max(n_items, 1)
        agg["n_items"] = n_items
        return agg

    # ================================================================
    # MAIN TRAINING LOOP
    # ================================================================

    async def train(self, prompt_source: Optional[PromptSource] = None):
        """Run the full training loop."""
        ps = prompt_source or self.prompt_source
        if ps is None:
            raise ValueError("No prompt source provided")

        log.info(
            "[OVEN] Firing up — %d steps, group=%d, batch=%d, teacher=%s",
            self.config.total_steps, self.config.group_size,
            self.config.batch_size, self.teacher_mgr.current_name,
        )
        log.info("[OVEN] Agents: %s", [
            f"{a.name}->{a.provider}/{a.model}" for a in self.config.agents
        ])

        for step in range(self.config.total_steps):
            prompts = ps.sample(self.config.batch_size)
            metrics = await self._step(step, prompts)
            self.metrics_history.append(metrics)

            # ---- teacher auto-swap ----
            metrics["kl_mean"] = metrics.get("kl_loss", 1.0)
            swapped = self.teacher_mgr.maybe_auto_swap(step, metrics)
            if swapped:
                log.info("[OVEN] Auto-swapped teacher at step %d", step)

            # ---- curriculum injection ----
            if (
                step % self.config.curriculum_interval == 0
                and step > 0
                and self.swarm.curriculum_agents
            ):
                recent_rewards = [
                    m.get("reward", 0.5)
                    for m in self.metrics_history[-self.config.curriculum_interval:]
                ]
                new_prompts = await self.swarm.get_curriculum_suggestions(
                    prompts, recent_rewards, metrics,
                )
                if new_prompts:
                    ps.inject(new_prompts)

            # ---- logging ----
            if step % 10 == 0:
                log.info(
                    "[OVEN] Step %5d | Loss: %.4f | Reward: %.3f | "
                    "KL: %.4f | Teacher: %s | Items: %d",
                    step,
                    metrics.get("total_loss", 0),
                    metrics.get("reward", 0),
                    metrics.get("kl_loss", 0),
                    self.teacher_mgr.current_name,
                    metrics.get("n_items", 0),
                )

        log.info("[OVEN] Training complete. %d steps.", self.config.total_steps)
        return self.metrics_history

    def _training_context(self, step: int) -> Dict:
        """Context sent to agents so they can adapt scoring."""
        return {
            "step": step,
            "total_steps": self.config.total_steps,
            "teacher": self.teacher_mgr.current_name,
            "recent_metrics": self.metrics_history[-5:] if self.metrics_history else [],
        }


# =============================================================================
# USAGE EXAMPLE
# =============================================================================

def example_usage():
    """
    Full wiring example.
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer

    # ---- shared tokenizer (Qwen2TokenizerFast) ----
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B", trust_remote_code=True)

    # ---- your policy model ----
    policy = AutoModelForCausalLM.from_pretrained(
        "your-model-path", trust_remote_code=True,
    ).to("cuda")

    # ---- config ----
    config = OvenConfig(
        group_size=8,
        batch_size=2,
        total_steps=5000,
        kl_beta=0.1,
        initial_teacher="small_qwen3_4b",
    )

    # ---- build oven ----
    oven = SuperintelligenceOven(
        config=config,
        policy_model=policy,
        tokenizer=tok,
    )

    # ---- wire into model.teacher / model.verifier ----
    oven.attach_to_model(policy)

    # ---- swap each agent independently ----
    oven.swap_agent("critic",     provider="claude",   model="claude-sonnet-4-20250514")
    oven.swap_agent("adversary",  provider="deepseek", model="deepseek-reasoner", logprobs=True)
    oven.swap_agent("specialist", provider="grok",     model="grok-3", logprobs=True, top_logprobs=20)
    oven.swap_agent("style",      provider="claude",   model="claude-haiku-4-5-20251001")
    oven.swap_agent("curriculum", provider="claude",   model="claude-sonnet-4-20250514")

    # ---- or swap all at once ----
    # oven.swap_all_agents(provider="deepseek", model="deepseek-chat")

    # ---- swap teacher mid-run ----
    # oven.swap_teacher("big_qwen3_8b_q4")

    # ---- register a new teacher at runtime ----
    # oven.teacher_mgr.register_teacher(TeacherConfig(
    #     name="big_qwen3_32b",
    #     kind="big",
    #     model_name="Qwen/Qwen3-32B",
    #     strategy="hybrid",
    # ))

    # ---- prompts ----
    prompts = PromptSource([
        "Explain quantum entanglement simply.",
        "Write a Python function to merge two sorted lists.",
        "What are the trade-offs of GRPO vs PPO?",
        "Derive the backpropagation update rule for a two-layer MLP.",
        "Write a short story about a robot learning to dream.",
    ])

    # ---- run ----
    asyncio.run(oven.train(prompts))


if __name__ == "__main__":
    example_usage()
