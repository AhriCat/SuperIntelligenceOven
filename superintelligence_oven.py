"""
SUPERINTELLIGENCE OVEN  v0.2
=============================
GRPO training loop with:
  - Local policy model (trainable, any architecture)
  - Hot-swappable local teachers (SmallCausalTeacher / BigCausalTeacher / SmallCausalDiffusionTeacher)
  - QwenEmbedVerifier as semantic reward signal (text-only, auto-skipped for non-text)
  - Remote agent swarm (1-5 agents, each with independently configurable frontier model)
  - Optional frontier logprob calibration (DeepSeek / Grok top-k)
  - Model-agnostic wrapper: pass any nn.Module + ModelDescription

Flat repo imports:
  - big_teacher.py          → BigCausalTeacher
  - small_causal_teacher.py → SmallCausalTeacher
  - d_teacher.py            → SmallCausalDiffusionTeacher
  - verifier.py             → QwenEmbedVerifier

Usage:
    from superintelligence_oven import bake
    oven = bake(model=your_model, name="my-model", description="...", prompts=[...])
    oven.run_sync()
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from collections import defaultdict
from contextlib import contextmanager
import asyncio
import aiohttp
import json
import os
import time
import logging
import random

# ---------------------------------------------------------------------------
# Imports from flat repo
# ---------------------------------------------------------------------------
from big_teacher import BigCausalTeacher
from small_causal_teacher import SmallCausalTeacher
from d_teacher import SmallCausalDiffusionTeacher
from verifier import QwenEmbedVerifier

log = logging.getLogger("oven")
log.setLevel(logging.INFO)
if not log.handlers:
    log.addHandler(logging.StreamHandler())


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


def resolve_endpoint(agent: "AgentConfig") -> str:
    if agent.endpoint:
        return agent.endpoint
    return PROVIDER_ENDPOINTS.get(agent.provider, agent.endpoint)


def resolve_api_key(agent: "AgentConfig") -> str:
    env = agent.api_key_env or PROVIDER_KEY_ENVS.get(agent.provider, "")
    return os.environ.get(env, "")


# =============================================================================
# CONFIG DATACLASSES
# =============================================================================

@dataclass
class AgentConfig:
    """Per-agent configuration — each agent can hit a different frontier model."""
    name: str
    role: str
    weight: float = 0.20
    provider: str = "claude"
    model: str = "claude-sonnet-4-20250514"
    endpoint: str = ""
    api_key_env: str = ""
    temperature: float = 0.1
    logprobs: bool = False
    top_logprobs: int = 20
    is_curriculum: bool = False


DEFAULT_AGENTS = [
    AgentConfig(
        name="critic",
        role="Score correctness, coherence, and instruction-following",
        weight=0.30, provider="claude", model="claude-sonnet-4-20250514",
    ),
    AgentConfig(
        name="adversary",
        role="Probe for failure modes, hallucinations, and reward-hackable outputs",
        weight=0.20, provider="claude", model="claude-sonnet-4-20250514",
    ),
    AgentConfig(
        name="specialist",
        role="Score domain-specific accuracy and depth",
        weight=0.20, provider="deepseek", model="deepseek-chat", logprobs=True,
    ),
    AgentConfig(
        name="style",
        role="Score clarity, tone, and formatting quality",
        weight=0.15, provider="grok", model="grok-3-mini", logprobs=True,
    ),
    AgentConfig(
        name="curriculum",
        role="Analyze training dynamics and suggest next training prompts",
        weight=0.15, provider="claude", model="claude-sonnet-4-20250514",
        is_curriculum=True,
    ),
]


@dataclass
class TeacherConfig:
    """Configuration for a local teacher (KL anchor)."""
    name: str
    kind: str                   # "small" | "big" | "diffusion"
    model_name: str
    strategy: str = "quantized" # big only: "quantized" | "offload" | "hybrid"
    priority: str = "default"
    hf_token: Optional[str] = None
    # Diffusion-specific
    joint_dim: int = 72
    intent_dim: int = 512
    timesteps: int = 100
    hidden_dim: int = 4096


DEFAULT_TEACHERS = [
    TeacherConfig(
        name="small_qwen3_4b", kind="small",
        model_name="Qwen/Qwen3-4B-Thinking-2507", priority="fast_iteration",
    ),
    TeacherConfig(
        name="big_qwen3_8b_q4", kind="big",
        model_name="Qwen/Qwen3-8B", strategy="quantized", priority="deep_reasoning",
    ),
    TeacherConfig(
        name="big_qwen3_8b_hybrid", kind="big",
        model_name="Qwen/Qwen3-8B", strategy="hybrid", priority="balanced",
    ),
]


@dataclass
class ModelDescription:
    """
    Describes the model being trained.
    Converted to a prompt and injected into every agent call so they
    know what kind of GRPO they're scoring for.
    """
    name: str = "unnamed-model"
    modality: str = "text"              # "text" | "diffusion" | "multimodal" | "motion"
    architecture: str = ""
    domain: str = ""
    description: str = ""
    output_format: str = "text"         # "text" | "trajectory" | "image" | "tokens"
    scoring_guidance: str = ""
    use_verifier: bool = True           # text-only; auto-False for non-text

    def to_agent_prompt(self) -> str:
        parts = [f"MODEL BEING TRAINED: {self.name}", f"Modality: {self.modality}"]
        if self.architecture:
            parts.append(f"Architecture: {self.architecture}")
        if self.domain:
            parts.append(f"Domain: {self.domain}")
        if self.description:
            parts.append(f"Description: {self.description}")
        parts.append(f"Output format: {self.output_format}")
        if self.scoring_guidance:
            parts.append(f"\nSCORING GUIDANCE:\n{self.scoring_guidance}")
        return "\n".join(parts)


@dataclass
class OvenConfig:
    # GRPO
    group_size: int = 8
    max_seq_len: int = 2048
    grpo_clip_eps: float = 0.2
    advantage_norm: bool = True

    # KL
    kl_beta: float = 0.1
    kl_anneal: bool = False
    kl_beta_min: float = 0.01

    # Verifier (text-only — auto-disabled for non-text via ModelDescription)
    verifier_weight: float = 0.1
    verifier_model: str = "Qwen/Qwen3-Embedding-0.6B"

    # Teachers
    teachers: List[TeacherConfig] = field(default_factory=lambda: DEFAULT_TEACHERS)
    initial_teacher: str = "small_qwen3_4b"
    teacher_swap_interval: int = 500
    teacher_kl_collapse_threshold: float = 0.01

    # Agent swarm
    agents: List[AgentConfig] = field(default_factory=lambda: DEFAULT_AGENTS)
    agent_timeout: float = 30.0
    agent_retry: int = 2

    # Frontier calibration
    frontier_calibration: bool = True
    calibration_interval: int = 50
    calibration_agent: str = "specialist"

    # Training
    lr: float = 1e-6
    batch_size: int = 4
    total_steps: int = 10000
    grad_clip: float = 1.0
    curriculum_interval: int = 100

    # Shared tokenizer
    tokenizer_name: str = "Qwen/Qwen2.5-0.5B"

    # Model description
    model_description: ModelDescription = field(default_factory=ModelDescription)

    # Device
    device: str = "cuda"


# =============================================================================
# TEACHER MANAGER
# =============================================================================

class TeacherManager:
    """
    Hot-swappable teacher pool.
    Wraps SmallCausalTeacher, BigCausalTeacher, SmallCausalDiffusionTeacher.
    Active teacher: `.current`
    """

    def __init__(self, config: OvenConfig, shared_tok=None):
        self.config = config
        self.shared_tok = shared_tok
        self.device = torch.device(config.device)
        self._pool: Dict[str, Any] = {}
        self._configs: Dict[str, TeacherConfig] = {
            t.name: t for t in config.teachers
        }
        self.current: Optional[Any] = None
        self.current_name: Optional[str] = None

    def _load(self, name: str) -> Any:
        if name in self._pool:
            return self._pool[name]
        tc = self._configs[name]
        if tc.kind == "small":
            teacher = SmallCausalTeacher(
                model_or_name=tc.model_name,
                device=str(self.device),
                tok=self.shared_tok,
            )
        elif tc.kind == "big":
            teacher = BigCausalTeacher(
                model_or_name=tc.model_name,
                device=str(self.device),
                tok=self.shared_tok,
                strategy=tc.strategy,
                max_seq_len=self.config.max_seq_len,
                hf_token=tc.hf_token,
            )
        elif tc.kind == "diffusion":
            teacher = SmallCausalDiffusionTeacher(
                joint_dim=tc.joint_dim,
                intent_dim=tc.intent_dim,
                timesteps=tc.timesteps,
                hidden_dim=tc.hidden_dim,
            )
            if hasattr(teacher, "enabled") and teacher.enabled:
                teacher.to(self.device)
        else:
            raise ValueError(f"Unknown teacher kind: {tc.kind!r}")
        self._pool[name] = teacher
        return teacher

    def swap_teacher(self, name: str):
        """Hot-swap. Evicts previous big/diffusion teacher from GPU."""
        if name not in self._configs:
            raise KeyError(
                f"Unknown teacher {name!r}. Available: {list(self._configs.keys())}"
            )
        if self.current is not None and hasattr(self.current, "evict_from_gpu"):
            self.current.evict_from_gpu()
        self.current = self._load(name)
        self.current_name = name
        log.info("[OVEN] Teacher swapped -> %s (%s)", name, self._configs[name].kind)

    def register_teacher(self, tc: TeacherConfig):
        """Add a new teacher config at runtime."""
        self._configs[tc.name] = tc
        log.info("[OVEN] Registered teacher: %s (kind=%s)", tc.name, tc.kind)

    @property
    def current_kind(self) -> Optional[str]:
        if self.current_name and self.current_name in self._configs:
            return self._configs[self.current_name].kind
        return None

    def list_teachers(self) -> Dict[str, str]:
        return {n: c.kind for n, c in self._configs.items()}

    @torch.inference_mode()
    def compute_kl(
        self,
        policy_logits: torch.Tensor,
        token_ids: torch.Tensor,
    ) -> torch.Tensor:
        """
        KL(policy || teacher) — full local distributions.
        For diffusion teachers: returns 0.0 (use trajectory_logprob separately).
        """
        if self.current is None or not self.current.enabled:
            return torch.tensor(0.0, device=self.device)

        if self.current_kind == "diffusion":
            return torch.tensor(0.0, device=self.device)

        if policy_logits.dim() == 2:
            policy_logits = policy_logits.unsqueeze(0)
        if token_ids.dim() == 1:
            token_ids = token_ids.unsqueeze(0)

        teacher_logits = self.current._forward(token_ids)

        policy_lp = F.log_softmax(policy_logits, dim=-1)
        teacher_lp = F.log_softmax(teacher_logits, dim=-1)
        teacher_p = teacher_lp.exp()

        kl = (teacher_p * (teacher_lp - policy_lp)).sum(dim=-1).mean()
        return kl

    def maybe_auto_swap(self, step: int, metrics: Dict) -> bool:
        if step % self.config.teacher_swap_interval != 0:
            return False
        kl_mean = metrics.get("kl_mean", 1.0)
        if self.current_name not in self._configs:
            return False
        current_cfg = self._configs[self.current_name]

        # small absorbed -> escalate to big
        if (
            current_cfg.kind == "small"
            and kl_mean < self.config.teacher_kl_collapse_threshold
        ):
            big_teachers = [n for n, c in self._configs.items() if c.kind == "big"]
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
    """Remote reward committee. Each agent independently swappable."""

    def __init__(self, config: OvenConfig):
        self.config = config
        self.agents: Dict[str, AgentConfig] = {a.name: a for a in config.agents}
        self.scoring_agents = [a for a in config.agents if not a.is_curriculum]
        self.curriculum_agents = [a for a in config.agents if a.is_curriculum]
        self.score_history: Dict[str, List] = defaultdict(list)
        self._model_description_prompt: str = ""

    def set_model_description(self, prompt: str):
        self._model_description_prompt = prompt

    # ---- per-agent swap ----

    def swap_agent_model(
        self, agent_name: str,
        provider: Optional[str] = None, model: Optional[str] = None,
        endpoint: Optional[str] = None, logprobs: Optional[bool] = None,
        top_logprobs: Optional[int] = None,
    ):
        if agent_name not in self.agents:
            raise KeyError(f"Unknown agent {agent_name!r}. Available: {list(self.agents.keys())}")
        a = self.agents[agent_name]
        if provider is not None:    a.provider = provider
        if model is not None:       a.model = model
        if endpoint is not None:    a.endpoint = endpoint
        if logprobs is not None:    a.logprobs = logprobs
        if top_logprobs is not None: a.top_logprobs = top_logprobs
        log.info("[OVEN] Agent '%s' -> %s/%s (logprobs=%s)", agent_name, a.provider, a.model, a.logprobs)

    def swap_all_agents_model(self, provider: str, model: str, **kw):
        for a in self.scoring_agents:
            self.swap_agent_model(a.name, provider=provider, model=model, **kw)

    # ---- scoring ----

    async def score_completions(
        self, prompt: str, completions: List[str], context: Optional[Dict] = None,
    ) -> Tuple[List[float], Dict]:
        tasks = [self._query_agent(a, prompt, completions, context) for a in self.scoring_agents]
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
            if meta.get("logprobs"):
                diagnostics["frontier_logprobs"][agent.name] = meta["logprobs"]

        if total_weight > 0:
            composite = [s / total_weight for s in composite]
        return composite, diagnostics

    async def _query_agent(
        self, agent: AgentConfig, prompt: str,
        completions: List[str], context: Optional[Dict],
    ) -> Tuple[List[float], Dict]:
        system_prompt = self._build_system_prompt(agent)
        user_msg = self._build_scoring_prompt(prompt, completions, context)
        endpoint = resolve_endpoint(agent)
        api_key = resolve_api_key(agent)

        if agent.provider == "claude":
            body = {
                "model": agent.model, "max_tokens": 1024,
                "temperature": agent.temperature, "system": system_prompt,
                "messages": [{"role": "user", "content": user_msg}],
            }
            headers = {
                "x-api-key": api_key, "anthropic-version": "2023-06-01",
                "content-type": "application/json",
            }
        else:
            body = {
                "model": agent.model, "temperature": agent.temperature,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_msg},
                ],
            }
            if agent.logprobs:
                body["logprobs"] = True
                body["top_logprobs"] = agent.top_logprobs
            headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}

        meta: Dict[str, Any] = {}
        for attempt in range(self.config.agent_retry + 1):
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.post(
                        endpoint, json=body, headers=headers,
                        timeout=aiohttp.ClientTimeout(total=self.config.agent_timeout),
                    ) as resp:
                        resp.raise_for_status()
                        data = await resp.json()
                scores = self._parse_response(agent, data, len(completions))
                if agent.logprobs and agent.provider != "claude":
                    meta["logprobs"] = self._extract_logprobs(data)
                return scores, meta
            except Exception as e:
                if attempt == self.config.agent_retry:
                    raise
                log.warning("[OVEN] Agent '%s' attempt %d failed: %s", agent.name, attempt + 1, e)
                await asyncio.sleep(1.0 * (attempt + 1))
        return [0.5] * len(completions), meta

    def _build_system_prompt(self, agent: AgentConfig) -> str:
        parts = [
            "You are a training reward agent.",
            f"Role: {agent.role}",
        ]
        if self._model_description_prompt:
            parts.insert(1, f"\n{self._model_description_prompt}\n")
        parts += [
            "Score each completion 0.0 to 1.0. Be precise — spread scores.",
            "Return ONLY a JSON array of floats, nothing else.",
        ]
        return "\n".join(parts)

    def _build_scoring_prompt(self, prompt, completions, context):
        parts = [f"PROMPT:\n{prompt}\n"]
        for i, c in enumerate(completions):
            parts.append(f"--- Completion {i} ---\n{c}\n")
        if context:
            parts.append(f"\nTRAINING CONTEXT:\n{json.dumps(context, default=str)}")
        return "\n".join(parts)

    def _parse_response(self, agent, data, expected):
        if agent.provider == "claude":
            text = data.get("content", [{}])[0].get("text", "[]")
        else:
            text = data["choices"][0]["message"]["content"]
        text = text.strip()
        if text.startswith("```"):
            text = text.split("\n", 1)[-1].rsplit("```", 1)[0]
        try:
            scores = json.loads(text)
        except json.JSONDecodeError:
            log.warning("[OVEN] Agent '%s' non-JSON: %s", agent.name, text[:200])
            return [0.5] * expected
        if not isinstance(scores, list) or len(scores) != expected:
            if isinstance(scores, list):
                scores = (scores + [0.5] * expected)[:expected]
            else:
                scores = [0.5] * expected
        return [max(0.0, min(1.0, float(s))) for s in scores]

    def _extract_logprobs(self, data):
        try:
            lp_content = data["choices"][0].get("logprobs", {}).get("content", [])
            if not lp_content:
                return None
            return {
                str(i): {e["token"]: e["logprob"] for e in t.get("top_logprobs", [])}
                for i, t in enumerate(lp_content)
            }
        except Exception:
            return None

    # ---- curriculum ----

    async def get_curriculum_suggestions(self, recent_prompts, recent_scores, metrics):
        if not self.curriculum_agents:
            return []
        agent = self.curriculum_agents[0]
        system_prompt = f"You are a curriculum designer for model training.\nRole: {agent.role}\n"
        if self._model_description_prompt:
            system_prompt += f"\n{self._model_description_prompt}\n"
        system_prompt += "Return ONLY a JSON array of 10 prompt strings."
        user_msg = (
            f"Recent prompts and mean scores:\n"
            f"{json.dumps(list(zip(recent_prompts, recent_scores)), indent=2)}\n\n"
            f"Metrics:\n{json.dumps(metrics, default=str, indent=2)}\n\n"
            f"Generate 10 new prompts targeting the model's weaknesses."
        )
        endpoint = resolve_endpoint(agent)
        api_key = resolve_api_key(agent)
        if agent.provider == "claude":
            body = {"model": agent.model, "max_tokens": 2048, "system": system_prompt,
                    "messages": [{"role": "user", "content": user_msg}]}
            headers = {"x-api-key": api_key, "anthropic-version": "2023-06-01", "content-type": "application/json"}
        else:
            body = {"model": agent.model, "messages": [
                {"role": "system", "content": system_prompt}, {"role": "user", "content": user_msg}]}
            headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(endpoint, json=body, headers=headers,
                                        timeout=aiohttp.ClientTimeout(total=60.0)) as resp:
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
            log.warning("[OVEN] Curriculum failed: %s", e)
            return []


# =============================================================================
# FRONTIER CALIBRATOR
# =============================================================================

class FrontierCalibrator:
    """Diagnostic drift detection from top-k logprobs. NOT a loss term."""

    def __init__(self, config: OvenConfig):
        self.enabled = config.frontier_calibration
        self.interval = config.calibration_interval
        self.source_agent = config.calibration_agent

    def compute_drift_diagnostic(self, policy_log_probs, frontier_top_k, tokenizer):
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
# PROMPT SOURCE
# =============================================================================

class PromptSource:
    def __init__(self, seed_prompts: List[str]):
        self.prompts = list(seed_prompts)
        self._injected = 0

    def sample(self, n: int) -> List[str]:
        return random.sample(self.prompts, min(n, len(self.prompts)))

    def inject(self, new_prompts: List[str]):
        self.prompts.extend(new_prompts)
        self._injected += len(new_prompts)
        log.info("[OVEN] Injected %d curriculum prompts (pool: %d)", len(new_prompts), len(self.prompts))

    def __len__(self):
        return len(self.prompts)


# =============================================================================
# SUPERINTELLIGENCE OVEN — main trainer
# =============================================================================

class SuperintelligenceOven:
    """
    Main training loop.

    Heat sources:
      1. Policy model logits (generation + gradient)
      2. Local teacher KL anchor (full distribution, hot-swappable)
      3. Agent swarm reward (black-box, per-agent frontier model)
      4. QwenEmbedVerifier semantic reward (local, text-only)
      5. Frontier logprob calibration (diagnostic top-k)

    Verifier is only used for text outputs. It catches garbage (repeated chars,
    noise, off-topic) before expensive API scoring — saving compute so teachers
    don't have to waste cycles on obviously bad completions.
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
        self.optimizer = torch.optim.AdamW(self.policy.parameters(), lr=config.lr)

        # ---- teacher manager (uses real imports) ----
        self.teacher_mgr = TeacherManager(config, shared_tok=self.tokenizer)
        self.teacher_mgr.swap_teacher(config.initial_teacher)

        # ---- verifier (text-only, auto-skipped for non-text) ----
        self.verifier = verifier
        if verifier is None and config.model_description.use_verifier:
            try:
                self.verifier = QwenEmbedVerifier(
                    model_name=config.verifier_model,
                    device=config.device,
                )
                log.info("[OVEN] QwenEmbedVerifier loaded: %s", config.verifier_model)
            except Exception:
                self.verifier = None
                log.warning("[OVEN] QwenEmbedVerifier not available — skipping")
        elif not config.model_description.use_verifier:
            self.verifier = None
            log.info("[OVEN] Verifier disabled (non-text modality: %s)", config.model_description.modality)

        # ---- agent swarm ----
        self.swarm = AgentSwarm(config)
        if config.model_description.description:
            self.swarm.set_model_description(config.model_description.to_agent_prompt())

        # ---- frontier calibrator ----
        self.calibrator = FrontierCalibrator(config)

        # ---- prompt source ----
        self.prompt_source = prompt_source

        # ---- metrics ----
        self.metrics_history: List[Dict] = []

    # ---- wire into model.teacher / model.verifier ----

    def attach_to_model(self, model):
        model.teacher = self.teacher_mgr.current
        if self.verifier is not None:
            model.verifier = self.verifier

    # ---- public swap API ----

    def swap_teacher(self, name: str):
        self.teacher_mgr.swap_teacher(name)

    def swap_agent(self, agent_name: str, **kw):
        self.swarm.swap_agent_model(agent_name, **kw)

    def swap_all_agents(self, provider: str, model: str, **kw):
        self.swarm.swap_all_agents_model(provider, model, **kw)

    # ---- generation ----

    def _generate_group(self, prompt: str, n: int) -> List[Dict]:
        group = []
        input_ids = self.tokenizer(
            prompt, return_tensors="pt", truncation=True,
            max_length=self.config.max_seq_len,
        )["input_ids"].to(self.device)
        prompt_len = input_ids.size(1)

        for _ in range(n):
            with torch.no_grad():
                output = self.policy.generate(
                    input_ids, do_sample=True, temperature=0.8, top_p=0.95,
                    max_new_tokens=self.config.max_seq_len // 2,
                    return_dict_in_generate=True, output_scores=True,
                    pad_token_id=(self.tokenizer.eos_token_id or self.tokenizer.pad_token_id),
                )
            gen_ids = output.sequences[0, prompt_len:]
            if len(output.scores) == 0:
                continue
            scores = torch.stack(output.scores, dim=0)
            effective_len = min(gen_ids.size(0), scores.size(0))
            gen_ids_eff = gen_ids[:effective_len]
            gen_log_probs = F.log_softmax(scores[:effective_len], dim=-1)
            token_log_probs = gen_log_probs.gather(-1, gen_ids_eff.unsqueeze(-1)).squeeze(-1)

            group.append({
                "text": self.tokenizer.decode(gen_ids_eff, skip_special_tokens=True),
                "full_ids": output.sequences[0, :prompt_len + effective_len],
                "gen_ids": gen_ids_eff,
                "gen_logits": scores[:effective_len],
                "gen_log_probs": token_log_probs,
                "prompt_len": prompt_len,
            })
        return group

    # ---- GRPO step ----

    async def _step(self, step_num: int, prompts: List[str]) -> Dict:
        step_metrics = defaultdict(list)
        total_loss = torch.tensor(0.0, device=self.device, requires_grad=True)
        n_items = 0
        last_diag = {}

        for prompt in prompts:
            group = self._generate_group(prompt, self.config.group_size)
            if not group:
                continue
            texts = [c["text"] for c in group]

            # ---- agent swarm scoring ----
            rewards, diag = await self.swarm.score_completions(
                prompt, texts, context=self._training_context(step_num),
            )
            last_diag = diag

            # ---- verifier blend (text-only, saves compute) ----
            # Verifier catches garbage so teachers/agents don't waste cycles
            # on obviously bad completions (repeated chars, noise, etc.)
            if self.verifier is not None and self.config.verifier_weight > 0:
                teacher = self.teacher_mgr.current
                gold = ""
                if teacher and teacher.enabled and hasattr(teacher, "draft"):
                    try:
                        gold_texts, _ = teacher.draft([prompt], max_new_tokens=128)
                        gold = gold_texts[0] if gold_texts else ""
                    except Exception:
                        gold = ""

                if gold:
                    vw = self.config.verifier_weight
                    # batch support for efficiency
                    v_scores = self.verifier.batch_support(gold, texts)
                    for i in range(len(texts)):
                        rewards[i] = (1.0 - vw) * rewards[i] + vw * v_scores[i]

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

                    eff_len = min(cur_logits.size(0), comp["gen_ids"].size(0))
                    cur_logits_eff = cur_logits[:eff_len]
                    gen_ids_eff = comp["gen_ids"][:eff_len]
                    old_lp_eff = comp["gen_log_probs"][:eff_len]

                    cur_lp = F.log_softmax(cur_logits_eff, dim=-1)
                    cur_token_lp = cur_lp.gather(-1, gen_ids_eff.unsqueeze(-1)).squeeze(-1)

                    seq_lp = cur_token_lp.sum()
                    old_seq_lp = old_lp_eff.sum().detach()

                    ratio = torch.exp(seq_lp - old_seq_lp)
                    clipped = torch.clamp(ratio, 1.0 - self.config.grpo_clip_eps, 1.0 + self.config.grpo_clip_eps)
                    policy_loss = -torch.min(ratio * adv, clipped * adv)

                    kl_loss = self.teacher_mgr.compute_kl(cur_logits_eff, gen_ids_eff)

                    if self.config.kl_anneal:
                        progress = step_num / max(self.config.total_steps, 1)
                        beta = self.config.kl_beta * (1.0 - progress) + self.config.kl_beta_min * progress
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
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.config.grad_clip)
            self.optimizer.step()

        # ---- frontier calibration ----
        if (self.calibrator.enabled and step_num % self.calibrator.interval == 0 and step_num > 0):
            frontier_lps = last_diag.get("frontier_logprobs", {}).get(self.config.calibration_agent)
            if frontier_lps and group:
                sample_lp = F.log_softmax(group[0]["gen_logits"], dim=-1)
                cal = self.calibrator.compute_drift_diagnostic(sample_lp, frontier_lps, self.tokenizer)
                step_metrics.update({k: [v] for k, v in cal.items()})
                if cal.get("frontier_drift"):
                    log.warning("[OVEN] ⚠️  Frontier drift at step %d", step_num)

        agg = {}
        for k, v in step_metrics.items():
            if isinstance(v, list) and v and isinstance(v[0], (int, float)):
                agg[k] = sum(v) / len(v)
            else:
                agg[k] = v
        agg["total_loss"] = total_loss.item() / max(n_items, 1)
        agg["n_items"] = n_items
        return agg

    # ---- main loop ----

    async def train(self, prompt_source: Optional[PromptSource] = None):
        ps = prompt_source or self.prompt_source
        if ps is None:
            raise ValueError("No prompt source provided")

        log.info(
            "[OVEN] 🔥 Firing up — %d steps, group=%d, batch=%d, teacher=%s, verifier=%s",
            self.config.total_steps, self.config.group_size,
            self.config.batch_size, self.teacher_mgr.current_name,
            "ON" if self.verifier else "OFF",
        )
        log.info("[OVEN] Agents: %s", [f"{a.name}->{a.provider}/{a.model}" for a in self.config.agents])
        log.info("[OVEN] Model: %s", self.config.model_description.to_agent_prompt())

        for step in range(self.config.total_steps):
            prompts = ps.sample(self.config.batch_size)
            metrics = await self._step(step, prompts)
            self.metrics_history.append(metrics)

            metrics["kl_mean"] = metrics.get("kl_loss", 1.0)
            self.teacher_mgr.maybe_auto_swap(step, metrics)

            if (step % self.config.curriculum_interval == 0 and step > 0 and self.swarm.curriculum_agents):
                recent_rewards = [m.get("reward", 0.5) for m in self.metrics_history[-self.config.curriculum_interval:]]
                new_prompts = await self.swarm.get_curriculum_suggestions(prompts, recent_rewards, metrics)
                if new_prompts:
                    ps.inject(new_prompts)

            if step % 10 == 0:
                log.info(
                    "[OVEN] Step %5d | Loss: %.4f | Reward: %.3f | KL: %.4f | Teacher: %s",
                    step, metrics.get("total_loss", 0), metrics.get("reward", 0),
                    metrics.get("kl_loss", 0), self.teacher_mgr.current_name,
                )

        log.info("[OVEN] 🔥 Training complete. %d steps.", self.config.total_steps)
        return self.metrics_history

    def _training_context(self, step):
        ctx = {
            "step": step, "total_steps": self.config.total_steps,
            "teacher": self.teacher_mgr.current_name,
            "recent_metrics": self.metrics_history[-5:] if self.metrics_history else [],
        }
        if self.config.model_description.description:
            ctx["model_description"] = self.config.model_description.to_agent_prompt()
        return ctx


# =============================================================================
# MODEL-AGNOSTIC WRAPPER
# =============================================================================

class OvenWrapper:
    """
    Simplified API around SuperintelligenceOven.
    Pass any model + ModelDescription, get a trainable oven.
    """

    def __init__(self, model, description, config, tokenizer=None, verifier=None, prompts=None):
        self.description = description
        self.config = config
        config.model_description = description
        if not description.use_verifier:
            config.verifier_weight = 0.0
            verifier = None
        self.oven = SuperintelligenceOven(
            config=config, policy_model=model, tokenizer=tokenizer,
            verifier=verifier, prompt_source=PromptSource(prompts) if prompts else None,
        )

    async def run(self, prompts=None):
        ps = PromptSource(prompts) if prompts else None
        return await self.oven.train(prompt_source=ps)

    def run_sync(self, prompts=None):
        return asyncio.run(self.run(prompts))

    def swap_teacher(self, name):
        self.oven.swap_teacher(name)

    def swap_agent(self, agent_name, **kw):
        self.oven.swap_agent(agent_name, **kw)

    def swap_all_agents(self, provider, model, **kw):
        self.oven.swap_all_agents(provider, model, **kw)

    def attach_to_model(self, model):
        self.oven.attach_to_model(model)

    @property
    def metrics(self):
        return self.oven.metrics_history

    def status(self):
        return {
            "model": self.description.name,
            "modality": self.description.modality,
            "domain": self.description.domain,
            "teacher": self.oven.teacher_mgr.current_name,
            "teacher_kind": self.oven.teacher_mgr.current_kind,
            "agents": {a.name: f"{a.provider}/{a.model}" for a in self.config.agents},
            "verifier_active": self.oven.verifier is not None,
            "steps_completed": len(self.oven.metrics_history),
            "total_steps": self.config.total_steps,
        }


def bake(
    model: nn.Module,
    name: str = "unnamed-model",
    modality: str = "text",
    architecture: str = "",
    domain: str = "",
    description: str = "",
    output_format: str = "text",
    scoring_guidance: str = "",
    use_verifier: bool = True,
    tokenizer=None,
    verifier=None,
    prompts: Optional[List[str]] = None,
    teacher: Optional[str] = None,
    teachers: Optional[List[TeacherConfig]] = None,
    agents: Optional[List[AgentConfig]] = None,
    group_size: int = 8,
    batch_size: int = 4,
    total_steps: int = 10000,
    lr: float = 1e-6,
    kl_beta: float = 0.1,
    device: str = "cuda",
    config: Optional[OvenConfig] = None,
) -> OvenWrapper:
    """
    One-call setup. Pass any model + description, get a trainable oven.

    The `description` and `scoring_guidance` are injected into every agent
    call so the reward committee knows what kind of GRPO they're doing.

    Examples:
        # Text LM
        oven = bake(
            model=my_lm, name="assistant-7b", modality="text",
            description="7B causal LM for chat.",
            scoring_guidance="Penalize hallucinations heavily.",
            prompts=["Explain X.", "Write Y."],
        )
        oven.run_sync()

        # Motion model (verifier auto-disabled)
        oven = bake(
            model=my_motion_model, name="egirl-v2", modality="motion",
            description="72-dim joint trajectory generator.",
            use_verifier=False,
            teacher="diffusion_motion",
            teachers=[TeacherConfig(name="diffusion_motion", kind="diffusion", model_name="d-teacher")],
            prompts=["wave hello", "dance excitedly"],
        )
        oven.run_sync()
    """
    model_desc = ModelDescription(
        name=name, modality=modality, architecture=architecture,
        domain=domain, description=description, output_format=output_format,
        scoring_guidance=scoring_guidance, use_verifier=use_verifier,
    )
    if config is None:
        config = OvenConfig(
            group_size=group_size, batch_size=batch_size, total_steps=total_steps,
            lr=lr, kl_beta=kl_beta, device=device, model_description=model_desc,
            teachers=teachers or DEFAULT_TEACHERS,
            agents=agents or DEFAULT_AGENTS,
            initial_teacher=teacher or "small_qwen3_4b",
        )
    else:
        config.model_description = model_desc
    return OvenWrapper(
        model=model, description=model_desc, config=config,
        tokenizer=tokenizer, verifier=verifier, prompts=prompts,
    )
