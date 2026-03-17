# 🔥 Superintelligence Oven

**Multi-agent GRPO training framework that doesn't need teacher logits from frontier models.**

Use Claude, DeepSeek, Grok, or any LLM API as a reward committee to train your local model — while a hot-swappable local teacher handles the KL anchor with full distributions. No logit access to frontier models required. Model-agnostic: works with text LLMs, diffusion models, motion generators, anything with a trainable forward pass.

---

## The Problem

You want to distill intelligence from frontier models into your local model, but:

- **Frontier APIs don't give you full logit distributions** — at best you get top-20 logprobs (DeepSeek, Grok), at worst you get nothing (Claude, GPT)
- **GRPO needs a reward signal, not teacher logits** — but most implementations conflate the two
- **Single-teacher distillation bottlenecks** on whatever that one teacher is bad at
- **Your model might not even be a text LM** — diffusion, motion, multimodal models need GRPO too

## The Solution

The Superintelligence Oven separates the training signal into independent channels:

| Signal | Source | What it does | Needs logits? |
|---|---|---|---|
| **KL Anchor** | Local teacher (hot-swappable) | Prevents policy collapse | ✅ Full distribution (local) |
| **Reward** | Remote agent swarm (1-5 agents) | Defines "what good looks like" | ❌ Black-box scores only |
| **Semantic Verification** | QwenEmbedVerifier (local) | Embedding-space quality check (text-only) | ❌ Cosine similarity |
| **Frontier Calibration** | DeepSeek/Grok top-k logprobs | Drift detection diagnostic | ⚠️ Optional, not in loss |
| **Curriculum Steering** | Curriculum agent (API) | Picks what to train on next | ❌ Just text |

The key insight: **logit-level guidance and reward-level guidance are separate channels.** Your local teacher handles the first. Frontier models handle the second. They don't need to be the same model.

## Architecture

```
┌──────────────────────────────────────────┐
│  Local (your GPU)                        │
│                                          │
│  Policy Model ──generates──► N completions
│       │                                  │
│  Local Teacher ──provides──► KL anchor   │
│  (hot-swappable: small ↔ big ↔ diffusion)│
│       │                                  │
│  QwenEmbedVerifier ──► semantic score    │
│  (text-only, auto-skipped for non-text)  │
└────────┼─────────────────────────────────┘
         │ completions sent out
         ▼
┌──────────────────────────────────────────┐
│  Remote Agent Swarm (async, parallel)    │
│                                          │
│  Agent 1: Critic ──────── Claude Sonnet  │
│  Agent 2: Adversary ───── DeepSeek R1    │
│  Agent 3: Specialist ──── Grok 3        │
│  Agent 4: Style ────────── Claude Haiku  │
│  Agent 5: Curriculum ──── Claude Sonnet  │
│                                          │
│  Each agent receives ModelDescription    │
│  so it knows what kind of GRPO this is   │
│                                          │
│  ──► weighted composite reward           │
└────────┼─────────────────────────────────┘
         │ rewards back
         ▼
┌──────────────────────────────────────────┐
│  GRPO Update                             │
│                                          │
│  advantages = group_normalize(rewards)   │
│  loss = -log_prob(policy) × advantages   │
│       + β × KL(policy ∥ local_teacher)   │
└──────────────────────────────────────────┘
```

## Quick Start

### Install

```bash
pip install -e .

# For quantized teachers:
pip install super_intelligence_oven -e ".[quantized]"

# For diffusion teachers:
pip install super_intelligence_oven -e ".[diffusion]"

# Everything:
pip install super_intelligence_oven -e ".[all]"
```

### The `bake()` Function — One Call Setup

```python
import asyncio
from transformers import AutoModelForCausalLM, AutoTokenizer
from superintelligence_oven import bake

tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B", trust_remote_code=True)
policy = AutoModelForCausalLM.from_pretrained("your-model", trust_remote_code=True).to("cuda")

oven = bake(
    model=policy,
    tokenizer=tok,

    # This description is injected into every agent call
    name="my-assistant-7b",
    modality="text",
    domain="general instruction following",
    description="A 7B causal LM for chat. Prioritize accuracy over verbosity.",
    scoring_guidance="Penalize hallucinations heavily. Reward concise answers.",

    prompts=["Explain quantum entanglement.", "Write a merge sort in Python."],
    group_size=8,
    total_steps=5000,
)

oven.run_sync()
```

The `description` and `scoring_guidance` fields become part of every agent's system prompt — this is how you tell the reward committee what kind of GRPO they're doing. A motion model gets different scoring criteria than a code model.

### Non-Text Models (Diffusion, Motion)

```python
from superintelligence_oven import bake, TeacherConfig

oven = bake(
    model=my_motion_model,
    name="egirl-motion-v2",
    modality="motion",
    domain="character animation",
    description="VALM motion generator, 72-dim joint trajectories.",
    output_format="trajectory",
    use_verifier=False,  # verifier is text-only — skip for motion
    teacher="diffusion_motion",
    teachers=[TeacherConfig(
        name="diffusion_motion",
        kind="diffusion",
        model_name="diffusion-teacher",
        joint_dim=72,
        intent_dim=512,
    )],
    prompts=["wave hello", "dance excitedly"],
)
```

### Swap Agent Models On The Fly

```python
oven.swap_agent("critic",     provider="claude",   model="claude-sonnet-4-20250514")
oven.swap_agent("adversary",  provider="deepseek", model="deepseek-reasoner", logprobs=True)
oven.swap_agent("specialist", provider="grok",     model="grok-3", logprobs=True, top_logprobs=20)
oven.swap_agent("style",      provider="claude",   model="claude-haiku-4-5-20251001")

# Or swap all at once
oven.swap_all_agents(provider="deepseek", model="deepseek-chat")
```

### Hot-Swap Local Teachers

```python
oven.swap_teacher("small_qwen3_4b")     # fast, early training
oven.swap_teacher("big_qwen3_8b_q4")    # deeper, after model absorbs small teacher

# Register new teachers at runtime
from superintelligence_oven import TeacherConfig
oven.oven.teacher_mgr.register_teacher(TeacherConfig(
    name="big_qwen3_32b",
    kind="big",
    model_name="Qwen/Qwen3-32B",
    strategy="hybrid",
))
```

### Wire Into Existing Code

```python
oven.attach_to_model(your_model)
# your_model.teacher = oven's active teacher
# your_model.verifier = oven's QwenEmbedVerifier
```

## Repo Structure

```
superintelligence-oven/
├── pyproject.toml
├── README.md
├── superintelligence_oven/
│   ├── __init__.py              # Public API
│   ├── config.py                # All dataclasses, constants, ModelDescription
│   ├── oven.py                  # SuperintelligenceOven (main training loop)
│   ├── wrapper.py               # OvenWrapper + bake() convenience function
│   ├── agents.py                # AgentSwarm (remote reward committee)
│   ├── verifier.py              # QwenEmbedVerifier (text-only semantic check)
│   ├── calibrator.py            # FrontierCalibrator (diagnostic drift detection)
│   ├── prompts.py               # PromptSource (with curriculum injection)
│   └── teachers/
│       ├── __init__.py
│       ├── manager.py           # TeacherManager (hot-swap, auto-escalation)
│       ├── small_causal.py      # SmallCausalTeacher (Qwen3-4B)
│       ├── big_causal.py        # BigCausalTeacher (quantized/offload/hybrid)
│       └── diffusion.py         # SmallCausalDiffusionTeacher (motion/trajectory)
└── examples/
    ├── train_text_model.py
    └── train_motion_model.py
```

## ModelDescription — Tell Agents What You're Training

The `ModelDescription` is the bridge between your model and the agent swarm. It gets injected into every agent's system prompt:

```python
from superintelligence_oven import ModelDescription

desc = ModelDescription(
    name="code-wizard-3b",
    modality="text",
    architecture="causal LM",
    domain="code generation",
    description="A 3B model specialized for Python and JavaScript code generation.",
    output_format="text",
    scoring_guidance="Score code for correctness first, then readability, then efficiency.",
    use_verifier=True,
)

# This becomes part of every agent's prompt:
print(desc.to_agent_prompt())
```

## Teacher Types

| Teacher | Kind | VRAM | Speed | Use Case |
|---|---|---|---|---|
| SmallCausalTeacher | `small` | ~2-8 GB | Fast | Text, early training |
| BigCausalTeacher (quantized) | `big` | ~4 GB/8B | Medium | Text, deeper KL signal |
| BigCausalTeacher (offload) | `big` | ~0 GB idle | Slow | VRAM-constrained |
| BigCausalTeacher (hybrid) | `big` | Adaptive | Medium | Balance |
| SmallCausalDiffusionTeacher | `diffusion` | ~1 GB | Fast | Motion/trajectory |

Auto-escalation: when KL between policy and small teacher collapses, the oven swaps to a big teacher automatically.

## Agent Roles

| Agent | Default | Scores? | Purpose |
|---|---|---|---|
| Critic | Claude Sonnet | ✅ | Correctness, coherence, instruction-following |
| Adversary | Claude Sonnet | ✅ | Failure modes, hallucinations, reward hacking |
| Specialist | DeepSeek | ✅ | Domain accuracy, reasoning depth |
| Style | Grok | ✅ | Clarity, tone, formatting |
| Curriculum | Claude Sonnet | ❌ | Steers what to train on next |

Each agent receives the `ModelDescription` so it knows whether it's scoring code, prose, or motion trajectories.

## Supported Providers

| Provider | Scoring | Logprobs | Env Variable |
|---|---|---|---|
| Claude (Anthropic) | ✅ | ❌ | `ANTHROPIC_API_KEY` |
| DeepSeek | ✅ | ✅ top-20 | `DEEPSEEK_API_KEY` |
| Grok (xAI) | ✅ | ✅ top-20 | `XAI_API_KEY` |
| OpenAI | ✅ | ✅ top-20 | `OPENAI_API_KEY` |

## Verifier (Text-Only)

The `QwenEmbedVerifier` uses Qwen3-Embedding-0.6B to compare policy completions against teacher drafts via cosine similarity. It catches garbage (repeated characters, noise, off-topic) before expensive API calls.

**Automatically disabled for non-text models** (motion, diffusion, image) via `use_verifier=False`.

## How GRPO Works Here

1. **Generate** N completions from your policy (you have full logit access locally)
2. **Score** via agent swarm (scalar rewards — no logits needed from APIs)
3. **Blend** verifier score (text-only, optional)
4. **Normalize** within group → advantages
5. **Update** policy via clipped gradient, anchored by KL to local teacher
6. **Calibrate** against frontier top-k logprobs (diagnostic, not in loss)
7. **Steer** curriculum agent suggests next prompts

## Requirements

- Python 3.9+
- PyTorch 2.0+
- `transformers`, `accelerate`, `aiohttp`
- At least one API key
- GPU with VRAM for policy + one teacher

## License

MIT

## Contributing

Open issues, PRs, or fork it. The point is everyone should be able to train better models without needing frontier API logits.

---

*Named because it bakes models from all sides simultaneously — KL anchor keeps it from burning, agent swarm controls the temperature, GRPO is the convection.*
