# verifier_qwen_embed.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Tuple, Union
from transformers import AutoTokenizer, AutoModel
from torch.utils.checkpoint import checkpoint
device = torch.device('cuda')

class QwenEmbedVerifier(nn.Module):
    """
    Qwen3 Embedding verifier.

    Two primary modes:
      1) Direct similarity: support(gold_cont, teacher_cont, ...)
         -> cosine(gold, teacher) mapped [-1,1]→[0,1]
      2) QA-style evidence: support(question=?, context=?, answer=?, evidence_spans=?)
         -> pick best evidence vs question, then cosine(answer, evidence) mapped to [0,1]

    Return:
      - by default: scalar support in [0,1]
      - if return_evidence=True and evidence path used: (support, best_evidence_str)
    """
    def __init__(self,
                 model_name: str = "Qwen/Qwen3-Embedding-0.6B",
                 device: str = "cuda",
                 normalize: bool = True):
        super().__init__()
        self.device = torch.device(device)
        self.tok = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.enc = AutoModel.from_pretrained(model_name, trust_remote_code=True).to(self.device)
        self.normalize = normalize
        self.eval()

    @torch.no_grad()
    def _embed(self, texts: List[str]) -> torch.Tensor:
        batch = self.tok(texts, padding=True, truncation=True, return_tensors="pt").to(self.device)
        out = self.enc(**batch)
        # prefer pooled if present, else CLS
        if hasattr(out, "pooler_output") and out.pooler_output is not None:
            E = out.pooler_output
        else:
            E = out.last_hidden_state[:, 0]
        if self.normalize:
            E = F.normalize(E, dim=-1)
        return E  # (B, D)

    @torch.no_grad()
    def support(self,
                gold_cont: Optional[str] = None,
                teacher_cont: Optional[str] = None,
                *,
                question: Optional[str] = None,
                context: Optional[str] = None,
                answer: Optional[str] = None,
                evidence_spans: Optional[List[str]] = None,
                return_evidence: bool = False) -> Union[float, Tuple[float, str]]:
        """
        Flexible API; trainer calls with gold_cont=..., teacher_cont=....

        Case A (trainer path):
            - gold_cont and teacher_cont provided -> direct similarity.
        Case B (QA path):
            - question + context (+optional evidence_spans) (+answer) -> evidence-based score.

        Returns:
            float in [0,1], or (float, best_evidence) if return_evidence=True and QA path used.
        """
        # ---------- Case A: direct similarity (trainer’s call) ----------
        if gold_cont is not None and teacher_cont is not None:
            E = self._embed([gold_cont, teacher_cont])  # (2, D)
            cos = torch.dot(E[0], E[1]).clamp(-1.0, 1.0).item()
            s01 = 0.5 * (cos + 1.0)
            return float(s01)

        # ---------- Case B: QA-style evidence selection ----------
        if question is None or (context is None and not evidence_spans):
            # insufficient info -> neutral score
            return 0.5 if not return_evidence else (0.5, "")

        # prepare candidate evidence
        if evidence_spans and len(evidence_spans) > 0:
            candidates = [e.strip() for e in evidence_spans if isinstance(e, str) and e.strip()]
        else:
            # naive sentence split fallback; replace with your own splitter if desired
            candidates = [x.strip() for x in (context or "").replace("\n", " ").split(". ") if x.strip()]

        if len(candidates) == 0:
            candidates = [context or ""]

        qE = self._embed([question]).squeeze(0)            # (D,)
        cE = self._embed(candidates)                        # (N, D)
        sims = cE @ qE                                      # (N,)
        idx = int(torch.argmax(sims).item())
        best_ev = candidates[idx]

        # choose which text to compare to evidence: prefer provided 'answer',
        # otherwise fallback to 'teacher_cont' or 'gold_cont' if either present.
        ref_txt = answer if (answer is not None) else (teacher_cont if teacher_cont is not None else gold_cont)
        if ref_txt is None:
            # neutral if nothing to compare
            return 0.5 if not return_evidence else (0.5, best_ev)

        aE = self._embed([ref_txt]).squeeze(0)             # (D,)
        eE = cE[idx]                                        # (D,)
        cos = torch.dot(aE, eE).clamp(-1.0, 1.0).item()
        s01 = 0.5 * (cos + 1.0)
        return (float(s01), best_ev) if return_evidence else float(s01)
