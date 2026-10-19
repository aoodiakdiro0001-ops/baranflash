"""Sparse Mixture-of-Experts router — real top-k gating over specialized experts.

Implements the described MoE pattern at tool scale: input tokens are scored
against expert "affinity" features, a softmax gate picks the top-k experts,
and only those run (the rest stay idle — that's the sparsity that buys speed).
Every routing decision is logged with its gate weights for transparency.
"""
import math
import re
import time
from typing import Any, Dict, List, Tuple

from . import config

# ── Expert feature probes (deterministic, language-aware AR/EN) ───────────
RX_SUMMARIZE = re.compile(r"(لخص|تلخيص|ملخص|summar\w*|tl;?dr)", re.I)
RX_TRANSLATE = re.compile(r"(ترجم|ترجمة|translate\w*|بالانجليز|بالعرب|بالعربي|to english|to arabic)", re.I)
RX_CODE = re.compile(r"(كود|برمج|دال[ةه]|خطأ برمجي|سكربت|code|bug|error|traceback|def |function|class |import |api|regex)", re.I)
RX_MATH = re.compile(r"(احسب|حسب|كم يساوي|calc\w*|equation|\d+\s*[\+\-\*\/x÷\^]\s*\d+)", re.I)
RX_DOCQA = re.compile(r"(ما ?(هي|هو)|مَ?ن|متى|أين|كيف|لماذا|كم|what|who|when|where|how|why|which)", re.I)
RX_DOCREF = re.compile(r"(الملف|المستند|الوثيق[ةه]|التقرير|المدخل|file|document|report|pdf|log)", re.I)

EXPERT_PATTERNS: Dict[str, List[re.Pattern]] = {
    "summarize": [RX_SUMMARIZE, RX_DOCREF],
    "translate": [RX_TRANSLATE],
    "code": [RX_CODE],
    "math": [RX_MATH],
    "doc_qa": [RX_DOCQA, RX_DOCREF],
    "general": [],
}

WEIGHTS: Dict[str, float] = {
    "summarize": 3.0, "translate": 4.0, "code": 3.2, "math": 3.4,
    "doc_qa": 1.6, "vision": 2.5, "general": 0.4,
}


class MoERouter:
    """Top-k sparse gate. Returns ordered (expert, weight) pairs + gate log."""

    def __init__(self, top_k: int = None):
        self.top_k = top_k or config.TOP_K_EXPERTS

    def features(self, text: str, has_image: bool = False) -> Dict[str, float]:
        f: Dict[str, float] = {}
        for expert, patterns in EXPERT_PATTERNS.items():
            score = 0.0
            for rx in patterns:
                m = rx.findall(text or "")
                if m:
                    score += 1.0 + 0.25 * min(len(m), 4)
            f[expert] = score * WEIGHTS.get(expert, 1.0)
        f["general"] += 0.4  # always a little affinity for the fallback expert
        if has_image:
            f["vision"] += 5.0
        return f

    def route(self, text: str, has_image: bool = False) -> Dict[str, Any]:
        t0 = time.time()
        f = self.features(text, has_image)
        # softmax gate
        mx = max(f.values())
        exps = {k: math.exp(v - mx) for k, v in f.items()}
        z = sum(exps.values())
        gates = {k: v / z for k, v in exps.items()}
        ranked = sorted(gates.items(), key=lambda kv: kv[1], reverse=True)[: self.top_k]
        # sparsity report: how many experts stayed idle
        return {
            "experts": ranked,                      # [(name, gate_weight), ...]
            "primary": ranked[0][0],
            "features": {k: round(v, 3) for k, v in f.items()},
            "idle_experts": len(config.EXPERTS) - len(ranked),
            "gate_ms": round((time.time() - t0) * 1000, 2),
        }


def expert_model(expert: str) -> str:
    return config.EXPERT_MODEL.get(expert, config.MODEL_FAST)
