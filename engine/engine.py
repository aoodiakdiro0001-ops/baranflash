"""BaranFlash orchestrator — the asymmetric MoE engine end-to-end."""
import hashlib
import os
import time
from typing import Any, Dict, Optional

from . import brain, config, memory
from .router import MoERouter, expert_model


class BaranFlash:
    def __init__(self, db_path: str = None):
        self.mem = memory.Memory(db_path)
        self.router = MoERouter()

    # ── PREFILL (fast path — no LLM) ──────────────────────────────────────
    def ingest(self, text: str, source: str = "inline") -> Dict[str, Any]:
        t0 = time.time()
        res = self.mem.ingest(text, source)
        res["prefill_ms"] = round((time.time() - t0) * 1000, 1)
        res["mode"] = "prefill-fast"
        return res

    def ingest_file(self, path: str) -> Dict[str, Any]:
        with open(path, "r", encoding="utf-8", errors="ignore") as fh:
            text = fh.read()
        return self.ingest(text, os.path.basename(path))

    def ingest_image(self, image_bytes: bytes, filename: str = "image") -> Dict[str, Any]:
        desc = brain.describe_image(image_bytes, filename)
        res = self.ingest(desc["text_for_index"], f"[img] {filename}")
        res["vision"] = desc["info"]
        return res

    # ── DECODE (deep path — routed expert + RAG) ──────────────────────────
    def ask(self, question: str, has_image: bool = False) -> Dict[str, Any]:
        t0 = time.time()
        gate = self.router.route(question, has_image)
        qhash = hashlib.sha256((question + ("+img" if has_image else "")).encode("utf-8")).hexdigest()[:16]
        cached = self.mem.cache_get(qhash)
        if cached:
            answer, meta = cached
            meta["cache"] = "HIT"
            meta["total_ms"] = round((time.time() - t0) * 1000, 1)
            return {"answer": answer, **meta}
        contexts = self.mem.search(question, top_n=6)
        primary = gate["primary"]
        out = brain.decode(primary, question, contexts, expert_model(primary))
        meta = {
            "router": {"primary": gate["primary"], "experts": gate["experts"],
                        "idle": gate["idle_experts"], "gate_ms": gate["gate_ms"]},
            "context_used": out["context_used"], "mode": out["mode"],
            "approx_tokens_in_window": memory.approx_tokens(" ".join(c["text"] for c in contexts)),
            "cache": "MISS",
        }
        self.mem.cache_put(qhash, out["answer"], {**meta, "cache": "MISS"})
        meta["total_ms"] = round((time.time() - t0) * 1000, 1)
        return {"answer": out["answer"], **meta}

    def ask_image(self, question: str, image_bytes: bytes, filename: str = "image") -> Dict[str, Any]:
        self.ingest_image(image_bytes, filename)
        gate = self.router.route(question or "وصف الصورة", has_image=True)
        contexts = self.mem.search(question or "وصف الصورة", top_n=4)
        out = brain.decode("vision", question or "وصف الصورة", contexts, expert_model("vision"))
        return {"answer": out["answer"], "router": {"primary": "vision", "experts": gate["experts"]},
                "mode": out["mode"], "cache": "MISS"}

    def stats(self) -> Dict[str, Any]:
        return self.mem.stats()
