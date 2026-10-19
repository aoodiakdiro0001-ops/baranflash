"""Asymmetric prefill/decode brain + vision expert.

* PREFILL (ingest): pure CPU — tokenize/index/summarize. Never calls an LLM.
  This mirrors the 8B-active-prefill asymmetry: cheap, fast, parallel.
* DECODE (answer): retrieval-augmented synthesis through the local Ollama
  model assigned to the routed expert; falls back to extractive composition
  when Ollama is unreachable (the tool never hard-fails).
* VISION: native image intake — dimensions/dominant colors via Pillow, OCR
  when tesseract exists, and the extracted text is indexed like any document.
"""
import base64
import io
import json
import re
import urllib.request
from typing import Any, Dict, List, Optional

from . import config, memory

OLLAMA_TIMEOUT = 90


def ollama_generate(model: str, prompt: str, temperature: float = None) -> Optional[str]:
    """Local generation through Ollama; returns None on any failure."""
    try:
        body = json.dumps({
            "model": model,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": temperature if temperature is not None else config.TEMPERATURE_DEEP},
        }).encode("utf-8")
        req = urllib.request.Request(
            config.OLLAMA_URL + "/api/generate", data=body,
            headers={"Content-Type": "application/json"}, method="POST",
        )
        with urllib.request.urlopen(req, timeout=OLLAMA_TIMEOUT) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        return (data.get("response") or "").strip() or None
    except Exception:
        return None


PROMPTS: Dict[str, str] = {
    "doc_qa": "أنت خبير تحليل مستندات. أجب عن السؤال اعتماداً فقط على المقاطع المقتبسة، واذكر أرقام المصادر [1] [2]. إن لم تكن الإجابة موجودة قل ذلك صراحة.\n\nالسؤال: {q}\n\nالمقاطع:\n{ctx}\n\nالإجابة (بالعربية، مختصرة، مع الاستشهادات):",
    "summarize": "لخّص المحتوى التالي في نقاط مركزة (حتى 7 نقاط) بالعربية:\n\n{ctx}\n\nالملخص:",
    "code": "أنت مهندس برمجيات. حلّل الكود/المشكلة التالية وأعطِ السبب الجذري والحل مع مثال:\n\n{q}\n\nالسياق:\n{ctx}\n\nالحل:",
    "math": "حل المسألة التالية خطوة بخطوة وأظهر النتيجة النهائية بوضوح:\n\n{q}\n\nالحل:",
    "translate": "ترجم النص التالي بدقة (عربي↔إنجليزي حسب اللغة المصدر):\n\n{ctx}\n\nالترجمة:",
    "vision": "وصّف محتوى الصورة المستخرج (نصوص/أبعاد/ألوان) وأجب:\n\n{q}\n\nالمستخرج:\n{ctx}\n\nالإجابة:",
    "general": "أجب بإيجاز ودقة:\n\n{q}\n\nسياق اختياري:\n{ctx}\n\nالإجابة:",
}


def decode(expert: str, question: str, contexts: List[Dict[str, Any]], model: str) -> Dict[str, Any]:
    """DECODE path: RAG synthesis. Extractive fallback if the LLM is down."""
    ctx_blocks = []
    for i, c in enumerate(contexts, 1):
        head = f"[{i}] ({c['source']}#c{c['ordinal']})\n"
        ctx_blocks.append(head + c["text"][:1200])
    ctx_text = "\n\n".join(ctx_blocks) if ctx_blocks else "(لا يوجد سياق مفهرس بعد — أجب من معرفتك العامة)"
    template = PROMPTS.get(expert, PROMPTS["general"])
    prompt = template.format(q=question, ctx=ctx_text)

    gen = ollama_generate(model, prompt)
    mode = "ollama:" + model
    if not gen:
        gen = extractive_answer(question, contexts)
        mode = "extractive-fallback"
    return {"answer": gen, "mode": mode, "context_used": len(contexts)}


def extractive_answer(question: str, contexts: List[Dict[str, Any]]) -> str:
    if not contexts:
        return "لا يوجد محتوى مفهرس يجيب على هذا السؤال بعد. أرسل ملفات/مستندات أولاً ثم اسأل من جديد."
    q_words = set(memory.tokenize(question))
    lines: List[str] = []
    for i, c in enumerate(contexts, 1):
        sents = [s.strip() for s in SECTION_SENT_SPLIT.split(c["text"]) if s.strip()]
        best = max(sents, key=lambda s: len(q_words & set(memory.tokenize(s))) if sents else 0, default="")
        if best:
            lines.append(f"[{i}] {best[:300]}")
    return "أقساط مقتبسة ذات صلة (وضع الاستخراج — Ollama غير متاح):\n" + "\n".join(lines[:4])


# ── Vision expert ─────────────────────────────────────────────────────────
def describe_image(image_bytes: bytes, filename: str = "image") -> Dict[str, Any]:
    """Native multimodal intake: metadata + dominant colors (+OCR if available)."""
    info: Dict[str, Any] = {"filename": filename, "bytes": len(image_bytes)}
    pil_ok = False
    try:
        from PIL import Image  # type: ignore
        img = Image.open(io.BytesIO(image_bytes))
        img = img.convert("RGB")
        info["width"], info["height"] = img.size
        small = img.resize((48, 48))
        pixels = list(small.getdata())
        # quantize to 4 bits/channel buckets
        buckets: Dict[tuple, int] = {}
        for r, g, b in pixels:
            key = (r >> 6, g >> 6, b >> 6)
            buckets[key] = buckets.get(key, 0) + 1
        top = sorted(buckets.items(), key=lambda kv: kv[1], reverse=True)[:3]
        info["dominant_colors_rgb"] = ["#%02x%02x%02x" % (k[0] * 64 + 32, k[1] * 64 + 32, k[2] * 64 + 32) for k, _ in top]
        pil_ok = True
    except Exception as e:
        info["pil_error"] = str(e)[:80]
    # OCR optional
    ocr_text = ""
    try:
        import pytesseract  # type: ignore
        from PIL import Image  # type: ignore
        ocr_text = pytesseract.image_to_string(Image.open(io.BytesIO(image_bytes)), lang="ara+eng").strip()
        info["ocr_chars"] = len(ocr_text)
    except Exception:
        info["ocr"] = "unavailable (brew install tesseract + pip3 install pytesseract pillow)"
    info["pil"] = pil_ok
    summary = (
        f"صورة {info.get('width','?')}×{info.get('height','?')} — ألوان سائدة: "
        f"{', '.join(info.get('dominant_colors_rgb', [])) or '?'}"
        + (f" — نص مستخرج: {ocr_text[:400]}" if ocr_text else "")
    )
    return {"info": info, "text_for_index": (f"{filename}\n{summary}\n{ocr_text}").strip(), "summary": summary}
