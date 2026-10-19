#!/usr/bin/env python3
"""BaranFlash selftest — end-to-end verification of every subsystem.
Run: python3 selftest.py   (exit 0 = all pass)
"""
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from engine import brain, config  # noqa: E402
from engine.engine import BaranFlash  # noqa: E402

PASS, FAIL = "✓", "✗"
results = []


def check(name: str, cond: bool, extra: str = "") -> None:
    results.append((name, cond))
    print(f"{PASS if cond else FAIL} {name}" + (f" — {extra}" if extra else ""))


def main() -> None:
    tmp = tempfile.mkdtemp(prefix="baranflash-test-")
    flash = BaranFlash(db_path=os.path.join(tmp, "test.db"))

    # 1) MoE router: sparse gating picks the right experts
    g1 = flash.router.route("لخص لي هذا التقرير الطويل")
    g2 = flash.router.route("def sort_arr(a): return sorted(a)  # اشرح هذا الكود")
    g3 = flash.router.route("احسب 144 * 12 + 88")
    g4 = flash.router.route("ما هي أهم النقاط في المستند؟")
    check("router: summarize picked", g1["primary"] == "summarize", str([(n, round(w, 2)) for n, w in g1["experts"]]))
    check("router: code picked", g2["primary"] == "code")
    check("router: math picked", g3["primary"] == "math")
    check("router: doc_qa picked", g4["primary"] == "doc_qa")
    check("router: sparse (idle experts exist)", g1["idle_experts"] >= 4)

    # 2) ingest (prefill): fast, no LLM, with dedupe cache
    needle_ar = "المشروع السري يُدعى صقر الصحراء ويعمل على خفض تكاليف الاستيراد بنسبة 37%."
    doc = ("تقرير مالي ربع سنوي.\n" + ("نص عام لا معنى له للتمرير. " * 40) + "\n" + needle_ar + "\n") * 3
    r1 = flash.ingest(doc, "تقرير-تجريبي.txt")
    r2 = flash.ingest(doc, "تقرير-تجريبي.txt")  # same content → KV-cache hit
    check("ingest: chunked", r1["chunks"] >= 2, f"{r1['chunks']} chunks")
    check("ingest: dedupe cache hit", r2.get("cached") is True)
    check("ingest: prefill fast (<2s, no LLM)", r1["prefill_ms"] < 2000, f"{r1['prefill_ms']}ms")

    # 3) retrieval: BM25 finds the needle with citation context
    hits = flash.mem.search("صقر الصحراء تكاليف الاستيراد")
    ok = any("صقر الصحراء" in h["text"] for h in hits)
    check("retrieval: BM25 finds needle", ok, f"top score {hits[0]['score'] if hits else 0}")

    # 4) decode path: answer composes (ollama or extractive fallback)
    out = flash.ask("ما اسم المشروع السري ونسبة خفض التكاليف؟")
    check("decode: answer non-empty", bool(out["answer"]))
    check("decode: routed expert recorded", "router" in out and out["router"]["primary"] in config.EXPERTS)
    print(f"   ↳ answer mode: {out['mode']} · {out['total_ms']}ms · cache={out['cache']}")
    print(f"   ↳ answer: {out['answer'][:120]}…")

    # 5) KV-cache second ask
    out2 = flash.ask("ما اسم المشروع السري ونسبة خفض التكاليف؟")
    check("kv-cache: second ask is HIT", out2.get("cache") == "HIT")

    # 6) vision expert: PIL path (metadata + colors) even without OCR
    try:
        from PIL import Image
        import io
        buf = io.BytesIO()
        Image.new("RGB", (320, 200), (30, 120, 200)).save(buf, "PNG")
        v = brain.describe_image(buf.getvalue(), "test.png")
        check("vision: dims read", v["info"].get("width") == 320)
        check("vision: dominant color read", bool(v["info"].get("dominant_colors_rgb")))
    except ImportError:
        print("- vision: PIL missing, skipped")

    # 7) asymmetric budgets sanity: prefill must beat decode time
    check("asymmetry: prefill_ms < decode_ms", r1["prefill_ms"] < out["total_ms"],
          f"{r1['prefill_ms']}ms vs {out['total_ms']}ms")

    # 8) stats
    st = flash.stats()
    check("stats: window limit ~1M tokens", st["window_limit_tokens"] >= 900_000,
          f"{st['window_limit_tokens']} tokens")

    fails = [n for n, ok in results if not ok]
    print(f"\n{'═' * 50}\n{len(results) - len(fails)}/{len(results)} passed" + (f" — FAILED: {fails}" if fails else " — ALL GREEN 🟢"))
    print(config.OWNER_CREDIT.splitlines()[0])
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
