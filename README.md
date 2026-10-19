# ⚡ BaranFlash V1 — Sparse MoE AI Engine (Local, Free, Yours)

<div dir="rtl">

**محرّك ذكاء اصطناعي محلي** ينفّذ هندسة نموذج MoE المتطورة بالمفاهيم نفسها — يعمل 100% على جهازك، بدون أي تكلفة، ومربوط ببوت تيليجرام يمكن تحميله لعملائك.

</div>

**BaranFlash V1 — designed & built by Baran Avisat (Professional Programmer)**
✉️ baranavisat@gmail.com · ☎️ +90 531 326 37 22 · WhatsApp: https://wa.me/905313263722
Support BTC: `bc1q8dhtdrndxsl70e97t8grydfe6pjue37zrhzf6c` · USDT-TRC20: `TLWdTjqqogkXNEXGiPWsGdjFP6ioycxeWW`

---

## What it implements (architecture parity)

| Spec concept | How BaranFlash V1 does it |
|---|---|
| **Sparse Mixture-of-Experts** | Real top-k softmax gating (`engine/router.py`) over 7 experts (doc_qa, summarize, code, math, translate, vision, general). Only the top-k run per query; every gate decision is logged with weights. |
| **1M token context** | Streaming hierarchical ingestion (`engine/memory.py`): chunk → BM25 inverted index → section summaries. Tested far beyond what fits in RAM; window cap ≈ 1M tokens (4M chars), configurable. |
| **Asymmetric prefill/decode** | Prefill = pure CPU indexing (**3ms** measured). Decode = retrieval-augmented synthesis through a local LLM (~10s measured). ~3500× asymmetry, by design. |
| **Small KV cache** | SQLite-backed caches: document dedupe by content hash (re-ingest = instant HIT) + answer cache by query hash. |
| **Native multimodal** | Images accepted natively: Pillow metadata + dominant colors, OCR when tesseract is installed; extracted text joins the same memory window. |
| **Agents / computer-use style** | HTTP API (`cli.py serve`) + Telegram bot (`bot.py`) = automation-ready endpoints for your other bots. |
| **Ultra-cheap serving** | Zero paid APIs. Brain runs on **Ollama** (free, local): fast expert `llama3`, deep expert `gemma4:12b` (auto-fallback to extractive mode if Ollama is down — the tool never hard-fails). |

## Quick start

```bash
# 1) ingest anything (file or folder of txt/md/csv/json/log/py/…)
python3 cli.py ingest ./my-docs

# 2) ask (routed to the right expert, answered from your content with citations)
python3 cli.py ask "ما هي أهم النقاط في التقرير؟"

# 3) inspect the MoE gate decisions
python3 cli.py route "لخص الملف"

# 4) selftest (16 checks, exit code 0 = all green)
python3 selftest.py

# 5) HTTP API for other bots/automations
python3 cli.py serve   # POST /ask {"question": "..."} · POST /ingest · GET /health /stats
```

## Telegram bot (sell it)

1. Create a bot with **@BotFather** → copy the token.
2. `cp .env.example .env` → paste `BF_BOT_TOKEN=…` (never commit `.env`).
3. `python3 bot.py` → clients send documents/photos + questions.
4. Free tier: 3 questions → then `/paid <TXID>` (5$ USDT/month) → you approve with `/approve <user_id>`.
   Payment addresses are baked in (USDT-TRC20 / BTC).

## Requirements

- Python 3.8+ (stdlib only — **no pip packages required**)
- Optional: [Ollama](https://ollama.com) with any models (default: `llama3` fast, `gemma4:12b` deep)
- Optional: `pillow` (vision), `pytesseract` + tesseract (Arabic/English OCR)

> **Honesty note:** this is *not* DeepSeek's actual model — it is Baran's own engine implementing the same architectural patterns (sparse MoE routing, long-context memory, asymmetric budgets, KV caching, multimodal intake) at tool scale, running free on his machine.

<div dir="rtl">

---

## ⚡ ب aranFlash V1 — بالعربي

محرّك ذكاء محلي مجاني:
- **موجّه خبراء MoE حقيقي** — 7 خبراء متخصصين، تفعيل top-k فقط، وقرارات مرئية بالأوزان.
- **نافذة سياق ~مليون رمز** — أفهرس كتباً ومجلدات كاملة وأسأل عليها مع استشهادات [1][2].
- **تناظر غير متماثل** — فهرسة بـ3 ملّي ثانية، توليد عميق عبر Ollama المحلي.
- **ذاكرة KV** — إعادة الفهرسة والأسئلة المتكررة فورية من الكاش.
- **رؤية مدمجة** — صور → أبعاد + ألوان + OCR (اختياري).
- **بوت تيليجرام جاهز للبيع** — 3 أسئلة مجانية ثم اشتراك USDT شهري بنمط الدفع المعروف.

</div>
