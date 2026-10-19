"""BaranFlash V1 — engine configuration (env-driven, zero paid deps)."""
import os
import json
from typing import Dict, List

try:  # python-dotenv optional; we read .env manually (stdlib only)
    with open(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env"), "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip())
except FileNotFoundError:
    pass

def env(key: str, default: str = "") -> str:
    return os.environ.get(key, default)

# ── Models (Ollama local HTTP) ────────────────────────────────────────────
# Asymmetric budgets: prefill never calls an LLM (fast path), decode uses the
# deep model for final synthesis (or falls back to extractive if Ollama down).
OLLAMA_URL = env("BF_OLLAMA_URL", "http://localhost:11434")
MODEL_DEEP = env("BF_MODEL_MAIN", "gemma4:12b")     # decode expert (deep synthesis)
MODEL_FAST = env("BF_MODEL_FAST", "llama3")          # quick expert (short answers)

# ── Long context ("1M tokens" emulated) ───────────────────────────────────
MAX_CONTEXT_CHARS = int(env("BF_MAX_CONTEXT_CHARS", "4000000"))  # ≈1M tokens @4 chars
CHUNK_CHARS = int(env("BF_CHUNK_CHARS", "900"))
CHUNK_OVERLAP = int(env("BF_CHUNK_OVERLAP", "120"))
SUMMARY_EVERY = int(env("BF_SUMMARY_EVERY", "8"))                # section summary cadence
BM25_K1 = float(env("BF_BM25_K1", "1.4"))
BM25_B = float(env("BF_BM25_B", "0.72"))

# ── Storage / cache (KV-cache emulation) ──────────────────────────────────
DATA_DIR = env("BF_DATA_DIR", os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data"))
DB_PATH = os.path.join(DATA_DIR, "flash.db")

# ── Router ────────────────────────────────────────────────────────────────
TOP_K_EXPERTS = int(env("BF_TOP_K", "2"))
TEMPERATURE_DEEP = float(env("BF_TEMP_DEEP", "0.4"))

# ── Telegram monetization (same pattern as baranexcel/baranstore) ─────────
BOT_TOKEN = env("BF_BOT_TOKEN", "")
ADMIN_ID = env("BF_ADMIN_ID", "5899220599")
FREE_LIMIT = int(env("BF_FREE_LIMIT", "3"))

PAYMENT_INFO: Dict[str, str] = {
    "btc": "bc1q8dhtdrndxsl70e97t8grydfe6pjue37zrhzf6c",
    "usdt_trc20": "TLWdTjqqogkXNEXGiPWsGdjFP6ioycxeWW",
    "whatsapp": "https://wa.me/905313263722",
    "email": "baranavisat@gmail.com",
    "phone": "+90 531 326 37 22",
}

OWNER_CREDIT = (
    "BaranFlash V1 — designed & built by Baran Avisat (Professional Programmer)\n"
    "✉️ baranavisat@gmail.com · ☎️ +90 531 326 37 22 · WhatsApp wa.me/905313263722\n"
    "Support BTC: bc1q8dhtdrndxsl70e97t8grydfe6pjue37zrhzf6c\n"
    "Support USDT-TRC20: TLWdTjqqogkXNEXGiPWsGdjFP6ioycxeWW"
)

EXPERTS: List[str] = ["doc_qa", "summarize", "code", "math", "translate", "vision", "general"]
EXPERT_MODEL: Dict[str, str] = {
    "doc_qa": MODEL_FAST,       # RAG-QA: fast model is enough (retrieval does the work)
    "summarize": MODEL_DEEP,    # deep synthesis benefits from the bigger model
    "code": MODEL_DEEP,
    "math": MODEL_FAST,
    "translate": MODEL_FAST,
    "vision": MODEL_DEEP,
    "general": MODEL_FAST,
}
