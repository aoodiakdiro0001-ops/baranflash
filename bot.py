#!/usr/bin/env python3
"""BaranFlash Telegram bot — stdlib only (urllib long-polling).

Users: send documents/photos to index them, then ask questions.
Monetization (same pattern as Baran's other bots):
  free questions per user (BF_FREE_LIMIT) → /paid <TXID> → admin /approve <user_id>.
The bot token lives in .env (BF_BOT_TOKEN) and is NEVER published.
"""
import json
import os
import sys
import time
import urllib.request
from typing import Any, Dict, Optional

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from engine import config  # noqa: E402
from engine.engine import BaranFlash  # noqa: E402

API = "https://api.telegram.org/bot" + config.BOT_TOKEN
STATE_PATH = os.path.join(config.DATA_DIR, "bot_state.json")

HELP = (
    "⚡ *BaranFlash V1* — محرك ذكاء محلي بهندسة MoE\n\n"
    "• أرسل مستنداً (txt/md/csv/json/log/py/js/html) ← يُفهرس في نافذة السياق\n"
    "• أرسل صورة ← يُحلّل ويُفهرس (خبير الرؤية)\n"
    "• اكتب سؤالك ← موجّه MoE يختار الخبير المناسب ويجيب مع الاستشهادات\n\n"
    f"الأسئلة المجانية: {config.FREE_LIMIT} — بعدها: /paid <رقم عملية USDT/BTC>\n"
    "الاشتراك: 5$ USDT شهرياً · دعم: " + config.PAYMENT_INFO["usdt_trc20"]
)


def load_state() -> Dict[str, Any]:
    try:
        with open(STATE_PATH, "r", encoding="utf-8") as fh:
            return json.load(fh)
    except Exception:
        return {"users": {}, "offset": 0}


def save_state(st: Dict[str, Any]) -> None:
    os.makedirs(config.DATA_DIR, exist_ok=True)
    with open(STATE_PATH, "w", encoding="utf-8") as fh:
        json.dump(st, fh)


def tg(method: str, payload: Optional[Dict[str, Any]] = None, timeout: int = 60) -> Dict[str, Any]:
    body = json.dumps(payload or {}).encode("utf-8")
    req = urllib.request.Request(API + "/" + method, data=body,
                                 headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def tg_file(file_id: str) -> bytes:
    info = tg("getFile", {"file_id": file_id})
    url = "https://api.telegram.org/file/bot" + config.BOT_TOKEN + "/" + info["result"]["file_path"]
    with urllib.request.urlopen(url, timeout=60) as resp:
        return resp.read()


def user_quota(st: Dict[str, Any], uid: int) -> Dict[str, Any]:
    u = st["users"].setdefault(str(uid), {"asks": 0, "premium_until": 0, "pending_tx": None})
    active = u["premium_until"] > time.time()
    return u, active


def handle_update(flash: BaranFlash, st: Dict[str, Any], upd: Dict[str, Any]) -> None:
    msg = upd.get("message") or upd.get("edited_message") or {}
    chat = msg.get("chat") or {}
    uid = (msg.get("from") or {}).get("id", 0)
    cid = chat.get("id")
    text = (msg.get("text") or "").strip()
    if not cid:
        return
    u, active = user_quota(st, uid)

    if text.startswith("/start") or text.startswith("/help"):
        tg("sendMessage", {"chat_id": cid, "text": HELP, "parse_mode": "Markdown"})
        return
    if text.startswith("/stats"):
        tg("sendMessage", {"chat_id": cid, "text": json.dumps(flash.stats(), ensure_ascii=False, indent=2)})
        return
    if text.startswith("/paid"):
        tx = text.replace("/paid", "").strip()
        if len(tx) < 6:
            tg("sendMessage", {"chat_id": cid, "text": "أرسل رقم العملية هكذا: /paid TXID-HERE"})
            return
        u["pending_tx"] = {"tx": tx, "uid": uid, "at": int(time.time())}
        save_state(st)
        tg("sendMessage", {"chat_id": cid, "text": "✅ استلمنا رقم العملية — قيد التحقق. سيتم تفعيل الاشتراك بعد موافقة الإدارة."})
        tg("sendMessage", {"chat_id": int(config.ADMIN_ID),
                           "text": f"💸 دفعة للتحقق\nuser: {uid}\ntx: {tx}\nللموافقة: /approve {uid}"})
        return
    if text.startswith("/approve"):
        if str(uid) != str(config.ADMIN_ID):
            tg("sendMessage", {"chat_id": cid, "text": "هذا الأمر للإدارة فقط."})
            return
        target = text.replace("/approve", "").strip()
        for _, uu in st["users"].items():
            if str(uu.get("uid")) == target or uu.get("pending_tx", {}).get("uid") is not None and str(uu.get("uid")) == target:
                uu["premium_until"] = time.time() + 30 * 86400
                uu["pending_tx"] = None
                tg("sendMessage", {"chat_id": int(target), "text": "🎉 تم تفعيل اشتراكك 30 يوماً — استمتع بالأسئلة غير المحدودة!"})
                break
        save_state(st)
        tg("sendMessage", {"chat_id": cid, "text": f"تم التفعيل للمستخدم {target}."})
        return

    # attachments
    doc = msg.get("document")
    photo = msg.get("photo")
    if doc:
        try:
            raw = tg_file(doc["file_id"])
            name = doc.get("file_name", "doc")
            r = flash.ingest(raw.decode("utf-8", "ignore"), name)
            tg("sendMessage", {"chat_id": cid, "text":
                f"📥 فُهرس «{name}»: {r['chunks']} مقطع / ~{r['tokens']} رمز (prefill {r['prefill_ms']}ms, cache={'HIT' if r.get('cached') else 'MISS'})"})
        except Exception as e:
            tg("sendMessage", {"chat_id": cid, "text": f"تعذّر فهرسة الملف: {str(e)[:120]}"})
        return
    if photo:
        try:
            best = photo[-1]
            raw = tg_file(best["file_id"])
            r = flash.ask_image("وصف الصورة", raw, name := f"tg-{best['file_unique_id']}.jpg")
            tg("sendMessage", {"chat_id": cid, "text": f"🖼️ [vision expert]\n{r['answer'][:1200]}"})
        except Exception as e:
            tg("sendMessage", {"chat_id": cid, "text": f"تعذّر تحليل الصورة: {str(e)[:120]}"})
        return
    if not text:
        return

    # plain question — quota check
    if not active and u["asks"] >= config.FREE_LIMIT:
        tg("sendMessage", {"chat_id": cid, "text":
            "🔒 انتهت أسئلتك المجانية. اشترك: 5$ USDT\n"
            "أرسل: /paid <TXID>\nUSDT-TRC20: " + config.PAYMENT_INFO["usdt_trc20"] +
            "\nBTC: " + config.PAYMENT_INFO["btc"]})
        return
    tg("sendChatAction", {"chat_id": cid, "action": "typing"})
    out = flash.ask(text)
    u["asks"] = u.get("asks", 0) + (0 if active else 1)
    save_state(st)
    r = out["router"]
    reply = (f"🧠 [{r['primary']}] ({', '.join(n for n, _ in r['experts'])}) "
             f"cache={out['cache']} · {out['total_ms']}ms\n\n{out['answer'][:3500]}")
    tg("sendMessage", {"chat_id": cid, "text": reply})


def main() -> None:
    if not config.BOT_TOKEN:
        print("✗ BF_BOT_TOKEN missing — put the bot token in .env (never commit it)")
        return
    flash = BaranFlash()
    st = load_state()
    print(f"⚡ BaranFlash bot polling… (free limit {config.FREE_LIMIT}, admin {config.ADMIN_ID})")
    print(config.OWNER_CREDIT)
    while True:
        try:
            res = tg("getUpdates", {"offset": st.get("offset", 0) + 1, "timeout": 30}, timeout=45)
            for upd in res.get("result", []):
                st["offset"] = upd["update_id"]
                save_state(st)
                try:
                    handle_update(flash, st, upd)
                except Exception as e:
                    print("handler error:", e)
        except Exception as e:
            print("poll error:", str(e)[:120])
            time.sleep(3)


if __name__ == "__main__":
    main()
