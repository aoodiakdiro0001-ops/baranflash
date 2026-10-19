#!/usr/bin/env python3
"""BaranFlash V1 — CLI. 
  python3 cli.py ingest <file|dir>     فهرسة ملف/مجلد (مسار سريع بلا LLM)
  python3 cli.py ask "سؤالك"          سؤال على المحتوى المفهرس (decode عميق)
  python3 cli.py route "نص"           اعرض قرار موجّه MoE فقط
  python3 cli.py stats                إحصائيات الذاكرة والكاش
  python3 cli.py serve                خادم HTTP محلي (صحة/سؤال/فهرسة)
"""
import json
import os
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from engine.engine import BaranFlash  # noqa: E402
from engine import config  # noqa: E402

DOC_EXT = {".txt", ".md", ".csv", ".json", ".log", ".py", ".js", ".ts", ".html", ".xml", ".yaml", ".yml", ".sql"}


def cmd_ingest(flash: BaranFlash, target: str) -> None:
    paths = []
    if os.path.isdir(target):
        for root, _, files in os.walk(target):
            for f in files:
                p = os.path.join(root, f)
                if os.path.splitext(f)[1].lower() in DOC_EXT:
                    paths.append(p)
    else:
        paths = [target]
    total_chunks = 0
    for p in paths:
        try:
            r = flash.ingest_file(p)
            total_chunks += r["chunks"]
            flag = "CACHED" if r.get("cached") else "indexed"
            print(f"  ✓ {os.path.basename(p)}: {r['chunks']} chunks / ~{r['tokens']} tokens "
                  f"[{flag}] prefill={r['prefill_ms']}ms")
        except Exception as e:
            print(f"  ✗ {p}: {e}")
    print(f"— {len(paths)} file(s), {total_chunks} chunks in window (limit ~{config.MAX_CONTEXT_CHARS // 4} tokens)")


def cmd_ask(flash: BaranFlash, question: str) -> None:
    out = flash.ask(question)
    print(f"\n🧠 [{out['router']['primary']}] experts={[(n, round(w,2)) for n, w in out['router']['experts']]} "
          f"idle={out['router']['idle']} cache={out['cache']} mode={out['mode']} {out['total_ms']}ms")
    print(f"{'─' * 62}\n{out['answer']}\n{'─' * 62}")
    print(f"context: {out['context_used']} chunks · ~{out['approx_tokens_in_window']} tokens in window")


def cmd_serve(flash: BaranFlash) -> None:
    port = int(config.env("BF_PORT", "8085"))

    class Handler(BaseHTTPRequestHandler):
        def _send(self, code: int, payload: dict) -> None:
            body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:
            if self.path.startswith("/health"):
                self._send(200, {"ok": True, "engine": "baranflash-v1", "owner": "Baran Avisat"})
            elif self.path.startswith("/stats"):
                self._send(200, flash.stats())
            else:
                self._send(404, {"error": "not found"})

        def do_POST(self) -> None:
            try:
                n = int(self.headers.get("Content-Length", "0") or "0")
                data = json.loads(self.rfile.read(n).decode("utf-8") or "{}")
            except Exception as e:
                self._send(400, {"error": f"bad json: {e}"})
                return
            if self.path.startswith("/ingest"):
                r = flash.ingest(data.get("text", ""), data.get("source", "api"))
                self._send(200, r)
            elif self.path.startswith("/ask"):
                r = flash.ask(data.get("question", ""))
                self._send(200, r)
            else:
                self._send(404, {"error": "not found"})

        def log_message(self, *a) -> None:  # silence
            pass

    print(f"⚡ BaranFlash HTTP on http://localhost:{port}  (POST /ask {json.dumps({'question': '...'})}, POST /ingest, GET /health /stats)")
    ThreadingHTTPServer(("127.0.0.1", port), Handler).serve_forever()


def main() -> None:
    flash = BaranFlash()
    args = sys.argv[1:]
    if not args:
        print(__doc__)
        print(config.OWNER_CREDIT)
        return
    cmd = args[0]
    if cmd == "ingest" and len(args) > 1:
        cmd_ingest(flash, args[1])
    elif cmd == "ask" and len(args) > 1:
        cmd_ask(flash, " ".join(args[1:]))
    elif cmd == "route" and len(args) > 1:
        g = flash.router.route(" ".join(args[1:]))
        print(json.dumps(g, ensure_ascii=False, indent=2))
    elif cmd == "stats":
        print(json.dumps(flash.stats(), ensure_ascii=False, indent=2))
    elif cmd == "serve":
        cmd_serve(flash)
    else:
        print(__doc__)


if __name__ == "__main__":
    main()
