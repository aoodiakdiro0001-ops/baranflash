"""Long-context memory — the "1M token window" emulation.

* Ingest any size document: chunk with overlap, index terms (BM25), keep
  hierarchical section summaries every N chunks. Chunk groups stream through
  the window instead of fitting in RAM.
* KV-cache emulation: SQLite-backed cache for (a) ingested documents by
  content hash (dedupe: re-ingest is a no-op) and (b) answers by query hash.
"""
import hashlib
import json
import math
import os
import re
import sqlite3
import time
from typing import Any, Dict, List, Optional, Tuple

from . import config

WORD_RX = re.compile(r"[^\W\d_]+|\d+", re.UNICODE)
SECTION_SENT_SPLIT = re.compile(r"(?<=[\.\!\?؟])\s+")

AR_DIACRITICS = re.compile(r"[\u064B-\u0652\u0670\u0640]")  # harakat + tatweel


def normalize(word: str) -> str:
    w = AR_DIACRITICS.sub("", word.lower())
    w = w.replace("أ", "ا").replace("إ", "ا").replace("آ", "ا")
    w = w.replace("ة", "ه").replace("ى", "ي")
    return w


def tokenize(text: str) -> List[str]:
    return [normalize(w) for w in WORD_RX.findall(text or "") if w]


def approx_tokens(text: str) -> int:
    return max(1, len(text) // 4)


class Memory:
    def __init__(self, db_path: str = None):
        os.makedirs(config.DATA_DIR, exist_ok=True)
        self.db = sqlite3.connect(db_path or config.DB_PATH)
        self.db.execute("PRAGMA journal_mode=WAL")
        self._schema()
        self._doc_len_cache: Dict[int, int] = {}
        self._avg_len: Optional[float] = None

    def _schema(self) -> None:
        self.db.executescript(
            """
            CREATE TABLE IF NOT EXISTS chunks (
              id INTEGER PRIMARY KEY AUTOINCREMENT,
              doc_hash TEXT NOT NULL,
              source TEXT NOT NULL,
              ordinal INTEGER NOT NULL,
              text TEXT NOT NULL,
              summary TEXT
            );
            CREATE INDEX IF NOT EXISTS idx_chunks_doc ON chunks(doc_hash);
            CREATE TABLE IF NOT EXISTS terms (
              term TEXT NOT NULL, chunk_id INTEGER NOT NULL, tf INTEGER NOT NULL,
              PRIMARY KEY (term, chunk_id)
            );
            CREATE TABLE IF NOT EXISTS docs (
              doc_hash TEXT PRIMARY KEY, source TEXT, chars INTEGER, tokens INTEGER,
              chunks INTEGER, ingested_at INTEGER
            );
            CREATE TABLE IF NOT EXISTS cache (  -- KV-cache emulation
              qhash TEXT PRIMARY KEY, answer TEXT, meta TEXT, created_at INTEGER
            );
            """
        )
        self.db.commit()

    # ── ingest (PREFILL — fast path, no LLM) ──────────────────────────────
    def ingest(self, text: str, source: str) -> Dict[str, Any]:
        if len(text) > config.MAX_CONTEXT_CHARS:
            text = text[: config.MAX_CONTEXT_CHARS]
        doc_hash = hashlib.sha256(text.encode("utf-8", "ignore")).hexdigest()[:16]
        row = self.db.execute("SELECT chunks FROM docs WHERE doc_hash=?", (doc_hash,)).fetchone()
        if row:  # KV-cache hit on document
            return {"doc_hash": doc_hash, "cached": True, "chunks": row[0],
                    "tokens": approx_tokens(text), "sections": 0}
        words = tokenize(text)
        # chunking with overlap
        step = config.CHUNK_CHARS - config.CHUNK_OVERLAP
        chunks: List[str] = []
        for i in range(0, max(1, len(text)), step):
            chunks.append(text[i: i + config.CHUNK_CHARS])
            if i + config.CHUNK_CHARS >= len(text):
                break
        # index chunks
        chunk_ids: List[int] = []
        for ordinal, chunk in enumerate(chunks):
            cur = self.db.execute(
                "INSERT INTO chunks(doc_hash, source, ordinal, text) VALUES (?,?,?,?)",
                (doc_hash, source, ordinal, chunk),
            )
            cid = cur.lastrowid
            chunk_ids.append(cid)
            tf: Dict[str, int] = {}
            for w in tokenize(chunk):
                tf[w] = tf.get(w, 0) + 1
            self.db.executemany(
                "INSERT OR REPLACE INTO terms(term, chunk_id, tf) VALUES (?,?,?)",
                [(w, cid, c) for w, c in tf.items()],
            )
        # hierarchical section summaries (extractive TF-IDF-lite, zero LLM)
        sections = 0
        for start in range(0, len(chunk_ids), config.SUMMARY_EVERY):
            group = chunk_ids[start: start + config.SUMMARY_EVERY]
            joined = " ".join(self.chunk_text(c) for c in group)
            self.db.execute("UPDATE chunks SET summary=? WHERE id=?",
                            (self._top_sentences(joined, 3), group[0]))
            sections += 1
        doc_row = (doc_hash, source, len(text), approx_tokens(text), len(chunks), int(time.time()))
        self.db.execute("INSERT OR REPLACE INTO docs VALUES (?,?,?,?,?,?)", doc_row)
        self.db.commit()
        self._avg_len = None
        return {"doc_hash": doc_hash, "cached": False, "chunks": len(chunks),
                "tokens": approx_tokens(text), "sections": sections}

    def _top_sentences(self, text: str, n: int) -> str:
        sents = [s.strip() for s in SECTION_SENT_SPLIT.split(text) if len(s.strip()) > 25]
        if not sents:
            return text[:280]
        freq: Dict[str, int] = {}
        for w in tokenize(text):
            freq[w] = freq.get(w, 0) + 1
        def score(s: str) -> float:
            ws = tokenize(s)
            return sum(freq.get(w, 0) for w in ws) / (1 + len(ws))
        return " … ".join(sorted(sents, key=score, reverse=True)[:n])[:600]

    # ── retrieval (BM25) ──────────────────────────────────────────────────
    def chunk_text(self, chunk_id: int) -> str:
        row = self.db.execute("SELECT text FROM chunks WHERE id=?", (chunk_id,)).fetchone()
        return row[0] if row else ""

    def search(self, query: str, top_n: int = 6) -> List[Dict[str, Any]]:
        q_terms = tokenize(query)
        if not q_terms:
            return []
        scores: Dict[int, float] = {}
        for t in q_terms:
            rows = self.db.execute("SELECT chunk_id, tf FROM terms WHERE term=?", (t,)).fetchall()
            df = len(rows)
            if not df:
                continue
            n_docs = max(1, self.db.execute("SELECT COUNT(*) FROM chunks").fetchone()[0])
            idf = math.log(1 + (n_docs - df + 0.5) / (df + 0.5))
            for cid, tf in rows:
                dl = self._doc_len_cache.get(cid) or len(tokenize(self.chunk_text(cid))) or 1
                self._doc_len_cache[cid] = dl
                if self._avg_len is None:
                    tot = self.db.execute("SELECT COUNT(*), COALESCE(SUM(LENGTH(text)),0) FROM chunks").fetchone()
                    self._avg_len = (tot[1] / max(1, tot[0])) or 400
                denom = tf + config.BM25_K1 * (1 - config.BM25_B + config.BM25_B * dl / self._avg_len)
                scores[cid] = scores.get(cid, 0.0) + idf * tf * (config.BM25_K1 + 1) / denom
        ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)[:top_n]
        out = []
        for cid, sc in ranked:
            meta = self.db.execute("SELECT source, ordinal FROM chunks WHERE id=?", (cid,)).fetchone()
            out.append({"chunk_id": cid, "score": round(sc, 3), "source": meta[0] if meta else "?",
                        "ordinal": meta[1] if meta else -1, "text": self.chunk_text(cid)})
        return out

    # ── KV-cache for answers ──────────────────────────────────────────────
    def cache_get(self, qhash: str) -> Optional[Tuple[str, Dict[str, Any]]]:
        row = self.db.execute("SELECT answer, meta FROM cache WHERE qhash=?", (qhash,)).fetchone()
        if row:
            return row[0], json.loads(row[1])
        return None

    def cache_put(self, qhash: str, answer: str, meta: Dict[str, Any]) -> None:
        self.db.execute("INSERT OR REPLACE INTO cache VALUES (?,?,?,?)",
                        (qhash, answer, json.dumps(meta, ensure_ascii=False), int(time.time())))
        self.db.commit()

    # ── stats ─────────────────────────────────────────────────────────────
    def stats(self) -> Dict[str, Any]:
        docs = self.db.execute("SELECT COUNT(*) FROM docs").fetchone()[0]
        chunks = self.db.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
        terms = self.db.execute("SELECT COUNT(DISTINCT term) FROM terms").fetchone()[0]
        cached = self.db.execute("SELECT COUNT(*) FROM cache").fetchone()[0]
        tokens = self.db.execute("SELECT COALESCE(SUM(tokens),0) FROM docs").fetchone()[0]
        return {"documents": docs, "chunks": chunks, "unique_terms": terms,
                "cached_queries": cached, "indexed_tokens": tokens,
                "window_limit_tokens": config.MAX_CONTEXT_CHARS // 4}
