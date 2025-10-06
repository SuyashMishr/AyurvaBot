"""Advanced chunking strategies for AyurvaBot RAG system.

Implements multiple strategies discussed in the improvement plan:

1. Sliding Window Chunking: maintains local continuity.
2. Recursive Structural Chunking: splits by headings / blank lines / paragraphs.
3. Semantic (Sentence) Chunking: groups sentences up to a target token length.
4. Hybrid Chunking (default): combines structural splits, then semantic packing with overlap.

Design Goals:
 - Preserve context (avoid over-fragmentation that harms semantic meaning)
 - Avoid overly large chunks that waste context window
 - Provide fast, dependency-light fallbacks (spaCy optional)
 - Provide lightweight summaries for prompt compression & reranking

Returned chunk objects: { 'text': str, 'summary': str }

NOTE: Token counting is approximated by splitting on whitespace. This keeps
ingestion fast; actual embedding model uses subword tokens but approximate
limits are adequate for chunk sizing heuristics.
"""

from __future__ import annotations

from typing import List, Dict, Iterable
import re

try:  # optional sentence boundary detection
    import spacy  # type: ignore
    _NLP = None
except Exception:  # pragma: no cover
    spacy = None  # type: ignore
    _NLP = None


HEADING_PATTERN = re.compile(r"^(#{1,6}\s+|[A-Z][A-Z\s]{6,}|\d+\.\s+)".strip())
MULTI_NEWLINE_PATTERN = re.compile(r"\n{2,}")


def _ensure_nlp():  # lazy load to avoid startup penalty
    global _NLP
    if _NLP is None and spacy is not None:
        try:  # pragma: no cover (heavy)
            _NLP = spacy.load("en_core_web_sm")
        except Exception:
            _NLP = False
    return _NLP


def _split_sentences(text: str) -> List[str]:
    nlp = _ensure_nlp()
    if nlp:
        try:  # pragma: no cover
            doc = nlp(text)
            return [s.text.strip() for s in doc.sents if s.text.strip()]
        except Exception:
            pass
    # Regex fallback (naive)
    parts = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9])", text.strip())
    return [p.strip() for p in parts if p.strip()]


def _summarize(text: str, max_chars: int = 180) -> str:
    # First sentence or truncated slice
    sentences = _split_sentences(text)
    if sentences:
        first = sentences[0]
        if len(first) <= max_chars:
            return first
    return (text[: max_chars].rsplit(" ", 1)[0] + "…") if len(text) > max_chars else text


def sliding_window_chunk(text: str, window_tokens: int = 140, overlap_tokens: int = 40) -> List[Dict]:
    tokens = text.split()
    chunks: List[Dict] = []
    if not tokens:
        return chunks
    start = 0
    while start < len(tokens):
        end = min(start + window_tokens, len(tokens))
        piece = " ".join(tokens[start:end]).strip()
        if piece:
            chunks.append({"text": piece, "summary": _summarize(piece)})
        if end == len(tokens):
            break
        start = end - overlap_tokens
        if start < 0:
            start = 0
        if start >= len(tokens):
            break
    return chunks


def structural_split(text: str) -> List[str]:
    # Split by blank lines first
    parts = [p.strip() for p in MULTI_NEWLINE_PATTERN.split(text) if p.strip()]
    refined: List[str] = []
    for p in parts:
        # Further split long blocks on headings or large uppercase sequences
        lines = p.splitlines()
        current: List[str] = []
        for line in lines:
            if HEADING_PATTERN.match(line.strip()) and current:
                refined.append(" ".join(current).strip())
                current = [line.strip()]
            else:
                current.append(line.strip())
        if current:
            refined.append(" ".join(current).strip())
    return [r for r in refined if r]


def semantic_pack(paragraphs: Iterable[str], target_tokens: int = 110, max_tokens: int = 180) -> List[Dict]:
    chunks: List[Dict] = []
    buffer: List[str] = []
    buffer_len = 0
    for para in paragraphs:
        ptoks = para.split()
        ptok_len = len(ptoks)
        # If single paragraph is huge, fallback to sliding window inside
        if ptok_len > max_tokens:
            if buffer:
                merged = " ".join(buffer).strip()
                if merged:
                    chunks.append({"text": merged, "summary": _summarize(merged)})
                buffer, buffer_len = [], 0
            sw = sliding_window_chunk(para, window_tokens=target_tokens, overlap_tokens=35)
            chunks.extend(sw)
            continue
        if buffer_len + ptok_len <= max_tokens:
            buffer.append(para)
            buffer_len += ptok_len
            # If we reached target range, flush
            if buffer_len >= target_tokens:
                merged = " ".join(buffer).strip()
                if merged:
                    chunks.append({"text": merged, "summary": _summarize(merged)})
                buffer, buffer_len = [], 0
        else:
            # flush current buffer
            if buffer:
                merged = " ".join(buffer).strip()
                if merged:
                    chunks.append({"text": merged, "summary": _summarize(merged)})
            buffer = [para]
            buffer_len = ptok_len
    if buffer:
        merged = " ".join(buffer).strip()
        if merged:
            chunks.append({"text": merged, "summary": _summarize(merged)})
    return chunks


def semantic_chunk(text: str, strategy: str = "hybrid", target_tokens: int = 110, max_tokens: int = 180, overlap: int = 35) -> List[Dict]:
    """Primary entry point used by knowledge base.

    strategy options:
        - 'sliding'
        - 'structural'
        - 'semantic'
        - 'hybrid' (structural -> semantic pack -> sliding enrichment)
    """
    text = (text or "").strip()
    if not text:
        return []
    if strategy == "sliding":
        return sliding_window_chunk(text, window_tokens=target_tokens, overlap_tokens=overlap)
    if strategy == "structural":
        paras = structural_split(text)
        return [{"text": p, "summary": _summarize(p)} for p in paras]
    if strategy == "semantic":
        sentences = _split_sentences(text)
        return semantic_pack(sentences, target_tokens=target_tokens, max_tokens=max_tokens)
    # hybrid
    paragraphs = structural_split(text)
    primary = semantic_pack(paragraphs, target_tokens=target_tokens, max_tokens=max_tokens)
    # Optional continuity enrichment: add sliding window over first large paragraph if content sparse
    if len(primary) <= 2 and len(text.split()) > max_tokens * 1.8:
        sw = sliding_window_chunk(text, window_tokens=max_tokens, overlap_tokens=overlap)
        # merge unique texts
        seen = set()
        merged = []
        for c in primary + sw:
            key = c['text'][:120]
            if key in seen:
                continue
            seen.add(key)
            merged.append(c)
        return merged
    return primary


__all__ = [
    "semantic_chunk",
    "sliding_window_chunk",
    "structural_split",
]
