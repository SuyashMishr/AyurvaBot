"""Query processing, normalization, expansion, and multi-query generation.

Goals:
 1. Disambiguate vague user queries.
 2. Expand with synonyms & domain terms (controlled to avoid drift).
 3. Generate multiple focused paraphrases (RAG Fusion style) for broader recall.
 4. Named entity injection (if entities extracted from original query).
"""

from __future__ import annotations

import re
from typing import List, Dict, Set

try:
    from nltk.corpus import wordnet as wn  # type: ignore
except Exception:  # pragma: no cover
    wn = None  # type: ignore


DOMAIN_SYNONYMS = {
    "fever": ["pyrexia", "temperature", "jwara"],
    "heart": ["cardiac", "cardiovascular"],
    "digestion": ["stomach", "gastric", "agni"],
    "cough": ["respiratory", "bronchial"],
    "immunity": ["ojas", "resistance"],
    "detox": ["panchakarma", "cleanse"],
}


STOP_CHARS_PATTERN = re.compile(r"[^a-zA-Z0-9\s-]")


class QueryProcessor:
    def __init__(self, enable_wordnet: bool = True, max_expansions: int = 4):
        self.enable_wordnet = enable_wordnet and wn is not None
        self.max_expansions = max_expansions

    # --- DISAMBIGUATION / ACRONYM HANDLING ---
    _ACRONYMS = {
        "BP": "blood pressure",
        "GI": "gastrointestinal",
        "CNS": "central nervous system",
    }
    _POLYSEMY_HINTS = {
        "cold": ["common cold", "low temperature"],
        "stress": ["mental stress", "physiological stress"],
        "pressure": ["blood pressure"],
    }

    def disambiguate(self, query: str) -> str:
        words = query.split()
        expanded = []
        for w in words:
            key = w.upper()
            if key in self._ACRONYMS:
                expanded.append(self._ACRONYMS[key])
            expanded.append(w)
        dis_q = " ".join(expanded)
        # Add clarifying hint if strongly polysemous single word queries
        if len(words) == 1 and words[0].lower() in self._POLYSEMY_HINTS:
            dis_q += " (" + " | ".join(self._POLYSEMY_HINTS[words[0].lower()][:2]) + ")"
        return dis_q

    def normalize(self, query: str) -> str:
        q = query.strip()
        q = STOP_CHARS_PATTERN.sub(" ", q)
        q = re.sub(r"\s+", " ", q)
        return q.strip()

    def _wordnet_synonyms(self, term: str) -> List[str]:
        if not self.enable_wordnet:
            return []
        syns: Set[str] = set()
        try:
            for syn in wn.synsets(term)[:2]:  # limit
                for lemma in syn.lemmas():
                    name = lemma.name().replace("_", " ")
                    if name.lower() != term.lower() and len(name) <= 20:
                        syns.add(name)
        except Exception:  # pragma: no cover
            pass
        return list(syns)[: self.max_expansions]

    def expand(self, query: str) -> Dict[str, List[str]]:
        base = self.normalize(query)
        tokens = [t for t in base.lower().split() if len(t) > 2]
        expansions: Set[str] = set()
        domain_hits: Set[str] = set()
        for tok in tokens:
            if tok in DOMAIN_SYNONYMS:
                for s in DOMAIN_SYNONYMS[tok][: self.max_expansions]:
                    expansions.add(s)
                domain_hits.add(tok)
            # limited wordnet lookups
            for wn_syn in self._wordnet_synonyms(tok):
                expansions.add(wn_syn)
        expansions = {e for e in expansions if e not in tokens}
        return {
            "normalized": base,
            "tokens": tokens,
            "domain_terms": list(domain_hits),
            "expansions": list(expansions),
        }

    def multi_queries(self, query: str, max_variants: int = 3) -> List[str]:
        # First normalize & disambiguate
        query = self.disambiguate(query)
        info = self.expand(query)
        base = info["normalized"]
        variants = [base]
        # variant1: add top 1-2 expansions inline
        if info["expansions"]:
            variants.append(base + " " + " ".join(info["expansions"][:2]))
        # variant2: if we have domain term synonyms, create structured query
        if info["domain_terms"]:
            variants.append(
                "; ".join(
                    f"{dt} OR {' OR '.join(DOMAIN_SYNONYMS.get(dt, [])[:2])}" for dt in info["domain_terms"]
                )
            )
        # Deduplicate & limit
        seen = set()
        final: List[str] = []
        for v in variants:
            vv = v.strip()
            if vv and vv.lower() not in seen:
                seen.add(vv.lower())
                final.append(vv)
            if len(final) >= max_variants:
                break
        return final


__all__ = ["QueryProcessor"]
