"""Named Entity & domain phrase extraction for filtering retrieval results.

Combines spaCy NER (if installed) with a light-weight domain gazetteer for
Ayurvedic herbs, conditions, and core concepts. Used for Named Entity Filtering (NEF).
"""

from __future__ import annotations

from typing import List, Dict, Set
import re

try:
    import spacy  # type: ignore
    _SPACY_OK = True
except Exception:  # pragma: no cover
    spacy = None  # type: ignore
    _SPACY_OK = False


HERB_TERMS = [
    "ashwagandha", "tulsi", "giloy", "brahmi", "turmeric", "triphala", "arjuna", "shatavari",
    "guggulu", "licorice", "mulethi", "hing", "ajwain", "fennel", "cumin", "ginger", "amla",
]
CONDITION_TERMS = [
    "fever", "jwara", "cough", "cold", "asthma", "digestion", "acidity", "insomnia", "stress",
    "immunity", "heart", "cardiac", "respiratory", "fatigue", "arthritis",
]
CORE_CONCEPTS = ["vata", "pitta", "kapha", "agni", "ojas", "ama", "panchakarma"]

GAZETTEER = set(HERB_TERMS + CONDITION_TERMS + CORE_CONCEPTS)


class EntityExtractor:
    def __init__(self, use_spacy: bool = False, max_chars: int = 4000) -> None:
        self.use_spacy = use_spacy and _SPACY_OK
        self.max_chars = max_chars
        self.model = None
        if self.use_spacy:
            try:  # attempt lightweight model load
                self.model = spacy.load("en_core_web_sm")  # pragma: no cover (heavy)
            except Exception:
                self.model = None
        # Precompile simple noun phrase heuristic
        self._np_pattern = re.compile(r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,2})\b")

    def extract(self, text: str) -> Dict[str, List[str]]:
        text_lower = text.lower()
        found_gazetteer: Set[str] = {g for g in GAZETTEER if g in text_lower}
        ner_entities: Set[str] = set()
        if self.model and len(text) <= self.max_chars:
            try:
                doc = self.model(text)
                for ent in doc.ents:
                    if ent.label_ in {"PERSON", "ORG", "GPE", "NORP", "FAC", "PRODUCT"}:
                        ner_entities.add(ent.text.strip())
            except Exception:  # pragma: no cover
                pass
        # simple noun phrase heuristic for capitalized domain words (bounded)
        sample = text[: self.max_chars]
        caps = re.findall(r"\b([A-Z][a-z]{3,})\b", sample)
        for c in caps:
            if c.lower() in GAZETTEER:
                found_gazetteer.add(c.lower())
        # Lightweight noun phrase extraction (capitalized sequences) - aids NEF
        for m in self._np_pattern.findall(sample):
            low = m.lower()
            for token in low.split():
                if token in GAZETTEER:
                    found_gazetteer.add(token)
        return {
            "gazetteer": sorted(found_gazetteer),
            "ner": sorted(ner_entities),
        }


__all__ = ["EntityExtractor"]
