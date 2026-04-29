from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

DEFINITION_KEYWORDS = [
    "what is", "what are", "what's", "define", "definition", "meaning of",
    "what does", "what do", "describe what", "what exactly",
]

EXPLANATORY_KEYWORDS = [
    "why", "explain", "because", "reason", "purpose of", "how does",
    "how do", "what causes", "elaborate", "what makes",
]

PROCEDURAL_KEYWORDS = [
    "how to", "steps to", "steps for", "procedure", "algorithm",
    "implement", "write a", "create a", "build a", "walk me through",
    "instructions for", "how can i",
]

QUESTION_WORDS = {"what", "why", "how", "when", "where", "who", "which", "is", "are", "can", "does", "do"}


@dataclass
class QueryFeatures:
    char_length: int
    token_count: int
    leading_question_word: Optional[str]
    is_definition: bool
    is_explanatory: bool
    is_procedural: bool

    def to_vector(self) -> List[float]:
        return [
            float(self.char_length),
            float(self.token_count),
            float(self.is_definition),
            float(self.is_explanatory),
            float(self.is_procedural),
            float(self.leading_question_word == "what"),
            float(self.leading_question_word == "why"),
            float(self.leading_question_word == "how"),
        ]


class QueryFeatureExtractor:
    def extract(self, query: str) -> QueryFeatures:
        q_lower = query.lower().strip()
        tokens = q_lower.split()
        leading_word = tokens[0] if tokens and tokens[0] in QUESTION_WORDS else None

        return QueryFeatures(
            char_length=len(query),
            token_count=len(tokens),
            leading_question_word=leading_word,
            is_definition=any(kw in q_lower for kw in DEFINITION_KEYWORDS),
            is_explanatory=any(kw in q_lower for kw in EXPLANATORY_KEYWORDS),
            is_procedural=any(kw in q_lower for kw in PROCEDURAL_KEYWORDS),
        )
