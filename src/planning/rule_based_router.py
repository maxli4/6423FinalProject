from __future__ import annotations

from copy import deepcopy
from typing import Tuple

from src.config import RAGConfig
from src.planning.planner import QueryPlanner
from src.planning.feature_extractor import QueryFeatureExtractor, QueryFeatures

_PROFILES: dict = {
    "definition": {
        "ranker_weights": {"faiss": 0.0, "bm25": 1.0, "index_keywords": 0.0},
        "top_k_delta": 0,
        "candidates_multiplier": 1.0,
    },
    "explanatory": {
        "ranker_weights": {"faiss": 0.7, "bm25": 0.3, "index_keywords": 0.0},
        "top_k_delta": 0,
        "candidates_multiplier": 1.0,
    },
    "procedural": {
        "ranker_weights": {"faiss": 0.6, "bm25": 0.4, "index_keywords": 0.0},
        "top_k_delta": 2,
        "candidates_multiplier": 1.5,
    },
    "other": {
        "ranker_weights": None,
        "top_k_delta": 0,
        "candidates_multiplier": 1.0,
    },
}


class RuleBasedRouter(QueryPlanner):
    def __init__(self, base_cfg: RAGConfig):
        super().__init__(base_cfg)
        self._extractor = QueryFeatureExtractor()

    @property
    def name(self) -> str:
        return "RuleBasedRouter"

    def classify(self, query: str) -> Tuple[str, QueryFeatures]:
        features = self._extractor.extract(query)
        if features.is_definition:
            return "definition", features
        if features.is_explanatory:
            return "explanatory", features
        if features.is_procedural:
            return "procedural", features
        return "other", features

    def plan(self, query: str) -> RAGConfig:
        query_type, _ = self.classify(query)
        cfg = deepcopy(self.base_cfg)
        profile = _PROFILES[query_type]

        if profile["ranker_weights"] is not None:
            weights = dict(profile["ranker_weights"])
            base_index_w = self.base_cfg.ranker_weights.get("index_keywords", 0.0)
            if base_index_w > 0:
                weights["index_keywords"] = base_index_w
            total = sum(weights.values())
            cfg.ranker_weights = {k: v / total for k, v in weights.items()}

        new_candidates = int(self.base_cfg.num_candidates * profile["candidates_multiplier"])
        cfg.num_candidates = max(self.base_cfg.num_candidates, new_candidates)
        cfg.top_k = min(self.base_cfg.top_k + profile["top_k_delta"], cfg.num_candidates)

        print(
            f"[RuleBasedRouter] type={query_type!r} "
            f"weights={cfg.ranker_weights} "
            f"top_k={cfg.top_k} candidates={cfg.num_candidates}"
        )
        return cfg
