from src.config import RAGConfig
from copy import deepcopy

from src.planning.planner import QueryPlanner


class HeuristicQueryPlanner(QueryPlanner):
    @property
    def name(self) -> str:
        return "HeuristicBasedPlanner"

    def __init__(self, base_cfg: RAGConfig):
        super().__init__(base_cfg)
        self.base_cfg = deepcopy(base_cfg)

    def classify(self, query: str) -> str:
        q = query.lower()
        if any(x in q for x in ["what is", "define", "definition"]):
            return "definition"
        if any(x in q for x in ["why", "explain", "because"]):
            return "explanatory"
        if any(x in q for x in ["how to", "steps", "procedure", "algorithm"]):
            return "procedural"
        return "other"

    def plan(self, query: str) -> RAGConfig:
        kind = self.classify(query)
        cfg = deepcopy(self.base_cfg)

        if kind == "definition":
            cfg.ranker_weights = {"faiss": 0.3, "bm25": 0.7}

        elif kind == "explanatory":
            cfg.ranker_weights = {"faiss": 0.7, "bm25": 0.3}

        elif kind == "procedural":
            cfg.num_candidates = max(cfg.num_candidates, cfg.top_k * 5)
            cfg.ranker_weights = {"faiss": 0.6, "bm25": 0.4}

        else:
            cfg.ranker_weights = {"faiss": 0.7, "bm25": 0.3}

        self._log_decision(cfg)
        return cfg
