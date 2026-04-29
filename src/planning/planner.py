from __future__ import annotations
from abc import ABC, abstractmethod
from copy import deepcopy

from src.config import RAGConfig


class QueryPlanner(ABC):
    def __init__(self, base_cfg: RAGConfig):
        self.base_cfg = deepcopy(base_cfg)

    @property
    @abstractmethod
    def name(self) -> str: ...

    @abstractmethod
    def plan(self, query: str) -> RAGConfig: ...

    def _log_decision(self, new_cfg: RAGConfig) -> None:
        pass
