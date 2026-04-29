from __future__ import annotations

import json
import time
from collections import Counter
from copy import deepcopy
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from src.config import RAGConfig
from src.planning.feature_extractor import QueryFeatureExtractor, QueryFeatures
from src.planning.planner import QueryPlanner
from src.planning.rule_based_router import RuleBasedRouter, _PROFILES

LABEL_ORDER = ["definition", "explanatory", "procedural", "other"]
DEFAULT_MODEL_PATH = "models/ml_router.pkl"
DEFAULT_DATA_PATH = "models/ml_router_training_data.jsonl"
DEFAULT_RETRAIN_INTERVAL = 50


class MLRouter(QueryPlanner):
    def __init__(
        self,
        base_cfg: RAGConfig,
        model_path: str = DEFAULT_MODEL_PATH,
        data_path: str = DEFAULT_DATA_PATH,
        retrain_interval: int = DEFAULT_RETRAIN_INTERVAL,
    ):
        super().__init__(base_cfg)
        self._extractor = QueryFeatureExtractor()
        self._fallback = RuleBasedRouter(base_cfg)
        self._model_path = model_path
        self._data_path = data_path
        self._retrain_interval = retrain_interval
        self._clf = None
        self._model_name: Optional[str] = None
        self._training_samples: List[Tuple[List[float], str]] = []
        self._samples_since_last_train: int = 0

        if Path(model_path).exists():
            self.load_model(model_path)
        if Path(data_path).exists():
            self._load_training_data(data_path)

    @property
    def name(self) -> str:
        return "MLRouter"

    def collect_sample(self, query: str, label: str) -> None:
        if label not in LABEL_ORDER:
            raise ValueError(f"Unknown label {label!r}. Must be one of {LABEL_ORDER}.")
        features = self._extractor.extract(query)
        self._training_samples.append((features.to_vector(), label))

    def collect_from_rule_based(self, query: str) -> str:
        query_type, _ = self._fallback.classify(query)
        self.collect_sample(query, query_type)
        return query_type

    def update(self, query: str, label: str) -> bool:
        self.collect_sample(query, label)
        self._samples_since_last_train += 1
        if self._samples_since_last_train >= self._retrain_interval:
            self.train(save=True)
            self._samples_since_last_train = 0
            return True
        return False

    def save_training_data(self, path: Optional[str] = None) -> None:
        out = path or self._data_path
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w") as f:
            for vec, label in self._training_samples:
                f.write(json.dumps({"features": vec, "label": label}) + "\n")

    def _load_training_data(self, path: str) -> None:
        with open(path) as f:
            for line in f:
                row = json.loads(line)
                self._training_samples.append((row["features"], row["label"]))

    def train(self, save: bool = True) -> dict:
        try:
            from sklearn.linear_model import LogisticRegression
            from sklearn.tree import DecisionTreeClassifier
            from sklearn.ensemble import RandomForestClassifier
            from sklearn.preprocessing import StandardScaler
            from sklearn.pipeline import Pipeline
            from sklearn.model_selection import cross_val_score
            import numpy as np
        except ImportError as e:
            raise ImportError("scikit-learn is required for MLRouter training.") from e

        if len(self._training_samples) < 4:
            raise ValueError(
                f"Need at least 4 training samples (got {len(self._training_samples)})."
            )

        X = [s[0] for s in self._training_samples]
        y = [s[1] for s in self._training_samples]

        candidates = {
            "logistic_regression": Pipeline([
                ("scaler", StandardScaler()),
                ("clf", LogisticRegression(max_iter=1000, C=1.0, random_state=42)),
            ]),
            "decision_tree": Pipeline([
                ("scaler", StandardScaler()),
                ("clf", DecisionTreeClassifier(max_depth=5, random_state=42)),
            ]),
            "random_forest": Pipeline([
                ("scaler", StandardScaler()),
                ("clf", RandomForestClassifier(n_estimators=20, max_depth=5, random_state=42)),
            ]),
        }

        min_class_count = min(Counter(y).values())
        n_splits = min(3, min_class_count) if min_class_count >= 2 else 0

        comparison: Dict[str, dict] = {}
        for model_name, pipeline in candidates.items():
            pipeline.fit(X, y)
            train_acc = pipeline.score(X, y)
            if n_splits >= 2:
                cv_scores = cross_val_score(pipeline, X, y, cv=n_splits, scoring="accuracy")
                cv_acc = float(np.mean(cv_scores))
            else:
                cv_acc = train_acc
            comparison[model_name] = {"train_accuracy": train_acc, "cv_accuracy": cv_acc}

        best_name = max(comparison, key=lambda k: comparison[k]["cv_accuracy"])
        self._clf = candidates[best_name]
        self._model_name = best_name

        print("[MLRouter] Model comparison:")
        for model_name, r in comparison.items():
            marker = " <- selected" if model_name == best_name else ""
            print(f"  {model_name}: train={r['train_accuracy']:.3f}  cv={r['cv_accuracy']:.3f}{marker}")

        if save:
            self.save_model()

        self._samples_since_last_train = 0
        return {
            "selected_model": best_name,
            "model_comparison": comparison,
            "n_samples": len(X),
        }

    def save_model(self, path: Optional[str] = None) -> None:
        import pickle
        out = path or self._model_path
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        with open(out, "wb") as f:
            pickle.dump((self._clf, self._model_name), f)
        print(f"[MLRouter] Model saved to {out}")

    def load_model(self, path: Optional[str] = None) -> None:
        import pickle
        src = path or self._model_path
        with open(src, "rb") as f:
            self._clf, self._model_name = pickle.load(f)
        print(f"[MLRouter] Loaded {self._model_name!r} from {src}")

    def _predict_type(self, features: QueryFeatures) -> Tuple[Optional[str], float]:
        if self._clf is None:
            return None, 0.0
        t0 = time.perf_counter()
        label = self._clf.predict([features.to_vector()])[0]
        return str(label), (time.perf_counter() - t0) * 1000

    def plan(self, query: str) -> RAGConfig:
        features = self._extractor.extract(query)
        query_type, latency_ms = self._predict_type(features)

        if query_type is None:
            print("[MLRouter] No trained model found. Falling back to RuleBasedRouter.")
            return self._fallback.plan(query)

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
            f"[MLRouter] model={self._model_name!r} type={query_type!r} "
            f"weights={cfg.ranker_weights} "
            f"top_k={cfg.top_k} candidates={cfg.num_candidates} "
            f"inference={latency_ms:.2f}ms"
        )
        return cfg
