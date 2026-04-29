import time
import pytest

from src.config import RAGConfig
from src.planning.feature_extractor import QueryFeatureExtractor
from src.planning.rule_based_router import RuleBasedRouter


def make_cfg(**overrides) -> RAGConfig:
    defaults = dict(
        embed_model="models/dummy.gguf",
        gen_model="models/dummy.gguf",
        top_k=10,
        num_candidates=50,
        ranker_weights={"faiss": 1.0, "bm25": 0.0, "index_keywords": 0.0},
        ensemble_method="rrf",
        rrf_k=60,
        chunk_size_in_chars=2000,
        chunk_overlap=300,
        max_gen_tokens=400,
        embedding_model_context_window=4096,
        rerank_mode="",
        rerank_top_k=5,
        use_hyde=False,
        hyde_max_tokens=300,
        enable_history=False,
        max_history_turns=3,
        use_indexed_chunks=False,
        use_double_prompt=False,
        enable_topic_extraction=False,
    )
    defaults.update(overrides)
    return RAGConfig(**defaults)


class TestQueryFeatureExtractor:
    def setup_method(self):
        self.extractor = QueryFeatureExtractor()

    @pytest.mark.parametrize("query,expected_flag", [
        ("What is a primary key?", "is_definition"),
        ("Why do we use indexes?", "is_explanatory"),
        ("How to create a join in SQL?", "is_procedural"),
    ])
    def test_category_flags(self, query, expected_flag):
        feat = self.extractor.extract(query)
        assert getattr(feat, expected_flag) is True

    def test_no_flags_for_generic_query(self):
        feat = self.extractor.extract("database performance")
        assert feat.is_definition is False
        assert feat.is_explanatory is False
        assert feat.is_procedural is False


class TestRuleBasedRouterClassification:
    def setup_method(self):
        self.router = RuleBasedRouter(make_cfg())

    @pytest.mark.parametrize("query,expected", [
        ("What is a B-tree?",                    "definition"),
        ("what does ACID stand for?",             "definition"),
        ("What is a database schema",             "definition"),
        ("Why do we use indexes?",                "explanatory"),
        ("How does a B+ tree index organize keys?","explanatory"),
        ("Explain the purpose of normalization",  "explanatory"),
        ("How to implement a hash join?",         "procedural"),
        ("Walk me through creating a transaction","procedural"),
        ("Steps to create a database index",      "procedural"),
        ("database performance",                  "other"),
        ("SQL optimizer",                         "other"),
        ("",                                      "other"),
    ])
    def test_classify(self, query, expected):
        query_type, _ = self.router.classify(query)
        assert query_type == expected, f"Expected {expected!r} for {query!r}, got {query_type!r}"

    def test_definition_beats_explanatory(self):
        query_type, _ = self.router.classify("explain what is an index")
        assert query_type == "definition"

    def test_explanatory_beats_procedural(self):
        query_type, _ = self.router.classify("explain how to implement locking")
        assert query_type == "explanatory"

    def test_all_caps(self):
        query_type, _ = self.router.classify("WHAT IS A B-TREE")
        assert query_type == "definition"


class TestRuleBasedRouterConfig:
    def setup_method(self):
        self.cfg = make_cfg()
        self.router = RuleBasedRouter(self.cfg)

    def test_definition_bm25_only(self):
        routed = self.router.plan("What is a B-tree?")
        assert routed.ranker_weights["faiss"] == pytest.approx(0.0)
        assert routed.ranker_weights["bm25"] == pytest.approx(1.0)

    def test_explanatory_faiss_dominant(self):
        routed = self.router.plan("Why do we use indexes?")
        assert routed.ranker_weights["faiss"] > routed.ranker_weights["bm25"]

    def test_procedural_expands_candidates_and_uses_both(self):
        routed = self.router.plan("How to implement a hash join?")
        assert routed.num_candidates > self.cfg.num_candidates
        assert routed.top_k >= self.cfg.top_k
        assert routed.ranker_weights.get("faiss", 0) > 0
        assert routed.ranker_weights.get("bm25", 0) > 0

    def test_other_preserves_base_weights(self):
        routed = self.router.plan("database performance")
        assert routed.ranker_weights == pytest.approx(self.cfg.ranker_weights)

    def test_base_config_not_mutated(self):
        original_weights = dict(self.cfg.ranker_weights)
        self.router.plan("What is a B-tree?")
        assert self.cfg.ranker_weights == original_weights

    def test_weights_sum_to_one(self):
        for query in ["What is a B-tree?", "Why do we use indexes?",
                      "How to implement a hash join?", "database performance"]:
            routed = self.router.plan(query)
            assert sum(routed.ranker_weights.values()) == pytest.approx(1.0)

    def test_top_k_never_exceeds_candidates(self):
        for query in ["What is a B-tree?", "How to implement a hash join?"]:
            routed = self.router.plan(query)
            assert routed.top_k <= routed.num_candidates

    def test_index_keywords_preserved_when_nonzero(self):
        cfg = make_cfg(ranker_weights={"faiss": 0.5, "bm25": 0.3, "index_keywords": 0.2})
        routed = RuleBasedRouter(cfg).plan("What is a B-tree?")
        assert routed.ranker_weights.get("index_keywords", 0.0) > 0.0
        assert sum(routed.ranker_weights.values()) == pytest.approx(1.0)

    def test_overhead_under_10ms(self):
        t0 = time.perf_counter()
        self.router.plan("What is a database index?")
        assert (time.perf_counter() - t0) * 1000 < 10.0


class TestMLRouter:
    TRAINING_QUERIES = [
        ("What is a B-tree?", "definition"),
        ("define primary key", "definition"),
        ("what does ACID mean?", "definition"),
        ("What are transactions?", "definition"),
        ("Why do we use indexes?", "explanatory"),
        ("explain normalization", "explanatory"),
        ("what causes a deadlock?", "explanatory"),
        ("Why is ACID important?", "explanatory"),
        ("How to implement a hash join?", "procedural"),
        ("steps to create an index", "procedural"),
        ("write a procedure for insertion", "procedural"),
        ("how to build a B-tree?", "procedural"),
        ("database performance", "other"),
        ("SQL optimizer", "other"),
        ("concurrency control", "other"),
        ("storage engine", "other"),
    ]

    def _make_trained_router(self, tmp_path):
        from src.planning.ml_router import MLRouter
        router = MLRouter(make_cfg(), model_path=str(tmp_path / "model.pkl"))
        for query, label in self.TRAINING_QUERIES:
            router.collect_sample(query, label)
        router.train(save=False)
        return router

    def test_fallback_without_model(self, tmp_path):
        pytest.importorskip("sklearn")
        from src.planning.ml_router import MLRouter
        router = MLRouter(make_cfg(), model_path=str(tmp_path / "nonexistent.pkl"))
        assert isinstance(router.plan("What is a B-tree?"), RAGConfig)

    def test_save_and_load_model(self, tmp_path):
        pytest.importorskip("sklearn")
        router = self._make_trained_router(tmp_path)
        model_path = str(tmp_path / "saved.pkl")
        router.save_model(model_path)
        from src.planning.ml_router import MLRouter
        router2 = MLRouter(make_cfg(), model_path=model_path)
        assert router2._model_name == router._model_name
        assert isinstance(router2.plan("What is a B-tree?"), RAGConfig)

    def test_save_and_load_training_data(self, tmp_path):
        pytest.importorskip("sklearn")
        from src.planning.ml_router import MLRouter
        router = MLRouter(make_cfg(), model_path=str(tmp_path / "model.pkl"))
        for query, label in self.TRAINING_QUERIES[:4]:
            router.collect_sample(query, label)
        data_path = str(tmp_path / "data.jsonl")
        router.save_training_data(data_path)
        router2 = MLRouter(make_cfg(), model_path=str(tmp_path / "model.pkl"), data_path=data_path)
        assert len(router2._training_samples) == 4

    def test_update_triggers_retrain_at_interval(self, tmp_path):
        pytest.importorskip("sklearn")
        from src.planning.ml_router import MLRouter
        router = MLRouter(make_cfg(), model_path=str(tmp_path / "model.pkl"), retrain_interval=4)
        for query, label in [
            ("What is a B-tree?", "definition"),
            ("Why do we use indexes?", "explanatory"),
            ("How to implement a hash join?", "procedural"),
            ("database performance", "other"),
        ]:
            router.update(query, label)
        assert router._samples_since_last_train == 0
        assert router._clf is not None

    def test_latency_under_50ms(self, tmp_path):
        pytest.importorskip("sklearn")
        router = self._make_trained_router(tmp_path)
        t0 = time.perf_counter()
        router.plan("What is a primary key?")
        assert (time.perf_counter() - t0) * 1000 < 50.0

    AGREEMENT_TEST_QUERIES = [
        "What is a checkpoint in database recovery?",
        "What is the difference between a primary and secondary index?",
        "What are phantom reads?",
        "Why does two-phase locking prevent non-serializable schedules?",
        "How does MVCC handle read-write conflicts?",
        "Explain what a dirty read is",
        "How to implement a nested loop join?",
        "Steps to write a recursive SQL query",
        "Walk me through creating a clustered index",
        "How to handle a deadlock in application code?",
        "database tuning",
        "execution plan analysis",
        "lock granularity",
        "Write a trigger in SQL",
        "Steps to roll back a failed transaction",
    ]

    def test_ml_router_agrees_with_rule_based(self, tmp_path):
        pytest.importorskip("sklearn")
        from src.planning.ml_router import MLRouter

        cfg = make_cfg()
        rule_router = RuleBasedRouter(cfg)
        ml_router = MLRouter(cfg, model_path=str(tmp_path / "model.pkl"))

        for query, label in self.TRAINING_QUERIES:
            ml_router.collect_sample(query, label)
        ml_router.train(save=False)

        agreement = sum(
            1 for query in self.AGREEMENT_TEST_QUERIES
            if rule_router.plan(query).ranker_weights ==
               pytest.approx(ml_router.plan(query).ranker_weights, abs=0.01)
        )
        rate = agreement / len(self.AGREEMENT_TEST_QUERIES)
        print(f"\nMLRouter agreement: {agreement}/{len(self.AGREEMENT_TEST_QUERIES)} = {rate:.0%}")
        assert rate >= 0.80, f"MLRouter agreement {rate:.0%} is below 80%"
