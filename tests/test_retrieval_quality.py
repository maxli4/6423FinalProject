import math
import time
import pathlib
import pytest

ARTIFACTS_DIR = pathlib.Path("index/sections")
EMBED_MODEL = "models/Qwen3-Embedding-4B-Q5_K_M.gguf"
INDEX_PREFIX = "textbook_index"

BENCHMARKS = [
    {"id": "acid_properties",          "query": "What are the ACID properties of transactions?",                   "query_type_expected": "definition",   "ideal": [1067, 1479, 1104, 1063, 1069]},
    {"id": "bptree",                   "query": "How does a B+ tree index organize keys and support search?",      "query_type_expected": "explanatory",  "ideal": [837, 867, 855, 863, 779]},
    {"id": "fd_normalization",         "query": "What are functional dependencies?",                               "query_type_expected": "definition",   "ideal": [387, 405, 411, 388, 401]},
    {"id": "sql_isolation",            "query": "What isolation guarantees does SQL provide by default?",          "query_type_expected": "definition",   "ideal": [1095, 1096, 1172, 1103, 1067]},
    {"id": "primary_foreign_keys",     "query": "Explain primary keys and foreign keys",                           "query_type_expected": "explanatory",  "ideal": [181, 58, 317, 316, 318]},
    {"id": "database_schema",          "query": "What is a database schema",                                       "query_type_expected": "definition",   "ideal": [19, 55, 383, 204, 447]},
    {"id": "aries_atomicity",          "query": "How does the recovery manager use ARIES to ensure atomicity",     "query_type_expected": "explanatory",  "ideal": [1265, 37, 1275, 1205, 1267]},
    {"id": "insert_bptree_steps",      "query": "Walk me through the steps to insert a key into a B+ tree",       "query_type_expected": "procedural",   "ideal": [847, 845, 840, 839, 891]},
    {"id": "begin_commit_transaction", "query": "Walk me through the steps to begin and commit a SQL transaction", "query_type_expected": "procedural",   "ideal": [1096, 174, 1065, 172, 1068]},
    {"id": "normalize_to_3nf",         "query": "Walk me through decomposing a relation into third normal form",   "query_type_expected": "procedural",   "ideal": [399, 383, 434, 394, 430]},
    {"id": "oltp_vs_analytics",        "query": "Contrast the goals of OLTP and data analytics",                  "query_type_expected": "other",        "ideal": [1692, 1694, 675, 4, 678]},
    {"id": "lossy_decomposition",      "query": "Show me what happens during a lossy decomposition",              "query_type_expected": "other",        "ideal": [382, 381, 390, 421, 436]},
]


def ndcg_at_k(retrieved_ids, ideal_ids, k=5):
    ideal_set = set(ideal_ids)
    dcg = sum(
        1.0 / math.log2(rank + 2)
        for rank, cid in enumerate(retrieved_ids[:k])
        if cid in ideal_set
    )
    idcg = sum(1.0 / math.log2(rank + 2) for rank in range(min(k, len(ideal_ids))))
    return dcg / idcg if idcg > 0 else 0.0


def mrr(retrieved_ids, ideal_ids):
    ideal_set = set(ideal_ids)
    for rank, cid in enumerate(retrieved_ids, 1):
        if cid in ideal_set:
            return 1.0 / rank
    return 0.0


def retrieve_chunk_ids(query, cfg, artifacts):
    from src.retriever import filter_retrieved_chunks
    from src.ranking.ranker import EnsembleRanker

    ranker = EnsembleRanker(
        ensemble_method=cfg.ensemble_method,
        weights=cfg.ranker_weights,
        rrf_k=int(cfg.rrf_k),
    )
    pool_n = max(cfg.num_candidates, cfg.top_k + 10)
    raw_scores = {}
    for r in artifacts["retrievers"]:
        if cfg.ranker_weights.get(r.name, 0) > 0:
            raw_scores[r.name] = r.get_scores(query, pool_n, artifacts["chunks"])
    ordered, _ = ranker.rank(raw_scores)
    return filter_retrieved_chunks(cfg, artifacts["chunks"], ordered)


pytestmark = pytest.mark.skipif(
    not ARTIFACTS_DIR.exists(),
    reason="Index artifacts not found — run index mode first",
)


@pytest.fixture(scope="module")
def artifacts():
    from src.retriever import FAISSRetriever, BM25Retriever, load_artifacts
    faiss_idx, bm25_idx, chunks, sources, meta = load_artifacts(ARTIFACTS_DIR, INDEX_PREFIX)
    retrievers = [FAISSRetriever(faiss_idx, EMBED_MODEL), BM25Retriever(bm25_idx)]
    return {"chunks": chunks, "retrievers": retrievers}


@pytest.fixture(scope="module")
def base_cfg():
    from src.config import RAGConfig
    return RAGConfig(
        embed_model=EMBED_MODEL,
        gen_model="models/qwen2.5-3b-instruct-q8_0.gguf",
        top_k=10,
        num_candidates=50,
        ranker_weights={"faiss": 1.0, "bm25": 0.0, "index_keywords": 0.0},
        ensemble_method="rrf",
        rrf_k=60,
        rerank_mode="",
        rerank_top_k=5,
        chunk_size_in_chars=2000,
        chunk_overlap=300,
        max_gen_tokens=400,
        embedding_model_context_window=4096,
        use_hyde=False,
        hyde_max_tokens=300,
        enable_history=False,
        max_history_turns=3,
        use_indexed_chunks=False,
        use_double_prompt=False,
        enable_topic_extraction=False,
    )


@pytest.fixture(scope="module")
def trained_ml_router(base_cfg):
    from src.planning.ml_router import MLRouter
    router = MLRouter(base_cfg, model_path="/tmp/test_ml_router.pkl")
    for bm in BENCHMARKS:
        router.collect_from_rule_based(bm["query"])
    router.train(save=False)
    return router


def test_all_query_types_have_benchmarks():
    types_covered = {b["query_type_expected"] for b in BENCHMARKS}
    for required in ("definition", "explanatory", "procedural", "other"):
        assert required in types_covered, f"No benchmark for query type {required!r}"


def test_procedural_routing_expands_candidates(base_cfg):
    from src.planning.rule_based_router import RuleBasedRouter
    router = RuleBasedRouter(base_cfg)
    for bm in [b for b in BENCHMARKS if b["query_type_expected"] == "procedural"]:
        routed = router.plan(bm["query"])
        assert routed.num_candidates > base_cfg.num_candidates
        assert routed.top_k >= base_cfg.top_k
        assert routed.ranker_weights.get("faiss", 0) > 0
        assert routed.ranker_weights.get("bm25", 0) > 0


def test_router_classifies_benchmarks_correctly(base_cfg):
    from src.planning.rule_based_router import RuleBasedRouter
    router = RuleBasedRouter(base_cfg)
    for bm in BENCHMARKS:
        actual, _ = router.classify(bm["query"])
        assert actual == bm["query_type_expected"], (
            f"{bm['id']!r}: expected {bm['query_type_expected']!r}, got {actual!r}"
        )


def test_ml_router_beats_fixed_average_ndcg(artifacts, base_cfg, trained_ml_router):
    fixed_scores, ml_scores = [], []
    for bm in BENCHMARKS:
        fixed_ids = retrieve_chunk_ids(bm["query"], base_cfg, artifacts)
        ml_ids = retrieve_chunk_ids(bm["query"], trained_ml_router.plan(bm["query"]), artifacts)
        fixed_scores.append(ndcg_at_k(fixed_ids, bm["ideal"]))
        ml_scores.append(ndcg_at_k(ml_ids, bm["ideal"]))

    avg_fixed = sum(fixed_scores) / len(fixed_scores)
    avg_ml = sum(ml_scores) / len(ml_scores)

    W = 100
    print(f"\n{'─'*W}")
    print(f"  {'Benchmark':<28} {'Type':<13} {'Fixed NDCG':>10} {'ML NDCG':>9} {'Winner':>8}")
    print(f"{'─'*W}")
    for bm, fs, ms in zip(BENCHMARKS, fixed_scores, ml_scores):
        winner = "ML ↑" if ms > fs + 0.01 else ("Fix ↑" if fs > ms + 0.01 else "TIE")
        print(f"  {bm['id']:<28} {bm['query_type_expected']:<13} {fs:>10.3f} {ms:>9.3f} {winner:>8}")
    print(f"{'─'*W}")
    print(f"  {'AVERAGE':<41} {avg_fixed:>10.3f} {avg_ml:>9.3f}")
    print(f"{'─'*W}")
    ml_wins = sum(1 for f, m in zip(fixed_scores, ml_scores) if m > f + 0.01)
    fix_wins = sum(1 for f, m in zip(fixed_scores, ml_scores) if f > m + 0.01)
    print(f"\n  ML wins: {ml_wins}/{len(BENCHMARKS)}  Fixed wins: {fix_wins}/{len(BENCHMARKS)}  Ties: {len(BENCHMARKS)-ml_wins-fix_wins}/{len(BENCHMARKS)}\n")

    assert avg_ml > avg_fixed, (
        f"MLRouter avg NDCG ({avg_ml:.3f}) did not beat fixed config ({avg_fixed:.3f})"
    )


def test_ml_router_wins_procedural_queries(artifacts, base_cfg, trained_ml_router):
    proc_bms = [b for b in BENCHMARKS if b["query_type_expected"] == "procedural"]
    fixed_scores, ml_scores = [], []
    for bm in proc_bms:
        fixed_ids = retrieve_chunk_ids(bm["query"], base_cfg, artifacts)
        ml_ids = retrieve_chunk_ids(bm["query"], trained_ml_router.plan(bm["query"]), artifacts)
        fixed_scores.append(ndcg_at_k(fixed_ids, bm["ideal"]))
        ml_scores.append(ndcg_at_k(ml_ids, bm["ideal"]))

    avg_fixed = sum(fixed_scores) / len(fixed_scores)
    avg_ml = sum(ml_scores) / len(ml_scores)
    print(f"\nProcedural NDCG@5 — Fixed: {avg_fixed:.3f}  ML: {avg_ml:.3f}")
    assert avg_ml > avg_fixed, (
        f"MLRouter did not improve procedural retrieval: ML={avg_ml:.3f} Fixed={avg_fixed:.3f}"
    )


def test_ml_router_wins_explanatory_queries(artifacts, base_cfg, trained_ml_router):
    exp_bms = [b for b in BENCHMARKS if b["query_type_expected"] == "explanatory"]
    fixed_scores, ml_scores = [], []
    for bm in exp_bms:
        fixed_ids = retrieve_chunk_ids(bm["query"], base_cfg, artifacts)
        ml_ids = retrieve_chunk_ids(bm["query"], trained_ml_router.plan(bm["query"]), artifacts)
        fixed_scores.append(ndcg_at_k(fixed_ids, bm["ideal"]))
        ml_scores.append(ndcg_at_k(ml_ids, bm["ideal"]))

    avg_fixed = sum(fixed_scores) / len(fixed_scores)
    avg_ml = sum(ml_scores) / len(ml_scores)
    print(f"\nExplanatory NDCG@5 — Fixed: {avg_fixed:.3f}  ML: {avg_ml:.3f}")
    assert avg_ml > avg_fixed, (
        f"MLRouter did not improve explanatory retrieval: ML={avg_ml:.3f} Fixed={avg_fixed:.3f}"
    )


def test_ml_router_overhead_is_negligible(trained_ml_router):
    router_times = []
    for bm in BENCHMARKS:
        t0 = time.perf_counter()
        trained_ml_router.plan(bm["query"])
        router_times.append((time.perf_counter() - t0) * 1000)

    avg_ms = sum(router_times) / len(router_times)
    print(f"\nAvg MLRouter overhead: {avg_ms:.2f}ms")
    assert avg_ms < 5.0, f"MLRouter adds {avg_ms:.2f}ms avg — exceeds 5ms threshold"


def test_definition_queries_skip_faiss(artifacts, base_cfg):
    from src.planning.rule_based_router import RuleBasedRouter

    class CallSpy:
        def __init__(self, wrapped):
            self._wrapped = wrapped
            self.call_count = 0

        def get_scores(self, *args, **kwargs):
            self.call_count += 1
            return self._wrapped.get_scores(*args, **kwargs)

        @property
        def name(self):
            return self._wrapped.name

    router = RuleBasedRouter(base_cfg)
    definition_bms = [b for b in BENCHMARKS if b["query_type_expected"] == "definition"]

    faiss_spy = CallSpy(artifacts["retrievers"][0])
    bm25_spy  = CallSpy(artifacts["retrievers"][1])
    spy_artifacts = {**artifacts, "retrievers": [faiss_spy, bm25_spy]}

    for bm in definition_bms:
        routed_cfg = router.plan(bm["query"])
        retrieve_chunk_ids(bm["query"], routed_cfg, spy_artifacts)

    print(f"\n  FAISS calls: {faiss_spy.call_count}  BM25 calls: {bm25_spy.call_count}")
    assert faiss_spy.call_count == 0, (
        f"FAISS was called {faiss_spy.call_count} times for definition queries — should be 0"
    )
    assert bm25_spy.call_count == len(definition_bms)
    for bm in definition_bms:
        routed_cfg = router.plan(bm["query"])
        assert routed_cfg.ranker_weights.get("faiss", 0.0) == pytest.approx(0.0)
