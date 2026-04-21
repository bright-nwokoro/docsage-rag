from app.core.retriever import rrf_fuse


def test_rrf_single_ranker_preserves_order():
    ranker = [("a", {}), ("b", {}), ("c", {})]
    fused = rrf_fuse([ranker], k=60, top_k=3)
    assert [x[0] for x in fused] == ["a", "b", "c"]


def test_rrf_both_rankers_agree():
    r1 = [("a", {}), ("b", {}), ("c", {})]
    r2 = [("a", {}), ("b", {}), ("c", {})]
    fused = rrf_fuse([r1, r2], k=60, top_k=3)
    assert [x[0] for x in fused] == ["a", "b", "c"]


def test_rrf_combines_disagreeing_rankers():
    r1 = [("a", {}), ("b", {}), ("c", {})]
    r2 = [("b", {}), ("a", {}), ("c", {})]
    fused = rrf_fuse([r1, r2], k=60, top_k=3)
    # a: 1/61 + 1/62 ≈ 0.03236
    # b: 1/62 + 1/61 ≈ 0.03236
    # c: 1/63 + 1/63 ≈ 0.03174
    # a and b tie for top — tie-break by id order doesn't matter, just ensure c is last.
    ids = [x[0] for x in fused]
    assert ids[2] == "c"
    assert set(ids[:2]) == {"a", "b"}


def test_rrf_chunks_only_in_one_ranker_included():
    r1 = [("a", {}), ("b", {})]
    r2 = [("c", {}), ("a", {})]
    fused = rrf_fuse([r1, r2], k=60, top_k=10)
    ids = {x[0] for x in fused}
    assert ids == {"a", "b", "c"}


def test_rrf_top_k_limits_output():
    r1 = [(f"id{i}", {}) for i in range(50)]
    fused = rrf_fuse([r1], k=60, top_k=5)
    assert len(fused) == 5


def test_rrf_preserves_payload():
    r1 = [("a", {"content": "hello", "page": 1})]
    fused = rrf_fuse([r1], k=60, top_k=1)
    assert fused[0][1]["content"] == "hello"
    assert fused[0][1]["page"] == 1
