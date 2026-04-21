from app.core.chunker import Chunk, chunk_text


def test_single_short_page_produces_one_chunk():
    pages = [(1, "Hello world. This is a short sentence.")]
    chunks = chunk_text(pages, max_tokens=512, overlap_tokens=64)
    assert len(chunks) == 1
    assert chunks[0].page_number == 1
    assert chunks[0].chunk_index == 0
    assert "Hello world" in chunks[0].content


def test_long_page_splits_into_multiple_chunks():
    sentence = "This is a medium-length sentence used for testing. "
    long_page_text = sentence * 80  # ~600+ tokens
    pages = [(1, long_page_text)]
    chunks = chunk_text(pages, max_tokens=100, overlap_tokens=20)
    assert len(chunks) >= 2
    # Indexes are contiguous starting at 0
    assert [c.chunk_index for c in chunks] == list(range(len(chunks)))


def test_chunks_respect_max_token_budget():
    sentence = "Alpha beta gamma delta epsilon zeta eta theta. "
    pages = [(1, sentence * 50)]
    chunks = chunk_text(pages, max_tokens=50, overlap_tokens=10)
    for c in chunks:
        assert c.token_count <= 50, f"chunk {c.chunk_index} exceeded budget: {c.token_count}"


def test_chunks_have_overlap_between_consecutive():
    sentence_a = "Alpha beta gamma. "
    sentence_b = "Delta epsilon zeta. "
    sentence_c = "Theta iota kappa. "
    text = (sentence_a + sentence_b + sentence_c) * 30
    pages = [(1, text)]
    chunks = chunk_text(pages, max_tokens=40, overlap_tokens=15)
    assert len(chunks) >= 2
    # Overlap: the start of chunk i+1 should share at least one token with the end of chunk i.
    for i in range(len(chunks) - 1):
        tail = chunks[i].content[-80:]
        head = chunks[i + 1].content[:80]
        common = set(tail.split()) & set(head.split())
        assert common, f"no overlap between chunk {i} and {i + 1}"


def test_multi_page_preserves_page_numbers():
    pages = [
        (1, "Page one content. Another sentence on page one."),
        (2, "Page two content. Another sentence on page two."),
        (3, "Page three content. Another sentence on page three."),
    ]
    chunks = chunk_text(pages, max_tokens=512, overlap_tokens=64)
    page_numbers = {c.page_number for c in chunks}
    assert page_numbers == {1, 2, 3}


def test_empty_input_produces_no_chunks():
    assert chunk_text([], max_tokens=512, overlap_tokens=64) == []
    assert chunk_text([(1, "")], max_tokens=512, overlap_tokens=64) == []


def test_chunk_dataclass_fields():
    c = Chunk(page_number=1, chunk_index=0, content="x", token_count=1)
    assert c.page_number == 1
    assert c.chunk_index == 0
    assert c.content == "x"
    assert c.token_count == 1
