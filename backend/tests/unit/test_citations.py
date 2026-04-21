from app.core.citations import (
    RAG_ANSWER_SCHEMA,
    RAG_SYSTEM_PROMPT,
    build_user_prompt,
    verify_citations,
)


def test_schema_has_required_shape():
    assert RAG_ANSWER_SCHEMA["type"] == "object"
    props = RAG_ANSWER_SCHEMA["properties"]
    assert "answer" in props and props["answer"]["type"] == "string"
    assert "citations" in props and props["citations"]["type"] == "array"
    item = props["citations"]["items"]
    for key in ("source", "page", "score"):
        assert key in item["properties"]


def test_system_prompt_mentions_no_invention():
    assert "invent" in RAG_SYSTEM_PROMPT.lower() or "never" in RAG_SYSTEM_PROMPT.lower()


def test_build_user_prompt_includes_question_and_sources():
    prompt = build_user_prompt(
        question="What is X?",
        chunks=[
            {"source": "a.pdf", "page": 1, "content": "X is a thing."},
            {"source": "b.pdf", "page": 3, "content": "Y is another thing."},
        ],
    )
    assert "What is X?" in prompt
    assert "a.pdf" in prompt and "page=1" in prompt
    assert "b.pdf" in prompt and "page=3" in prompt
    assert "X is a thing." in prompt


def test_verify_citations_drops_invalid():
    retrieved = [
        {"source": "a.pdf", "page": 1},
        {"source": "b.pdf", "page": 3},
    ]
    citations = [
        {"source": "a.pdf", "page": 1, "score": 0.9},
        {"source": "c.pdf", "page": 7, "score": 0.8},  # not retrieved
        {"source": "b.pdf", "page": 99, "score": 0.7},  # wrong page
    ]
    verified = verify_citations(citations, retrieved)
    assert len(verified) == 1
    assert verified[0]["source"] == "a.pdf"
    assert verified[0]["page"] == 1


def test_verify_citations_handles_empty():
    assert verify_citations([], []) == []
    assert verify_citations([{"source": "a.pdf", "page": 1, "score": 0.5}], []) == []
