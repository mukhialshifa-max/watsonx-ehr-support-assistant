import pytest

from app.knowledge_base import FaqEntry, KnowledgeBase, tokenize


@pytest.fixture(scope="module")
def kb():
    return KnowledgeBase.from_file()


def test_tokenize_removes_stop_words_and_stems():
    assert tokenize("How do I reschedule the appointments?") == ["reschedule", "appointment"]


@pytest.mark.parametrize(
    "question, expected_id",
    [
        ("I forgot my password and I'm locked out", "login-01"),
        ("how to reschedule a patient appointment", "sched-01"),
        ("patient registered twice, duplicate record", "reg-01"),
        ("labels are not printing", "print-01"),
        ("the EHR is down what do we do", "downtime-01"),
        ("I charted on the wrong patient", "chart-01"),
    ],
)
def test_search_finds_the_right_article(kb, question, expected_id):
    results = kb.search(question)
    assert results, "expected at least one result"
    assert results[0].entry.id == expected_id


def test_search_returns_nothing_for_unrelated_question(kb):
    assert kb.search("what is the weather in Toronto tomorrow") == []


def test_results_are_sorted_and_limited(kb):
    results = kb.search("patient appointment record", top_k=2)
    assert len(results) <= 2
    assert results == sorted(results, key=lambda r: r.score, reverse=True)


def test_empty_knowledge_base_is_rejected():
    with pytest.raises(ValueError):
        KnowledgeBase([])


def test_custom_entries():
    kb = KnowledgeBase([FaqEntry("x", "Test", "How do I reset the printer?", "Turn it off and on.")])
    assert kb.search("reset printer")[0].entry.id == "x"
