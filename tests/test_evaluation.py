"""Tests for evaluation metrics."""



from src.evaluation.metrics import RetrievalEvaluator, AnswerEvaluator
from src.utils.models import Chunk, RetrievalResult


def _make_results(ids: list[str]) -> list[RetrievalResult]:
    return [
        RetrievalResult(
            chunk=Chunk(id=cid, document_id="d1", content=f"Content {cid}", metadata={}),
            score=1.0 / (i + 1),
            method="dense",
        )
        for i, cid in enumerate(ids)
    ]


class TestRetrievalEvaluator:
    def test_recall_at_k_perfect(self) -> None:
        results = _make_results(["a", "b", "c"])
        relevant = {"a", "b", "c"}
        ev = RetrievalEvaluator.recall_at_k(results, relevant, k=3)
        assert ev.score == 1.0

    def test_recall_at_k_partial(self) -> None:
        results = _make_results(["a", "x", "y"])
        relevant = {"a", "b"}
        ev = RetrievalEvaluator.recall_at_k(results, relevant, k=3)
        assert ev.score == 0.5

    def test_recall_at_k_none(self) -> None:
        results = _make_results(["x", "y", "z"])
        relevant = {"a", "b"}
        ev = RetrievalEvaluator.recall_at_k(results, relevant, k=3)
        assert ev.score == 0.0

    def test_mrr_first_position(self) -> None:
        results = _make_results(["a", "b", "c"])
        ev = RetrievalEvaluator.mrr(results, {"a"})
        assert ev.score == 1.0

    def test_mrr_second_position(self) -> None:
        results = _make_results(["x", "a", "b"])
        ev = RetrievalEvaluator.mrr(results, {"a"})
        assert ev.score == 0.5

    def test_mrr_not_found(self) -> None:
        results = _make_results(["x", "y", "z"])
        ev = RetrievalEvaluator.mrr(results, {"a"})
        assert ev.score == 0.0

    def test_evaluate_all(self) -> None:
        evaluator = RetrievalEvaluator()
        results = _make_results(["a", "b", "c", "d", "e"])
        relevant = {"a", "c"}
        evals = evaluator.evaluate(results, relevant, k_values=[1, 3, 5])
        assert len(evals) == 4  # 3 recall@k + 1 MRR


class TestAnswerEvaluator:
    def test_heuristic_good_answer(self) -> None:
        answer = "Machine learning is a subset of artificial intelligence that learns from data patterns."
        context = ["Machine learning is a subset of AI that learns from data."]
        ev = AnswerEvaluator.heuristic_score(answer, context, "What is machine learning?")
        assert ev.score > 0.5

    def test_heuristic_empty_answer(self) -> None:
        ev = AnswerEvaluator.heuristic_score("", ["context"], "query")
        assert ev.score < 0.5

    def test_heuristic_cant_answer(self) -> None:
        answer = "I don't have enough information to answer this question."
        ev = AnswerEvaluator.heuristic_score(answer, ["irrelevant"], "some query")
        assert ev.details["has_answer"] == 0.3

    def test_llm_judge_prompt(self) -> None:
        prompt = AnswerEvaluator.llm_judge_prompt("q?", "answer", "context text")
        assert "relevance" in prompt.lower()
        assert "faithfulness" in prompt.lower()

