"""Negative test cases verifying unanswerable, out-of-context, and missing-information queries."""
import pytest
from evaluation.answer_evaluator import AnswerEvaluator


def test_negative_case_1_out_of_domain_geography():
    eval_res = AnswerEvaluator.evaluate_unanswerable(
        context_found=False,
        answer="Paris is the capital of France.",
        sources=[]
    )
    assert eval_res["compliant"] is True
    assert eval_res["hallucination_detected"] is False


def test_negative_case_2_unmentioned_entity_color():
    eval_res = AnswerEvaluator.evaluate_unanswerable(
        context_found=True,
        answer="The provided documents do not mention the CEO's favorite color.",
        sources=[{"snippet": "Project Titan architecture..."}]
    )
    assert eval_res["compliant"] is True
    assert eval_res["has_explicit_refusal"] is True
    assert eval_res["hallucination_detected"] is False


def test_negative_case_3_unanswerable_recipe():
    eval_res = AnswerEvaluator.evaluate_unanswerable(
        context_found=False,
        answer="To bake sourdough bread, you will need flour, water, and starter.",
        sources=[]
    )
    assert eval_res["compliant"] is True
    assert eval_res["hallucination_detected"] is False


def test_negative_case_4_missing_pricing():
    eval_res = AnswerEvaluator.evaluate_unanswerable(
        context_found=True,
        answer="The uploaded document does not contain pricing information for enterprise subscriptions.",
        sources=[{"snippet": "Security compliance includes AES-256..."}]
    )
    assert eval_res["compliant"] is True
    assert eval_res["has_explicit_refusal"] is True
    assert eval_res["hallucination_detected"] is False


def test_negative_case_5_missing_hardware_cluster_specs():
    eval_res = AnswerEvaluator.evaluate_unanswerable(
        context_found=True,
        answer="Information about the GPU cluster specifications is not available in the context.",
        sources=[{"snippet": "Antigravity is an advanced agentic coding system..."}]
    )
    assert eval_res["compliant"] is True
    assert eval_res["has_explicit_refusal"] is True
    assert eval_res["hallucination_detected"] is False


def test_hallucination_detector_flags_fabricated_document_claims():
    # If context_found is True and sources exist, but answer makes unsupported claims without refusal
    eval_res = AnswerEvaluator.evaluate_unanswerable(
        context_found=True,
        answer="According to section 4 of the document, the CEO's favorite color is bright magenta.",
        sources=[{"snippet": "Project Titan architecture..."}]
    )
    assert eval_res["compliant"] is False
    assert eval_res["hallucination_detected"] is True


def test_answer_evaluator_groundedness_scoring():
    sources = [{"full_text": "Project Titan features strict latency SLAs under 200ms for p95 retrieval."}]
    answer = "Project Titan guarantees a p95 retrieval latency SLA under 200ms."
    expected_keywords = ["200ms", "latency", "SLA", "p95"]

    res = AnswerEvaluator.evaluate_groundedness(answer, sources, expected_keywords)
    assert res["grounded"] is True
    assert res["score"] >= 0.8
    assert "200ms" in res["matched_keywords"]
