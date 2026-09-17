"""Answer Grounding & Hallucination Evaluator."""
import json
import logging
import re
from pathlib import Path
from typing import Dict, Any, List, Optional

logger = logging.getLogger("answer_evaluator")


class AnswerEvaluator:
    """
    Evaluates generated answers against retrieved context and expected ground-truth facts.
    Provides deterministic checks for context presence, answerability compliance, and groundedness.
    """

    @staticmethod
    def evaluate_groundedness(
        answer: str,
        sources: List[Dict[str, Any]],
        expected_keywords: List[str]
    ) -> Dict[str, Any]:
        """
        Check if the answer contains expected key terms and is grounded in the retrieved sources.
        """
        if not answer:
            return {"grounded": False, "score": 0.0, "matched_keywords": [], "missing_keywords": expected_keywords}

        answer_lower = answer.lower()
        matched = []
        missing = []

        for kw in expected_keywords:
            if kw.lower() in answer_lower:
                matched.append(kw)
            else:
                missing.append(kw)

        keyword_coverage = len(matched) / len(expected_keywords) if expected_keywords else 1.0

        # Check if sources actually contain the key content
        sources_text = " ".join([s.get("full_text", s.get("snippet", "")) for s in sources]).lower()
        supported_facts = [kw for kw in matched if kw.lower() in sources_text]
        source_support_rate = len(supported_facts) / len(matched) if matched else 1.0

        is_grounded = (keyword_coverage >= 0.5) and (source_support_rate >= 0.5)
        overall_score = round(keyword_coverage * 0.5 + source_support_rate * 0.5, 3)

        return {
            "grounded": is_grounded,
            "score": overall_score,
            "matched_keywords": matched,
            "missing_keywords": missing,
            "keyword_coverage": round(keyword_coverage, 3),
            "source_support_rate": round(source_support_rate, 3)
        }

    @staticmethod
    def evaluate_unanswerable(
        context_found: bool,
        answer: str,
        sources: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Verify that out-of-domain or unanswerable queries do not hallucinate false document context.
        """
        # Compliant if either context_found is False or answer explicitly indicates lack of information
        explicit_refusal_patterns = [
            r"not (?:mentioned|mention|found|provided|contain|included|available)",
            r"(?:do|does|did) not (?:state|contain|mention|have|include|specify)",
            r"cannot (?:find|answer|determine|be found)",
            r"no (?:information|context|documents|mention)",
            r"without (?:document )?context",
            r"unsupported|unavailable"
        ]

        has_explicit_refusal = any(
            re.search(pat, answer.lower()) is not None
            for pat in explicit_refusal_patterns
        )

        is_compliant = (not context_found) or has_explicit_refusal or (len(sources) == 0)

        return {
            "compliant": is_compliant,
            "context_found": context_found,
            "has_explicit_refusal": has_explicit_refusal,
            "hallucination_detected": not is_compliant
        }
