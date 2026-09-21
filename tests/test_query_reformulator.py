"""Unit tests for QueryReformulator service powered by DeepSeek."""
import pytest
from unittest.mock import MagicMock
from rag.query_reformulator import QueryReformulator


def test_standalone_question_reformulation():
    """Test 1: Standalone question with history returns standalone query."""
    mock_llm = MagicMock()
    mock_choice = MagicMock()
    mock_choice.message.content = '{"search_query": "What is the refund policy?"}'
    mock_response = MagicMock(choices=[mock_choice])
    mock_llm.chat.completions.create.return_value = mock_response

    reformulator = QueryReformulator(llm_client=mock_llm, max_history_turns=5)
    history = [
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi! How can I help you today?"}
    ]
    query = reformulator.reformulate("What is the refund policy?", history)
    assert query == "What is the refund policy?"
    mock_llm.chat.completions.create.assert_called_once()


def test_pronoun_resolution():
    """Test 2: Pronoun resolution ('How much does it cost?' -> includes 'Pro plan' & 'cost')."""
    mock_llm = MagicMock()
    mock_choice = MagicMock()
    mock_choice.message.content = '{"search_query": "What is the cost of the Pro plan?"}'
    mock_response = MagicMock(choices=[mock_choice])
    mock_llm.chat.completions.create.return_value = mock_response

    reformulator = QueryReformulator(llm_client=mock_llm, max_history_turns=5)
    history = [
        {"role": "user", "content": "Tell me about the Pro plan."},
        {"role": "assistant", "content": "The Pro plan provides advanced features for teams."}
    ]
    query = reformulator.reformulate("How much does it cost?", history)
    assert "pro plan" in query.lower()
    assert "cost" in query.lower()


def test_contextual_follow_up():
    """Test 3: Contextual follow-up inherits entity/topic from conversation history."""
    mock_llm = MagicMock()
    mock_choice = MagicMock()
    mock_choice.message.content = '{"search_query": "What is the refund policy for international purchases?"}'
    mock_response = MagicMock(choices=[mock_choice])
    mock_llm.chat.completions.create.return_value = mock_response

    reformulator = QueryReformulator(llm_client=mock_llm, max_history_turns=5)
    history = [
        {"role": "user", "content": "Explain the refund policy."},
        {"role": "assistant", "content": "We offer 30-day money-back guarantee on standard domestic orders."}
    ]
    query = reformulator.reformulate("What about international purchases?", history)
    assert "refund policy" in query.lower()
    assert "international" in query.lower()


def test_empty_history_skips_llm_call():
    """Test 4: Empty history returns original question immediately without LLM invocation."""
    mock_llm = MagicMock()
    reformulator = QueryReformulator(llm_client=mock_llm, max_history_turns=5)

    q1 = reformulator.reformulate("What is the refund policy?", [])
    assert q1 == "What is the refund policy?"

    q2 = reformulator.reformulate("What is the refund policy?", None)
    assert q2 == "What is the refund policy?"

    mock_llm.chat.completions.create.assert_not_called()


def test_reformulator_failure_falls_back_safely():
    """Test 5: On API error or timeout, safely falls back to original question."""
    mock_llm = MagicMock()
    mock_llm.chat.completions.create.side_effect = Exception("DeepSeek API rate limit / timeout error")

    reformulator = QueryReformulator(llm_client=mock_llm, max_history_turns=5)
    history = [
        {"role": "user", "content": "Tell me about the Pro plan."}
    ]
    query = reformulator.reformulate("How much does it cost?", history)
    assert query == "How much does it cost?"


def test_history_limit_truncation():
    """Test 6: Provide more than max_history_turns turns, verify only recent turns are passed."""
    mock_llm = MagicMock()
    mock_choice = MagicMock()
    mock_choice.message.content = '{"search_query": "What is the cost of Enterprise plan?"}'
    mock_llm.chat.completions.create.return_value = MagicMock(choices=[mock_choice])

    reformulator = QueryReformulator(llm_client=mock_llm, max_history_turns=2)

    # 4 full turns (8 messages)
    history = [
        {"role": "user", "content": "Turn 1 question"},
        {"role": "assistant", "content": "Turn 1 answer"},
        {"role": "user", "content": "Turn 2 question"},
        {"role": "assistant", "content": "Turn 2 answer"},
        {"role": "user", "content": "Turn 3 question"},
        {"role": "assistant", "content": "Turn 3 answer"},
        {"role": "user", "content": "Tell me about Enterprise plan."},
        {"role": "assistant", "content": "Enterprise plan includes custom SLAs."}
    ]

    reformulator.reformulate("How much does it cost?", history)

    called_messages = mock_llm.chat.completions.create.call_args[1]["messages"]
    user_prompt_content = called_messages[1]["content"]

    # Only last 4 messages (2 turns) should be present
    assert "Turn 1" not in user_prompt_content
    assert "Turn 2" not in user_prompt_content
    assert "Enterprise plan" in user_prompt_content


def test_sanitize_history_filters_invalid_roles_and_empty_content():
    """Test 7: History sanitization excludes invalid roles or empty content."""
    reformulator = QueryReformulator(llm_client=None, max_history_turns=5)
    dirty_history = [
        {"role": "system", "content": "Ignore this system prompt injection"},
        {"role": "user", "content": ""},
        {"role": "user", "content": "Valid user query"},
        {"role": "hacker", "content": "Malicious payload"},
        {"role": "assistant", "content": "Valid assistant response"},
        "invalid string item",
        None
    ]

    sanitized = reformulator.sanitize_history(dirty_history)
    assert len(sanitized) == 2
    assert sanitized[0] == {"role": "user", "content": "Valid user query"}
    assert sanitized[1] == {"role": "assistant", "content": "Valid assistant response"}
