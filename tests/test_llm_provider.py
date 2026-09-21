"""Unit tests for DeepSeek LLM Provider abstraction and client initialization."""
import os
import pytest
from unittest.mock import MagicMock, patch
from openai import AuthenticationError, RateLimitError, APIConnectionError, APIError
from rag.llm import (
    get_llm_client,
    get_llm_config,
    format_llm_error,
    run_diagnostic_probe,
    DEFAULT_LLM_PROVIDER,
    DEFAULT_LLM_MODEL,
    DEFAULT_LLM_BASE_URL,
)


def test_default_llm_configuration():
    """Verify default model is deepseek-chat, provider is deepseek, and base_url is https://api.deepseek.com."""
    assert DEFAULT_LLM_PROVIDER == "deepseek"
    assert DEFAULT_LLM_MODEL == "deepseek-chat"
    assert DEFAULT_LLM_BASE_URL == "https://api.deepseek.com"


def test_get_llm_config_resolution():
    """Verify DEEPSEEK_API_KEY environment variable resolution."""
    with patch.dict(os.environ, {"DEEPSEEK_API_KEY": "dsk-test-key-12345"}, clear=True):
        config = get_llm_config()
        assert config["api_key"] == "dsk-test-key-12345"
        assert config["base_url"] == "https://api.deepseek.com"
        assert config["model"] == "deepseek-chat"
        assert config["provider"] == "deepseek"


def test_get_llm_client_missing_key_raises_error():
    """Verify that attempting to create a client without DEEPSEEK_API_KEY raises a clean ValueError."""
    with patch.dict(os.environ, {}, clear=True):
        with pytest.raises(ValueError) as exc_info:
            get_llm_client()
        assert "DEEPSEEK_API_KEY is not configured" in str(exc_info.value)
        # Verify no secret pattern is in error message
        assert "dsk-" not in str(exc_info.value)


def test_get_llm_client_initialization():
    """Verify that get_llm_client creates an OpenAI client with DeepSeek base_url."""
    with patch.dict(os.environ, {"DEEPSEEK_API_KEY": "dsk-valid-key-999"}, clear=True):
        client = get_llm_client()
        assert client is not None
        assert str(client.base_url).rstrip("/") == "https://api.deepseek.com"
        assert client.api_key == "dsk-valid-key-999"


def test_format_llm_error_authentication():
    """Verify 401 AuthenticationError is formatted without leaking secrets."""
    mock_request = MagicMock()
    mock_response = MagicMock()
    mock_response.status_code = 401
    err = AuthenticationError("Incorrect API key provided: dsk-secret-12345", response=mock_response, body=None)
    msg, code = format_llm_error(err)
    assert code == 401
    assert "Authentication failed" in msg
    assert "dsk-secret-12345" not in msg


def test_format_llm_error_rate_limit():
    """Verify RateLimitError is formatted cleanly."""
    mock_response = MagicMock()
    mock_response.status_code = 429
    err = RateLimitError("Rate limit reached for model deepseek-chat", response=mock_response, body=None)
    msg, code = format_llm_error(err)
    assert code == 429
    assert "rate limit or quota" in msg


def test_format_llm_error_connection():
    """Verify APIConnectionError is formatted cleanly."""
    mock_request = MagicMock()
    err = APIConnectionError(request=mock_request)
    msg, code = format_llm_error(err)
    assert code == 503
    assert "Failed to connect" in msg


def test_format_llm_error_bad_request_400():
    """Verify 400 BadRequestError produces formatted diagnostic output with sanitized body."""
    err = APIError(
        "Error code: 400 - {'error': 'Invalid parameter temperature'}",
        request=MagicMock(),
        body={"error": "Invalid parameter temperature"}
    )
    setattr(err, "status_code", 400)
    msg, code = format_llm_error(err)
    assert code == 400
    assert "DeepSeek API 400 Bad Request:" in msg
    assert "Invalid parameter temperature" in msg


def test_format_llm_error_insufficient_balance_402():
    """Verify 402 Insufficient Balance produces helpful error message with portal link."""
    err = APIError(
        "Error code: 402 - {'error': 'Insufficient Balance'}",
        request=MagicMock(),
        body={"error": "Insufficient Balance"}
    )
    setattr(err, "status_code", 402)
    msg, code = format_llm_error(err)
    assert code == 402
    assert "Insufficient Balance" in msg
    assert "platform.deepseek.com" in msg


def test_openai_and_xai_api_key_not_required():
    """Verify that system functions with ONLY DEEPSEEK_API_KEY and does not require OPENAI_API_KEY or XAI_API_KEY."""
    with patch.dict(os.environ, {"DEEPSEEK_API_KEY": "dsk-exclusive-key"}, clear=True):
        assert os.getenv("OPENAI_API_KEY") is None
        assert os.getenv("XAI_API_KEY") is None
        config = get_llm_config()
        assert config["api_key"] == "dsk-exclusive-key"
        assert config["provider"] == "deepseek"
