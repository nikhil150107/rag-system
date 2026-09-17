"""Unit tests for xAI Grok LLM Provider abstraction and client initialization."""
import os
import pytest
from unittest.mock import MagicMock, patch
from openai import AuthenticationError, RateLimitError, APIConnectionError, APIError
from rag.llm import (
    get_llm_client,
    get_llm_config,
    format_llm_error,
    DEFAULT_LLM_MODEL,
    DEFAULT_LLM_BASE_URL,
)


def test_default_llm_configuration():
    """Verify default model is grok-4.20-0309-non-reasoning and base_url is https://api.x.ai/v1."""
    assert DEFAULT_LLM_MODEL == "grok-4.20-0309-non-reasoning"
    assert DEFAULT_LLM_BASE_URL == "https://api.x.ai/v1"


def test_get_llm_config_resolution():
    """Verify XAI_API_KEY environment variable resolution."""
    with patch.dict(os.environ, {"XAI_API_KEY": "xai-test-key-12345"}, clear=True):
        config = get_llm_config()
        assert config["api_key"] == "xai-test-key-12345"
        assert config["base_url"] == "https://api.x.ai/v1"
        assert config["model"] == "grok-4.20-0309-non-reasoning"
        assert config["provider"] == "xai_grok"


def test_get_llm_client_missing_key_raises_error():
    """Verify that attempting to create a client without XAI_API_KEY raises a clean ValueError."""
    with patch.dict(os.environ, {}, clear=True):
        with pytest.raises(ValueError) as exc_info:
            get_llm_client()
        assert "XAI_API_KEY is not configured" in str(exc_info.value)
        # Verify no fake key is logged
        assert "xai-" not in str(exc_info.value)


def test_get_llm_client_initialization():
    """Verify that get_llm_client creates an OpenAI client with xAI base_url."""
    with patch.dict(os.environ, {"XAI_API_KEY": "xai-valid-key-999"}, clear=True):
        client = get_llm_client()
        assert client is not None
        assert str(client.base_url) == "https://api.x.ai/v1/"
        assert client.api_key == "xai-valid-key-999"


def test_format_llm_error_authentication():
    """Verify 401/403 AuthenticationError is formatted without leaking secrets."""
    mock_request = MagicMock()
    mock_response = MagicMock()
    mock_response.status_code = 401
    err = AuthenticationError("Incorrect API key provided: xai-secret-12345", response=mock_response, body=None)
    msg, code = format_llm_error(err)
    assert code == 401
    assert "Authentication failed" in msg
    assert "xai-secret-12345" not in msg


def test_format_llm_error_rate_limit():
    """Verify RateLimitError is formatted cleanly."""
    mock_response = MagicMock()
    mock_response.status_code = 429
    err = RateLimitError("Rate limit reached for model grok", response=mock_response, body=None)
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


def test_openai_api_key_not_required():
    """Verify that system functions with ONLY XAI_API_KEY and does not look for OPENAI_API_KEY."""
    with patch.dict(os.environ, {"XAI_API_KEY": "xai-exclusive-key"}, clear=True):
        assert os.getenv("OPENAI_API_KEY") is None
        config = get_llm_config()
        assert config["api_key"] == "xai-exclusive-key"
