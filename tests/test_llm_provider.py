"""Unit tests for Ollama (Local) and DeepSeek (Cloud) LLM Provider abstraction and client initialization."""
import os
import pytest
from unittest.mock import MagicMock, patch
from openai import AuthenticationError, RateLimitError, APIConnectionError, APIError
from rag.llm import (
    get_llm_client,
    get_llm_config,
    format_llm_error,
    run_diagnostic_probe,
    check_ollama_health,
    normalize_ollama_base_url,
    DEFAULT_LLM_PROVIDER,
    DEFAULT_LLM_MODEL,
    DEFAULT_LLM_BASE_URL,
    DEFAULT_OLLAMA_MODEL,
    DEFAULT_OLLAMA_BASE_URL,
    DEFAULT_DEEPSEEK_MODEL,
    DEFAULT_DEEPSEEK_BASE_URL,
)


def test_default_llm_configuration():
    """Verify default provider is ollama, model is llama3.2:3b, and base_url is http://localhost:11434/v1."""
    assert DEFAULT_LLM_PROVIDER == "ollama"
    assert DEFAULT_LLM_MODEL == "llama3.2:3b"
    assert DEFAULT_LLM_BASE_URL == "http://localhost:11434/v1"
    assert DEFAULT_OLLAMA_MODEL == "llama3.2:3b"
    assert DEFAULT_OLLAMA_BASE_URL == "http://localhost:11434/v1"
    assert DEFAULT_DEEPSEEK_MODEL == "deepseek-chat"
    assert DEFAULT_DEEPSEEK_BASE_URL == "https://api.deepseek.com"


def test_normalize_ollama_base_url():
    """Verify normalize_ollama_base_url appends /v1 if missing and preserves trailing slashes properly."""
    assert normalize_ollama_base_url("http://localhost:11434") == "http://localhost:11434/v1"
    assert normalize_ollama_base_url("http://localhost:11434/") == "http://localhost:11434/v1"
    assert normalize_ollama_base_url("http://localhost:11434/v1") == "http://localhost:11434/v1"
    assert normalize_ollama_base_url("http://127.0.0.1:11434") == "http://127.0.0.1:11434/v1"
    assert normalize_ollama_base_url("") == DEFAULT_OLLAMA_BASE_URL


def test_get_llm_config_ollama_default():
    """Verify default config resolution produces Ollama configuration without requiring an API key."""
    with patch.dict(os.environ, {}, clear=True):
        config = get_llm_config()
        assert config["provider"] == "ollama"
        assert config["model"] == "llama3.2:3b"
        assert config["base_url"] == "http://localhost:11434/v1"
        assert config["api_key"] == "ollama"


def test_get_llm_config_deepseek_resolution():
    """Verify DEEPSEEK_API_KEY environment variable resolution when provider is set to deepseek."""
    with patch.dict(os.environ, {"LLM_PROVIDER": "deepseek", "DEEPSEEK_API_KEY": "dsk-test-key-12345"}, clear=True):
        config = get_llm_config()
        assert config["provider"] == "deepseek"
        assert config["api_key"] == "dsk-test-key-12345"
        assert config["base_url"] == "https://api.deepseek.com"
        assert config["model"] == "deepseek-chat"


def test_get_llm_client_ollama_no_key_required():
    """Verify that creating an Ollama client requires no API key and connects to local endpoint."""
    with patch.dict(os.environ, {}, clear=True):
        client = get_llm_client(provider="ollama")
        assert client is not None
        assert str(client.base_url).rstrip("/") == "http://localhost:11434/v1"
        assert client.api_key == "ollama"


def test_get_llm_client_deepseek_missing_key_raises_error():
    """Verify that creating a DeepSeek client without DEEPSEEK_API_KEY raises a clean ValueError."""
    with patch.dict(os.environ, {}, clear=True):
        with pytest.raises(ValueError) as exc_info:
            get_llm_client(provider="deepseek")
        assert "DEEPSEEK_API_KEY is not configured" in str(exc_info.value)
        assert "dsk-" not in str(exc_info.value)


def test_get_llm_client_deepseek_initialization():
    """Verify that get_llm_client creates an OpenAI client with DeepSeek base_url."""
    with patch.dict(os.environ, {"DEEPSEEK_API_KEY": "dsk-valid-key-999"}, clear=True):
        client = get_llm_client(provider="deepseek")
        assert client is not None
        assert str(client.base_url).rstrip("/") == "https://api.deepseek.com"
        assert client.api_key == "dsk-valid-key-999"


def test_check_ollama_health_mocked_success():
    """Verify check_ollama_health returns status and model presence when Ollama is running."""
    mock_resp = MagicMock()
    mock_resp.status = 200
    mock_resp.read.return_value = b'{"models": [{"name": "llama3.2:3b"}, {"name": "nomic-embed-text:latest"}]}'
    mock_resp.__enter__.return_value = mock_resp

    with patch("urllib.request.urlopen", return_value=mock_resp):
        health = check_ollama_health(base_url="http://localhost:11434", target_model="llama3.2:3b")
        assert health["reachable"] is True
        assert health["target_model_present"] is True
        assert "llama3.2:3b" in health["models"]
        assert health["error"] is None


def test_check_ollama_health_mocked_unreachable():
    """Verify check_ollama_health handles connection failure gracefully without raising."""
    with patch("urllib.request.urlopen", side_effect=Exception("Connection refused")):
        health = check_ollama_health(base_url="http://localhost:11434", target_model="llama3.2:3b")
        assert health["reachable"] is False
        assert health["target_model_present"] is False
        assert "Connection refused" in health["error"]


def test_format_llm_error_ollama_connection_503():
    """Verify APIConnectionError for Ollama yields helpful instructions to start Ollama."""
    mock_request = MagicMock()
    err = APIConnectionError(request=mock_request)
    msg, code = format_llm_error(err, endpoint="http://localhost:11434/v1", provider="ollama")
    assert code == 503
    assert "Failed to connect to local Ollama instance" in msg
    assert "ollama serve" in msg


def test_format_llm_error_ollama_model_not_found_404():
    """Verify 404 for Ollama suggests running ollama pull."""
    err = APIError(
        "model 'llama3.2:3b' not found, try pulling it first",
        request=MagicMock(),
        body={"error": "model 'llama3.2:3b' not found, try pulling it first"}
    )
    setattr(err, "status_code", 404)
    msg, code = format_llm_error(err, model="llama3.2:3b", endpoint="http://localhost:11434/v1", provider="ollama")
    assert code == 404
    assert "ollama pull llama3.2:3b" in msg


def test_format_llm_error_deepseek_authentication():
    """Verify 401 AuthenticationError for DeepSeek is formatted without leaking secrets."""
    mock_request = MagicMock()
    mock_response = MagicMock()
    mock_response.status_code = 401
    err = AuthenticationError("Incorrect API key provided: dsk-secret-12345", response=mock_response, body=None)
    msg, code = format_llm_error(err, provider="deepseek")
    assert code == 401
    assert "DeepSeek Authentication failed" in msg
    assert "dsk-secret-12345" not in msg


def test_format_llm_error_deepseek_rate_limit():
    """Verify RateLimitError for DeepSeek is formatted cleanly."""
    mock_response = MagicMock()
    mock_response.status_code = 429
    err = RateLimitError("Rate limit reached for model deepseek-chat", response=mock_response, body=None)
    msg, code = format_llm_error(err, provider="deepseek")
    assert code == 429
    assert "rate limit or quota" in msg


def test_format_llm_error_deepseek_insufficient_balance_402():
    """Verify 402 Insufficient Balance for DeepSeek produces helpful error message with portal link."""
    err = APIError(
        "Error code: 402 - {'error': 'Insufficient Balance'}",
        request=MagicMock(),
        body={"error": "Insufficient Balance"}
    )
    setattr(err, "status_code", 402)
    msg, code = format_llm_error(err, provider="deepseek")
    assert code == 402
    assert "Insufficient Balance" in msg
    assert "platform.deepseek.com" in msg
