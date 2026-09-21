"""LLM Provider abstraction and client factory for Ollama (Local) and DeepSeek (Cloud) APIs."""
import os
import re
import logging
import urllib.request
import json
from typing import Optional, Dict, Any, Tuple, List
from openai import OpenAI, AuthenticationError, RateLimitError, APIConnectionError, APIError

logger = logging.getLogger(__name__)

DEFAULT_LLM_PROVIDER = "ollama"

# Ollama Defaults (Local)
DEFAULT_OLLAMA_MODEL = "llama3.2:3b"
DEFAULT_OLLAMA_BASE_URL = "http://localhost:11434/v1"

# DeepSeek Defaults (Cloud)
DEFAULT_DEEPSEEK_MODEL = "deepseek-chat"
DEFAULT_DEEPSEEK_BASE_URL = "https://api.deepseek.com"

# Generic defaults
DEFAULT_LLM_MODEL = DEFAULT_OLLAMA_MODEL
DEFAULT_LLM_BASE_URL = DEFAULT_OLLAMA_BASE_URL


def normalize_ollama_base_url(url: str) -> str:
    """Ensure Ollama base URL includes the /v1 path for OpenAI SDK compatibility."""
    if not url:
        return DEFAULT_OLLAMA_BASE_URL
    clean = url.rstrip("/")
    if not clean.endswith("/v1"):
        clean = f"{clean}/v1"
    return clean


def check_ollama_health(base_url: str = "http://localhost:11434", target_model: str = DEFAULT_OLLAMA_MODEL) -> Dict[str, Any]:
    """
    Check if local Ollama daemon is reachable and list installed models.
    """
    # Strip /v1 if present for root api call
    root_url = base_url.rstrip("/").replace("/v1", "")
    tags_url = f"{root_url}/api/tags"
    try:
        req = urllib.request.Request(tags_url, headers={"User-Agent": "rag-system"})
        with urllib.request.urlopen(req, timeout=3.0) as resp:
            if resp.status == 200:
                data = json.loads(resp.read().decode("utf-8"))
                models_list = [m.get("name", "") for m in data.get("models", [])]
                model_present = any(target_model in m for m in models_list)
                return {
                    "reachable": True,
                    "models": models_list,
                    "target_model_present": model_present,
                    "error": None
                }
    except Exception as e:
        return {
            "reachable": False,
            "models": [],
            "target_model_present": False,
            "error": str(e)
        }
    return {"reachable": False, "models": [], "target_model_present": False, "error": "Unknown error"}


def get_llm_config(provider: Optional[str] = None) -> Dict[str, Any]:
    """
    Resolve LLM provider configuration from environment variables or parameters.
    Supported providers: 'ollama', 'deepseek'.
    """
    resolved_provider = (provider or os.getenv("LLM_PROVIDER", DEFAULT_LLM_PROVIDER)).lower().strip()

    if resolved_provider == "ollama":
        api_key = os.getenv("OLLAMA_API_KEY", "ollama")
        base_url_raw = os.getenv("OLLAMA_BASE_URL") or os.getenv("LLM_BASE_URL") or DEFAULT_OLLAMA_BASE_URL
        base_url = normalize_ollama_base_url(base_url_raw)
        model = os.getenv("OLLAMA_MODEL") or os.getenv("LLM_MODEL") or DEFAULT_OLLAMA_MODEL

        return {
            "provider": "ollama",
            "api_key": api_key.strip() if api_key else "ollama",
            "base_url": base_url,
            "model": model.strip() if model else DEFAULT_OLLAMA_MODEL
        }
    elif resolved_provider == "deepseek":
        api_key = os.getenv("DEEPSEEK_API_KEY") or os.getenv("LLM_API_KEY")
        base_url = (os.getenv("LLM_BASE_URL") or DEFAULT_DEEPSEEK_BASE_URL).rstrip("/")
        model = os.getenv("LLM_MODEL") or DEFAULT_DEEPSEEK_MODEL

        return {
            "provider": "deepseek",
            "api_key": api_key.strip() if api_key else None,
            "base_url": base_url,
            "model": model.strip() if model else DEFAULT_DEEPSEEK_MODEL
        }
    else:
        # Generic OpenAI-compatible fallback
        api_key = os.getenv("LLM_API_KEY") or "dummy-key"
        base_url = (os.getenv("LLM_BASE_URL") or DEFAULT_DEEPSEEK_BASE_URL).rstrip("/")
        model = os.getenv("LLM_MODEL") or DEFAULT_DEEPSEEK_MODEL

        return {
            "provider": resolved_provider,
            "api_key": api_key.strip() if api_key else "dummy-key",
            "base_url": base_url,
            "model": model.strip() if model else DEFAULT_DEEPSEEK_MODEL
        }


def get_llm_client(
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    provider: Optional[str] = None
) -> OpenAI:
    """
    Initialize and return an OpenAI-compatible client.
    For Ollama: no API key required (uses 'ollama').
    For DeepSeek: validates DEEPSEEK_API_KEY.
    """
    config = get_llm_config(provider=provider)
    resolved_provider = config["provider"]
    resolved_base_url = base_url or config["base_url"]

    if resolved_provider == "ollama":
        resolved_key = api_key or config["api_key"] or "ollama"
        resolved_base_url = normalize_ollama_base_url(resolved_base_url)
        return OpenAI(
            api_key=resolved_key,
            base_url=resolved_base_url
        )

    if resolved_provider == "deepseek":
        resolved_key = api_key or config["api_key"]
        if not resolved_key:
            raise ValueError(
                "DEEPSEEK_API_KEY is not configured. Please set DEEPSEEK_API_KEY in environment or Streamlit Secrets."
            )
        return OpenAI(
            api_key=resolved_key,
            base_url=resolved_base_url
        )

    # Generic provider
    resolved_key = api_key or config["api_key"] or "dummy-key"
    return OpenAI(
        api_key=resolved_key,
        base_url=resolved_base_url
    )


def _sanitize_error_text(text: str) -> str:
    """Strip out any API keys, auth tokens, or sensitive header strings."""
    if not text:
        return ""
    text = re.sub(r'dsk-[A-Za-z0-9_\-]+', '[REDACTED_API_KEY]', text)
    text = re.sub(r'sk-[A-Za-z0-9_\-]+', '[REDACTED_API_KEY]', text)
    text = re.sub(r'xai-[A-Za-z0-9_\-]+', '[REDACTED_API_KEY]', text)
    text = re.sub(r'(Bearer\s+)[A-Za-z0-9_\-\.]+', r'\1[REDACTED]', text, flags=re.IGNORECASE)
    text = re.sub(r'(api[_-]?key[\'":\s=]+)[\'"]?[A-Za-z0-9_\-\.]+[\'"]?', r'\1[REDACTED]', text, flags=re.IGNORECASE)
    text = re.sub(r'(Authorization[\'":\s=]+)[\'"]?[A-Za-z0-9_\-\.\s]+[\'"]?', r'\1[REDACTED]', text, flags=re.IGNORECASE)
    return text.strip()


def format_llm_error(
    error: Exception,
    model: Optional[str] = None,
    endpoint: Optional[str] = None,
    provider: Optional[str] = None
) -> Tuple[str, int]:
    """
    Sanitize and categorize LLM provider errors without exposing sensitive API keys.
    Returns (user_facing_message, http_status_code).
    """
    is_ollama = (provider == "ollama") or (endpoint and ("11434" in endpoint or "localhost" in endpoint))

    if isinstance(error, AuthenticationError):
        if is_ollama:
            return ("Ollama local inference does not require authentication, but received 401 Unauthorized.", 401)
        return ("DeepSeek Authentication failed. Please verify your DEEPSEEK_API_KEY in Streamlit Secrets.", 401)

    if isinstance(error, RateLimitError):
        if is_ollama:
            return ("Local Ollama instance is overloaded or resource-constrained. Please try again shortly.", 429)
        return ("DeepSeek rate limit or quota exceeded. Please try again shortly.", 429)

    if isinstance(error, APIConnectionError):
        if is_ollama:
            return (
                f"Failed to connect to local Ollama instance at {endpoint or DEFAULT_OLLAMA_BASE_URL}. "
                "Please ensure Ollama is installed and running ('ollama serve' or open Ollama application).",
                503
            )
        return ("Failed to connect to DeepSeek API endpoint (https://api.deepseek.com). Please check network connectivity.", 503)

    if isinstance(error, APIError):
        status_code = getattr(error, "status_code", 500) or 500
        body = getattr(error, "body", None)
        msg_detail = ""
        if isinstance(body, dict):
            err_obj = body.get("error")
            if isinstance(err_obj, dict):
                msg_detail = err_obj.get("message") or str(err_obj)
            elif isinstance(err_obj, str):
                msg_detail = err_obj
            elif "message" in body:
                msg_detail = str(body["message"])

        if not msg_detail:
            msg_detail = getattr(error, "message", "") or str(error)

        sanitized_msg = _sanitize_error_text(msg_detail)
        meta_info = []
        if model:
            meta_info.append(f"Model: `{model}`")
        if endpoint:
            meta_info.append(f"Endpoint: `{endpoint}`")
        meta_str = f" ({', '.join(meta_info)})" if meta_info else ""

        if is_ollama:
            if status_code == 404 or "not found" in sanitized_msg.lower():
                return (f"Ollama model '{model}' was not found{meta_str}. Please run `ollama pull {model}` in your terminal.", 404)
            return (f"Ollama Provider Error (Status {status_code}){meta_str}:\n{sanitized_msg}", status_code)

        # DeepSeek error categorization
        if status_code == 402 or "insufficient balance" in sanitized_msg.lower() or "balance" in sanitized_msg.lower():
            return (f"DeepSeek Insufficient Balance{meta_str}:\n{sanitized_msg}\nPlease check your account balance at platform.deepseek.com.", 402)
        if status_code == 400:
            return (f"DeepSeek API 400 Bad Request{meta_str}:\n{sanitized_msg}", 400)
        return (f"DeepSeek Provider Error (Status {status_code}){meta_str}:\n{sanitized_msg}", status_code)

    sanitized_generic = _sanitize_error_text(str(error))
    return (f"An unexpected error occurred during LLM processing: {sanitized_generic}", 500)


def run_diagnostic_probe(
    client: OpenAI,
    model: str = DEFAULT_LLM_MODEL,
    provider: str = DEFAULT_LLM_PROVIDER
) -> Dict[str, Any]:
    """
    Execute progressive API probes to test model, sampling parameters,
    and structured output compatibility on Ollama or DeepSeek API.
    """
    endpoint_str = str(getattr(client, "base_url", DEFAULT_OLLAMA_BASE_URL))
    results = {"model": model, "endpoint": endpoint_str, "provider": provider, "steps": {}}

    # Step 1: Minimal request
    try:
        r1 = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": "Say hello in one short sentence."}]
        )
        out1 = r1.choices[0].message.content.strip() if r1.choices else ""
        results["steps"]["1_minimal"] = {
            "status": "PASS",
            "output": out1
        }
    except Exception as e:
        err_msg, code = format_llm_error(e, model=model, endpoint=endpoint_str, provider=provider)
        results["steps"]["1_minimal"] = {"status": "FAIL", "error": err_msg, "code": code}
        return results

    # Step 2: Sampling parameters (temperature=0.3, max_tokens=500)
    try:
        r2 = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": "Say hello in one short sentence."}],
            temperature=0.3,
            max_tokens=500
        )
        out2 = r2.choices[0].message.content.strip() if r2.choices else ""
        results["steps"]["2_sampling_params"] = {
            "status": "PASS",
            "output": out2
        }
    except Exception as e:
        err_msg, code = format_llm_error(e, model=model, endpoint=endpoint_str, provider=provider)
        results["steps"]["2_sampling_params"] = {"status": "FAIL", "error": err_msg, "code": code}
        return results

    # Step 3: Structured output (json_object)
    try:
        r3 = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a JSON assistant. Output valid JSON."},
                {"role": "user", "content": "Return a JSON object with key 'status' and value 'ok'."}
            ],
            response_format={"type": "json_object"},
            temperature=0.0,
            max_tokens=150
        )
        out3 = r3.choices[0].message.content.strip() if r3.choices else ""
        results["steps"]["3_structured_json"] = {
            "status": "PASS",
            "output": out3
        }
    except Exception as e:
        err_msg, code = format_llm_error(e, model=model, endpoint=endpoint_str, provider=provider)
        results["steps"]["3_structured_json"] = {"status": "FAIL", "error": err_msg, "code": code}

    # Step 4: Grounded context prompt
    try:
        r4 = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "Answer ONLY using provided document context."},
                {"role": "user", "content": "DOCUMENT CONTEXT:\n[Source 1: test.txt]\nProject Titan has 99.9% uptime.\n\nUSER QUESTION:\nWhat is the uptime?"}
            ],
            temperature=0.3,
            max_tokens=500
        )
        out4 = r4.choices[0].message.content.strip() if r4.choices else ""
        results["steps"]["4_grounded_prompt"] = {
            "status": "PASS",
            "output": out4
        }
    except Exception as e:
        err_msg, code = format_llm_error(e, model=model, endpoint=endpoint_str, provider=provider)
        results["steps"]["4_grounded_prompt"] = {"status": "FAIL", "error": err_msg, "code": code}

    return results
