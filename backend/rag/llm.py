"""LLM Provider abstraction and client factory for xAI Grok API."""
import os
import re
import logging
from typing import Optional, Dict, Any, Tuple
from openai import OpenAI, AuthenticationError, RateLimitError, APIConnectionError, APIError

logger = logging.getLogger(__name__)

DEFAULT_LLM_MODEL = "grok-4.20-0309-non-reasoning"
DEFAULT_LLM_BASE_URL = "https://api.x.ai/v1"


def get_llm_config() -> Dict[str, Any]:
    """
    Resolve LLM provider configuration from environment variables.
    Precedence: XAI_API_KEY -> LLM_API_KEY.
    """
    api_key = os.getenv("XAI_API_KEY") or os.getenv("LLM_API_KEY")
    base_url = os.getenv("LLM_BASE_URL") or os.getenv("XAI_BASE_URL", DEFAULT_LLM_BASE_URL)
    model = os.getenv("LLM_MODEL") or os.getenv("GROK_MODEL", DEFAULT_LLM_MODEL)

    return {
        "api_key": api_key.strip() if api_key else None,
        "base_url": base_url.rstrip("/"),
        "model": model.strip() if model else DEFAULT_LLM_MODEL,
        "provider": "xai_grok"
    }


def get_llm_client(
    api_key: Optional[str] = None,
    base_url: Optional[str] = None
) -> OpenAI:
    """
    Initialize and return an OpenAI-compatible client pointed to xAI Grok API.
    Raises ValueError if API key is not configured.
    """
    config = get_llm_config()
    resolved_key = api_key or config["api_key"]
    resolved_base_url = base_url or config["base_url"]

    if not resolved_key:
        raise ValueError(
            "XAI_API_KEY is not configured. Please set XAI_API_KEY in environment or Streamlit Secrets."
        )

    return OpenAI(
        api_key=resolved_key,
        base_url=resolved_base_url
    )


def _sanitize_error_text(text: str) -> str:
    """Strip out any API keys, auth tokens, or sensitive header strings."""
    if not text:
        return ""
    text = re.sub(r'xai-[A-Za-z0-9_\-]+', '[REDACTED_API_KEY]', text)
    text = re.sub(r'sk-[A-Za-z0-9_\-]+', '[REDACTED_API_KEY]', text)
    text = re.sub(r'(Bearer\s+)[A-Za-z0-9_\-\.]+', r'\1[REDACTED]', text, flags=re.IGNORECASE)
    text = re.sub(r'(api[_-]?key[\'":\s=]+)[\'"]?[A-Za-z0-9_\-\.]+[\'"]?', r'\1[REDACTED]', text, flags=re.IGNORECASE)
    text = re.sub(r'(Authorization[\'":\s=]+)[\'"]?[A-Za-z0-9_\-\.\s]+[\'"]?', r'\1[REDACTED]', text, flags=re.IGNORECASE)
    return text.strip()


def format_llm_error(
    error: Exception,
    model: Optional[str] = None,
    endpoint: Optional[str] = None
) -> Tuple[str, int]:
    """
    Sanitize and categorize LLM provider errors without exposing sensitive API keys.
    Returns (user_facing_message, http_status_code).
    """
    if isinstance(error, AuthenticationError):
        return ("xAI Grok Authentication failed. Please verify your XAI_API_KEY in Streamlit Secrets.", 401)
    if isinstance(error, RateLimitError):
        return ("xAI Grok rate limit or quota exceeded. Please try again shortly.", 429)
    if isinstance(error, APIConnectionError):
        return ("Failed to connect to xAI Grok API endpoint (https://api.x.ai/v1). Please check network connectivity.", 503)
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

        if status_code == 400:
            return (f"xAI API 400 Bad Request{meta_str}:\n{sanitized_msg}", 400)
        return (f"xAI Grok Provider Error (Status {status_code}){meta_str}:\n{sanitized_msg}", status_code)

    sanitized_generic = _sanitize_error_text(str(error))
    return (f"An unexpected error occurred during LLM processing: {sanitized_generic}", 500)


def run_diagnostic_probe(
    client: OpenAI,
    model: str = DEFAULT_LLM_MODEL
) -> Dict[str, Any]:
    """
    Execute progressive API probes to test model, sampling parameters,
    and structured output compatibility on xAI Grok API.
    """
    results = {"model": model, "endpoint": "https://api.x.ai/v1/chat/completions", "steps": {}}

    # Step 1: Minimal request
    try:
        r1 = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": "Say hello in one short sentence."}]
        )
        results["steps"]["1_minimal"] = {
            "status": "PASS",
            "output": r1.choices[0].message.content.strip()
        }
    except Exception as e:
        err_msg, code = format_llm_error(e, model=model)
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
        results["steps"]["2_sampling_params"] = {
            "status": "PASS",
            "output": r2.choices[0].message.content.strip()
        }
    except Exception as e:
        err_msg, code = format_llm_error(e, model=model)
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
        results["steps"]["3_structured_json"] = {
            "status": "PASS",
            "output": r3.choices[0].message.content.strip()
        }
    except Exception as e:
        err_msg, code = format_llm_error(e, model=model)
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
        results["steps"]["4_grounded_prompt"] = {
            "status": "PASS",
            "output": r4.choices[0].message.content.strip()
        }
    except Exception as e:
        err_msg, code = format_llm_error(e, model=model)
        results["steps"]["4_grounded_prompt"] = {"status": "FAIL", "error": err_msg, "code": code}

    return results
