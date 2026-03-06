"""
Gamma Vector — API Clients
===============
Unified wrappers for Anthropic, OpenAI, Google Gemini, and LM Studio APIs.
All experiment scripts import from here instead of duplicating API code.
"""

import os
import re
import time


# ──────────────────────────────────────────────────
# Retry Logic
# ──────────────────────────────────────────────────

def call_with_retry(func, *args, max_retries: int = 3, base_delay: int = 5, **kwargs):
    """Wrapper with exponential backoff for rate-limit and transient error resilience."""
    for attempt in range(max_retries):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            err_str = str(e).lower()
            retryable = (
                "429" in str(e)
                or "rate" in err_str
                or "overloaded" in err_str
                or "disconnect" in err_str
                or "500" in str(e)
                or "503" in str(e)
                or "server error" in err_str
                or "timeout" in err_str
            )
            if attempt < max_retries - 1 and retryable:
                delay = base_delay * (2 ** attempt)
                print(f"  \u26a0 Transient error, retrying in {delay}s ({e})...")
                time.sleep(delay)
            else:
                raise


# ──────────────────────────────────────────────────
# LM Studio Utilities
# ──────────────────────────────────────────────────

# Module-level cache: cleared on model switch
_lm_model_cache: dict[tuple[str, str], str] = {}


def strip_think(text: str) -> str:
    """Remove <think>...</think> reasoning tags from local model output."""
    if not text:
        return ""
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r"<think>.*$", "", text, flags=re.DOTALL | re.IGNORECASE)
    return text.strip()


def invalidate_lm_model_cache():
    """Clear the LM Studio model resolution cache (call on model switch)."""
    _lm_model_cache.clear()


def resolve_lm_studio_model(config: dict) -> str:
    """Resolve configured model_id against actually loaded LM Studio model.

    LM Studio model names include quantization suffixes. This function
    queries /v1/models and finds the best match.

    Returns the actual model ID to use in API calls.
    Raises RuntimeError if LM Studio is unreachable or no match found.
    """
    base_url = config.get("base_url", "http://localhost:1234/v1")
    target = config["model_id"].lower()
    cache_key = (base_url, target)

    if cache_key in _lm_model_cache:
        return _lm_model_cache[cache_key]

    from openai import OpenAI

    try:
        client = OpenAI(base_url=base_url, api_key="not-needed")
        models_response = client.models.list()
        available = [m.id for m in models_response.data]
    except Exception as e:
        raise RuntimeError(f"Cannot reach LM Studio at {base_url}: {e}")

    if not available:
        raise RuntimeError(f"LM Studio at {base_url} has no models loaded.")

    # Exact match
    for mid in available:
        if mid.lower() == target:
            _lm_model_cache[cache_key] = mid
            return mid

    # Substring match
    matches = [mid for mid in available if target in mid.lower()]
    if len(matches) == 1:
        print(f"  \u2139 Model resolved: '{config['model_id']}' \u2192 '{matches[0]}'")
        _lm_model_cache[cache_key] = matches[0]
        return matches[0]
    elif len(matches) > 1:
        print(f"  \u26a0 Multiple matches for '{config['model_id']}': {matches}")
        print(f"    Using first match: '{matches[0]}'")
        _lm_model_cache[cache_key] = matches[0]
        return matches[0]

    # Fallback: if only one model is loaded, use it
    if len(available) == 1:
        print(f"  \u26a0 No match for '{config['model_id']}', using only loaded model: '{available[0]}'")
        _lm_model_cache[cache_key] = available[0]
        return available[0]

    raise RuntimeError(
        f"No model matching '{config['model_id']}' found in LM Studio.\n"
        f"  Available: {available}\n"
        f"  Please load the correct model or adjust model_id in config."
    )


# ──────────────────────────────────────────────────
# API Call Functions
# ──────────────────────────────────────────────────

def call_anthropic(messages: list[dict], model_id: str, max_tokens: int = 2048,
                   system: str | None = None) -> str:
    """Call Anthropic API with conversation history."""
    from anthropic import Anthropic
    client = Anthropic()

    def _call():
        kwargs = dict(model=model_id, max_tokens=max_tokens, messages=messages)
        if system:
            kwargs["system"] = system
        response = client.messages.create(**kwargs)
        return response.content[0].text

    return call_with_retry(_call)


def call_openai(messages: list[dict], model_id: str, max_tokens: int = 2048) -> str:
    """Call OpenAI API with conversation history."""
    from openai import OpenAI
    client = OpenAI()

    oai_messages = [{"role": msg["role"], "content": msg["content"]} for msg in messages]

    def _call():
        response = client.chat.completions.create(
            model=model_id,
            max_tokens=max_tokens,
            messages=oai_messages,
        )
        return response.choices[0].message.content

    return call_with_retry(_call)


def call_google(messages: list[dict], model_id: str, max_tokens: int = 2048) -> str:
    """Call Google Gemini API with conversation history."""
    from google import genai
    from google.genai import types

    api_key = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
    client = genai.Client(api_key=api_key)

    contents = []
    for msg in messages:
        role = "user" if msg["role"] == "user" else "model"
        contents.append(types.Content(
            role=role,
            parts=[types.Part.from_text(text=msg["content"])],
        ))

    def _call():
        response = client.models.generate_content(
            model=model_id,
            contents=contents,
            config=types.GenerateContentConfig(max_output_tokens=max_tokens),
        )
        return response.text

    return call_with_retry(_call)


def call_lm_studio(messages: list[dict], model_id: str, base_url: str,
                    max_tokens: int = 2048) -> str:
    """Call local model via LM Studio (OpenAI-compatible API)."""
    from openai import OpenAI
    client = OpenAI(base_url=base_url, api_key="not-needed")

    oai_messages = [{"role": msg["role"], "content": msg["content"]} for msg in messages]

    def _call():
        response = client.chat.completions.create(
            model=model_id,
            max_tokens=max_tokens,
            messages=oai_messages,
            temperature=0.0,
        )
        content = response.choices[0].message.content or ""
        return strip_think(content)

    return call_with_retry(_call)


# ──────────────────────────────────────────────────
# Unified Router
# ──────────────────────────────────────────────────

def call_model(messages: list[dict], model_key: str, models_config: dict,
               max_tokens: int = 2048) -> str:
    """Route to appropriate API based on model configuration.

    Args:
        messages: Conversation history
        model_key: Key in models_config (e.g. "claude", "gpt4o")
        models_config: Dict of model configurations
        max_tokens: Maximum response tokens
    """
    config = models_config[model_key]
    provider = config["provider"]
    model_id = config["model_id"]

    if provider == "anthropic":
        return call_anthropic(messages, model_id, max_tokens)
    elif provider == "openai":
        return call_openai(messages, model_id, max_tokens)
    elif provider == "google":
        return call_google(messages, model_id, max_tokens)
    elif provider == "lm_studio":
        base_url = config.get("base_url", "http://localhost:1234/v1")
        resolved_id = resolve_lm_studio_model(config)
        return call_lm_studio(messages, resolved_id, base_url, max_tokens)
    else:
        raise ValueError(f"Unknown provider: {provider}")


def check_api_available(model_key: str, models_config: dict) -> bool:
    """Check if API is available for a model.

    For lm_studio models, also checks that ANTHROPIC_API_KEY is set
    (needed for judge scoring).
    """
    config = models_config[model_key]
    provider = config["provider"]

    if provider == "anthropic":
        return bool(os.environ.get("ANTHROPIC_API_KEY"))
    elif provider == "openai":
        return bool(os.environ.get("OPENAI_API_KEY"))
    elif provider == "google":
        return bool(os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY"))
    elif provider == "lm_studio":
        if not os.environ.get("ANTHROPIC_API_KEY"):
            print(f"  \u26a0 {model_key}: LM Studio erreichbar, aber ANTHROPIC_API_KEY "
                  f"fehlt (noetig fuer Judge). Ueberspringe.")
            return False
        try:
            resolve_lm_studio_model(config)
            return True
        except RuntimeError:
            return False
    return False
