"""
BotHunter AI Insights Module

Provides LLM-powered plain-English explanations for why a node was flagged.
Provider is pluggable: set AI_PROVIDER and AI_API_KEY in your .env file.

Supported providers:
  openai    — OpenAI Chat Completions (gpt-4o-mini, gpt-4o, etc.)
  gemini    — Google Gemini (gemini-1.5-flash, gemini-1.5-pro)
  anthropic — Anthropic Claude (claude-3-haiku-20240307, etc.)
  ollama    — Local Ollama server (llama3, mistral, etc.) — no API key needed

Usage:
    from ai_insights import explain_node, is_ai_enabled

    if is_ai_enabled():
        explanation = explain_node(node_features, label)
        st.info(explanation)
"""

import json
import logging
import os

log = logging.getLogger(__name__)

# ─── Config loading ────────────────────────────────────────────────────────────

def _load_config() -> dict:
    path = os.path.join(os.path.dirname(__file__), "config.json")
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return {}

_CFG = _load_config()
_AI_CFG = _CFG.get("ai", {})

PROVIDER: str = os.environ.get("AI_PROVIDER", _AI_CFG.get("provider", "openai"))
API_KEY:  str = os.environ.get("AI_API_KEY", "")
MODEL:    str = os.environ.get("AI_MODEL", _AI_CFG.get("model", "gpt-4o-mini"))
ENABLED:  bool = os.environ.get("AI_ENABLED", str(_AI_CFG.get("enabled", False))).lower() == "true"
MAX_TOKENS: int = _AI_CFG.get("max_explanation_tokens", 200)
OLLAMA_BASE: str = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")


def is_ai_enabled() -> bool:
    """Returns True if AI explanations are configured and enabled."""
    if not ENABLED:
        return False
    if PROVIDER == "ollama":
        return True   # Ollama runs locally, no key needed
    return bool(API_KEY)


# ─── Prompt builder ───────────────────────────────────────────────────────────

_DISPLAY_NAMES = {
    "bot":            "Star Bot",
    "engagement_pod": "Engagement Group",
    "influencer":     "Influencer",
    "organic":        "Human",
    "normal":         "Human",
}

def _build_prompt(features: dict, label: str) -> str:
    display = _DISPLAY_NAMES.get(label, label)
    return (
        f"You are a social network security analyst. A bot detection algorithm "
        f"classified a Twitter/social-media account as '{display}'. "
        f"Here are the account's graph-theoretic features:\n"
        f"  - K-Core number: {features.get('k_core', '?')} "
        f"    (how embedded it is in a dense cluster)\n"
        f"  - Out-degree (following): {features.get('out_deg', '?')}\n"
        f"  - In-degree (followers): {features.get('in_deg', '?')}\n"
        f"  - Local clustering coefficient: {features.get('clustering', '?'):.3f} "
        f"    (0 = followers are strangers, 1 = followers all know each other)\n"
        f"  - Risk score: {features.get('risk_score', '?'):.4f} (0 = safe, 1 = highly suspicious)\n\n"
        f"In 2-3 concise sentences, explain in plain English why this account was "
        f"classified as '{display}' and what that means for platform integrity. "
        f"Be specific about which features drove the classification."
    )


# ─── Provider implementations ─────────────────────────────────────────────────

def _explain_openai(prompt: str) -> str:
    try:
        from openai import OpenAI
        client = OpenAI(api_key=API_KEY)
        resp = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=MAX_TOKENS,
            temperature=0.4,
        )
        return resp.choices[0].message.content.strip()
    except ImportError:
        return "Install the `openai` package: `pip install openai`"
    except Exception as e:
        log.error("OpenAI explain_node failed: %s", e)
        return f"AI explanation unavailable ({type(e).__name__})."


def _explain_gemini(prompt: str) -> str:
    try:
        import google.generativeai as genai
        genai.configure(api_key=API_KEY)
        model = genai.GenerativeModel(MODEL)
        resp = model.generate_content(prompt)
        return resp.text.strip()
    except ImportError:
        return "Install the `google-generativeai` package: `pip install google-generativeai`"
    except Exception as e:
        log.error("Gemini explain_node failed: %s", e)
        return f"AI explanation unavailable ({type(e).__name__})."


def _explain_anthropic(prompt: str) -> str:
    try:
        import anthropic
        client = anthropic.Anthropic(api_key=API_KEY)
        msg = client.messages.create(
            model=MODEL,
            max_tokens=MAX_TOKENS,
            messages=[{"role": "user", "content": prompt}],
        )
        return msg.content[0].text.strip()
    except ImportError:
        return "Install the `anthropic` package: `pip install anthropic`"
    except Exception as e:
        log.error("Anthropic explain_node failed: %s", e)
        return f"AI explanation unavailable ({type(e).__name__})."


def _explain_ollama(prompt: str) -> str:
    """Call a local Ollama server (no API key required)."""
    try:
        import httpx
        resp = httpx.post(
            f"{OLLAMA_BASE}/api/generate",
            json={"model": MODEL, "prompt": prompt, "stream": False},
            timeout=60,
        )
        resp.raise_for_status()
        return resp.json().get("response", "").strip()
    except ImportError:
        return "Install the `httpx` package: `pip install httpx`"
    except Exception as e:
        log.error("Ollama explain_node failed: %s", e)
        return f"AI explanation unavailable ({type(e).__name__})."


# ─── Public API ───────────────────────────────────────────────────────────────

def explain_node(features: dict, label: str) -> str:
    """
    Return a plain-English explanation of why a node was classified with the given label.

    Args:
        features: Dict from classify_nodes() — must contain k_core, out_deg, in_deg,
                  clustering, risk_score.
        label:    Internal classification label (bot, engagement_pod, influencer, etc.)

    Returns:
        Human-readable explanation string, or an error message if AI is unavailable.
    """
    if not is_ai_enabled():
        return (
            "AI explanations are disabled. Set AI_ENABLED=true and provide an "
            "AI_API_KEY in your .env file to enable this feature."
        )

    prompt = _build_prompt(features, label)

    dispatchers = {
        "openai":    _explain_openai,
        "gemini":    _explain_gemini,
        "anthropic": _explain_anthropic,
        "ollama":    _explain_ollama,
    }

    fn = dispatchers.get(PROVIDER.lower())
    if fn is None:
        supported = ", ".join(dispatchers)
        return f"Unknown AI provider '{PROVIDER}'. Supported: {supported}."

    return fn(prompt)
