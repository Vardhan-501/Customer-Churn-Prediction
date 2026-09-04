import json
import os
from typing import Any

import requests

SUMMARY_SCHEMA = {
    "type": "object",
    "properties": {
        "summary": {"type": "string"},
        "risk_factors": {"type": "array", "items": {"type": "string"}},
        "recommended_next_steps": {"type": "array", "items": {"type": "string"}},
        "caveat": {"type": "string"},
    },
    "required": ["summary", "risk_factors", "recommended_next_steps", "caveat"],
    "additionalProperties": False,
}

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2:3b")


def build_verified_facts(result: dict[str, Any], explanation_df, actions: list[dict[str, str]], threshold: float = 0.60) -> dict[str, Any]:
    risk_factors = explanation_df.loc[explanation_df["impact"] > 0, "display_feature"].head(3).tolist()
    next_steps = [item["action"] for item in actions[:3]]
    return {
        "risk_band": result["risk_band"],
        "probability": round(float(result["probability"]), 4),
        "threshold": threshold,
        "top_risk_factors": risk_factors,
        "recommended_actions": next_steps,
        "limitations": [
            "Model explanations describe associations, not causes.",
            "Prediction should be reviewed by a human.",
        ],
    }


def ollama_available() -> bool:
    try:
        response = requests.get(f"{OLLAMA_URL}/api/tags", timeout=3)
        return response.ok
    except requests.RequestException:
        return False


def openai_available() -> bool:
    return bool(os.getenv("OPENAI_API_KEY"))


def ai_summary_available() -> bool:
    return ollama_available() or openai_available()


def _prompt(facts: dict[str, Any]) -> str:
    return (
        "You are a careful customer-retention analyst. Use only the verified facts below. "
        "Do not invent metrics, causes, discounts, or customer history. State that model "
        "explanations describe associations rather than causation. Return JSON with exactly "
        "the fields summary, risk_factors, recommended_next_steps, and caveat.\n\n"
        + json.dumps(facts)
    )


def _validate_summary(summary: dict[str, Any]) -> dict[str, Any]:
    required = {"summary", "risk_factors", "recommended_next_steps", "caveat"}
    if set(summary) != required:
        raise ValueError("The summary did not match the required JSON fields")
    if not isinstance(summary["summary"], str) or not isinstance(summary["caveat"], str):
        raise ValueError("Summary and caveat must be strings")
    if not isinstance(summary["risk_factors"], list) or not isinstance(summary["recommended_next_steps"], list):
        raise ValueError("Risk factors and next steps must be arrays")
    return summary


def _generate_with_ollama(facts: dict[str, Any], model: str) -> dict[str, Any]:
    response = requests.post(
        f"{OLLAMA_URL}/api/chat",
        json={
            "model": model,
            "messages": [{"role": "user", "content": _prompt(facts)}],
            "stream": False,
            "format": SUMMARY_SCHEMA,
            "options": {"temperature": 0},
        },
        timeout=120,
    )
    response.raise_for_status()
    return _validate_summary(json.loads(response.json()["message"]["content"]))


def _generate_with_openai(facts: dict[str, Any], model: str = "gpt-5-mini") -> dict[str, Any]:
    from openai import OpenAI

    response = OpenAI().chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are a careful customer-retention analyst. Output JSON only."},
            {"role": "user", "content": _prompt(facts)},
        ],
        response_format={
            "type": "json_schema",
            "json_schema": {"name": "retention_summary", "strict": True, "schema": SUMMARY_SCHEMA},
        },
        max_completion_tokens=800,
    )
    return _validate_summary(json.loads(response.choices[0].message.content))


def generate_business_summary(facts: dict[str, Any]) -> dict[str, Any]:
    """Use local Ollama first, then OpenAI only when explicitly configured."""
    if ollama_available():
        return _generate_with_ollama(facts, OLLAMA_MODEL)
    if openai_available():
        return _generate_with_openai(facts)
    raise RuntimeError("No summary provider is available. Start Ollama or configure OPENAI_API_KEY.")
