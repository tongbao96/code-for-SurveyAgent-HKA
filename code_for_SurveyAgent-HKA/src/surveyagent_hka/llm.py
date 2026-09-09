from __future__ import annotations

import json
import re
import time
from threading import BoundedSemaphore
from typing import Any, Dict, Optional

from .config import ModelConfig


LLMClient = Any


def strip_json_fence(text: str) -> str:
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s*```$", "", text)
    return text.strip()


def parse_json_object(text: str) -> Dict[str, Any]:
    cleaned = strip_json_fence(text)
    try:
        value = _load_json(cleaned)
    except json.JSONDecodeError as full_error:
        start, end = cleaned.find("{"), cleaned.rfind("}")
        if start < 0 or end <= start:
            raise full_error
        value = _load_json(cleaned[start:end + 1])
    if not isinstance(value, dict):
        raise ValueError("LLM JSON response must be an object")
    return value


def _load_json(text: str) -> Any:
    """Parse strict JSON first, then tolerate raw controls inside strings.

    Some OpenAI-compatible gateways return otherwise valid JSON with literal
    newlines or tabs embedded in long generated strings.  `strict=False`
    accepts those characters without relaxing object/array/number syntax.
    """
    try:
        return json.loads(text)
    except json.JSONDecodeError as strict_error:
        try:
            return json.loads(text, strict=False)
        except json.JSONDecodeError:
            raise strict_error


class OpenAIClient:
    def __init__(self, config: ModelConfig, api_key: str, base_url: Optional[str] = None):
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise RuntimeError("Install the 'openai' package to use the LLM agents") from exc
        kwargs: Dict[str, Any] = {"api_key": api_key, "timeout": config.timeout_seconds}
        if base_url:
            kwargs["base_url"] = base_url
        self._client = OpenAI(**kwargs)
        self._config = config
        self._request_slots = BoundedSemaphore(config.max_concurrent_requests)

    def _complete(self, prompt: str, json_mode: bool) -> str:
        with self._request_slots:
            return self._complete_with_retries(prompt, json_mode)

    def _complete_with_retries(self, prompt: str, json_mode: bool) -> str:
        error: Optional[Exception] = None
        for attempt in range(1, self._config.max_retries + 1):
            try:
                if self._config.api_mode == "responses":
                    kwargs: Dict[str, Any] = {
                        "model": self._config.model,
                        "input": prompt,
                        "temperature": self._config.temperature,
                    }
                    if json_mode:
                        kwargs["text"] = {"format": {"type": "json_object"}}
                    response = self._client.responses.create(**kwargs)
                    return (response.output_text or "").strip()
                kwargs = {
                    "model": self._config.model,
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": self._config.temperature,
                }
                response = self._client.chat.completions.create(**kwargs)
                return (response.choices[0].message.content or "").strip()
            except Exception as exc:  # SDK/provider exceptions vary.
                error = exc
                if attempt < self._config.max_retries:
                    time.sleep(min(2 ** (attempt - 1), 8))
        raise RuntimeError(f"LLM request failed after {self._config.max_retries} attempts: {error}")

    def text(self, prompt: str) -> str:
        return self._complete(prompt, json_mode=False)

    def json(self, prompt: str) -> Dict[str, Any]:
        error: Optional[Exception] = None
        for attempt in range(1, self._config.max_retries + 1):
            # Transport failures have already exhausted the controlled retry
            # policy inside `_complete_with_retries`.  Do not multiply those
            # attempts here; only regenerate when the model returned text that
            # could not be parsed as the requested JSON object.
            text = self._complete(prompt, json_mode=True)
            try:
                return parse_json_object(text)
            except (json.JSONDecodeError, ValueError) as exc:
                error = exc
                if attempt < self._config.max_retries:
                    time.sleep(min(attempt, 4))
        raise RuntimeError(f"Could not parse an LLM JSON response: {error}")
