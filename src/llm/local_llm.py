"""
local_llm.py

Local LLM backend for the ADHD classroom simulation.

Calls a local OpenAI-compatible chat/completions endpoint (e.g.
Ollama, vLLM, llama.cpp server, LM Studio) and returns the raw
assistant message text. Works with any model exposed at a
``/v1/chat/completions`` endpoint — Qwen, Llama, Mistral, etc.

The contract is intentionally minimal:
  - POST to ``{base_url}/v1/chat/completions``
  - Body: ``{"model": ..., "messages": [{"role": "user", "content": prompt}], ...}``
  - Response: ``{"choices": [{"message": {"content": "..."}}]}``
  - Returns ``choices[0].message.content`` as a plain string.

This matches the OpenAI-compatible API that Ollama (``ollama serve``),
vLLM (``--api-key token --served-model-name ...``), and llama.cpp
(``--host 0.0.0.0``) all expose out of the box.
"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request

from src.cache.response_cache import ResponseCache
from src.llm.backend import LLMBackend


class LocalLLMBackend(LLMBackend):
    """OpenAI-compatible local LLM backend.

    Parameters
    ----------
    model : str
        Model name as registered on the local server (e.g. ``"qwen2.5:7b"``).
    base_url : str
        Base URL of the local server. Default ``http://localhost:11434``
        (Ollama default).
    timeout : int
        HTTP request timeout in seconds.
    temperature : float
        Sampling temperature. 0.0 = greedy/deterministic.
    max_tokens : int
        Maximum tokens to generate.
    cache_dir : str
        Directory for response caching.
    cache_enabled : bool
        Whether to cache responses.
    retry_attempts : int
        Number of retry attempts on failure.
    retry_delay : float
        Seconds between retries.
    """

    def __init__(
        self,
        model: str = "qwen2.5:7b",
        base_url: str = "http://localhost:11434",
        timeout: int = 120,
        temperature: float = 0.0,
        max_tokens: int = 4096,
        cache_dir: str = ".cache/local_llm",
        cache_enabled: bool = True,
        retry_attempts: int = 2,
        retry_delay: float = 2.0,
        api_key: str | None = None,
    ):
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.temperature = temperature
        self.max_tokens = int(__import__("os").environ.get("OMC_LLM_MAX_TOKENS", max_tokens))
        self.cache = ResponseCache(cache_dir=cache_dir, enabled=cache_enabled)
        self.retry_attempts = retry_attempts
        self.retry_delay = retry_delay
        self.api_key = api_key

    def generate(self, prompt: str) -> str:
        """Send prompt to local LLM and return response text.

        Uses the response cache when enabled. Retries on transient
        failures (connection errors, timeouts, malformed responses).
        """
        cached = self.cache.get(prompt, self._cache_context())
        if cached is not None:
            return cached

        response = self._call_local(prompt)
        self.cache.set(prompt, self._cache_context(), response)
        return response

    def generate_raw(self, prompt: str) -> str:
        """Send prompt without schema enforcement. Same as generate()."""
        return self.generate(prompt)

    def _cache_context(self) -> str:
        return json.dumps(
            {
                "backend": "local_llm",
                "model": self.model,
                "base_url": self.base_url,
                "temperature": self.temperature,
                "max_tokens": self.max_tokens,
            },
            sort_keys=True,
        )

    def _call_local(self, prompt: str) -> str:
        """Call the local OpenAI-compatible endpoint with retries."""
        url = f"{self.base_url}/v1/chat/completions"
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }

        last_error = "Local LLM returned an empty or invalid response."

        for attempt in range(self.retry_attempts):
            try:
                _headers = {"Content-Type": "application/json"}
                if self.api_key:
                    _headers["Authorization"] = f"Bearer {self.api_key}"
                # httpx + wall-clock watchdog: httpx의 phase별 timeout이
                # stale/half-open 연결(서버가 응답을 영영 안 주는 경우)에서
                # 안 걸리는 사례를 대비해, 별도 watchdog 스레드가 hard
                # deadline 초과 시 client.close()로 진행중 소켓을 깨워 hang을
                # 강제 종료한다(연결도 닫혀 daemon thread 누수 없음).
                import httpx as _httpx
                import threading as _threading
                _client = _httpx.Client(
                    timeout=_httpx.Timeout(self.timeout, connect=10.0)
                )
                _hard_deadline = float(self.timeout) + 15.0
                _wd_done = _threading.Event()

                def _watchdog():
                    if not _wd_done.wait(_hard_deadline):
                        try:
                            _client.close()
                        except Exception:
                            pass

                _wd = _threading.Thread(target=_watchdog, daemon=True)
                _wd.start()
                try:
                    _resp = _client.post(url, json=payload, headers=_headers)
                    _resp.raise_for_status()
                    body = _resp.json()
                finally:
                    _wd_done.set()
                    _wd.join(timeout=1.0)
                    try:
                        _client.close()
                    except Exception:
                        pass

                content = self._extract_content(body)
                if content:
                    return content

                last_error = f"Empty content in response: {json.dumps(body)[:200]}"

            except urllib.error.URLError as e:
                last_error = f"Connection error: {e}"
            except json.JSONDecodeError as e:
                last_error = f"Malformed JSON response: {e}"
            except TimeoutError:
                last_error = f"Request timed out after {self.timeout}s"
            except Exception as e:
                last_error = f"Unexpected error: {type(e).__name__}: {e}"

            if attempt < self.retry_attempts - 1:
                time.sleep(self.retry_delay)

        raise RuntimeError(
            f"Local LLM failed after {self.retry_attempts} attempts: {last_error}"
        )

    @staticmethod
    def _extract_content(body: dict) -> str:
        """Extract assistant message content from OpenAI-format response."""
        try:
            choices = body.get("choices", [])
            if not choices:
                return ""
            message = choices[0].get("message", {})
            return (message.get("content") or "").strip()
        except (KeyError, IndexError, TypeError):
            return ""
