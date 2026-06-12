"""Slice 31: backend factory / dispatch tests.

Tests prove:
  * factory returns correct backend type for each supported name
  * local_llm is selectable with custom params
  * existing backend names still resolve correctly
  * unknown backend names fail with ValueError
  * kwargs are forwarded to constructors
  * mock backend works without args
"""

import pytest

from src.llm import (
    create_backend,
    LLMBackend,
    ClaudeCodeBackend,
    CodexCLIBackend,
    LocalLLMBackend,
    MockTeacherBackend,
)


def test_create_claude_cli():
    backend = create_backend("claude_cli")
    assert isinstance(backend, ClaudeCodeBackend)
    assert isinstance(backend, LLMBackend)


def test_create_codex_cli():
    backend = create_backend("codex_cli")
    assert isinstance(backend, CodexCLIBackend)
    assert isinstance(backend, LLMBackend)


def test_create_local_llm_defaults():
    backend = create_backend("local_llm")
    assert isinstance(backend, LocalLLMBackend)
    assert backend.model == "qwen2.5:7b"
    assert backend.base_url == "http://localhost:11434"


def test_create_local_llm_with_custom_params():
    backend = create_backend(
        "local_llm",
        model="llama3:8b",
        base_url="http://192.168.1.50:8080",
        temperature=0.7,
        max_tokens=2048,
        timeout=60,
    )
    assert isinstance(backend, LocalLLMBackend)
    assert backend.model == "llama3:8b"
    assert backend.base_url == "http://192.168.1.50:8080"
    assert backend.temperature == 0.7
    assert backend.max_tokens == 2048
    assert backend.timeout == 60


def test_create_mock():
    backend = create_backend("mock")
    assert isinstance(backend, MockTeacherBackend)
    assert isinstance(backend, LLMBackend)


def test_unknown_backend_raises():
    with pytest.raises(ValueError, match="Unknown LLM backend"):
        create_backend("nonexistent_backend")


def test_unknown_backend_lists_available():
    with pytest.raises(ValueError, match="claude_cli"):
        create_backend("bad_name")


def test_extra_kwargs_ignored_gracefully():
    # Extra kwargs not in the constructor signature should be ignored
    backend = create_backend(
        "local_llm",
        model="qwen2.5:7b",
        nonexistent_param="should_be_ignored",
    )
    assert isinstance(backend, LocalLLMBackend)
    assert not hasattr(backend, "nonexistent_param")


def test_codex_cli_with_kwargs():
    backend = create_backend(
        "codex_cli",
        cache_enabled=False,
        retry_attempts=1,
        timeout=30,
    )
    assert isinstance(backend, CodexCLIBackend)
    assert backend.retry_attempts == 1
    assert backend.timeout == 30


def test_all_backends_are_llm_backend():
    for name in ("claude_cli", "codex_cli", "local_llm", "mock"):
        backend = create_backend(name)
        assert isinstance(backend, LLMBackend), f"{name} is not LLMBackend"
