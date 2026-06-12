from src.llm.backend import LLMBackend
from src.llm.claude_code_backend import ClaudeCodeBackend
from src.llm.codex_cli_backend import CodexCLIBackend
from src.llm.local_llm import LocalLLMBackend
from src.llm.mock_backend import MockTeacherBackend

__all__ = [
    "LLMBackend",
    "ClaudeCodeBackend",
    "CodexCLIBackend",
    "LocalLLMBackend",
    "MockTeacherBackend",
    "create_backend",
]


def create_backend(
    name: str,
    **kwargs,
) -> LLMBackend:
    """Create an LLM backend by name.

    Slice 31: config-driven backend factory. Maps a short string name
    to a concrete ``LLMBackend`` subclass. Any extra ``kwargs`` are
    forwarded to the backend constructor.

    Supported names:
      - ``"claude_cli"`` → ``ClaudeCodeBackend``
      - ``"codex_cli"`` → ``CodexCLIBackend``
      - ``"local_llm"`` → ``LocalLLMBackend``
      - ``"mock"``      → ``MockTeacherBackend``

    For ``local_llm``, useful kwargs include:
      - ``model``: model name on local server (default ``"qwen2.5:7b"``)
      - ``base_url``: server URL (default ``"http://localhost:11434"``)
      - ``timeout``: request timeout in seconds
      - ``temperature``: sampling temperature
      - ``max_tokens``: generation cap

    Raises ``ValueError`` for unknown backend names.
    """
    registry: dict[str, type[LLMBackend]] = {
        "claude_cli": ClaudeCodeBackend,
        "codex_cli": CodexCLIBackend,
        "local_llm": LocalLLMBackend,
        "mock": MockTeacherBackend,
    }

    cls = registry.get(name)
    if cls is None:
        available = ", ".join(sorted(registry.keys()))
        raise ValueError(
            f"Unknown LLM backend: {name!r}. Available: {available}"
        )

    # MockTeacherBackend takes no constructor args
    if cls is MockTeacherBackend:
        return cls()

    # Filter kwargs to only those accepted by the constructor
    import inspect
    sig = inspect.signature(cls.__init__)
    valid_params = set(sig.parameters.keys()) - {"self"}
    filtered = {k: v for k, v in kwargs.items() if k in valid_params}
    return cls(**filtered)
