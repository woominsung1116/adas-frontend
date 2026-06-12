"""Slice 30: local LLM backend tests.

Tests prove:
  * backend construction with defaults and custom params
  * successful generation from a mocked local HTTP response
  * malformed response handling
  * transport failure handling
  * empty content handling
  * cache behavior
  * conforms to LLMBackend interface
"""

import json
import tempfile
from http.server import HTTPServer, BaseHTTPRequestHandler
from threading import Thread

from src.llm.backend import LLMBackend
from src.llm.local_llm import LocalLLMBackend


# ---------------------------------------------------------------------------
# Mock HTTP server
# ---------------------------------------------------------------------------


def _make_handler(response_body: dict | None = None, status: int = 200):
    """Create a request handler that returns a fixed response."""
    body = response_body or {
        "choices": [{"message": {"content": "test response from local LLM"}}]
    }

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            content_length = int(self.headers.get("Content-Length", 0))
            self.rfile.read(content_length)  # consume body
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(body).encode())

        def log_message(self, format, *args):
            pass  # suppress server logs in test output

    return Handler


def _run_with_server(handler_cls, fn):
    """Start a local HTTP server, run fn(port), then shut down."""
    server = HTTPServer(("127.0.0.1", 0), handler_cls)
    port = server.server_address[1]
    thread = Thread(target=server.serve_forever)
    thread.daemon = True
    thread.start()
    try:
        return fn(port)
    finally:
        server.shutdown()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_construction_defaults():
    backend = LocalLLMBackend()
    assert backend.model == "qwen2.5:7b"
    assert backend.base_url == "http://localhost:11434"
    assert backend.temperature == 0.0
    assert backend.max_tokens == 4096
    assert backend.retry_attempts == 3


def test_construction_custom():
    backend = LocalLLMBackend(
        model="llama3:8b",
        base_url="http://192.168.1.100:8080",
        temperature=0.7,
        max_tokens=2048,
        timeout=60,
    )
    assert backend.model == "llama3:8b"
    assert backend.base_url == "http://192.168.1.100:8080"
    assert backend.temperature == 0.7


def test_conforms_to_llm_backend_interface():
    backend = LocalLLMBackend()
    assert isinstance(backend, LLMBackend)
    assert hasattr(backend, "generate")
    assert hasattr(backend, "generate_raw")


def test_successful_generation():
    handler = _make_handler({
        "choices": [{"message": {"content": "hello from qwen"}}]
    })

    def _test(port):
        backend = LocalLLMBackend(
            base_url=f"http://127.0.0.1:{port}",
            cache_enabled=False,
            retry_attempts=1,
        )
        result = backend.generate("test prompt")
        assert result == "hello from qwen"

    _run_with_server(handler, _test)


def test_generate_raw_delegates_to_generate():
    handler = _make_handler({
        "choices": [{"message": {"content": "raw response"}}]
    })

    def _test(port):
        backend = LocalLLMBackend(
            base_url=f"http://127.0.0.1:{port}",
            cache_enabled=False,
            retry_attempts=1,
        )
        result = backend.generate_raw("test prompt")
        assert result == "raw response"

    _run_with_server(handler, _test)


def test_malformed_response_raises():
    handler = _make_handler({"not_choices": []})

    def _test(port):
        backend = LocalLLMBackend(
            base_url=f"http://127.0.0.1:{port}",
            cache_enabled=False,
            retry_attempts=1,
            retry_delay=0.0,
        )
        try:
            backend.generate("test prompt")
            assert False, "Should have raised RuntimeError"
        except RuntimeError as e:
            assert "failed after" in str(e).lower()

    _run_with_server(handler, _test)


def test_empty_content_raises():
    handler = _make_handler({
        "choices": [{"message": {"content": ""}}]
    })

    def _test(port):
        backend = LocalLLMBackend(
            base_url=f"http://127.0.0.1:{port}",
            cache_enabled=False,
            retry_attempts=1,
            retry_delay=0.0,
        )
        try:
            backend.generate("test prompt")
            assert False, "Should have raised RuntimeError"
        except RuntimeError as e:
            assert "empty content" in str(e).lower()

    _run_with_server(handler, _test)


def test_connection_failure_raises():
    backend = LocalLLMBackend(
        base_url="http://127.0.0.1:1",  # nothing listening
        cache_enabled=False,
        retry_attempts=1,
        retry_delay=0.0,
        timeout=2,
    )
    try:
        backend.generate("test prompt")
        assert False, "Should have raised RuntimeError"
    except RuntimeError as e:
        assert "connection error" in str(e).lower()


def test_cache_returns_cached_response():
    call_count = 0

    class CountingHandler(BaseHTTPRequestHandler):
        def do_POST(self):
            nonlocal call_count
            call_count += 1
            body = {"choices": [{"message": {"content": f"response_{call_count}"}}]}
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(body).encode())

        def log_message(self, format, *args):
            pass

    def _test(port):
        with tempfile.TemporaryDirectory() as tmpdir:
            backend = LocalLLMBackend(
                base_url=f"http://127.0.0.1:{port}",
                cache_dir=tmpdir,
                cache_enabled=True,
                retry_attempts=1,
            )
            r1 = backend.generate("same prompt")
            r2 = backend.generate("same prompt")
            assert r1 == r2 == "response_1"
            assert call_count == 1  # second call was cached

    _run_with_server(CountingHandler, _test)


def test_extract_content_handles_edge_cases():
    assert LocalLLMBackend._extract_content({}) == ""
    assert LocalLLMBackend._extract_content({"choices": []}) == ""
    assert LocalLLMBackend._extract_content({"choices": [{}]}) == ""
    assert LocalLLMBackend._extract_content(
        {"choices": [{"message": {"content": "ok"}}]}
    ) == "ok"
