"""Smoke tests for the HTTP shim. Stub mode only, no network."""

from __future__ import annotations

import json
import threading
import time
import urllib.error
import urllib.request

import pytest

from frontier_router.router import Router
from frontier_router.server import make_handler


@pytest.fixture
def server():
    """Spin up the HTTP server on an ephemeral port for the duration of a test."""
    from http.server import ThreadingHTTPServer

    router = Router(mode="stub")
    handler = make_handler(router)
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    port = httpd.server_address[1]
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()

    base = f"http://127.0.0.1:{port}"
    # Wait briefly for the server to be ready
    for _ in range(50):
        try:
            urllib.request.urlopen(f"{base}/health", timeout=1).read()
            break
        except (urllib.error.URLError, ConnectionError):
            time.sleep(0.02)
    yield base
    httpd.shutdown()
    httpd.server_close()


def _post(base: str, path: str, body: dict) -> tuple[int, dict]:
    req = urllib.request.Request(
        base + path,
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        resp = urllib.request.urlopen(req, timeout=5)
        return resp.status, json.loads(resp.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read())


def _get(base: str, path: str) -> tuple[int, dict]:
    resp = urllib.request.urlopen(base + path, timeout=5)
    return resp.status, json.loads(resp.read())


def test_health(server):
    status, body = _get(server, "/health")
    assert status == 200
    assert body["status"] == "ok"


def test_capabilities(server):
    status, body = _get(server, "/capabilities")
    assert status == 200
    names = [c["name"] for c in body["capabilities"]]
    assert "code_generation" in names
    assert "realtime_x_timeline" in names


def test_chat_completions_routes_code(server):
    status, body = _post(
        server,
        "/v1/chat/completions",
        {
            "model": "auto",
            "messages": [{"role": "user", "content": "Refactor this login helper"}],
        },
    )
    assert status == 200
    assert body["choices"][0]["message"]["role"] == "assistant"
    metadata = body["x_frontier_router"]
    assert metadata["provider"] == "anthropic"
    assert metadata["capability"] == "code_generation"
    assert metadata["stub"] is True


def test_chat_completions_routes_x_to_xai(server):
    _, body = _post(
        server,
        "/v1/chat/completions",
        {
            "model": "auto",
            "messages": [{"role": "user", "content": "What's trending on X today?"}],
        },
    )
    assert body["x_frontier_router"]["provider"] == "xai"
    assert body["x_frontier_router"]["capability"] == "realtime_x_timeline"


def test_chat_completions_explicit_provider_via_model(server):
    _, body = _post(
        server,
        "/v1/chat/completions",
        {
            "model": "grok",  # routes through the alias map -> xai
            "messages": [{"role": "user", "content": "anything"}],
        },
    )
    assert body["x_frontier_router"]["provider"] == "xai"


def test_chat_completions_uses_system_prompt_from_messages(server):
    _, body = _post(
        server,
        "/v1/chat/completions",
        {
            "model": "auto",
            "messages": [
                {"role": "system", "content": "Be terse."},
                {"role": "user", "content": "Refactor this"},
            ],
        },
    )
    assert body["choices"][0]["message"]["content"]


def test_chat_completions_400_on_no_messages(server):
    status, body = _post(server, "/v1/chat/completions", {"model": "auto"})
    assert status == 400
    assert "messages" in body["error"]["message"]


def test_route_native_endpoint(server):
    status, body = _post(
        server,
        "/route",
        {"task": "Summarize this report", "context": {"file": "x.pdf"}},
    )
    assert status == 200
    assert body["provider"] == "google"
    assert body["capability"] == "massive_doc_analysis"
    assert body["stub"] is True
    assert "explain" in body
    assert body["response"]


def test_route_400_on_missing_task(server):
    status, body = _post(server, "/route", {})
    assert status == 400


def test_unknown_path(server):
    status, body = _post(server, "/v1/nope", {})
    assert status == 404
