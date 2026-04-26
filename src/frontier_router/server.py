"""OpenAI-compatible HTTP shim around the Router.

This is the practical interface, clients that already speak the OpenAI
chat completions protocol (the SDK, curl, Cursor, Continue, your code)
can point ``base_url`` at this server and get capability-aware routing
without changing anything else.

Endpoints:

- ``POST /v1/chat/completions`` , OpenAI-compatible. Body = standard
  request shape; we extract the last user message as the task, flatten
  earlier messages into ``system_prompt``, route, and return a response
  in OpenAI's response shape.
- ``POST /route``               , Native shape, returns the routing
  decision *and* the response. Useful when the caller wants both.
- ``GET /capabilities``         , JSON dump of the capability matrix.
- ``GET /health``               , Liveness probe.

Stdlib-only (``http.server`` + ``json``); no FastAPI/Starlette dependency.

Run::

    python -m frontier_router.server --host 0.0.0.0 --port 8080
    # or via the CLI:
    frontier-router serve --port 8080

Then point any OpenAI client at it::

    from openai import OpenAI
    client = OpenAI(base_url="http://localhost:8080/v1", api_key="unused")
    client.chat.completions.create(
        model="auto",  # router picks
        messages=[{"role": "user", "content": "Refactor this function"}],
    )
"""

from __future__ import annotations

import argparse
import json
import logging
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

from frontier_router import __version__
from frontier_router.capabilities import (
    CAPABILITY_DESCRIPTIONS,
    CAPABILITY_MAP,
)
from frontier_router.router import Router

logger = logging.getLogger("frontier_router.server")


def make_handler(router: Router) -> type[BaseHTTPRequestHandler]:
    """Return a request-handler class bound to a configured Router."""

    class Handler(BaseHTTPRequestHandler):
        server_version = f"frontier-router/{__version__}"

        # --- response helpers -------------------------------------------- #

        def _send_json(self, status: int, payload: dict) -> None:
            body = json.dumps(payload).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(body)

        def _read_json(self) -> dict:
            length = int(self.headers.get("Content-Length", "0"))
            if length <= 0:
                return {}
            raw = self.rfile.read(length)
            try:
                return json.loads(raw)
            except json.JSONDecodeError as e:
                raise _BadRequest(f"invalid JSON body: {e}") from e

        def log_message(self, fmt, *args):
            logger.info("%s - %s", self.address_string(), fmt % args)

        # --- routing ----------------------------------------------------- #

        def do_OPTIONS(self):  # noqa: N802 (BaseHTTPRequestHandler API)
            self.send_response(204)
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
            self.send_header("Access-Control-Allow-Headers", "Content-Type, Authorization")
            self.end_headers()

        def do_GET(self):  # noqa: N802
            try:
                if self.path == "/health":
                    return self._send_json(200, {"status": "ok", "version": __version__})
                if self.path == "/capabilities":
                    return self._send_json(200, _matrix_payload())
                if self.path in {"/", "/v1", "/v1/"}:
                    return self._send_json(200, {
                        "service": "frontier-router",
                        "version": __version__,
                        "endpoints": [
                            "POST /v1/chat/completions",
                            "POST /route",
                            "GET /capabilities",
                            "GET /health",
                        ],
                    })
                return self._send_json(404, {"error": {"message": "not found"}})
            except Exception as e:
                logger.exception("GET %s failed", self.path)
                return self._send_json(500, {"error": {"message": str(e)}})

        def do_POST(self):  # noqa: N802
            try:
                if self.path == "/v1/chat/completions":
                    return self._handle_chat_completions()
                if self.path == "/route":
                    return self._handle_route()
                return self._send_json(404, {"error": {"message": "not found"}})
            except _BadRequest as e:
                return self._send_json(400, {"error": {"message": str(e)}})
            except ValueError as e:
                return self._send_json(400, {"error": {"message": str(e)}})
            except RuntimeError as e:
                # All providers in the chain failed.
                return self._send_json(502, {"error": {"message": str(e)}})
            except Exception as e:
                logger.exception("POST %s failed", self.path)
                return self._send_json(500, {"error": {"message": str(e)}})

        def _handle_chat_completions(self):
            body = self._read_json()
            messages = body.get("messages")
            if not messages:
                raise _BadRequest("messages is required")

            task, system_prompt = _split_messages(messages)
            context: dict[str, Any] = body.get("context") or {}
            if system_prompt and "system_prompt" not in context:
                context["system_prompt"] = system_prompt
            for k in ("max_tokens", "model"):
                if k in body and k not in context:
                    context[k] = body[k]

            # `model` in the request can also be a hard provider override, e.g.
            # "frontier:grok" or just one of {anthropic,openai,google,xai}.
            override = _explicit_provider(body.get("model"))
            result = router.query(task, context=context, provider=override)

            self._send_json(200, _openai_envelope(result, body.get("model") or "auto"))

        def _handle_route(self):
            body = self._read_json()
            task = body.get("task")
            if not task or not isinstance(task, str):
                raise _BadRequest("task is required and must be a string")
            context = body.get("context") or {}
            override = body.get("provider")
            result = router.query(task, context=context, provider=override)
            self._send_json(200, {
                "provider": result.provider,
                "capability": result.capability.value,
                "confidence": result.confidence,
                "stub": result.stub,
                "fallback_chain": result.fallback_chain,
                "explain": result.explain(),
                "response": result.response,
            })

    return Handler


class _BadRequestError(ValueError):
    pass


# Backwards-compat alias used elsewhere in this module.
_BadRequest = _BadRequestError


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


_PROVIDER_NAMES = {"anthropic", "openai", "google", "xai", "grok", "claude", "gpt", "gemini"}
_PROVIDER_ALIASES = {
    "claude": "anthropic", "gpt": "openai", "gemini": "google", "grok": "xai",
}


def _explicit_provider(model: str | None) -> str | None:
    """If ``model`` names a provider, return its canonical id; else None."""
    if not model:
        return None
    m = model.strip().lower()
    if m.startswith("frontier:"):
        m = m.split(":", 1)[1]
    if m in _PROVIDER_ALIASES:
        return _PROVIDER_ALIASES[m]
    if m in {"anthropic", "openai", "google", "xai"}:
        return m
    return None


def _split_messages(messages: list[dict]) -> tuple[str, str | None]:
    """Pull the last user message out as task; concatenate prior system text.

    Multimodal content arrays are flattened to text. Anything we can't
    interpret is dropped, the goal is "just enough" OpenAI compatibility.
    """
    system_parts: list[str] = []
    last_user: str | None = None
    for m in messages:
        role = m.get("role")
        content = m.get("content")
        text = _content_to_text(content)
        if role == "system":
            if text:
                system_parts.append(text)
        elif role == "user":
            last_user = text
        elif role == "assistant" and last_user is None and text:
            # Prepend assistant turns so the task carries conversational context.
            system_parts.append(f"(assistant) {text}")
    if last_user is None:
        raise _BadRequest("no user message found")
    system_prompt = "\n\n".join(system_parts) if system_parts else None
    return last_user, system_prompt


def _content_to_text(content) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        out = []
        for piece in content:
            if isinstance(piece, dict):
                if piece.get("type") == "text" and "text" in piece:
                    out.append(piece["text"])
            elif isinstance(piece, str):
                out.append(piece)
        return "".join(out)
    return ""


def _openai_envelope(result, requested_model: str) -> dict:
    return {
        "id": f"chatcmpl-{uuid.uuid4().hex}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": requested_model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": result.response},
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
        "x_frontier_router": {
            "provider": result.provider,
            "capability": result.capability.value,
            "confidence": result.confidence,
            "stub": result.stub,
            "fallback_chain": result.fallback_chain,
            "explain": result.explain(),
        },
    }


def _matrix_payload() -> dict:
    return {
        "version": __version__,
        "capabilities": [
            {
                "name": cap.value,
                "description": CAPABILITY_DESCRIPTIONS.get(cap, ""),
                "primary": chain[0] if chain else None,
                "fallbacks": chain[1:] if chain else [],
            }
            for cap, chain in CAPABILITY_MAP.items()
        ],
    }


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def serve(host: str = "127.0.0.1", port: int = 8080, mode: str = "stub",
          llm_fallback: bool = False, real: bool = False) -> None:
    """Start the HTTP server. Blocks until interrupted.

    Args:
        host: bind address.
        port: bind port.
        mode: passed straight to Router. "stub" by default for safe demos.
        llm_fallback: enable LLM-as-router fallback.
        real: shorthand to set mode="real" *and* use enhanced providers.
    """
    if real:
        mode = "real"
        from frontier_router.real_providers import enhanced_providers
        router = Router(mode=mode, llm_fallback=llm_fallback,
                        providers=enhanced_providers())
    else:
        router = Router(mode=mode, llm_fallback=llm_fallback)

    handler = make_handler(router)
    httpd = ThreadingHTTPServer((host, port), handler)
    logger.info("frontier-router serving on http://%s:%d (mode=%s)", host, port, mode)
    print(f"frontier-router serving on http://{host}:{port} (mode={mode})")
    print("  POST /v1/chat/completions   (OpenAI-compatible)")
    print("  POST /route                 (native: returns routing + response)")
    print("  GET  /capabilities          (capability matrix)")
    print("  GET  /health")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nshutting down")
    finally:
        httpd.server_close()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="frontier-router-server")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--mode", default="stub", choices=["stub", "real", "hybrid"])
    parser.add_argument("--llm-fallback", action="store_true")
    parser.add_argument(
        "--real",
        action="store_true",
        help=(
            "Shorthand for --mode real + use enhanced "
            "(image-gen / live-search / file-upload) providers."
        ),
    )
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )

    serve(host=args.host, port=args.port, mode=args.mode,
          llm_fallback=args.llm_fallback, real=args.real)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
