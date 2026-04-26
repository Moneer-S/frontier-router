"""Command-line interface for frontier-router.

Install the package and use via the `frontier-router` entry point, or run
this module directly: `python -m frontier_router.cli`.
"""

from __future__ import annotations

import argparse
import sys

from frontier_router import __version__
from frontier_router.capabilities import (
    ALL_PROVIDERS,
    CAPABILITY_DESCRIPTIONS,
    CAPABILITY_MAP,
    Capability,
)
from frontier_router.router import Router

# Friendly CLI names → internal provider IDs. Common product names map to the
# canonical identifier so users can write `--provider grok` instead of `xai`.
PROVIDER_ALIASES = {
    "claude": "anthropic",
    "anthropic": "anthropic",
    "gpt": "openai",
    "chatgpt": "openai",
    "openai": "openai",
    "gemini": "google",
    "google": "google",
    "grok": "xai",
    "xai": "xai",
}


def main(argv: list[str] | None = None) -> int:
    # Subcommand split-out: `frontier-router serve ...` boots the HTTP shim.
    # We keep argparse simple by sniffing argv[0] before building the parser.
    if argv is None:
        argv = sys.argv[1:]
    if argv and argv[0] == "serve":
        from frontier_router.server import main as serve_main
        return serve_main(argv[1:])

    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.version:
        print(f"frontier-router {__version__}")
        return 0

    if args.matrix:
        _print_matrix()
        return 0

    if not args.task:
        parser.error("a task argument is required (or use --matrix / --version / serve)")

    context = _parse_context(args.context)
    provider = _resolve_provider(args.provider) if args.provider else None

    if args.explain:
        _print_explain(args.task, context, args.mode, args.llm_fallback, provider)
        return 0

    # In real/hybrid mode, use the enhanced providers so capability-specific
    # endpoints (image gen, X live search, doc upload, PDF attachment) actually
    # fire. In stub mode the default stubs are correct.
    providers = None
    if args.mode in {"real", "hybrid"}:
        from frontier_router.real_providers import enhanced_providers
        providers = enhanced_providers()
    router = Router(mode=args.mode, llm_fallback=args.llm_fallback, providers=providers)
    try:
        result = router.query(args.task, context=context, provider=provider)
    except ValueError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    except RuntimeError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1

    print(result.response)
    print()
    print(result.explain(), file=sys.stderr)
    return 0


# ---------------------------------------------------------------------------
# Argparse
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="frontier-router",
        description="Capability-aware routing across frontier AI models.",
    )
    parser.add_argument("task", nargs="?", help="The prompt / query text.")
    parser.add_argument(
        "--provider",
        help=f"Explicit provider override. One of: {', '.join(sorted(set(PROVIDER_ALIASES)))}.",
    )
    parser.add_argument(
        "--mode",
        default="stub",
        choices=["stub", "real", "hybrid"],
        help="Run mode. stub=no API calls, real=call SDKs, hybrid=call if keys present.",
    )
    parser.add_argument(
        "--explain",
        action="store_true",
        help="Print the routing decision only; do not call any provider.",
    )
    parser.add_argument(
        "--context",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Add a context key/value (repeatable). Example: --context priority=latency",
    )
    parser.add_argument(
        "--llm-fallback",
        action="store_true",
        help="Use the LLM classifier when rules confidence is low.",
    )
    parser.add_argument(
        "--matrix", action="store_true", help="Print the capability matrix and exit."
    )
    parser.add_argument("--version", action="store_true", help="Print version and exit.")
    return parser


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _parse_context(items: list[str]) -> dict:
    context: dict = {}
    for item in items:
        if "=" not in item:
            raise SystemExit(f"--context expected KEY=VALUE, got {item!r}")
        key, _, value = item.partition("=")
        context[key.strip()] = _coerce(value.strip())
    return context


def _coerce(value: str):
    """Lightly coerce common scalars; leave everything else as a string."""
    if value.isdigit():
        return int(value)
    lowered = value.lower()
    if lowered in {"true", "false"}:
        return lowered == "true"
    return value


def _resolve_provider(name: str) -> str:
    canonical = PROVIDER_ALIASES.get(name.lower())
    if canonical is None:
        valid = ", ".join(sorted(set(PROVIDER_ALIASES)))
        raise SystemExit(f"unknown provider {name!r}. Valid: {valid}")
    return canonical


def _print_matrix() -> None:
    header = f"{'Capability':<32} {'Primary':<12} Fallbacks"
    print(header)
    print("-" * len(header))
    for cap in Capability:
        chain = CAPABILITY_MAP.get(cap, [])
        if not chain:
            continue
        primary = chain[0]
        fallbacks = ", ".join(chain[1:]) if len(chain) > 1 else "-"
        print(f"{cap.value:<32} {primary:<12} {fallbacks}")


def _print_explain(
    task: str,
    context: dict,
    mode: str,
    llm_fallback: bool,
    provider: str | None,
) -> None:
    if provider is not None:
        if provider not in ALL_PROVIDERS:
            raise SystemExit(f"unknown provider {provider!r}")
        print(
            f"explicit provider override: {provider} "
            f"| capability: {Capability.GENERAL.value} "
            f"(bypassed classifier) | confidence: 1.00"
        )
        return

    from frontier_router.routing.rules import classify as classify_rules

    capability, confidence = classify_rules(task, context)
    if confidence < 0.7 and llm_fallback:
        from frontier_router.routing.llm import classify as classify_llm
        capability, confidence = classify_llm(task, context, mode=mode)

    chain = CAPABILITY_MAP.get(capability, CAPABILITY_MAP[Capability.GENERAL])
    desc = CAPABILITY_DESCRIPTIONS.get(capability, capability.value)
    print(
        f"would route to {chain[0]} "
        f"| capability: {capability.value} ({desc}) "
        f"| confidence: {confidence:.2f} "
        f"| chain: {' -> '.join(chain)}"
    )


if __name__ == "__main__":
    raise SystemExit(main())
