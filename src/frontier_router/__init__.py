"""frontier-router: capability-aware routing across frontier AI models."""

from frontier_router.capabilities import CAPABILITY_MAP, Capability
from frontier_router.router import Router, RouterResult

__version__ = "0.1.0"

__all__ = [
    "Router",
    "RouterResult",
    "Capability",
    "CAPABILITY_MAP",
    "__version__",
]
