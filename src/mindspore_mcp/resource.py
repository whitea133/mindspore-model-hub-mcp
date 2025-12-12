"""Resource definitions for MCP."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Callable, Dict

# central registry for resources: uri -> function
RESOURCE_REGISTRY: Dict[str, Callable] = {}


def resource(uri: str) -> Callable:
    """Decorator to tag and register a function as an MCP resource with the given URI."""

    def decorator(func: Callable) -> Callable:
        RESOURCE_REGISTRY[uri] = func
        setattr(func, "__mcp_resource_uri__", uri)
        return func

    return decorator


MODELS_PATH = Path(__file__).resolve().parents[2] / "data" / "mindspore_official_models.json"


@resource("mindspore://models/official")
def get_official_models() -> dict:
    """Return the full official models registry JSON."""
    with MODELS_PATH.open("r", encoding="utf-8") as fh:
        return json.load(fh)
