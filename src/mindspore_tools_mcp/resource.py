"""Resource definitions for MCP."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Callable, Dict, Any

# central registry for resources: uri -> function
RESOURCE_REGISTRY: Dict[str, Callable] = {}


def resource(uri: str) -> Callable:
    """Decorator to tag and register a function as an MCP resource with the given URI."""

    def decorator(func: Callable) -> Callable:
        RESOURCE_REGISTRY[uri] = func
        setattr(func, "__mcp_resource_uri__", uri)
        return func

    return decorator


DATA_ROOT = Path(__file__).resolve().parents[2] / "data"
MODELS_PATH = DATA_ROOT / "mindspore_official_models.json"
OPMAP_CONSISTENT = DATA_ROOT / "pytorch_ms_api_mapping_consistent.json"
OPMAP_DIFF = DATA_ROOT / "pytorch_ms_api_mapping_diff.json"
OPMAP_SECTION_CONS_DIR = DATA_ROOT / "convert" / "consistent"
OPMAP_SECTION_DIFF_DIR = DATA_ROOT / "convert" / "diff"


def _load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


@resource("mindspore://models/official")
def get_official_models() -> dict:
    """Return the full official models registry JSON."""
    return _load_json(MODELS_PATH)


@resource("mindspore://opmap/pytorch/consistent")
def get_opmap_pytorch_consistent() -> dict:
    """Return the full PyTorch→MindSpore API mapping (consistent only)."""
    return _load_json(OPMAP_CONSISTENT)


@resource("mindspore://opmap/pytorch/diff")
def get_opmap_pytorch_diff() -> dict:
    """Return the PyTorch→MindSpore API mapping entries with differences."""
    return _load_json(OPMAP_DIFF)


def _load_sections(folder: Path) -> Dict[str, Any]:
    if not folder.exists():
        return {}
    sections: Dict[str, Any] = {}
    for file in folder.glob("*.json"):
        try:
            sections[file.stem] = _load_json(file)
        except Exception:
            continue
    return sections


@resource("mindspore://opmap/pytorch/sections/consistent")
def get_opmap_pytorch_sections_consistent() -> Dict[str, Any]:
    """Return per-section consistent mappings."""
    return _load_sections(OPMAP_SECTION_CONS_DIR)


@resource("mindspore://opmap/pytorch/sections/diff")
def get_opmap_pytorch_sections_diff() -> Dict[str, Any]:
    """Return per-section diff mappings."""
    return _load_sections(OPMAP_SECTION_DIFF_DIR)
