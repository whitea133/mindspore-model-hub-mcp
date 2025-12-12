"""MindSpore MCP tools backed by the official models JSON."""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any

MODEL_FILE = Path(__file__).resolve().parents[1] / "data" / "mindspore_official_models.json"


@lru_cache(maxsize=1)
def _load_registry() -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Load the registry payload and model list from disk."""
    try:
        with MODEL_FILE.open("r", encoding="utf-8") as handle:
            payload: Any = json.load(handle)
    except FileNotFoundError as exc:
        raise RuntimeError(f"Missing model registry file: {MODEL_FILE}") from exc
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Invalid JSON payload in model registry: {MODEL_FILE}") from exc

    if not isinstance(payload, dict) or "models" not in payload:
        raise RuntimeError("Model registry JSON must be an object with a 'models' array")
    models = payload.get("models", [])
    if not isinstance(models, list):
        raise RuntimeError("Model registry 'models' field must be a list")
    return payload, models


def list_models(
    group: str | None = None,
    category: str | None = None,
    task: str | None = None,
    suite: str | None = None,
    q: str | None = None,
) -> list[dict[str, Any]]:
    """List models with optional filters on group/category/task/suite or a name/id substring."""
    _, models = _load_registry()

    def match(model: dict[str, Any]) -> bool:
        if group and model.get("group", "").lower() != group.lower():
            return False
        if category and model.get("category", "").lower() != category.lower():
            return False
        if suite and model.get("suite", "").lower() != suite.lower():
            return False
        if task:
            tasks = [t.lower() for t in model.get("task", []) if isinstance(t, str)]
            if task.lower() not in tasks:
                return False
        if q:
            q_lower = q.lower()
            if q_lower not in model.get("id", "").lower() and q_lower not in model.get("name", "").lower():
                return False
        return True

    filtered = [m for m in models if match(m)]

    # 仅返回核心字段，避免多余数据噪声
    projection_keys = {"id", "name", "group", "category", "task", "suite", "variants", "links", "dataset", "metrics", "hardware"}
    projected: list[dict[str, Any]] = []
    for m in filtered:
        projected.append({k: m.get(k) for k in projection_keys if k in m})
    return projected


def get_model_info(model_id: str) -> dict[str, Any]:
    """Return the full record of a model by id or name (case-insensitive)."""
    payload, models = _load_registry()
    needle = model_id.lower()
    for m in models:
        if m.get("id", "").lower() == needle or m.get("name", "").lower() == needle:
            return m
    raise ValueError(f"Model '{model_id}' not found in registry (version={payload.get('version')})")
