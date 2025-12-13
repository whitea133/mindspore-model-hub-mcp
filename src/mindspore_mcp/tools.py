"""MindSpore MCP tools backed by the official models JSON."""

from __future__ import annotations

import json
from functools import lru_cache
import re
from pathlib import Path
from typing import Any

from mindspore_mcp.resource import get_official_models

MODEL_FILE = Path(__file__).resolve().parents[2] / "data" / "mindspore_official_models.json"
OPMAP_CONSISTENT_FILE = Path(__file__).resolve().parents[2] / "data" / "pytorch_ms_api_mapping_consistent.json"
OPMAP_DIFF_FILE = Path(__file__).resolve().parents[2] / "data" / "pytorch_ms_api_mapping_diff.json"
OPMAP_SECTION_CONS_DIR = Path(__file__).resolve().parents[2] / "data" / "convert" / "consistent"
OPMAP_SECTION_DIFF_DIR = Path(__file__).resolve().parents[2] / "data" / "convert" / "diff"


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


def fetch_official_models() -> dict:
    """Wrapper tool to return the full official models registry JSON."""
    return get_official_models()


def _load_json(path: Path) -> Any:
    try:
        with path.open("r", encoding="utf-8") as fh:
            return json.load(fh)
    except FileNotFoundError as exc:
        raise RuntimeError(f"Missing mapping file: {path}") from exc
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Invalid JSON payload in mapping: {path}") from exc


def _load_section_map(folder: Path) -> dict[str, Any]:
    data: dict[str, Any] = {}
    if not folder.exists():
        return data
    for file in folder.glob("*.json"):
        try:
            data[file.stem] = _load_json(file)
        except Exception:
            continue
    return data


def query_op_mapping(op: str, section: str | None = None) -> dict[str, list[dict[str, str]]]:
    """Query PyTorch→MindSpore API mapping (supports section filter and fuzzy match).

    Args:
        op: PyTorch API name or substring (e.g., "torch.addmm" or "addmm").
        section: Optional section (e.g., "torch", "torchvision") to narrow the search scope.

    Returns:
        {"consistent": [...], "diff": [...]} matched entries.
    """
    key = op.lower()
    def match_row(row: dict[str, Any]) -> bool:
        pt = row.get("pytorch", "").lower()
        return key in pt

    # load data
    base_cons = _load_json(OPMAP_CONSISTENT_FILE)
    base_diff = _load_json(OPMAP_DIFF_FILE)
    cons_items = base_cons.get("items", []) if isinstance(base_cons, dict) else []
    diff_items = base_diff.get("items", []) if isinstance(base_diff, dict) else []

    sec_cons_all = _load_section_map(OPMAP_SECTION_CONS_DIR)
    sec_diff_all = _load_section_map(OPMAP_SECTION_DIFF_DIR)

    if section:
        cons_sec = sec_cons_all.get(section, {})
        diff_sec = sec_diff_all.get(section, {})
        if cons_sec and isinstance(cons_sec, dict):
            cons_items = cons_items + cons_sec.get("items", [])
        if diff_sec and isinstance(diff_sec, dict):
            diff_items = diff_items + diff_sec.get("items", [])
    else:
        # 未指定 section 时合并所有分表，覆盖更多映射（如 torch_tensor 等）
        for sec in sec_cons_all.values():
            if isinstance(sec, dict):
                cons_items += sec.get("items", [])
        for sec in sec_diff_all.values():
            if isinstance(sec, dict):
                diff_items += sec.get("items", [])

    cons_hits = [r for r in cons_items if match_row(r)]
    diff_hits = [r for r in diff_items if match_row(r)]

    return {
        "consistent": cons_hits,
        "diff": diff_hits,
    }


def translate_pytorch_code(code: str, section: str | None = None) -> dict[str, Any]:
    """Translate PyTorch code snippets using the mapping tables.

    - Consistent entries: replace with MindSpore API.
    - Diff entries: do not replace, record warnings for manual review.

    Args:
        code: Original PyTorch code string.
        section: Optional section (e.g., "torchvision", "torchaudio") to limit search scope.

    Returns:
        {
            "translated": translated code string,
            "replacements": [...],  # applied replacements
            "warnings": [...],      # hits on diff entries
        }
    """
    # 加载映射数据
    base_cons = _load_json(OPMAP_CONSISTENT_FILE)
    base_diff = _load_json(OPMAP_DIFF_FILE)
    cons_items = base_cons.get("items", []) if isinstance(base_cons, dict) else []
    diff_items = base_diff.get("items", []) if isinstance(base_diff, dict) else []

    if section:
        sec_cons_all = _load_section_map(OPMAP_SECTION_CONS_DIR)
        sec_diff_all = _load_section_map(OPMAP_SECTION_DIFF_DIR)
        cons_sec = sec_cons_all.get(section, {})
        diff_sec = sec_diff_all.get(section, {})
        if cons_sec and isinstance(cons_sec, dict):
            cons_items = cons_sec.get("items", cons_items)
        if diff_sec and isinstance(diff_sec, dict):
            diff_items = diff_sec.get("items", diff_items)

    # 为避免部分匹配，按长度排序，使用严格的边界（不匹配子串）
    def sorted_items(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return sorted(items, key=lambda r: len(r.get("pytorch", "")), reverse=True)

    translated = code
    annotated = code  # 带注释提示的版本
    replacements: list[dict[str, Any]] = []
    warnings: list[dict[str, Any]] = []

    SHAPE_HINT_APIS = {
        "torch.addmm",
        "torch.mm",
        "torch.matmul",
        "torch.bmm",
    }

    # 处理一致项：直接替换
    for row in sorted_items(cons_items):
        pt = row.get("pytorch", "")
        ms = row.get("mindspore", "")
        if not pt or not ms:
            continue
        pattern = rf"(?<![\w.]){re.escape(pt)}(?![\w.])"
        new_code, count = re.subn(pattern, ms, translated)
        if count > 0:
            translated = new_code
            entry = {k: row.get(k) for k in ("section", "pytorch", "mindspore", "description")}
            entry["count"] = count
            replacements.append(entry)

    # 差异项：不替换，记录命中
    for row in sorted_items(diff_items):
        pt = row.get("pytorch", "")
        if not pt:
            continue
        pattern = rf"(?<![\w.]){re.escape(pt)}(?![\w.])"
        matches = list(re.finditer(pattern, translated))
        if not matches:
            continue
        entry = {k: row.get(k) for k in ("section", "pytorch", "mindspore", "description")}
        entry["count"] = len(matches)
        # 形状提示
        if pt in SHAPE_HINT_APIS:
            entry["shape_hint"] = "check input/output shapes (expects matrix/matched dims)"
        warnings.append(entry)
        # 在 annotated 中为每个命中插入 TODO 注释，保留原调用
        desc = row.get("description") or "diff"
        ms = row.get("mindspore") or "mindspore.*"
        def add_comment(match: re.Match[str]) -> str:
            return f"# TODO: check mapping {pt} -> {ms}: {desc}\n{match.group(0)}"
        annotated = re.sub(pattern, add_comment, annotated)

    return {
        "translated": translated,
        "annotated": annotated,
        "replacements": replacements,
        "warnings": warnings,
    }
