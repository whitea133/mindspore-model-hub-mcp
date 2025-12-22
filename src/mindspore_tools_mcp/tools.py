"""MindSpore MCP 工具集合，基于官方模型与映射表。"""

from __future__ import annotations

import json
from functools import lru_cache
import re
from pathlib import Path
from typing import Any

from mindspore_tools_mcp.resource import get_official_models

MODEL_FILE = Path(__file__).resolve().parents[2] / "data" / "mindspore_official_models.json"
OPMAP_CONSISTENT_FILE = Path(__file__).resolve().parents[2] / "data" / "pytorch_ms_api_mapping_consistent.json"
OPMAP_DIFF_FILE = Path(__file__).resolve().parents[2] / "data" / "pytorch_ms_api_mapping_diff.json"
OPMAP_SECTION_CONS_DIR = Path(__file__).resolve().parents[2] / "data" / "convert" / "consistent"
OPMAP_SECTION_DIFF_DIR = Path(__file__).resolve().parents[2] / "data" / "convert" / "diff"

SHAPE_HINT_APIS = {
    "torch.addmm",
    "torch.mm",
    "torch.matmul",
    "torch.bmm",
}


@lru_cache(maxsize=1)
def _load_registry() -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """加载模型清单及列表。"""
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
    """列出模型，可按 group/category/task/suite 或名称关键字过滤。"""
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
    """按 id 或 name（不区分大小写）返回完整模型记录。"""
    payload, models = _load_registry()
    needle = model_id.lower()
    for m in models:
        if m.get("id", "").lower() == needle or m.get("name", "").lower() == needle:
            return m
    raise ValueError(f"Model '{model_id}' not found in registry (version={payload.get('version')})")


def fetch_official_models() -> dict:
    """返回官方模型清单的完整 JSON。"""
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


def _sorted_items(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """按 PyTorch API 长度排序，优先匹配更长的名称。"""
    return sorted(items, key=lambda r: len(r.get("pytorch", "")), reverse=True)


def _collect_mapping_items(section: str | None = None) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """汇总一致/差异映射条目，可选按 section 限定。"""
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
    else:
        # 未指定 section 时合并所有分表，覆盖更多映射（如 torch_tensor 等）
        sec_cons_all = _load_section_map(OPMAP_SECTION_CONS_DIR)
        sec_diff_all = _load_section_map(OPMAP_SECTION_DIFF_DIR)
        for sec in sec_cons_all.values():
            if isinstance(sec, dict):
                cons_items += sec.get("items", [])
        for sec in sec_diff_all.values():
            if isinstance(sec, dict):
                diff_items += sec.get("items", [])

    return _sorted_items(cons_items), _sorted_items(diff_items)


def _count_occurrences(text: str, target: str) -> int:
    """统计边界安全的目标符号出现次数。"""
    if not target:
        return 0
    pattern = rf"(?<![\w.]){re.escape(target)}(?![\w.])"
    return len(list(re.finditer(pattern, text)))


def query_op_mapping(op: str, section: str | None = None) -> dict[str, list[dict[str, str]]]:
    """查询 PyTorch→MindSpore API 映射（支持 section 过滤与模糊匹配）。

    Args:
        op: PyTorch API 名称或子串（如 "torch.addmm" 或 "addmm"）。
        section: 可选 section（如 "torch"、"torchvision"）缩小搜索范围。

    Returns:
        {"consistent": [...], "diff": [...]} 匹配条目。
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


def diagnose_translation(original_code: str, translated_code: str, section: str | None = None) -> dict[str, Any]:
    """诊断 LLM 翻译结果：基于映射表检查替换是否到位。

    Args:
        original_code: 原始 PyTorch 代码。
        translated_code: LLM 翻译后的 MindSpore 代码。
        section: 可选 section（如 "torchvision"）用于缩小映射范围。

    Returns:
        {
            "applied_mappings": [...],   # 原文命中的一致映射及替换计数
            "missing_mappings": [...],   # 原文命中但译文未出现的映射
            "diff_hits": [...],          # 差异映射命中
            "extra_calls": [...],        # 译文出现但原文未触发的 MindSpore API
            "annotated": "...",          # 在原文标注 TODO 的版本
        }
    """
    cons_items, diff_items = _collect_mapping_items(section)

    applied: list[dict[str, Any]] = []
    missing: list[dict[str, Any]] = []
    extra: list[dict[str, Any]] = []
    diff_hits: list[dict[str, Any]] = []
    annotated = original_code

    def base_entry(row: dict[str, Any]) -> dict[str, Any]:
        return {k: row.get(k) for k in ("section", "pytorch", "mindspore", "description")}

    # 一致映射：检查源代码命中与译文替换情况
    for row in cons_items:
        pt = row.get("pytorch", "")
        ms = row.get("mindspore", "")
        if not pt or not ms:
            continue
        source_count = _count_occurrences(original_code, pt)
        translated_count = _count_occurrences(translated_code, ms) if translated_code else 0
        if source_count == 0 and translated_count == 0:
            continue
        entry = base_entry(row)
        entry["source_count"] = source_count
        entry["translated_count"] = translated_count
        applied.append(entry)
        if source_count > 0 and translated_count == 0:
            missing.append(entry)
        if source_count == 0 and translated_count > 0:
            extra_entry = entry.copy()
            extra_entry["note"] = "MindSpore API present but no matching PyTorch call found"
            extra.append(extra_entry)

    # 差异映射：仅提示，不自动替换
    for row in diff_items:
        pt = row.get("pytorch", "")
        if not pt:
            continue
        source_count = _count_occurrences(original_code, pt)
        if source_count == 0:
            continue
        entry = base_entry(row)
        entry["count"] = source_count
        if pt in SHAPE_HINT_APIS:
            entry["shape_hint"] = "check input/output shapes (expects matrix/matched dims)"
        diff_hits.append(entry)
        pattern = rf"(?<![\w.]){re.escape(pt)}(?![\w.])"
        desc = row.get("description") or "diff"
        ms = row.get("mindspore") or "mindspore.*"
        def add_comment(match: re.Match[str]) -> str:
            return f"# TODO: check mapping {pt} -> {ms}: {desc}\n{match.group(0)}"
        annotated = re.sub(pattern, add_comment, annotated)

    # 对未替换的命中添加标注，便于人工复核
    for miss in missing:
        pt = miss.get("pytorch") or ""
        ms = miss.get("mindspore") or "mindspore.*"
        if not pt:
            continue
        pattern = rf"(?<![\w.]){re.escape(pt)}(?![\w.])"
        def add_comment(match: re.Match[str]) -> str:
            return f"# TODO: replace {pt} -> {ms} per mapping\n{match.group(0)}"
        annotated = re.sub(pattern, add_comment, annotated)

    return {
        "applied_mappings": applied,
        "missing_mappings": missing,
        "diff_hits": diff_hits,
        "extra_calls": extra,
        "annotated": annotated,
    }
