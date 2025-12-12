#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
从官方页面拉取 MindSpore 模型列表并生成规范化 JSON。

来源：
https://www.mindspore.cn/docs/zh-CN/r2.3.1/note/official_models.html
输出：
data/mindspore_official_models.json
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests
from bs4 import BeautifulSoup, Tag

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/122.0.0.0 Safari/537.36"
    )
}
INDEX_URL = "https://www.mindspore.cn/docs/zh-CN/r2.3.1/note/official_models.html"
OUTPUT = Path(__file__).resolve().parent.parent / "data" / "mindspore_official_models.json"
RETRY = 3
TIMEOUT = 15


@dataclass
class ModelRow:
    """Standardized model information."""

    id: str
    name: str
    group: str
    category: str
    task: List[str]
    suite: Optional[str]
    variants: List[str]
    links: Dict[str, Optional[str]]
    metrics: Optional[Dict[str, Any]] = None
    dataset: Optional[str] = None
    hardware: Optional[Dict[str, Optional[bool]]] = None

    def as_dict(self) -> Dict[str, Any]:
        """Convert to serializable dictionary."""
        return {
            "id": self.id,
            "name": self.name,
            "group": self.group,
            "category": self.category,
            "task": self.task,
            "suite": self.suite,
            "variants": self.variants,
            "links": self.links,
            "metrics": self.metrics,
            "dataset": self.dataset,
            "hardware": self.hardware,
        }


def get_soup(url: str) -> BeautifulSoup:
    """Fetch HTML with retries."""
    parser = "lxml"
    try:
        BeautifulSoup("<html></html>", parser)  # type: ignore[arg-type]
    except Exception:
        parser = "html.parser"
    for attempt in range(1, RETRY + 1):
        try:
            resp = requests.get(url, headers=HEADERS, timeout=TIMEOUT)
            resp.raise_for_status()
            resp.encoding = resp.apparent_encoding or "utf-8"
            return BeautifulSoup(resp.text, parser)
        except Exception as exc:  # noqa: BLE001
            print(f"[WARN] fetch failed {attempt}/{RETRY}: {exc}")
            if attempt == RETRY:
                raise
            time.sleep(2)
    raise RuntimeError("Unreachable fetch loop")


def normalize_text(text: str) -> str:
    """Collapse whitespace for stable parsing."""
    return " ".join(text.split())


def parse_llm(soup: BeautifulSoup) -> List[ModelRow]:
    section = soup.find("section", {"id": "大语言模型"})
    if not section:
        return []
    table = section.find("table")
    if not table:
        return []
    rows: List[ModelRow] = []
    for tr in table.find_all("tr")[1:]:
        cells = tr.find_all("td")
        if len(cells) < 2:
            continue
        model_cell = cells[0]
        model_name = normalize_text(model_cell.get_text())
        card_link = model_cell.find("a").get("href") if model_cell.find("a") else None
        variants_text = cells[1].get_text()
        variants = [normalize_text(v) for v in variants_text.split(",") if normalize_text(v)]
        rows.append(
            ModelRow(
                id=model_name.lower(),
                name=model_name,
                group="领域套件与扩展包",
                category="大语言模型",
                task=["text-generation"],
                suite="mindformers",
                variants=variants,
                links={"card": card_link, "config": None},
            )
        )
    return rows


def parse_image_classification(soup: BeautifulSoup) -> List[ModelRow]:
    section = soup.find("section", {"id": "图像分类骨干类"})
    if not section:
        return []
    table = section.find("table")
    if not table:
        return []
    rows: List[ModelRow] = []
    for tr in table.find_all("tr")[1:]:
        cells = tr.find_all("td")
        if len(cells) < 3:
            continue
        model_name = normalize_text(cells[0].get_text())
        acc_text = normalize_text(cells[1].get_text())
        config_link = cells[2].find("a").get("href") if cells[2].find("a") else None
        try:
            acc_value = float(acc_text)
        except ValueError:
            acc_value = None
        rows.append(
            ModelRow(
                id=model_name.lower(),
                name=model_name,
                group="领域套件与扩展包",
                category="图像分类（骨干类）",
                task=["image-classification"],
                suite="mindcv",
                variants=[model_name],
                links={"card": None, "config": config_link},
                metrics={"acc@1": acc_value} if acc_value is not None else None,
                dataset="ImageNet-1K",
            )
        )
    return rows


def parse_ocr(soup: BeautifulSoup) -> List[ModelRow]:
    ocr_section = soup.find("section", {"id": "ocr"})
    if not ocr_section:
        return []
    mapping = {
        "文本检测": ("OCR/文本检测", "text-detection"),
        "文本识别": ("OCR/文本识别", "text-recognition"),
        "文本方向分类": ("OCR/文本方向分类", "text-orientation"),
    }
    rows: List[ModelRow] = []
    for sub_id, (category_name, task_id) in mapping.items():
        sub_sec = ocr_section.find("section", {"id": sub_id})
        if not sub_sec:
            continue
        table = sub_sec.find("table")
        if not table:
            continue
        for tr in table.find_all("tr")[1:]:
            cells = tr.find_all("td")
            if len(cells) < 4:
                continue
            model_cell = cells[0]
            model_name = normalize_text(model_cell.get_text())
            card_link = model_cell.find("a").get("href") if model_cell.find("a") else None
            dataset = normalize_text(cells[1].get_text())
            metric_text = normalize_text(cells[2].get_text())
            try:
                fscore = float(metric_text)
            except ValueError:
                fscore = None
            config_link = cells[3].find("a").get("href") if cells[3].find("a") else None
            rows.append(
                ModelRow(
                    id=model_name.lower(),
                    name=model_name,
                    group="领域套件与扩展包",
                    category=category_name,
                    task=[task_id],
                    suite="mindocr",
                    variants=[model_name],
                    links={"card": card_link, "config": config_link},
                    metrics={"fscore": fscore} if fscore is not None else None,
                    dataset=dataset or None,
                )
            )
    return rows


def parse_object_detection(soup: BeautifulSoup) -> List[ModelRow]:
    det_sec = soup.find("section", {"id": "目标检测"})
    if not det_sec:
        return []
    yolo_sec = det_sec.find("section", {"id": "yolo系列"})
    if not yolo_sec:
        return []
    table = yolo_sec.find("table")
    if not table:
        return []
    rows: List[ModelRow] = []
    for tr in table.find_all("tr")[1:]:
        cells = tr.find_all("td")
        if len(cells) < 4:
            continue
        model_cell = cells[0]
        model_name = normalize_text(model_cell.get_text())
        card_link = model_cell.find("a").get("href") if model_cell.find("a") else None
        dataset = normalize_text(cells[1].get_text())
        metric_text = normalize_text(cells[2].get_text())
        config_link = cells[3].find("a").get("href") if cells[3].find("a") else None
        try:
            map_val = float(metric_text)
        except ValueError:
            map_val = None
        # 根据链接猜测使用 mindcv 或 mindocr
        suite = "mindocr" if config_link and "mindocr" in config_link else "mindcv"
        rows.append(
            ModelRow(
                id=model_name.lower(),
                name=model_name,
                group="领域套件与扩展包",
                category="目标检测/YOLO系列",
                task=["object-detection"],
                suite=suite,
                variants=[model_name],
                links={"card": card_link, "config": config_link},
                metrics={"map": map_val} if map_val is not None else None,
                dataset=dataset or None,
            )
        )
    return rows


def parse_reinforcement_learning(soup: BeautifulSoup) -> List[ModelRow]:
    rl_sec = soup.find("section", {"id": "强化学习"})
    if not rl_sec:
        return []
    table = rl_sec.find("table")
    if not table:
        return []
    rows: List[ModelRow] = []
    for tr in table.find_all("tr")[1:]:
        cells = tr.find_all("td")
        if len(cells) < 7:
            continue
        model_cell = cells[0]
        model_name = normalize_text(model_cell.get_text())
        card_link = model_cell.find("a").get("href") if model_cell.find("a") else None
        dataset = normalize_text(cells[6].get_text())
        score_text = normalize_text(cells[7].get_text()) if len(cells) > 7 else ""
        try:
            score_val = float(score_text)
        except ValueError:
            score_val = None
        rows.append(
            ModelRow(
                id=model_name.lower(),
                name=model_name,
                group="领域套件与扩展包",
                category="强化学习",
                task=["reinforcement-learning"],
                suite="mindrl",
                variants=[model_name],
                links={"card": card_link, "config": None},
                metrics={"score": score_val} if score_val is not None else None,
                dataset=dataset or None,
            )
        )
    return rows


def parse_recommendation(soup: BeautifulSoup) -> List[ModelRow]:
    rec_sec = soup.find("section", {"id": "推荐"})
    if not rec_sec:
        return []
    table = rec_sec.find("table")
    if not table:
        return []
    rows: List[ModelRow] = []
    for tr in table.find_all("tr")[1:]:
        cells = tr.find_all("td")
        if len(cells) < 5:
            continue
        model_name = normalize_text(cells[0].get_text())
        dataset = normalize_text(cells[1].get_text())
        auc_text = normalize_text(cells[2].get_text())
        mindrec_link = cells[3].find("a").get("href") if cells[3].find("a") else None
        ms_link = cells[4].find("a").get("href") if cells[4].find("a") else None
        try:
            auc_val = float(auc_text)
        except ValueError:
            auc_val = None
        rows.append(
            ModelRow(
                id=model_name.lower(),
                name=model_name,
                group="领域套件与扩展包",
                category="推荐",
                task=["recommendation"],
                suite="mindrec",
                variants=[model_name],
                links={"mindrec": mindrec_link, "mindspore": ms_link},
                metrics={"auc": auc_val} if auc_val is not None else None,
                dataset=dataset or None,
            )
        )
    return rows


def parse_scientific_suite(soup: BeautifulSoup) -> List[ModelRow]:
    sci_sec = soup.find("section", {"id": "科学计算套件"})
    if not sci_sec:
        return []
    table = sci_sec.find("table")
    if not table:
        return []
    rows: List[ModelRow] = []
    for tr in table.find_all("tr")[1:]:
        cells = tr.find_all("td")
        if len(cells) < 5:
            continue
        domain = normalize_text(cells[0].get_text())
        model_cell = cells[1]
        model_name = normalize_text(model_cell.get_text())
        card_link = model_cell.find("a").get("href") if model_cell.find("a") else None
        impl_link = cells[2].find("a").get("href") if cells[2].find("a") else None
        ascend_text = normalize_text(cells[3].get_text())
        gpu_text = normalize_text(cells[4].get_text())
        rows.append(
            ModelRow(
                id=model_name.lower(),
                name=model_name,
                group="科学计算套件",
                category=domain,
                task=["scientific-computing"],
                suite="mindscience",
                variants=[model_name],
                links={"card": card_link, "implementation": impl_link},
                metrics=None,
                dataset=None,
                hardware={"ascend": "✅" in ascend_text if ascend_text else None, "gpu": "✅" in gpu_text if gpu_text else None},
            )
        )
    return rows


def build_payload(models: List[ModelRow]) -> Dict[str, Any]:
    """Assemble final JSON payload."""
    return {
        "version": "r2.3.1",
        "source": INDEX_URL,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "count": len(models),
        "models": [m.as_dict() for m in models],
    }


def main() -> None:
    print("[INFO] fetching official models page...")
    soup = get_soup(INDEX_URL)

    models: List[ModelRow] = []
    models.extend(parse_llm(soup))
    models.extend(parse_image_classification(soup))
    models.extend(parse_ocr(soup))
    models.extend(parse_object_detection(soup))
    models.extend(parse_reinforcement_learning(soup))
    models.extend(parse_recommendation(soup))
    models.extend(parse_scientific_suite(soup))

    payload = build_payload(models)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT.open("w", encoding="utf-8") as fp:
        json.dump(payload, fp, ensure_ascii=False, indent=2)
    print(f"[INFO] wrote {payload['count']} models -> {OUTPUT}")


if __name__ == "__main__":
    main()
