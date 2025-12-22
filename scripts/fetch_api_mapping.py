#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Fetch PyTorch ↔ MindSpore API mapping table and split into consistent vs differing entries.

Output:
- data/pytorch_ms_api_mapping_consistent.json
- data/pytorch_ms_api_mapping_diff.json
"""

from __future__ import annotations

import json
import time
import re
from pathlib import Path
from typing import List, Dict, Tuple, Set

import requests
from bs4 import BeautifulSoup

URL = "https://www.mindspore.cn/docs/en/master/note/api_mapping/pytorch_api_mapping.html"
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/122.0.0.0 Safari/537.36"
    )
}
OUTPUT_DIR = Path(__file__).resolve().parents[1] / "data"
OUTPUT_CONSISTENT = OUTPUT_DIR / "pytorch_ms_api_mapping_consistent.json"
OUTPUT_DIFF = OUTPUT_DIR / "pytorch_ms_api_mapping_diff.json"
OUTPUT_SECTIONS_DIR = OUTPUT_DIR / "convert"
OUTPUT_SECTIONS_CONS = OUTPUT_SECTIONS_DIR / "consistent"
OUTPUT_SECTIONS_DIFF = OUTPUT_SECTIONS_DIR / "diff"


def fetch_html(url: str) -> str:
    """Fetch HTML content."""
    resp = requests.get(url, headers=HEADERS, timeout=20)
    resp.raise_for_status()
    resp.encoding = resp.apparent_encoding or "utf-8"
    return resp.text


def clean_text(text: str) -> str:
    """Strip non-printable/strange characters (e.g., link icons)."""
    return "".join(ch for ch in text if ch.isprintable() and ch not in {"\uf0c1"}).strip()


def extract_version_hints(texts: List[str]) -> List[str]:
    """Extract version hints (PyTorch/MindSpore/Python/torchaudio) from section headings or visible text."""
    hints: List[str] = []
    for raw in texts:
        text = clean_text(raw)
        if not text:
            continue
        lower = text.lower()
        if any(k in lower for k in ["pytorch", "mindspore", "python", "torchaudio"]):
            if any(ver in lower for ver in ["api", "version", "torch", "mindspore", "python", "torchaudio", "0.", "1.", "2."]):
                hints.append(text)
    # Deduplicate while preserving order
    seen: Set[str] = set()
    uniq: List[str] = []
    for h in hints:
        if h not in seen:
            uniq.append(h)
            seen.add(h)
    return uniq[:15]  # keep a few for context


def parse_mapping(html: str) -> Tuple[List[Dict[str, str]], List[str]]:
    """Parse mapping rows from the HTML table and collect version hints."""
    soup = BeautifulSoup(html, "html.parser")
    rows: List[Dict[str, str]] = []
    section_texts: List[str] = []
    for table in soup.find_all("table"):
        heading = table.find_previous(["h1", "h2", "h3", "h4"])
        section = clean_text(heading.get_text(strip=True)) if heading else ""
        if section:
            section_texts.append(section)
        # also record table header text (可能包含 'PyTorch 2.1 APIs' 等)
        header_text = clean_text(" ".join(th.get_text(strip=True) for th in table.find_all("th")))
        if header_text:
            section_texts.append(header_text)
        headers = [th.get_text(strip=True).lower() for th in table.find_all("th")]
        if len(headers) < 3:
            continue
        first_ok = any(
            key in headers[0]
            for key in ["pytorch", "torchaudio", "torchvision", "torchtext"]
        )
        second_ok = "mindspore" in headers[1]
        if not (first_ok and second_ok):
            continue
        for tr in table.find_all("tr")[1:]:
            cells = [td.get_text(strip=True) for td in tr.find_all("td")]
            if len(cells) < 3:
                continue
            rows.append(
                {
                    "section": section,
                    "header": header_text,
                    "pytorch": cells[0],
                    "mindspore": cells[1],
                    "description": cells[2],
                }
            )
    # collect version hints from sections and visible text
    version_hints = extract_version_hints(section_texts)
    if not version_hints:
        version_hints = extract_version_hints(list(soup.stripped_strings))
    return rows, version_hints


def split_rows(rows: List[Dict[str, str]]) -> tuple[List[Dict[str, str]], List[Dict[str, str]]]:
    """Split rows into consistent vs differing based on description text."""
    consistent: List[Dict[str, str]] = []
    diff: List[Dict[str, str]] = []
    for row in rows:
        desc = row.get("description", "").strip().lower()
        if desc == "consistent":
            consistent.append(row)
        else:
            diff.append(row)
    return consistent, diff


def dump_with_meta(path: Path, items: List[Dict[str, str]], total: int, diff_count: int, version_hints: List[str]) -> None:
    payload = {
        "meta": {
            "source": URL,
            "fetched_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "total_rows": total,
            "diff_rows": diff_count,
            "version_hints": version_hints,
        },
        "items": items,
    }
    with path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=2)


def slugify(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    text = text.strip("_")
    return text or "unknown"


def main() -> None:
    print(f"[INFO] fetching mapping table from {URL}")
    html = fetch_html(URL)
    rows, version_hints = parse_mapping(html)
    consistent, diff = split_rows(rows)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_SECTIONS_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_SECTIONS_CONS.mkdir(parents=True, exist_ok=True)
    OUTPUT_SECTIONS_DIFF.mkdir(parents=True, exist_ok=True)
    dump_with_meta(OUTPUT_CONSISTENT, consistent, len(rows), len(diff), version_hints)
    dump_with_meta(OUTPUT_DIFF, diff, len(rows), len(diff), version_hints)

    # split by section into per-section files
    sections: Dict[str, List[Dict[str, str]]] = {}
    for r in rows:
        sec = r.get("section") or "unknown"
        sections.setdefault(sec, []).append(r)
    for sec, items in sections.items():
        sec_consistent, sec_diff = split_rows(items)
        base = slugify(sec)
        sec_hints = []
        # collect section/header hints to meta
        if sec:
            sec_hints.append(sec)
        # header field may carry version wording
        headers = {clean_text(r.get("header", "")) for r in items if r.get("header")}
        for h in headers:
            if h:
                sec_hints.append(h)
        sec_hints = extract_version_hints(sec_hints) or sec_hints
        dump_with_meta(
            OUTPUT_SECTIONS_CONS / f"{base}_consistent.json",
            sec_consistent,
            len(items),
            len(sec_diff),
            sec_hints,
        )
        dump_with_meta(
            OUTPUT_SECTIONS_DIFF / f"{base}_diff.json",
            sec_diff,
            len(items),
            len(sec_diff),
            sec_hints,
        )

    print(f"[INFO] total rows: {len(rows)}, consistent: {len(consistent)}, diff: {len(diff)}")
    print(f"[INFO] version hints: {version_hints}")
    print(f"[INFO] written: {OUTPUT_CONSISTENT}")
    print(f"[INFO] written: {OUTPUT_DIFF}")


if __name__ == "__main__":
    main()
