#!/usr/bin/env python3
"""公开数据集准备：下载 → sha256 校验 → 固定子集切片。

数据源：
- LoCoMo（snap-research/locomo, data/locomo10.json, ACL 2024）：
  走 GitHub API blob 通道（部分网络下 raw.githubusercontent.com 不可达）；
  哈希锁死为首次下载时固定的版本，防止上游漂移。
- SummEval（Yale-LILY/SummEval, model_annotations_v1.json）：
  官方托管在 GCS，匿名访问可能被拒（403）；下载失败时支持手动放置：
  把文件放到 evaluation/public/data/sources/model_annotations_v1.json 再重跑。

产物写入 evaluation/public/data/（已 .gitignore，原始数据不入库）：
- sources/     原始文件（下载或手动放置）
- subsets/     固定子集（逐字节确定性，报告引用其哈希保证可复现）
- manifest.json 记录每个文件的来源与哈希

用法：
  uv run python -m evaluation.public.prepare_datasets            # 准备全部
  uv run python -m evaluation.public.prepare_datasets --only locomo
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import json
import sys
import urllib.request
from pathlib import Path
from typing import Any

from evaluation.public.locomo_adapter import (
    DEFAULT_CONV_INDICES,
    DEFAULT_MAX_PER_CATEGORY,
    build_locomo_subset,
)

HERE = Path(__file__).resolve().parent
DATA_DIR = HERE / "data"
SOURCES_DIR = DATA_DIR / "sources"
SUBSETS_DIR = DATA_DIR / "subsets"
MANIFEST_PATH = DATA_DIR / "manifest.json"

# LoCoMo 锁定版本（首次下载时记录；上游漂移会在哈希校验处硬失败）
LOCOMO_META_URL = (
    "https://api.github.com/repos/snap-research/locomo/contents/data/locomo10.json"
)
LOCOMO_BLOB_URL = "https://api.github.com/repos/snap-research/locomo/git/blobs/{sha}"
LOCOMO_SHA256 = "79fa87e90f04081343b8c8debecb80a9a6842b76a7aa537dc9fdf651ea698ff4"

# SummEval 官方托管地址（可能 403）；失败时降级为手动放置
SUMMEVAL_URLS = [
    "https://storage.googleapis.com/sfr-summarization-repo-research/"
    "model_annotations_v1.json",
]
SUMMEVAL_FILENAME = "model_annotations_v1.json"
SUMMEVAL_CALIBRATION_SIZE = 200
# 人工一致性标签阈值：专家均分 >= 4.0（5 分制）视为事实一致
SUMMEVAL_CONSISTENCY_THRESHOLD = 4.0


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _load_manifest() -> dict[str, Any]:
    if MANIFEST_PATH.exists():
        return json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    return {}


def _save_manifest(manifest: dict[str, Any]) -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    MANIFEST_PATH.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8")


def _http_get(url: str, timeout: int = 300) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "kamaclaude-eval"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read()


# GitHub contents API 对 >1MB 文件不返回内容，改走 git blob API（base64）
def download_locomo(dest: Path) -> bytes:
    if dest.exists():
        data = dest.read_bytes()
        if _sha256(data) == LOCOMO_SHA256:
            print(f"locomo: 复用已有 {dest}（哈希匹配）")
            return data
        print("locomo: 已有文件哈希不匹配，重新下载")
    meta = json.loads(_http_get(LOCOMO_META_URL, timeout=30))
    blob = json.loads(_http_get(LOCOMO_BLOB_URL.format(sha=meta["sha"])))
    data = base64.b64decode(blob["content"])
    digest = _sha256(data)
    if digest != LOCOMO_SHA256:
        raise SystemExit(
            f"locomo10.json 哈希漂移：期望 {LOCOMO_SHA256}，实际 {digest}。"
            "若确认为上游更新，需人工复核后更新 LOCOMO_SHA256 常量。")
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(data)
    print(f"locomo: 下载完成 {len(data)} bytes -> {dest}")
    return data


# SummEval：依次尝试远程地址；全部失败时检查手动放置路径
def fetch_summeval(dest: Path) -> bytes | None:
    if dest.exists() and dest.stat().st_size > 0:
        print(f"summeval: 复用已有 {dest}")
        return dest.read_bytes()
    for url in SUMMEVAL_URLS:
        try:
            data = _http_get(url)
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(data)
            print(f"summeval: 下载完成 {len(data)} bytes -> {dest}")
            return data
        except Exception as exc:  # noqa: BLE001 — 源不可达时降级为手动放置
            print(f"summeval: {url} 下载失败（{exc.__class__.__name__}）")
    print(f"summeval: 远程源不可达。请手动放置 {dest} 后重跑；"
          "裁判校准将降级为 LoCoMo EM 自校准。")
    return None


# SummEval → 裁判校准子集：前 N 条含专家标注的样本，转二分类一致性标签
def build_summeval_calibration(raw: list[dict[str, Any]]) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    for entry in raw:
        ann = entry.get("expert_annotations") or entry.get("annotations")
        scores = [d.get("consistency") for d in ann
                  if isinstance(d, dict) and d.get("consistency") is not None] \
            if isinstance(ann, list) else []
        if not scores or not entry.get("decoded") or not entry.get("text"):
            continue
        items.append({
            "id": f"summeval-{len(items):03d}",
            "source": entry["text"],
            "summary": entry["decoded"],
            "human_consistent": sum(scores) / len(scores)
            >= SUMMEVAL_CONSISTENCY_THRESHOLD,
            "human_mean_score": round(sum(scores) / len(scores), 2),
        })
        if len(items) >= SUMMEVAL_CALIBRATION_SIZE:
            break
    return items


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = "".join(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n"
                      for r in rows)
    path.write_text(payload, encoding="utf-8")
    return _sha256(payload.encode("utf-8"))


def prepare_locomo(manifest: dict[str, Any]) -> None:
    raw_bytes = download_locomo(SOURCES_DIR / "locomo10.json")
    raw = json.loads(raw_bytes)
    convs, qa = build_locomo_subset(raw, DEFAULT_CONV_INDICES,
                                    DEFAULT_MAX_PER_CATEGORY)
    conv_hash = _write_jsonl(SUBSETS_DIR / "locomo_convs.jsonl", convs)
    qa_hash = _write_jsonl(SUBSETS_DIR / "locomo_qa.jsonl", qa)
    manifest["locomo"] = {
        "source": LOCOMO_META_URL,
        "source_sha256": LOCOMO_SHA256,
        "conv_indices": list(DEFAULT_CONV_INDICES),
        "max_per_category": DEFAULT_MAX_PER_CATEGORY,
        "convs": len(convs),
        "questions": len(qa),
        "subset_convs_sha256": conv_hash,
        "subset_qa_sha256": qa_hash,
    }
    print(f"locomo: 子集 {len(convs)} 段对话 / {len(qa)} 道题"
          f"（qa sha256 {qa_hash[:12]}…）")


def prepare_summeval(manifest: dict[str, Any]) -> None:
    data = fetch_summeval(SOURCES_DIR / SUMMEVAL_FILENAME)
    if data is None:
        return
    raw = json.loads(data)
    calibration = build_summeval_calibration(raw)
    calib_hash = _write_jsonl(SUBSETS_DIR / "summeval_calibration.jsonl",
                              calibration)
    manifest["summeval"] = {
        "source": SUMMEVAL_URLS[0],
        "source_sha256": _sha256(data),
        "calibration_size": len(calibration),
        "subset_sha256": calib_hash,
    }
    print(f"summeval: 校准子集 {len(calibration)} 条"
          f"（sha256 {calib_hash[:12]}…）")


def main() -> int:
    parser = argparse.ArgumentParser(description="prepare public eval datasets")
    parser.add_argument("--only", choices=["locomo", "summeval"], default=None)
    args = parser.parse_args()

    manifest = _load_manifest()
    try:
        if args.only in (None, "locomo"):
            prepare_locomo(manifest)
        if args.only in (None, "summeval"):
            prepare_summeval(manifest)
    except SystemExit:
        raise
    except Exception as exc:  # noqa: BLE001 — 统一给出降级指引
        print(f"准备失败: {exc}")
        print("离线降级：从可联网环境获取数据后放入 "
              f"{SOURCES_DIR}/，再重跑本脚本（会直接复用并校验哈希）。")
        return 1
    _save_manifest(manifest)
    print(f"manifest: {MANIFEST_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
