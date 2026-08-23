#!/usr/bin/env python3
"""Tier1：compact 静态信息保留率评测 harness。

离线模式（默认，零 API 成本，可进 CI）：
  - 数据集校验：所有预埋事实必须落在 history 区（最近 3 轮之外）
  - 确定性预处理后的字面事实保留率（错误逐字 / 路径进 stub / ID 在 tool input）
  - 预处理压缩比（真实 tokenizer 计量）
  门禁：字面事实预处理保留率必须 100%——确定性手段不允许丢字面信息。

在线模式（--live，需要 ANTHROPIC_API_KEY）：
  - 端到端压缩比、字面保留率、语义保留率（LLM-as-judge 逐事实判定）
  - 六章节摘要结构完整性
  - 保留率按埋点深度分桶（early/mid/deep）输出衰减曲线

用法：
  uv run python scripts/run_compact_eval.py                 # 离线
  uv run python scripts/run_compact_eval.py --live          # 含 LLM 摘要质量
  uv run python scripts/run_compact_eval.py --live --limit 4
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv  # noqa: E402

load_dotenv(ROOT / ".env")  # --live 需要 ANTHROPIC_API_KEY

# 默认模型跟随项目配置（.env 的 KAMA_LLM_DEFAULT_MODEL），避免硬编码失效模型名
_DEFAULT_MODEL = os.environ.get("KAMA_LLM_DEFAULT_MODEL", "claude-sonnet-4-6")

from evaluation.compact.generate_cases import (  # noqa: E402
    DEFAULT_OUT,
    build_case_messages,
    load_cases,
)

from kama_claude.core.compact.compactor import (  # noqa: E402
    TAIL_ROUNDS,
    _find_tail_boundary,
    _messages_to_text,
    _preprocess_for_summary,
)
from kama_claude.core.llm.tokens import TOKENIZER_NAME, count_messages_tokens  # noqa: E402

_JUDGE_PROMPT = """\
You are auditing a compressed conversation summary for information loss.

FACT (planted earlier in the conversation):
{fact}

SUMMARY produced by the compactor:
{summary}

Question: is this fact preserved in the summary — either verbatim, or with \
enough detail that a continuing agent could act on it? For IDs, file paths \
and error messages "preserved" requires the exact value to be recoverable.

Answer YES or NO on the first line, then one sentence of justification.
"""

_SECTION_RE = re.compile(r"^##\s*[1-6][\.\s]", re.MULTILINE)


# 单个 case 的离线指标：预处理阶段字面事实保留率 + 压缩比 + 埋点位置校验
def evaluate_case_offline(case: dict[str, Any]) -> dict[str, Any]:
    messages = build_case_messages(case)
    split = _find_tail_boundary(messages, tail_rounds=TAIL_ROUNDS)
    history = messages[:split]

    # 数据集校验：每个事实必须埋在 history 区，否则评测无效；
    # 第 r 轮真实用户消息 = 消息列表索引 3*(r-1)（每轮固定 3 条消息）
    facts_in_history = [3 * (f["round"] - 1) < split
                        for f in case["planted_facts"]]

    processed_text = _messages_to_text(_preprocess_for_summary(history))
    literals = [f for f in case["planted_facts"] if f["check"] == "literal"]
    literal_hits = [f for f in literals if f["text"] in processed_text]

    before = count_messages_tokens(history)
    after = count_messages_tokens(_preprocess_for_summary(history))
    return {
        "case_id": case["id"],
        "facts_in_history": all(facts_in_history),
        "literal_total": len(literals),
        "literal_hits": len(literal_hits),
        "literal_retention": len(literal_hits) / len(literals) if literals else 1.0,
        "before_tokens": before,
        "after_tokens": after,
        "preprocess_ratio": round(after / before, 3) if before else 0.0,
    }


# 单个语义事实的 LLM-as-judge 判定：YES/NO 取首行
async def _judge_fact(provider: Any, summary: str, fact: str) -> bool:
    from kama_claude.core.events.bus import EventBus

    response = await provider.chat(
        messages=[{"role": "user",
                   "content": _JUDGE_PROMPT.format(fact=fact, summary=summary)}],
        tool_schemas=[],
        bus=EventBus(),
        run_id="eval-judge",
        step=0,
        system="You are a strict audit assistant.",
    )
    first_line = response.text.strip().splitlines()[0].upper() if response.text else ""
    return "YES" in first_line


# 单个 case 的在线指标：真实压缩后的字面/语义保留率、结构完整性、压缩比
async def evaluate_case_live(
    case: dict[str, Any], provider: Any, judge: Any
) -> dict[str, Any]:
    from kama_claude.core.compact.compactor import Compactor
    from kama_claude.core.events.bus import EventBus

    messages = build_case_messages(case)
    compactor = Compactor(EventBus(), Path("."), case["id"])
    result = await compactor.compact_messages(messages, provider)
    if result is None:
        return {"case_id": case["id"], "error": "compaction skipped"}

    searchable = (result.summary_text
                  + _messages_to_text(result.tail_messages or [])).lower()
    literals = [f for f in case["planted_facts"] if f["check"] == "literal"]
    semantics = [f for f in case["planted_facts"] if f["check"] == "semantic"]

    lit_detail = {f["key"]: f["text"].lower() in searchable for f in literals}
    sem_detail: dict[str, bool] = {}
    for f in semantics:
        sem_detail[f["key"]] = await _judge_fact(judge, result.summary_text,
                                                 f["text"])

    tail_tokens = count_messages_tokens(result.tail_messages or [])
    after = result.summary_tokens + tail_tokens
    return {
        "case_id": case["id"],
        "before_tokens": result.original_token_estimate,
        "after_tokens": after,
        "compression_ratio": round(result.original_token_estimate / after, 1)
        if after else 0.0,
        "six_sections_present": len(set(_SECTION_RE.findall(result.summary_text)))
        == 6,
        "literal_retention": sum(lit_detail.values()) / len(literals)
        if literals else 1.0,
        "semantic_retention": sum(sem_detail.values()) / len(semantics)
        if semantics else 1.0,
        "literal_detail": lit_detail,
        "semantic_detail": sem_detail,
        "facts": {f["key"]: f for f in case["planted_facts"]},
    }


def _aggregate(rows: list[dict[str, Any]], key: str) -> float:
    vals = [r[key] for r in rows if key in r and not r.get("error")]
    return round(sum(vals) / len(vals), 3) if vals else 0.0


# 深度衰减曲线：按 early/mid/deep 分桶统计保留率
def _depth_decay(rows: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    buckets: dict[str, list[bool]] = {}
    for r in rows:
        if r.get("error"):
            continue
        for f in r.get("facts", {}).values():
            detail = r.get("literal_detail") if f["check"] == "literal" \
                else r.get("semantic_detail")
            kept = bool(detail and detail.get(f["key"]))
            buckets.setdefault(f["depth"], []).append(kept)
    return {
        depth: {"n": len(hits), "retention": round(sum(hits) / len(hits), 3)}
        for depth, hits in sorted(buckets.items())
    }


def _write_reports(out_dir: Path, report: dict[str, Any], stem: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"{stem}.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    lines = [f"# Tier1 Compact 静态保留率报告（{report['mode']}）", ""]
    off = report["offline"]
    lines += [
        f"- tokenizer: {report['tokenizer']}",
        f"- 数据集: {report['dataset']}（{report['cases']} cases，"
        f"sha256 `{report['dataset_sha256'][:12]}…`）",
        "",
        "## 离线指标（确定性预处理）",
        "",
        f"- 字面事实预处理保留率: {off['literal_retention']:.0%}"
        f"（门禁 = 100%：{'PASS' if off['gate_passed'] else 'FAIL'}）",
        f"- 预处理压缩比（后/前）: {off['avg_preprocess_ratio']:.0%}",
        f"- 埋点位置校验（全部落在 history 区）: "
        f"{'✓' if off['all_facts_in_history'] else '✗'}",
    ]
    if "live" in report:
        live = report["live"]
        lines += [
            "",
            f"## 在线指标（model={live['model']}）",
            "",
            f"- 端到端压缩比: {live['avg_compression_ratio']}x",
            f"- 字面保留率: {live['literal_retention']:.0%}",
            f"- 语义保留率（LLM-as-judge）: {live['semantic_retention']:.0%}",
            f"- 六章节结构完整: {live['structure_ok']}/{live['cases_run']}",
            "",
            "### 保留率随埋点深度衰减",
            "",
            "| 深度桶 | 事实数 | 保留率 |",
            "|---|---:|---:|",
        ]
        for depth, stat in live["depth_decay"].items():
            lines.append(f"| {depth} | {stat['n']} | {stat['retention']:.0%} |")
    lines += [
        "",
        "> 口径声明：本基准为受控合成数据（事实埋点位置与类型可控），用于验证",
        "> 压缩策略的相对质量与回归门禁，不代表生产环境真实会话效果。",
        "",
    ]
    (out_dir / f"{stem}.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Tier1 compact retention eval")
    parser.add_argument("--dataset", default=str(DEFAULT_OUT))
    parser.add_argument("--output-dir", default=str(ROOT / "output" / "evaluation"))
    parser.add_argument("--live", action="store_true")
    parser.add_argument("--model", default=_DEFAULT_MODEL)
    parser.add_argument("--judge-model", default=None,
                        help="语义判定模型，默认与 --model 相同")
    parser.add_argument("--limit", type=int, default=0,
                        help="在线模式下只跑前 N 个 case（0=全部）")
    args = parser.parse_args()

    dataset_path = Path(args.dataset)
    if not dataset_path.exists():
        print(f"dataset not found: {dataset_path} —— 请先运行 "
              "uv run python -m evaluation.compact.generate_cases")
        return 1
    cases = load_cases(dataset_path)
    digest = __import__("hashlib").sha256(
        dataset_path.read_bytes()).hexdigest()

    offline_rows = [evaluate_case_offline(c) for c in cases]
    off_report = {
        "literal_retention": _aggregate(offline_rows, "literal_retention"),
        "avg_preprocess_ratio": _aggregate(offline_rows, "preprocess_ratio"),
        "all_facts_in_history": all(r["facts_in_history"] for r in offline_rows),
        "gate_passed": all(r["literal_hits"] == r["literal_total"]
                           for r in offline_rows),
    }
    report: dict[str, Any] = {
        "mode": "offline+live" if args.live else "offline",
        "generated_at": datetime.now(UTC).isoformat(),
        "tokenizer": TOKENIZER_NAME,
        "dataset": str(dataset_path),
        "dataset_sha256": digest,
        "cases": len(cases),
        "offline": off_report,
        "offline_rows": offline_rows,
    }

    print(f"# Tier1 Compact Eval ({len(cases)} cases, tokenizer={TOKENIZER_NAME})")
    print(f"- 字面事实预处理保留率: {off_report['literal_retention']:.0%}"
          f"（门禁 {'PASS' if off_report['gate_passed'] else 'FAIL'}）")
    print(f"- 预处理压缩比: {off_report['avg_preprocess_ratio']:.0%}")
    print(f"- 埋点全部在 history 区: {'✓' if off_report['all_facts_in_history'] else '✗'}")

    if args.live:
        from kama_claude.core.llm.provider import AnthropicProvider

        provider = AnthropicProvider(args.model)
        judge = provider if not args.judge_model \
            else AnthropicProvider(args.judge_model)
        selected = cases[:args.limit] if args.limit else cases
        live_rows = []
        for i, case in enumerate(selected):
            row = asyncio.run(evaluate_case_live(case, provider, judge))
            live_rows.append(row)
            if row.get("error"):
                print(f"  [{i + 1}/{len(selected)}] {case['id']}: {row['error']}")
            else:
                print(f"  [{i + 1}/{len(selected)}] {case['id']}: "
                      f"ratio={row['compression_ratio']}x "
                      f"literal={row['literal_retention']:.0%} "
                      f"semantic={row['semantic_retention']:.0%}")
        ok_rows = [r for r in live_rows if not r.get("error")]
        report["live"] = {
            "model": args.model,
            "cases_run": len(ok_rows),
            "avg_compression_ratio": _aggregate(ok_rows, "compression_ratio"),
            "literal_retention": _aggregate(ok_rows, "literal_retention"),
            "semantic_retention": _aggregate(ok_rows, "semantic_retention"),
            "structure_ok": sum(1 for r in ok_rows if r["six_sections_present"]),
            "depth_decay": _depth_decay(ok_rows),
            "rows": live_rows,
        }
        live = report["live"]
        print(f"- 端到端压缩比: {live['avg_compression_ratio']}x")
        print(f"- 字面保留率: {live['literal_retention']:.0%} "
              f"语义保留率: {live['semantic_retention']:.0%} "
              f"结构完整: {live['structure_ok']}/{live['cases_run']}")

    _write_reports(Path(args.output_dir), report, "compact-eval-report")
    print(f"report: {Path(args.output_dir) / 'compact-eval-report.md'}")

    ok = off_report["gate_passed"] and off_report["all_facts_in_history"]
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
