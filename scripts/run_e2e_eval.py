#!/usr/bin/env python3
"""Tier2：端到端任务评测 harness——压缩 → 续跑生成 → 确定性校验闭环。

每个任务验证一件事：早轮预埋的关键信息（约束 / ID / 报错 / 待办）在
compact 之后，仅凭摘要 + 最近 3 轮，续跑的模型还能不能正确使用它。

离线模式（默认，零 API 成本）：
  - 任务集 schema 校验、事实埋点位置校验（必须落在 history 区）
  - 管道连通性由单测覆盖（tests/unit/test_e2e_eval.py 用脚本化 provider
    驱动完整 AgentLoop + Compactor）

在线模式（--live，需要 ANTHROPIC_API_KEY，每任务 2~3 次调用）：
  ① 构造预埋事实的合成历史 → ② 真实 compact → ③ 仅凭摘要续跑提问
  → ④ 确定性校验（contains / regex / LLM-as-judge）

用法：
  uv run python scripts/run_e2e_eval.py           # 离线校验
  uv run python scripts/run_e2e_eval.py --live    # 真实续跑评测
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
    SCENARIOS,
    build_case_messages,
    make_case,
)

from kama_claude.core.compact.compactor import (  # noqa: E402
    TAIL_ROUNDS,
    _find_tail_boundary,
)
from kama_claude.core.llm.tokens import count_messages_tokens  # noqa: E402

TASKS_PATH = ROOT / "evaluation" / "e2e" / "tasks.jsonl"

_CONTINUATION_TEMPLATE = """\
Another engineer compressed the earlier conversation into the summary above. \
Continue from that summary alone — do not assume anything that is not in it.
Original goal: {goal}

Now answer: {question}
Be direct and specific."""

_JUDGE_SYSTEM = "You are a strict audit assistant. Answer YES or NO only on the first line."


def load_tasks(path: Path = TASKS_PATH) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()]


# 任务 → 合成 case：复用 Tier1 的场景模板与埋点规则
def task_to_case(task: dict[str, Any]) -> dict[str, Any]:
    return make_case(task["scenario"], task["total_real_rounds"], seed=1,
                     idx=0, case_id=task["id"])


def focus_fact(task: dict[str, Any]) -> dict[str, Any]:
    case = task_to_case(task)
    for f in case["planted_facts"]:
        if f["key"] == task["focus_fact"]:
            return f
    raise ValueError(f"task {task['id']}: focus_fact {task['focus_fact']} not found")


# 离线校验：schema 完整性 + 埋点必须落在 history 区（否则评测无效）
def validate_task_offline(task: dict[str, Any]) -> dict[str, Any]:
    case = task_to_case(task)
    messages = build_case_messages(case)
    split = _find_tail_boundary(messages, tail_rounds=TAIL_ROUNDS)
    fact = focus_fact(task)

    # 第 r 轮真实用户消息 = 消息列表索引 3*(r-1)（每轮固定 3 条消息）
    planted_idx = 3 * (fact["round"] - 1)
    checks_ok = len(task.get("checks", [])) > 0 and all(
        c.get("kind") in {"contains", "regex", "judge"} for c in task["checks"])
    return {
        "task_id": task["id"],
        "schema_ok": checks_ok and task.get("scenario") in SCENARIOS,
        "fact_in_history": planted_idx < split,
        "fact": fact,
    }


# 单个 check 的执行：contains / regex 为确定性校验，judge 走 LLM-as-judge
async def _run_check(
    check: dict[str, Any], answer: str, fact_text: str, judge: Any
) -> dict[str, Any]:
    kind = check["kind"]
    if kind == "contains":
        value = check["value"].format(fact=fact_text)
        return {"kind": kind, "passed": value.lower() in answer.lower(),
                "detail": value}
    if kind == "regex":
        pattern = check["pattern"].format(fact=re.escape(fact_text))
        hit = re.search(pattern, answer, re.DOTALL) is not None
        return {"kind": kind, "passed": hit, "detail": check["pattern"]}
    # judge：把问题与模型答案一起交给裁判模型
    from kama_claude.core.events.bus import EventBus

    question = check["judge_question"].format(fact=fact_text)
    response = await judge.chat(
        messages=[{"role": "user",
                   "content": f"{question}\n\nANSWER:\n{answer}"}],
        tool_schemas=[],
        bus=EventBus(),
        run_id="e2e-judge",
        step=0,
        system=_JUDGE_SYSTEM,
    )
    first = response.text.strip().splitlines()[0].upper() if response.text else ""
    return {"kind": kind, "passed": "YES" in first,
            "detail": response.text.strip()[:120]}


# 单个任务的在线评测：压缩 → 续跑 → 校验
async def evaluate_task_live(
    task: dict[str, Any], provider: Any, judge: Any
) -> dict[str, Any]:
    from kama_claude.core.compact.compactor import Compactor
    from kama_claude.core.events.bus import EventBus

    case = task_to_case(task)
    fact = focus_fact(task)
    messages = build_case_messages(case)

    compactor = Compactor(EventBus(), Path("."), task["id"])
    result = await compactor.compact_messages(messages, provider)
    if result is None:
        return {"task_id": task["id"], "error": "compaction skipped"}

    fact_in_summary = fact["text"].lower() in result.summary_text.lower()
    continuation = [
        {"role": "user", "content": result.summary_text},
        {"role": "assistant",
         "content": "Understood, I'll continue from this summary."},
        {"role": "user", "content": _CONTINUATION_TEMPLATE.format(
            goal=SCENARIOS[task["scenario"]]["goal"],
            question=task["continuation_prompt"])},
    ]
    response = await provider.chat(
        messages=continuation, tool_schemas=[], bus=EventBus(),
        run_id=f"e2e-{task['id']}", step=0,
        system="You are continuing a task from a handoff summary.",
    )
    answer = response.text or ""

    check_rows = [await _run_check(c, answer, fact["text"], judge)
                  for c in task["checks"]]
    return {
        "task_id": task["id"],
        "title": task["title"],
        "focus_fact": fact["text"],
        "fact_in_summary": fact_in_summary,
        "summary_tokens": result.summary_tokens,
        "compression_ratio": round(
            result.original_token_estimate
            / max(result.summary_tokens
                  + count_messages_tokens(result.tail_messages or []), 1), 1),
        "checks": check_rows,
        "passed": all(c["passed"] for c in check_rows),
        "answer_preview": answer[:300],
    }


def _write_reports(out_dir: Path, report: dict[str, Any]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "e2e-eval-report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8")
    lines = ["# Tier2 端到端任务评测报告", ""]
    lines += [f"- 模式: {report['mode']}", f"- 生成时间: {report['generated_at']}",
              ""]
    if "offline" in report:
        lines += ["## 离线校验", "",
                  "| 任务 | schema | 埋点在 history 区 |", "|---|---|---|"]
        for r in report["offline"]:
            lines.append(f"| {r['task_id']} | {'✓' if r['schema_ok'] else '✗'} "
                         f"| {'✓' if r['fact_in_history'] else '✗'} |")
    if "live" in report:
        live = report["live"]
        lines += ["", f"## 在线结果（model={live['model']}）", "",
                  f"**任务通过率: {live['pass_rate']:.0%}"
                  f"（{live['passed']}/{live['total']}）**", "",
                  "| 任务 | 焦点事实 | 事实进摘要 | 校验 | 结果 |",
                  "|---|---|---|---|---|"]
        for r in live["rows"]:
            if r.get("error"):
                lines.append(f"| {r['task_id']} | — | — | — | ERROR: {r['error']} |")
                continue
            checks = " ".join("✓" if c["passed"] else "✗" for c in r["checks"])
            lines.append(
                f"| {r['task_id']} | {r['focus_fact'][:40]}… "
                f"| {'✓' if r['fact_in_summary'] else '✗'} | {checks} "
                f"| {'PASS' if r['passed'] else 'FAIL'} |")
    lines += [
        "",
        "> 口径声明：历史对话为受控合成数据（事实埋点位置可控），续跑生成为真实",
        "> LLM 调用。本评测验证\"压缩后信息仍可用于续跑\"，不代表生产会话效果；",
        "> 全工具循环下的重读率指标需真实工具环境，列入后续扩展。",
        "",
    ]
    (out_dir / "e2e-eval-report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Tier2 e2e compact eval")
    parser.add_argument("--tasks", default=str(TASKS_PATH))
    parser.add_argument("--output-dir", default=str(ROOT / "output" / "evaluation"))
    parser.add_argument("--live", action="store_true")
    parser.add_argument("--model", default=_DEFAULT_MODEL)
    parser.add_argument("--judge-model", default=None)
    args = parser.parse_args()

    tasks = load_tasks(Path(args.tasks))
    offline_rows = [validate_task_offline(t) for t in tasks]
    report: dict[str, Any] = {
        "mode": "offline+live" if args.live else "offline",
        "generated_at": datetime.now(UTC).isoformat(),
        "tasks": len(tasks),
        "offline": [{k: v for k, v in r.items() if k != "fact"}
                    for r in offline_rows],
    }

    print(f"# Tier2 E2E Eval ({len(tasks)} tasks)")
    for r in offline_rows:
        print(f"- {r['task_id']}: schema={'✓' if r['schema_ok'] else '✗'} "
              f"fact_in_history={'✓' if r['fact_in_history'] else '✗'}")

    if args.live:
        from kama_claude.core.llm.provider import AnthropicProvider

        provider = AnthropicProvider(args.model)
        judge = provider if not args.judge_model \
            else AnthropicProvider(args.judge_model)
        rows = []
        for i, task in enumerate(tasks):
            row = asyncio.run(evaluate_task_live(task, provider, judge))
            rows.append(row)
            status = "ERROR" if row.get("error") \
                else ("PASS" if row["passed"] else "FAIL")
            print(f"  [{i + 1}/{len(tasks)}] {task['id']}: {status}")
        passed = sum(1 for r in rows if r.get("passed"))
        report["live"] = {
            "model": args.model,
            "total": len(rows),
            "passed": passed,
            "pass_rate": passed / len(rows) if rows else 0.0,
            "rows": rows,
        }
        print(f"- 任务通过率: {report['live']['pass_rate']:.0%}")

    _write_reports(Path(args.output_dir), report)
    print(f"report: {Path(args.output_dir) / 'e2e-eval-report.md'}")

    ok = all(r["schema_ok"] and r["fact_in_history"] for r in offline_rows)
    if args.live and "live" in report:
        ok = ok and report["live"]["pass_rate"] == 1.0
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
