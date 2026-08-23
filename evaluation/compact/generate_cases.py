"""Tier1 数据集生成器：构造带预埋事实（planted facts）的合成多轮对话。

设计要点：
- 每个 case 预埋 6 类事实：字面类（path / id / error，可字符串精确校验）
  与语义类（constraint / decision / todo，需 LLM-as-judge 判定）；
- 所有事实的埋点轮次 <= total_real_rounds - 4，保证落在 history 区
  （最近 TAIL_ROUNDS=3 轮之外），必须经过摘要才能幸存——测的是压缩质量
  而不是"恰好在 tail 里"；
- 深度分层：early(<34%) / mid(34-67%) / deep(>67%)，用于绘制
  保留率随埋点深度的衰减曲线；
- 完全确定性：同一份代码生成的数据集逐字节一致（无随机源）。

用法：
  uv run python -m evaluation.compact.generate_cases            # 写入默认路径
  uv run python -m evaluation.compact.generate_cases --out x.jsonl
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from kama_claude.core.compact.compactor import TAIL_ROUNDS

SCHEMA_VERSION = 1
DEFAULT_OUT = Path(__file__).resolve().parent / "cases.jsonl"
SEEDS = (1, 2)
SIZES = (10, 14, 18)  # 真实用户轮数（最小 10 轮保证 6 个埋点位都在 history 区）

# 四个场景模板：提供目标话术与六类事实的取值（确定性，随场景变化）
SCENARIOS: dict[str, dict[str, Any]] = {
    "session-persistence": {
        "goal": "Refactor the session persistence layer to survive daemon restarts.",
        "path": "src/app/session_store.py",
        "id": "sess-4f8a1c02",
        "error": "ConnectionRefusedError: [Errno 111] daemon port 7437",
        "constraint": (
            "All new functions must include type hints and a one-line docstring."
        ),
        "decision": "Use SQLite instead of JSON files for the session cache.",
        "todo": "Record this in the TODO list: add retry with exponential backoff "
        "to the socket client.",
        "filler_cmd": "pytest tests/session -q",
    },
    "retry-client": {
        "goal": "Make the HTTP retry client respect server Retry-After headers.",
        "path": "src/app/http_client.py",
        "id": "sess-7b2e9d31",
        "error": "TimeoutError: upstream gateway exceeded 30s limit",
        "constraint": "Never log full authentication tokens; mask them as ***.",
        "decision": "Cap retry attempts at 3 with exponential backoff from 2s.",
        "todo": "Record this in the TODO list: expose per-request timeout "
        "overrides in the client config.",
        "filler_cmd": "curl -sS https://api.internal/healthz",
    },
    "event-bridge": {
        "goal": "Bridge child agent events into the parent event bus.",
        "path": "src/app/event_bridge.py",
        "id": "sess-9c0d4e77",
        "error": "RuntimeError: EventBus subscriber list modified during publish",
        "constraint": "Event handlers must never block; enqueue and return.",
        "decision": "Forward only topic 'progress' events to reduce parent bus load.",
        "todo": "Record this in the TODO list: unsubscribe the bridge when the "
        "child run finishes.",
        "filler_cmd": "python -m app.event_bridge --selftest",
    },
    "cost-tracker": {
        "goal": "Track LLM token cost per run and persist it to cost.jsonl.",
        "path": "src/app/cost_tracker.py",
        "id": "sess-2e6f0ab9",
        "error": "ValueError: usage.output_tokens missing from provider response",
        "constraint": "All money values must be rounded to 6 decimal places.",
        "decision": "Price cache-read tokens at 10% of the normal input rate.",
        "todo": "Record this in the TODO list: add a CLI flag to print the "
        "cumulative session cost.",
        "filler_cmd": "uv run pytest tests/cost -q",
    },
}

# 六类事实：字面类走字符串精确匹配，语义类交给 LLM-as-judge
FACT_TYPES = ("constraint", "critical_id", "error", "path", "decision", "todo")
LITERAL_TYPES = frozenset({"critical_id", "error", "path"})


# 事实埋点轮次分配：六类事实轮次互异（避免同轮覆盖），全部落在 history 区；
# 大尺寸下 early/mid/deep 三个深度桶各有覆盖（total>=10 时安全轮次 >= 6）
def _fact_rounds(total: int) -> dict[str, int]:
    safe_last = total - TAIL_ROUNDS - 1  # 再留一轮缓冲，确保必在 history（>=6）
    return {
        "constraint": 1,
        "critical_id": 2,
        "error": 3,
        "path": 4,
        "decision": 5,
        "todo": safe_last,
    }


# 深度桶口径：relative = round / total_rounds；<0.34 early，<0.67 mid，其余 deep
def depth_bucket(rel: float) -> str:
    if rel < 0.34:
        return "early"
    if rel < 0.67:
        return "mid"
    return "deep"


def _fact_text(ftype: str, sc: dict[str, Any]) -> str:
    return {
        "constraint": sc["constraint"],
        "critical_id": sc["id"],
        "error": sc["error"],
        "path": sc["path"],
        "decision": sc["decision"],
        "todo": sc["todo"],
    }[ftype]


# 构造单个 case：供批量生成（generate_cases）与 Tier2 任务（复用场景模板）共用
def make_case(scenario: str, total: int, seed: int, idx: int,
              case_id: str | None = None) -> dict[str, Any]:
    sc = SCENARIOS[scenario]
    rounds = _fact_rounds(total)
    facts = [
        {
            "key": ftype,
            "text": _fact_text(ftype, sc),
            "type": ftype,
            "check": "literal" if ftype in LITERAL_TYPES else "semantic",
            "round": rounds[ftype],
            "depth": depth_bucket(rounds[ftype] / total),
        }
        for ftype in FACT_TYPES
    ]
    return {
        "schema_version": SCHEMA_VERSION,
        "id": case_id or f"cc-{scenario}-r{total}-s{seed}",
        "idx": idx,
        "seed": seed,
        "scenario": scenario,
        "total_real_rounds": total,
        "planted_facts": facts,
    }


def generate_cases() -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []
    idx = 0
    for scenario in SCENARIOS:
        for total in SIZES:
            for seed in SEEDS:
                idx += 1
                cases.append(make_case(scenario, total, seed, idx))
    return cases


# 由 case 构造完整消息列表：每个真实轮次 = user + assistant(tool_use) + tool_result；
# 字面事实分别埋进 tool_use input（id）、错误 tool_result（error）、
# read_file 路径（path）；语义事实埋进用户消息文本
def build_case_messages(case: dict[str, Any]) -> list[dict[str, Any]]:
    sc = SCENARIOS[case["scenario"]]
    facts_by_round = {f["round"]: f for f in case["planted_facts"]}
    messages: list[dict[str, Any]] = []
    filler_bash = "".join(
        f"line {i:04d}: ok status=200 payload=" + "x" * 60 + "\n" for i in range(300)
    )
    filler_code = "".join(f"def fn_{i}():\n    return {i}\n" for i in range(400))

    for r in range(1, case["total_real_rounds"] + 1):
        fact = facts_by_round.get(r)
        tu_id = f"{case['id']}-tu{r}"

        user_text = sc["goal"] if r == 1 else f"Step {r}: continue with the plan."
        if fact is not None and fact["check"] == "semantic":
            user_text += f" Important: {fact['text']}"
        messages.append({"role": "user", "content": user_text})

        if fact is not None and fact["type"] == "critical_id":
            tool_name, tool_input = "bash", {"command": f"kama attach {sc['id']}"}
        elif fact is not None and fact["type"] == "path":
            tool_name, tool_input = "read_file", {"path": sc["path"]}
        else:
            tool_name, tool_input = "bash", {"command": sc["filler_cmd"]}
        messages.append(
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": f"Working on step {r}."},
                    {"type": "tool_use", "id": tu_id, "name": tool_name,
                     "input": tool_input},
                ],
            }
        )

        if fact is not None and fact["type"] == "error":
            result: dict[str, Any] = {
                "type": "tool_result",
                "tool_use_id": tu_id,
                "is_error": True,
                "content": (
                    f"Traceback (most recent call last):\n  {sc['error']}\n"
                    "command failed with exit code 1"
                ),
            }
        elif tool_name == "read_file":
            result = {
                "type": "tool_result",
                "tool_use_id": tu_id,
                "content": f"# {sc['path']}\n{filler_code}",
            }
        else:
            result = {"type": "tool_result", "tool_use_id": tu_id,
                      "content": filler_bash}
        messages.append({"role": "user", "content": [result]})

    return messages


def write_cases(path: Path, cases: list[dict[str, Any]]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = "".join(json.dumps(c, ensure_ascii=False, sort_keys=True) + "\n"
                      for c in cases)
    path.write_text(payload, encoding="utf-8")
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def load_cases(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()]


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description="generate Tier1 compact eval cases")
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    args = parser.parse_args()

    cases = generate_cases()
    digest = write_cases(Path(args.out), cases)
    print(f"cases: {len(cases)} -> {args.out}")
    print(f"sha256: {digest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
