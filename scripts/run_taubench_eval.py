#!/usr/bin/env python3
"""Tier3：τ-bench 压缩续跑双臂评测（M2 任务完成率保留 + M4 续跑可靠性）。

A 臂（基线）：τ-bench 官方 ToolCallingAgent，不压缩
B 臂（压缩）：CompactingToolCallingAgent，历史 token 超阈值时强制
             注入 KamaClaude Compactor（最近 3 轮原样保留）

用法：
  uv run python scripts/run_taubench_eval.py                  # 冒烟：3 任务 × 2 臂
  uv run python scripts/run_taubench_eval.py --num-tasks 20   # 完整子集
  uv run python scripts/run_taubench_eval.py --repeat 3       # pass^3（同任务 3 次全通率）

τ-bench 是可选依赖（故意不进 pyproject，避免 git 源进 uv.lock 破坏
uv sync --frozen / CI）。未安装时本脚本打印安装指引并退出，
不影响 make eval 与既有门禁。

口径声明：
  - 被测对象是"压缩器"而非模型：任务环境、评分（数据库状态校验）
    全部复用 τ-bench 官方实现，仅对照压缩引入的 Δ 指标
  - 用户模拟器存在方差：固定 --seed，pass^1 单次结果为观测值，
    门禁只作用于 Δpass^1（≥ -5pp）且压缩臂 ≥ 基线 80%
  - --max-cost 为总成本熔断（agent + user simulator 累计美元）
  - 报告写 output/evaluation/taubench-eval-report.{json,md}
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv  # noqa: E402

load_dotenv(ROOT / ".env")  # 需要 ANTHROPIC_API_KEY（litellm 与压缩器共用）

# 默认模型跟随项目配置（.env 的 KAMA_LLM_DEFAULT_MODEL），避免硬编码失效模型名
_DEFAULT_MODEL = os.environ.get("KAMA_LLM_DEFAULT_MODEL", "claude-sonnet-4-6")

try:  # τ-bench 可选依赖，缺失时优雅降级
    from tau_bench.agents.tool_calling_agent import ToolCallingAgent
    from tau_bench.envs import get_env
    from tau_bench.envs.user import UserStrategy
    _TAU_BENCH_IMPORT_ERROR = ""
except ImportError as _exc:  # pragma: no cover
    _TAU_BENCH_IMPORT_ERROR = str(_exc)

REPORT_DIR = ROOT / "output" / "evaluation"

# M2 门禁：Δpass^1 = pass^1(压缩) − pass^1(基线) 的允许下限
DELTA_PASS_GATE = -0.05
# M2 门禁：压缩臂绝对值不得低于基线的 80%
COMPACT_FLOOR_RATIO = 0.8

_INSTALL_HINT = """\
τ-bench 未安装（可选依赖，故意不进 pyproject，避免 git 源进 uv.lock
破坏 uv sync --frozen / CI）。请手动安装：

  uv pip install git+https://github.com/sierra-research/tau-bench.git

git 网络不可达时，可走 GitHub API tarball 本地安装：

  curl -sL https://api.github.com/repos/sierra-research/tau-bench/tarball/main \\
      -o /tmp/tau-bench.tar.gz
  mkdir -p /tmp/tau-bench-src && \\
      tar xzf /tmp/tau-bench.tar.gz -C /tmp/tau-bench-src --strip-components=1
  uv pip install /tmp/tau-bench-src
"""


def _tool_call_count(messages: list[dict[str, Any]]) -> int:
    """领域工具调用次数（role==tool 消息数），M4 重读率的代理分子。"""
    return sum(1 for m in messages if m.get("role") == "tool")


def _user_cost(info: dict[str, Any]) -> float:
    val = info.get("user_cost")
    try:
        return float(val)
    except (TypeError, ValueError):
        return 0.0


def _p95(values: list[float]) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(len(ordered) * 0.95))]


# 跑单任务单臂；基线用官方 Agent，压缩臂用注入 Compactor 的子类。
# 两臂共用同一个 env 实例（solve 内部 reset 会重建数据库状态）。
def run_arm(arm: str, task_index: int, args: argparse.Namespace,
            env: Any, compact_provider: Any) -> dict[str, Any]:
    if arm == "baseline":
        agent = ToolCallingAgent(env.tools_info, env.wiki,
                                 args.model, args.provider)
    else:
        from evaluation.e2e.taubench_adapter import CompactingToolCallingAgent
        agent = CompactingToolCallingAgent(
            env.tools_info, env.wiki, args.model, args.provider,
            compact_provider,
            compact_token_threshold=args.compact_threshold)
    started = time.monotonic()
    result = agent.solve(env, task_index=task_index,
                         max_num_steps=args.max_steps)
    record: dict[str, Any] = {
        "arm": arm,
        "task_index": task_index,
        "reward": result.reward,
        "passed": result.reward >= 0.999,  # 数据库状态校验全通过
        "agent_cost": round(result.total_cost, 6),
        "user_cost": round(_user_cost(result.info), 6),
        "wall_s": round(time.monotonic() - started, 1),
        "tool_calls": _tool_call_count(result.messages),
        "num_messages": len(result.messages),
    }
    if arm == "compact":
        record["compactions"] = agent.compactions
    return record


def _pass_rate(rows: list[dict[str, Any]]) -> float:
    return sum(r["passed"] for r in rows) / len(rows) if rows else 0.0


# pass^k：同一任务的全部 repeat 均通过才算通过（τ-bench 官方口径）
def _pass_k(rows: list[dict[str, Any]]) -> float:
    by_task: dict[int, list[bool]] = {}
    for r in rows:
        by_task.setdefault(r["task_index"], []).append(r["passed"])
    if not by_task:
        return 0.0
    return sum(all(v) for v in by_task.values()) / len(by_task)


# M2/M3/M4/M5 聚合：Δpass^1 与门禁、工具调用重读代理、压缩统计、成本
def aggregate(records: list[dict[str, Any]], args: argparse.Namespace,
              truncated: bool) -> dict[str, Any]:
    base = [r for r in records if r["arm"] == "baseline"]
    comp = [r for r in records if r["arm"] == "compact"]
    base_p1, comp_p1 = _pass_rate(base), _pass_rate(comp)
    delta = comp_p1 - base_p1
    base_tools = sum(r["tool_calls"] for r in base)
    comp_tools = sum(r["tool_calls"] for r in comp)
    compactions = [c for r in comp for c in r.get("compactions", [])]
    failed = [c for c in compactions if c.get("failed")]
    ok = [c for c in compactions if not c.get("failed")]
    token_ratios = [c["tokens_before"] / c["tokens_after"] for c in ok
                    if c.get("tokens_after")]
    return {
        "runs": {"baseline": len(base), "compact": len(comp),
                 "truncated_by_cost": truncated},
        "m2_task_completion": {
            "baseline_pass1": round(base_p1, 3),
            "compact_pass1": round(comp_p1, 3),
            "delta_pass1": round(delta, 3),
            "baseline_passk": round(_pass_k(base), 3),
            "compact_passk": round(_pass_k(comp), 3),
            "gate_delta_passed": delta >= DELTA_PASS_GATE,
            "gate_floor_passed": comp_p1 >= COMPACT_FLOOR_RATIO * base_p1,
        },
        "m4_resume_reliability": {
            "tool_calls_baseline": base_tools,
            "tool_calls_compact": comp_tools,
            # 重读率代理：压缩臂相对基线额外发起的工具查询倍率
            "reread_ratio": round(comp_tools / base_tools, 3) if base_tools else None,
        },
        "m3_m5_compaction": {
            "attempts": len(compactions),
            "failures": len(failed),
            "failure_rate_gate_passed": not failed,  # M5 硬门禁：失败率 = 0
            "latency_p95_s": round(_p95([c["latency_s"] for c in compactions]), 2),
            "avg_token_ratio": round(sum(token_ratios) / len(token_ratios), 2)
            if token_ratios else None,
        },
        "m3_cost": {
            "agent_usd": round(sum(r["agent_cost"] for r in records), 4),
            "user_sim_usd": round(sum(r["user_cost"] for r in records), 4),
            "total_usd": round(sum(r["agent_cost"] + r["user_cost"]
                                   for r in records), 4),
            "max_cost_usd": args.max_cost,
        },
    }


def write_reports(report: dict[str, Any]) -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    (REPORT_DIR / "taubench-eval-report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    m2 = report["metrics"]["m2_task_completion"]
    m4 = report["metrics"]["m4_resume_reliability"]
    m5 = report["metrics"]["m3_m5_compaction"]
    cost = report["metrics"]["m3_cost"]
    meta = report["meta"]
    md = f"""# τ-bench 压缩续跑双臂评测报告

生成时间：{meta['generated_at']}
环境：{meta['env']}（task_split={meta['task_split']}, seed={meta['seed']}）
模型：agent={meta['model']} user_sim={meta['user_model']} provider={meta['provider']}
任务：{meta['task_indices']}，repeat={meta['repeat']}，压缩阈值={meta['compact_threshold']} tokens

## M2 任务完成率保留

| 指标 | 基线 | 压缩 | 判定 |
|---|---|---|---|
| pass^1 | {m2['baseline_pass1']} | {m2['compact_pass1']} | Δ={m2['delta_pass1']} |
| pass^{meta['repeat']} | {m2['baseline_passk']} | {m2['compact_passk']} | — |

- Δpass^1 门禁（≥ {DELTA_PASS_GATE}）：{'通过' if m2['gate_delta_passed'] else '**未通过**'}
- 压缩臂 ≥ 基线 {COMPACT_FLOOR_RATIO:.0%} 门禁：{'通过' if m2['gate_floor_passed'] else '**未通过**'}

## M4 续跑可靠性

- 工具调用数：基线 {m4['tool_calls_baseline']} / 压缩 {m4['tool_calls_compact']}
  （重读率代理倍率 {m4['reread_ratio']}，>1.5 报警）

## M3/M5 压缩开销

- 压缩尝试 {m5['attempts']} 次，失败 {m5['failures']} 次
  （失败率=0 门禁：{'通过' if m5['failure_rate_gate_passed'] else '**未通过**'}）
- 压缩延迟 p95：{m5['latency_p95_s']}s；平均压缩比：{m5['avg_token_ratio']}x
- 成本：agent ${cost['agent_usd']} + user_sim ${cost['user_sim_usd']}
  = ${cost['total_usd']}（熔断线 ${cost['max_cost_usd']}）
  {'（已触发熔断，结果不完整）' if report['metrics']['runs']['truncated_by_cost'] else ''}

## 口径声明

- 本报告仅度量"压缩器"对任务完成率的影响：任务环境、评分（数据库状态
  校验）全部复用 τ-bench 官方实现，分数不与社区绝对榜直接对比
- 用户模拟器引入方差：单次结果为观测值，门禁只作用于 Δ 指标
- 逐任务明细见 taubench-eval-report.json
"""
    (REPORT_DIR / "taubench-eval-report.md").write_text(md, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="τ-bench 压缩续跑双臂评测（M2/M3/M4/M5）")
    parser.add_argument("--env", default="retail", choices=["retail", "airline"],
                        help="τ-bench 环境（默认 retail，airline 工具约束更严")
    parser.add_argument("--num-tasks", type=int, default=3,
                        help="任务数（冒烟 3；完整子集建议 20）")
    parser.add_argument("--task-indices", type=int, nargs="*", default=None,
                        help="显式任务索引（优先于 --num-tasks）")
    parser.add_argument("--repeat", type=int, default=1,
                        help="每任务重复次数（3 时可出 pass^3）")
    parser.add_argument("--seed", type=int, default=42,
                        help="随机种子（任务抽样与用户模拟方差控制）")
    parser.add_argument("--model", default=_DEFAULT_MODEL,
                        help="agent 模型（litellm 名称）")
    parser.add_argument("--user-model", default=None,
                        help="用户模拟器模型（默认同 --model）")
    parser.add_argument("--provider", default="anthropic",
                        help="litellm provider（agent 与 user simulator 共用）")
    parser.add_argument("--task-split", default="train",
                        help="τ-bench 任务分割（官方默认 train）")
    parser.add_argument("--compact-threshold", type=int, default=6000,
                        help="触发压缩的历史 token 阈值")
    parser.add_argument("--max-steps", type=int, default=30,
                        help="单任务最大步数（官方默认 30）")
    parser.add_argument("--max-cost", type=float, default=5.0,
                        help="总成本熔断（美元，含 user simulator）")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if _TAU_BENCH_IMPORT_ERROR:
        print(_INSTALL_HINT)
        print(f"(ImportError: {_TAU_BENCH_IMPORT_ERROR})")
        sys.exit(2)

    random.seed(args.seed)
    from kama_claude.core.llm.provider import AnthropicProvider
    compact_provider = AnthropicProvider(args.model)

    env = get_env(
        env_name=args.env,
        user_strategy=UserStrategy.LLM,
        user_model=args.user_model or args.model,
        task_split=args.task_split,
        user_provider=args.provider,
    )
    n_tasks = len(env.tasks)
    indices = args.task_indices or list(range(min(args.num_tasks, n_tasks)))
    bad = [i for i in indices if not 0 <= i < n_tasks]
    if bad:
        raise SystemExit(f"任务索引越界（共 {n_tasks} 个任务）：{bad}")

    print(f"τ-bench 双臂评测：env={args.env} tasks={indices} "
          f"repeat={args.repeat} model={args.model}")
    records: list[dict[str, Any]] = []
    total_cost = 0.0
    truncated = False
    for ti in indices:
        for rep in range(args.repeat):
            for arm in ("baseline", "compact"):
                if total_cost >= args.max_cost:
                    truncated = True
                    break
                print(f"[{arm}] task={ti} rep={rep} ...", flush=True)
                rec = run_arm(arm, ti, args, env, compact_provider)
                total_cost += rec["agent_cost"] + rec["user_cost"]
                records.append(rec)
                print(f"  reward={rec['reward']} passed={rec['passed']} "
                      f"tool_calls={rec['tool_calls']} "
                      f"wall={rec['wall_s']}s 累计=${total_cost:.4f}")
            if truncated:
                break
        if truncated:
            print(f"已触发成本熔断 ${args.max_cost}，提前结束")
            break

    if not records:
        raise SystemExit("没有可运行的任务记录（可能成本熔断线为 0）")
    report = {
        "meta": {
            "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
            "env": args.env, "task_split": args.task_split,
            "seed": args.seed, "task_indices": indices,
            "repeat": args.repeat, "model": args.model,
            "user_model": args.user_model or args.model,
            "provider": args.provider,
            "compact_threshold": args.compact_threshold,
        },
        "metrics": aggregate(records, args, truncated),
        "records": records,
    }
    write_reports(report)
    m2 = report["metrics"]["m2_task_completion"]
    m5 = report["metrics"]["m3_m5_compaction"]
    print("\n==== 结果 ====")
    print(f"pass^1: baseline={m2['baseline_pass1']} compact={m2['compact_pass1']} "
          f"Δ={m2['delta_pass1']}")
    print(f"Δpass^1 门禁(≥{DELTA_PASS_GATE}): "
          f"{'PASS' if m2['gate_delta_passed'] else 'FAIL'}")
    print(f"压缩失败率=0 门禁: {'PASS' if m5['failure_rate_gate_passed'] else 'FAIL'}")
    print(f"报告：{REPORT_DIR}/taubench-eval-report.{{json,md}}")


if __name__ == "__main__":
    main()
