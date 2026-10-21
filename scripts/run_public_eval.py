#!/usr/bin/env python3
"""Tier0：公开数据集（LoCoMo）压缩保留率评测 harness。

离线模式（默认，零 API 成本，可进 CI）：
  - 子集完整性校验：哈希与 manifest 一致、角色严格交替、
    所有题目证据落在 history 区（与合成集埋点校验同口径）

在线模式（--live，需要 ANTHROPIC_API_KEY）：
  - 每段对话：真实 compact → 仅凭摘要+最近 3 轮回答金标 QA
  - 确定性 EM/F1 为主，裁判仅作非单跳题的辅助口径
  - --with-baseline：同一批题在完整未压缩历史上作答（上界对照，
    保留率 = 压缩后正确数 / 未压缩正确数，消除题目难度噪声）
  - 同时输出 M5：压缩失败率（硬门禁 = 0）与压缩延迟

裁判校准（--calibrate，M6）：
  - SummEval 校准子集可用时：裁判判定与人工事实一致性标签的 Cohen's kappa
  - 不可用时降级为 LoCoMo 自校准：裁判判定与确定性 EM 判定的一致性

用法：
  uv run python scripts/run_public_eval.py                 # 离线校验
  uv run python scripts/run_public_eval.py --live
  uv run python scripts/run_public_eval.py --live --with-baseline
  uv run python scripts/run_public_eval.py --calibrate
"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import sys
import time
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

from evaluation.public.locomo_adapter import (  # noqa: E402
    cohens_kappa,
    deterministic_pass,
    evidence_in_history,
)

from kama_claude.core.llm.tokens import TOKENIZER_NAME, count_messages_tokens  # noqa: E402

DATA_DIR = ROOT / "evaluation" / "public" / "data"
SUBSETS_DIR = DATA_DIR / "subsets"
MANIFEST_PATH = DATA_DIR / "manifest.json"

_QA_PROMPT = """\
Answer the question using ONLY information from the conversation so far. \
Be brief — one short phrase or sentence. If the information is not \
available, say "I don't know".

QUESTION: {question}
Answer:"""

_JUDGE_QA_PROMPT = """\
You are auditing whether an answer is correct.

QUESTION: {question}
GOLD ANSWER: {gold}
MODEL ANSWER: {answer}

Is the model answer semantically correct (same fact, paraphrase allowed)? \
Answer YES or NO on the first line, then one sentence of justification."""

_JUDGE_CONSISTENCY_PROMPT = """\
You are auditing a summary for factual consistency with its source.

SOURCE DOCUMENT:
{source}

SUMMARY:
{summary}

Is the summary factually consistent with the source (no fabricated, \
contradicted or altered facts)? Answer YES or NO on the first line, \
then one sentence of justification."""

# 裁判可信度门禁：与人工/确定性口径的 Cohen's kappa 下限
KAPPA_GATE = 0.6


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()]


# 加载子集并做哈希校验；缺文件时给出准备指引
def load_subsets() -> tuple[list[dict[str, Any]], list[dict[str, Any]],
                            dict[str, Any]]:
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8")) \
        if MANIFEST_PATH.exists() else {}
    convs_path = SUBSETS_DIR / "locomo_convs.jsonl"
    qa_path = SUBSETS_DIR / "locomo_qa.jsonl"
    if not convs_path.exists() or not qa_path.exists():
        raise SystemExit(
            f"子集缺失（{SUBSETS_DIR}）——请先运行 "
            "uv run python -m evaluation.public.prepare_datasets")
    locomo_meta = manifest.get("locomo", {})
    if locomo_meta.get("subset_convs_sha256") \
            and _sha256_file(convs_path) != locomo_meta["subset_convs_sha256"]:
        raise SystemExit("locomo_convs.jsonl 哈希与 manifest 不一致，请重新准备数据集")
    if locomo_meta.get("subset_qa_sha256") \
            and _sha256_file(qa_path) != locomo_meta["subset_qa_sha256"]:
        raise SystemExit("locomo_qa.jsonl 哈希与 manifest 不一致，请重新准备数据集")
    return _load_jsonl(convs_path), _load_jsonl(qa_path), manifest


# 离线校验：角色交替、首条为 user、证据全部落在 history 区
def validate_offline(convs: list[dict[str, Any]],
                     qa: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    conv_by_id = {c["conv_id"]: c for c in convs}
    for c in convs:
        messages = c["messages"]
        alternating = all(messages[i]["role"] != messages[i + 1]["role"]
                          for i in range(len(messages) - 1))
        rows.append({
            "scope": c["conv_id"],
            "check": "role_alternation",
            "passed": bool(messages) and messages[0]["role"] == "user" and alternating,
            "detail": f"{len(messages)} messages, "
                      f"{count_messages_tokens(messages)} tokens",
        })
    qa_rows = []
    for q in qa:
        c = conv_by_id.get(q["conv_id"])
        ok = bool(c) and evidence_in_history(c["messages"], c["dia_index"],
                                             q["evidence"])
        qa_rows.append(ok)
    rows.append({
        "scope": "qa",
        "check": "evidence_in_history",
        "passed": all(qa_rows) if qa_rows else False,
        "detail": f"{sum(qa_rows)}/{len(qa_rows)} questions in history region",
    })
    return rows


async def _judge_first_line(judge: Any, prompt: str, run_id: str) -> bool:
    from kama_claude.core.events.bus import EventBus

    response = await judge.chat(
        messages=[{"role": "user", "content": prompt}],
        tool_schemas=[],
        bus=EventBus(),
        run_id=run_id,
        step=0,
        system="You are a strict audit assistant.",
    )
    first = response.text.strip().splitlines()[0].upper() if response.text else ""
    return "YES" in first


# 向对话追加提问；末条若为 user 需先补 assistant 占位（保持严格交替）
def _append_question(messages: list[dict[str, Any]], question: str,
                     ) -> list[dict[str, Any]]:
    out = list(messages)
    if out and out[-1]["role"] == "user":
        out.append({"role": "assistant", "content": "Okay."})
    out.append({"role": "user", "content": _QA_PROMPT.format(question=question)})
    return out


# 单题评分：确定性优先；非单跳题确定性失败时由裁判辅助判定
async def _score_answer(provider: Any, judge: Any, context_messages: list[dict[str, Any]],
                        q: dict[str, Any], arm: str) -> dict[str, Any]:
    from kama_claude.core.events.bus import EventBus

    response = await provider.chat(
        messages=_append_question(context_messages, q["question"]),
        tool_schemas=[],
        bus=EventBus(),
        run_id=f"public-{arm}-{q['id']}",
        step=0,
        system="Answer from the given conversation only.",
    )
    answer = response.text or ""
    det = deterministic_pass(answer, q["answer"])
    method = "deterministic"
    passed = det
    if not det and q["category"] != 1:
        passed = await _judge_first_line(
            judge, _JUDGE_QA_PROMPT.format(question=q["question"], gold=q["answer"],
                                           answer=answer), f"judge-{arm}-{q['id']}")
        method = "judge"
    return {"id": q["id"], "category": q["category"],
            "category_name": q["category_name"], "passed": passed,
            "method": method, "answer_preview": answer[:160]}


# 单段对话的在线评测：真实压缩 → 压缩臂 +（可选）未压缩基线臂
async def evaluate_conv_live(conv: dict[str, Any], qa: list[dict[str, Any]],
                             provider: Any, judge: Any,
                             with_baseline: bool) -> dict[str, Any]:
    from kama_claude.core.compact.compactor import Compactor
    from kama_claude.core.events.bus import EventBus

    messages = conv["messages"]
    conv_qa = [q for q in qa if q["conv_id"] == conv["conv_id"]]
    compactor = Compactor(EventBus(), Path("."), conv["conv_id"])

    t0 = time.monotonic()
    result = await compactor.compact_messages(messages, provider)
    compact_latency = time.monotonic() - t0
    if result is None:
        return {"conv_id": conv["conv_id"], "error": "compaction failed",
                "compact_latency_s": round(compact_latency, 1)}

    compact_ctx = [
        {"role": "user", "content": result.summary_text},
        {"role": "assistant", "content": "Understood, I'll continue from this summary."},
    ]
    compact_rows = [await _score_answer(provider, judge, compact_ctx, q, "compact")
                    for q in conv_qa]
    baseline_rows = []
    if with_baseline:
        baseline_rows = [await _score_answer(provider, judge, messages, q, "baseline")
                         for q in conv_qa]

    tail_tokens = count_messages_tokens(result.tail_messages or [])
    return {
        "conv_id": conv["conv_id"],
        "before_tokens": result.original_token_estimate,
        "after_tokens": result.summary_tokens + tail_tokens,
        "compression_ratio": round(result.original_token_estimate
                                   / max(result.summary_tokens + tail_tokens, 1), 1),
        "compact_latency_s": round(compact_latency, 1),
        "questions": len(conv_qa),
        "compact": compact_rows,
        "baseline": baseline_rows,
    }


def _accuracy(rows: list[dict[str, Any]]) -> dict[str, float]:
    by_cat: dict[str, list[bool]] = {}
    for r in rows:
        by_cat.setdefault(r["category_name"], []).append(r["passed"])
    out = {cat: round(sum(v) / len(v), 3) for cat, v in sorted(by_cat.items())}
    all_pass = [r["passed"] for r in rows]
    out["overall"] = round(sum(all_pass) / len(all_pass), 3) if all_pass else 0.0
    return out


# SummEval 裁判校准：裁判判定与人工一致性标签的 kappa
async def calibrate_summeval(judge: Any, limit: int) -> dict[str, Any]:
    calib_path = SUBSETS_DIR / "summeval_calibration.jsonl"
    if not calib_path.exists():
        return {"available": False,
                "reason": "SummEval 校准子集不可用，请用 --calibrate 的 "
                          "LoCoMo 自校准或手动放置数据后重跑准备脚本"}
    items = _load_jsonl(calib_path)[:limit or None]
    judge_verdicts: list[bool] = []
    for i, item in enumerate(items):
        verdict = await _judge_first_line(
            judge, _JUDGE_CONSISTENCY_PROMPT.format(source=item["source"],
                                                    summary=item["summary"]),
            f"calib-{item['id']}")
        judge_verdicts.append(verdict)
        print(f"  calib [{i + 1}/{len(items)}] {item['id']}: "
              f"judge={verdict} human={item['human_consistent']}")
    human = [bool(it["human_consistent"]) for it in items]
    kappa = cohens_kappa(judge_verdicts, human)
    agree = sum(1 for a, b in zip(judge_verdicts, human) if a == b) / len(human)
    return {"available": True, "dataset": "summeval", "n": len(items),
            "agreement": round(agree, 3), "kappa": round(kappa, 3),
            "gate_passed": kappa >= KAPPA_GATE}


# LoCoMo 自校准（无 SummEval 时的降级）：裁判判定与确定性判定的 kappa
async def calibrate_locomo(provider: Any, judge: Any,
                           convs: list[dict[str, Any]],
                           qa: list[dict[str, Any]], limit: int,
                           ) -> dict[str, Any]:
    from kama_claude.core.compact.compactor import Compactor
    from kama_claude.core.events.bus import EventBus

    conv_by_id = {c["conv_id"]: c for c in convs}
    singles = [q for q in qa if q["category"] == 1][:limit]
    if not singles:
        return {"available": False, "reason": "无单跳题可用于自校准"}
    compact_ctx: dict[str, list[dict[str, Any]]] = {}
    det_verdicts: list[bool] = []
    judge_verdicts: list[bool] = []
    for q in singles:
        cid = q["conv_id"]
        if cid not in compact_ctx:
            result = await Compactor(EventBus(), Path("."), cid).compact_messages(
                conv_by_id[cid]["messages"], provider)
            if result is None:
                continue
            compact_ctx[cid] = [
                {"role": "user", "content": result.summary_text},
                {"role": "assistant",
                 "content": "Understood, I'll continue from this summary."},
            ]
        # 直接取模型答案，分别用确定性口径与裁判口径各判一次（不走 fallback，
        # 保证两个判定基于同一完整答案，避免截断预览影响裁判）
        response = await provider.chat(
            messages=_append_question(compact_ctx[cid], q["question"]),
            tool_schemas=[], bus=EventBus(),
            run_id=f"calib-{q['id']}", step=0,
            system="Answer from the given conversation only.",
        )
        answer = response.text or ""
        det_verdicts.append(deterministic_pass(answer, q["answer"]))
        judge_verdicts.append(await _judge_first_line(
            judge, _JUDGE_QA_PROMPT.format(question=q["question"], gold=q["answer"],
                                           answer=answer),
            f"calibjudge-{q['id']}"))
    kappa = cohens_kappa(judge_verdicts, det_verdicts)
    agree = sum(1 for a, b in zip(judge_verdicts, det_verdicts) if a == b) \
        / len(det_verdicts) if det_verdicts else 0.0
    return {"available": True, "dataset": "locomo-self", "n": len(det_verdicts),
            "agreement": round(agree, 3), "kappa": round(kappa, 3),
            "gate_passed": kappa >= KAPPA_GATE}


def _write_reports(out_dir: Path, report: dict[str, Any]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "public-eval-report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8")
    lines = ["# Tier0 公开数据集（LoCoMo）评测报告", "",
             f"- 模式: {report['mode']}",
             f"- tokenizer: {report['tokenizer']}",
             f"- 数据集: LoCoMo 子集（{report['convs']} 段对话 / "
             f"{report['questions']} 道题，qa sha256 "
             f"`{report['qa_sha256'][:12]}…`）", "",
             "## 离线校验", "",
             "| 对象 | 检查 | 结果 |", "|---|---|---|"]
    for r in report["offline"]:
        lines.append(f"| {r['scope']} | {r['check']} | "
                     f"{'✓' if r['passed'] else '✗'} {r['detail']} |")
    if "live" in report:
        live = report["live"]
        lines += ["", f"## 在线结果（model={live['model']}）", ""]
        if live.get("compact_failed"):
            lines.append(f"**压缩失败 {live['compact_failed']} 段（门禁 = 0，FAIL）**")
        lines += [
            f"- 压缩失败率: {live['compact_failed']}/{live['convs_run']}（门禁 = 0："
            f"{'PASS' if live['compact_failed'] == 0 else 'FAIL'}）",
            f"- 压缩延迟: 均值 {live['avg_latency_s']}s / 最大 {live['max_latency_s']}s",
            f"- 端到端压缩比: {live['avg_compression_ratio']}x",
            "", "### M1 信息保留质量（按类别准确率）", "",
            "| 类别 | 压缩后 | 未压缩基线 |", "|---|---:|---:|",
        ]
        for cat in sorted(set(list(live["compact_acc"].keys())
                              + list(live.get("baseline_acc", {}).keys()))):
            b = live.get("baseline_acc", {}).get(cat)
            lines.append(f"| {cat} | {live['compact_acc'].get(cat, 0):.0%} "
                         f"| {f'{b:.0%}' if b is not None else '—'} |")
        if "retention" in live:
            lines += ["", f"- M1 保留率（压缩后正确 / 基线正确）: "
                      f"{live['retention']:.0%}"]
    if "calibration" in report:
        cal = report["calibration"]
        lines += ["", "## M6 裁判可信度", ""]
        if not cal.get("available"):
            lines.append(f"- 不可用: {cal['reason']}")
        else:
            lines += [
                f"- 校准集: {cal['dataset']}（n={cal['n']}）",
                f"- 一致率: {cal['agreement']:.0%}",
                f"- Cohen's kappa: {cal['kappa']}（门禁 >= {KAPPA_GATE}："
                f"{'PASS' if cal['gate_passed'] else 'FAIL'}）",
                "- kappa 不达标时语义类指标不可引用，仅引用确定性口径",
            ]
    lines += ["", "> 口径声明：LoCoMo 为生活对话域公开数据，用于外部有效性观测；",
              "> 与合成集双轨运行、不合并总分。确定性 EM 为主，裁判为辅口径。", ""]
    (out_dir / "public-eval-report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Tier0 public dataset eval")
    parser.add_argument("--output-dir", default=str(ROOT / "output" / "evaluation"))
    parser.add_argument("--live", action="store_true")
    parser.add_argument("--with-baseline", action="store_true",
                        help="追加未压缩历史的上界对照臂")
    parser.add_argument("--limit", type=int, default=0,
                        help="每段对话最多评测的题数（0=全部）")
    parser.add_argument("--calibrate", action="store_true")
    parser.add_argument("--calibrate-limit", type=int, default=50)
    parser.add_argument("--model", default=_DEFAULT_MODEL)
    parser.add_argument("--judge-model", default=None)
    args = parser.parse_args()

    convs, qa, manifest = load_subsets()
    offline_rows = validate_offline(convs, qa)
    qa_sha = manifest.get("locomo", {}).get("subset_qa_sha256", "")
    report: dict[str, Any] = {
        "mode": "offline+live" if args.live else "offline",
        "generated_at": datetime.now(UTC).isoformat(),
        "tokenizer": TOKENIZER_NAME,
        "convs": len(convs),
        "questions": len(qa),
        "qa_sha256": qa_sha,
        "offline": offline_rows,
    }

    print(f"# Tier0 Public Eval ({len(convs)} convs, {len(qa)} questions)")
    for r in offline_rows:
        print(f"- {r['scope']}/{r['check']}: "
              f"{'✓' if r['passed'] else '✗'} {r['detail']}")

    ok = all(r["passed"] for r in offline_rows)

    if args.live or args.calibrate:
        from kama_claude.core.llm.provider import AnthropicProvider

        provider = AnthropicProvider(args.model)
        judge = provider if not args.judge_model \
            else AnthropicProvider(args.judge_model)

    if args.live:
        live_convs = []
        for c in convs:
            row = asyncio.run(evaluate_conv_live(c, qa, provider, judge,
                                                 args.with_baseline))
            live_convs.append(row)
            if row.get("error"):
                print(f"  {c['conv_id']}: ERROR {row['error']}")
            else:
                acc = _accuracy(row["compact"])
                print(f"  {c['conv_id']}: ratio={row['compression_ratio']}x "
                      f"accuracy={acc['overall']:.0%} "
                      f"latency={row['compact_latency_s']}s")
        ok_rows = [r for r in live_convs if not r.get("error")]
        compact_rows = [q for r in ok_rows for q in r["compact"]]
        baseline_rows = [q for r in ok_rows for q in r["baseline"]]
        compact_acc = _accuracy(compact_rows)
        baseline_acc = _accuracy(baseline_rows) if baseline_rows else None
        compact_total = sum(1 for q in compact_rows if q["passed"])
        baseline_total = sum(1 for q in baseline_rows if q["passed"]) \
            if baseline_rows else None
        live_report: dict[str, Any] = {
            "model": args.model,
            "convs_run": len(live_convs),
            "compact_failed": sum(1 for r in live_convs if r.get("error")),
            "avg_latency_s": round(sum(r["compact_latency_s"] for r in ok_rows)
                                   / len(ok_rows), 1) if ok_rows else 0.0,
            "max_latency_s": max((r["compact_latency_s"] for r in ok_rows),
                                 default=0.0),
            "avg_compression_ratio": round(sum(r["compression_ratio"]
                                               for r in ok_rows)
                                           / len(ok_rows), 1) if ok_rows else 0.0,
            "compact_acc": compact_acc,
            "rows": live_convs,
        }
        if baseline_acc is not None:
            live_report["baseline_acc"] = baseline_acc
            live_report["retention"] = round(compact_total / max(baseline_total, 1), 3)
        report["live"] = live_report
        ok = ok and live_report["compact_failed"] == 0

    if args.calibrate:
        calib_path = SUBSETS_DIR / "summeval_calibration.jsonl"
        if calib_path.exists():
            cal = asyncio.run(calibrate_summeval(judge, args.calibrate_limit))
        else:
            cal = asyncio.run(calibrate_locomo(provider, judge, convs, qa,
                                               args.calibrate_limit))
        report["calibration"] = cal
        print(f"- 裁判校准: {cal}")

    _write_reports(Path(args.output_dir), report)
    print(f"report: {Path(args.output_dir) / 'public-eval-report.md'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
