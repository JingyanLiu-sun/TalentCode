#!/usr/bin/env python3
"""Compact 质量评测基准。

离线模式（默认，无需 API Key）：评测确定性预处理阶段的指标
  - 预处理压缩比（真实 tokenizer 计量）
  - 错误块保留率（[ERROR-VERBATIM] 标记且原文不丢）
  - 代码文件 stub 替换正确性（路径保留、内容被索引 stub 替换）
  - tail 边界正确性（最近 TAIL_ROUNDS 轮完整保留）

在线模式（--live，需要 ANTHROPIC_API_KEY，会产生少量费用）：
  - 端到端压缩比（原文 vs 摘要+tail）
  - 摘要信息保留率：预埋在历史中的关键事实出现在摘要里的比例

用法：
  uv run python scripts/benchmark_compact.py            # 离线指标
  uv run python scripts/benchmark_compact.py --live     # 含 LLM 摘要质量
  uv run python scripts/benchmark_compact.py --live --repeat 3
"""
from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path
from typing import Any

from kama_claude.core.compact.compactor import (
    TAIL_ROUNDS,
    Compactor,
    _find_tail_boundary,
    _messages_to_text,
    _preprocess_for_summary,
)
from kama_claude.core.llm.tokens import TOKENIZER_NAME, count_messages_tokens

# 预埋在历史中的关键事实：摘要质量评测会检查它们是否被保留
PLANTED_FACTS: dict[str, str] = {
    "file_path_1": "src/kama_claude/core/compact/compactor.py",
    "file_path_2": "src/kama_claude/core/session/store.py",
    "decision": "use sqlite for session cache",
    "error_marker": "ConnectionRefusedError: port 7437",
    "critical_id": "sess-9f3ab2c1",
    "remaining_todo": "add retry backoff to socket client",
}

_LONG_OUTPUT = "line %04d: ok status=200 payload=" + "x" * 80 + "\n"


# 构造一段带预埋事实的合成对话：8 轮真实用户消息，含超长 tool_result、
# read_file 代码文件结果和一个错误结果
def build_fixture() -> list[dict[str, Any]]:
    messages: list[dict[str, Any]] = [
        {"role": "user", "content": (
            "Goal: refactor session persistence. Constraint: "
            + PLANTED_FACTS["decision"] + ". Session id is "
            + PLANTED_FACTS["critical_id"] + "."
        )},
        {"role": "assistant", "content": [
            {"type": "text", "text": "I'll start by inspecting the store module."},
            {"type": "tool_use", "id": "tu1", "name": "read_file",
             "input": {"path": PLANTED_FACTS["file_path_2"]}},
        ]},
        {"role": "user", "content": [
            {"type": "tool_result", "tool_use_id": "tu1",
             "content": "# store.py\n" + "def f():\n    pass\n" * 300},
        ]},
        {"role": "assistant", "content": [
            {"type": "text", "text": "Now running the test suite."},
            {"type": "tool_use", "id": "tu2", "name": "bash",
             "input": {"command": "pytest tests/unit -q"}},
        ]},
        {"role": "user", "content": [
            {"type": "tool_result", "tool_use_id": "tu2",
             "content": "".join(_LONG_OUTPUT % i for i in range(400))},
        ]},
        {"role": "user", "content": (
            "The daemon failed to start, here is the log: "
            + PLANTED_FACTS["error_marker"]
        )},
        {"role": "assistant", "content": [
            {"type": "text", "text": "Retrying the connection."},
            {"type": "tool_use", "id": "tu3", "name": "bash",
             "input": {"command": "kama-core"}},
        ]},
        {"role": "user", "content": [
            {"type": "tool_result", "tool_use_id": "tu3", "is_error": True,
             "content": "Traceback (most recent call last):\n  "
             + PLANTED_FACTS["error_marker"] + "\nretry limit exceeded"},
        ]},
        {"role": "user", "content": "Ok, now please update the compactor module."},
        {"role": "assistant", "content": [
            {"type": "text", "text": "Reading it first."},
            {"type": "tool_use", "id": "tu4", "name": "read_file",
             "input": {"path": PLANTED_FACTS["file_path_1"]}},
        ]},
        {"role": "user", "content": [
            {"type": "tool_result", "tool_use_id": "tu4",
             "content": "# compactor.py\n" + "class C:\n    pass\n" * 200},
        ]},
        {"role": "user", "content": (
            "After you finish, remember: " + PLANTED_FACTS["remaining_todo"] + "."
        )},
    ]
    return messages


# 离线指标：确定性预处理阶段的质量，零 LLM 成本
def measure_offline(messages: list[dict[str, Any]]) -> dict[str, Any]:
    before = count_messages_tokens(messages)
    processed = _preprocess_for_summary(messages)
    after = count_messages_tokens(processed)

    # 错误块保留：预处理后仍含 [ERROR-VERBATIM] 且原始错误文本不丢
    error_ok = False
    for msg in processed:
        content = msg.get("content")
        if not isinstance(content, list):
            continue
        for block in content:
            if block.get("type") == "tool_result" and block.get("is_error"):
                text = str(block.get("content", ""))
                error_ok = (
                    text.startswith("[ERROR-VERBATIM]")
                    and PLANTED_FACTS["error_marker"] in text
                )

    # stub 替换：两个 read_file 代码文件结果都应变成含路径的索引 stub
    stub_hits = 0
    for path in (PLANTED_FACTS["file_path_1"], PLANTED_FACTS["file_path_2"]):
        for msg in processed:
            content = msg.get("content")
            if not isinstance(content, list):
                continue
            for block in content:
                text = str(block.get("content", ""))
                if "File previously read" in text and path in text:
                    stub_hits += 1

    # tail 边界：切分点之后恰好包含最近 TAIL_ROUNDS 轮真实用户消息
    split = _find_tail_boundary(messages, tail_rounds=TAIL_ROUNDS)
    tail = messages[split:]
    tail_user_rounds = sum(
        1 for m in tail
        if m.get("role") == "user" and not (
            isinstance(m.get("content"), list)
            and all(b.get("type") == "tool_result" for b in m["content"])
        )
    )
    real_user_rounds = sum(
        1 for m in messages
        if m.get("role") == "user" and not (
            isinstance(m.get("content"), list)
            and all(b.get("type") == "tool_result" for b in m["content"])
        )
    )

    return {
        "before_tokens": before,
        "after_preprocess_tokens": after,
        "preprocess_ratio": round(after / before, 3) if before else 0.0,
        "error_preserved": error_ok,
        "stub_replaced": stub_hits,
        "stub_expected": 2,
        "tail_rounds_kept": TAIL_ROUNDS,
        "real_user_rounds": real_user_rounds,
        "tail_boundary_valid": split > 0 and tail_user_rounds == TAIL_ROUNDS,
    }


# 在线指标：真实 LLM 摘要后的压缩比与信息保留率
async def measure_live(
    messages: list[dict[str, Any]], model: str, repeat: int
) -> dict[str, Any]:
    from kama_claude.core.events.bus import EventBus
    from kama_claude.core.llm.provider import AnthropicProvider

    provider = AnthropicProvider(model)
    ratios: list[float] = []
    retentions: list[float] = []

    for i in range(repeat):
        silent = EventBus()  # 静默总线：benchmark 不关心事件
        compactor = Compactor(silent, Path("."), "benchmark")
        result = await compactor.compact_messages(messages, provider)
        if result is None:
            print(f"  trial {i + 1}: compaction skipped (history too short)")
            continue

        before = count_messages_tokens(messages)
        tail_tokens = count_messages_tokens(result.tail_messages or [])
        after = result.summary_tokens + tail_tokens
        ratios.append(before / after if after else 0.0)

        searchable = result.summary_text + _messages_to_text(result.tail_messages or [])
        searchable_lower = searchable.lower()
        hits = sum(1 for f in PLANTED_FACTS.values() if f.lower() in searchable_lower)
        retentions.append(hits / len(PLANTED_FACTS))
        print(
            f"  trial {i + 1}: ratio={ratios[-1]:.1f}x "
            f"retention={retentions[-1]:.0%} "
            f"summary={result.summary_tokens} tokens"
        )

    if not ratios:
        return {"error": "all trials skipped"}
    return {
        "model": model,
        "repeat": repeat,
        "avg_compression_ratio": round(sum(ratios) / len(ratios), 1),
        "avg_retention": round(sum(retentions) / len(retentions), 3),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="compact quality benchmark")
    parser.add_argument("--live", action="store_true", help="调用真实 LLM（需 API Key）")
    parser.add_argument("--repeat", type=int, default=1, help="在线模式重复次数")
    parser.add_argument("--model", default="claude-sonnet-4-6")
    args = parser.parse_args()

    messages = build_fixture()
    print(f"# Compact Benchmark (tokenizer={TOKENIZER_NAME})\n")

    off = measure_offline(messages)
    print("## 离线指标（确定性预处理）")
    print(f"- 预处理压缩比: {off['before_tokens']} -> {off['after_preprocess_tokens']}"
          f" tokens ({off['preprocess_ratio']:.0%})")
    print(f"- 错误块保留: {'✓' if off['error_preserved'] else '✗'}")
    print(f"- 代码文件 stub 替换: {off['stub_replaced']}/{off['stub_expected']}")
    print(f"- tail 边界: 保留最近 {off['tail_rounds_kept']} 轮"
          f"（共 {off['real_user_rounds']} 轮真实对话）"
          f" {'✓' if off['tail_boundary_valid'] else '✗'}")

    failed = (
        not off["error_preserved"]
        or off["stub_replaced"] != off["stub_expected"]
        or not off["tail_boundary_valid"]
    )

    if args.live:
        print("\n## 在线指标（LLM 摘要质量）")
        live = asyncio.run(measure_live(messages, args.model, args.repeat))
        if "error" in live:
            print(f"- {live['error']}")
            return 1
        print(f"- 平均压缩比: {live['avg_compression_ratio']}x")
        print(f"- 平均信息保留率: {live['avg_retention']:.0%}"
              f"（{len(PLANTED_FACTS)} 个预埋事实）")

    print()
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
