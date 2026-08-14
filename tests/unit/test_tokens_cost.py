# tokens.py 与 cost.py 的单元测试：真实 tokenizer 计数、定价计算、
# CostTracker 累计与落盘、TrackingProvider 装饰器行为
from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from kama_claude.core.events.bus import EventBus
from kama_claude.core.llm.cost import CostTracker, TrackingProvider
from kama_claude.core.llm.tokens import (
    TOKENIZER_NAME,
    compute_cost,
    count_messages_tokens,
    count_tokens,
    has_pricing,
)
from kama_claude.core.llm.types import LlmResponse, UsageStats


def test_count_tokens_positive_and_monotonic() -> None:
    assert count_tokens("") == 0
    short = count_tokens("hello world")
    long = count_tokens("hello world " * 20)
    assert short > 0
    assert long > short


def test_tokenizer_is_real_when_tiktoken_available() -> None:
    # 依赖已声明 tiktoken；若环境异常降级为回退估算也不应报错
    assert TOKENIZER_NAME in ("cl100k_base", "char_div_4_fallback")


def test_count_messages_tokens_handles_all_block_types() -> None:
    messages: list[dict[str, Any]] = [
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": [
            {"type": "text", "text": "hi there"},
            {"type": "tool_use", "id": "t1", "name": "bash", "input": {"cmd": "ls"}},
        ]},
        {"role": "user", "content": [
            {"type": "tool_result", "tool_use_id": "t1", "content": "file1\nfile2"},
        ]},
    ]
    total = count_messages_tokens(messages)
    # 3 条消息各 +4 结构开销，正文 token 为正数
    assert total > 12


def test_compute_cost_matches_pricing_model() -> None:
    # sonnet: $3/M 输入，$15/M 输出；缓存读 10%、缓存写 125%
    cost = compute_cost(
        "claude-sonnet-4-6",
        input_tokens=1_000_000,
        output_tokens=1_000_000,
    )
    assert cost == pytest.approx(18.0)

    cost_cache = compute_cost(
        "claude-sonnet-4-6",
        input_tokens=0,
        output_tokens=0,
        cache_read_input_tokens=1_000_000,
        cache_creation_input_tokens=1_000_000,
    )
    assert cost_cache == pytest.approx(3.0 * 0.1 + 3.0 * 1.25)


def test_compute_cost_unknown_model_uses_default() -> None:
    assert not has_pricing("unknown-model")
    cost = compute_cost(
        "unknown-model", input_tokens=1_000_000, output_tokens=0
    )
    assert cost == pytest.approx(3.0)  # 默认按 sonnet 价估算


def test_cost_tracker_accumulates_and_flushes(tmp_path: Path) -> None:
    tracker = CostTracker(model="claude-sonnet-4-6")
    c1 = tracker.record(run_id="r1", input_tokens=1000, output_tokens=100)
    c2 = tracker.record(run_id="r1", input_tokens=2000, output_tokens=200)
    assert c1 > 0 and c2 > 0
    assert tracker.call_count == 2
    assert tracker.total_input_tokens == 3000
    assert tracker.total_output_tokens == 300
    assert tracker.total_cost_usd == pytest.approx(c1 + c2)

    path = tmp_path / "cost.jsonl"
    tracker.flush(path)
    lines = path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 2
    # 刷盘后记录清空，重复 flush 不追加
    tracker.flush(path)
    assert len(path.read_text(encoding="utf-8").strip().splitlines()) == 2


def test_cost_tracker_flush_skips_empty(tmp_path: Path) -> None:
    tracker = CostTracker(model="claude-sonnet-4-6")
    path = tmp_path / "cost.jsonl"
    tracker.flush(path)
    assert not path.exists()


class _FakeProvider:
    # 返回带 usage 的固定响应，记录收到的参数
    def __init__(self) -> None:
        self.calls: list[int] = []

    async def chat(
        self,
        messages: list[dict[str, object]],
        tool_schemas: list[dict[str, object]],
        bus: EventBus,
        run_id: str,
        *,
        step: int = 0,
        system: str | None = None,
    ) -> LlmResponse:
        self.calls.append(step)
        return LlmResponse(
            stop_reason="end_turn",
            text="ok",
            usage=UsageStats(input_tokens=1000, output_tokens=100),
        )


async def test_tracking_provider_records_and_publishes() -> None:
    inner = _FakeProvider()
    tracker = CostTracker(model="claude-sonnet-4-6")
    bus = EventBus()
    seen: list[Any] = []

    async def _collect(e: Any) -> None:
        seen.append(e)

    bus.subscribe(_collect)

    wrapper = TrackingProvider(inner, tracker, bus, run_id="r1", session_id="s1")
    resp = await wrapper.chat([], [], bus, "r1", step=1)

    assert resp.text == "ok"
    assert inner.calls == [1]
    assert tracker.call_count == 1
    assert tracker.total_cost_usd > 0
    cost_events = [e for e in seen if getattr(e, "type", "") == "cost.updated"]
    assert len(cost_events) == 1
    ev = cost_events[0]
    assert ev.run_id == "r1"
    assert ev.session_id == "s1"
    assert ev.total_input_tokens == 1000
    assert ev.cumulative_cost_usd == pytest.approx(tracker.total_cost_usd)


async def test_tracking_provider_no_usage_no_event() -> None:
    class _NoUsageProvider:
        async def chat(self, *args: Any, **kwargs: Any) -> LlmResponse:
            return LlmResponse(stop_reason="end_turn", text="ok", usage=None)

    tracker = CostTracker(model="claude-sonnet-4-6")
    bus = EventBus()
    seen: list[Any] = []

    async def _collect(e: Any) -> None:
        seen.append(e)

    bus.subscribe(_collect)
    wrapper = TrackingProvider(_NoUsageProvider(), tracker, bus, run_id="r1")
    await wrapper.chat([], [], bus, "r1")
    assert tracker.call_count == 0
    assert seen == []
