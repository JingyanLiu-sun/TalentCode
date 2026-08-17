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


# 功能：验证 count_tokens 基本性质：空串为 0、非空为正、文本越长 token 越多
# 设计：用单调性兜底 tokenizer 回归（如误退化为常量返回），不依赖具体 token 数
def test_count_tokens_positive_and_monotonic() -> None:
    assert count_tokens("") == 0
    short = count_tokens("hello world")
    long = count_tokens("hello world " * 20)
    assert short > 0
    assert long > short


# 功能：验证 tokenizer 标识为真实编码或声明的回退方案之一
# 设计：tiktoken 已是声明依赖，正常应加载 cl100k_base；允许回退保证降级环境模块仍可导入
def test_tokenizer_is_real_when_tiktoken_available() -> None:
    # 依赖已声明 tiktoken；若环境异常降级为回退估算也不应报错
    assert TOKENIZER_NAME in ("cl100k_base", "char_div_4_fallback")


# 功能：验证 count_messages_tokens 覆盖 str、text/tool_use 块、tool_result 三种内容形态
# 设计：混合 user/assistant 消息，断言总数含每消息结构开销且为正，防止新块类型漏计或崩溃
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


# 功能：验证 sonnet 模型定价计算精确值，含缓存读/写折扣
# 设计：1M tokens 全量输入输出验证基础价（$3+$15=18）；纯缓存输入验证 10%/125% 系数
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


# 功能：验证未知模型回退到默认价而非抛异常
# 设计：费用统计不应因新模型名导致 run 崩溃；用 has_pricing 断言走的是回退路径
def test_compute_cost_unknown_model_uses_default() -> None:
    assert not has_pricing("unknown-model")
    cost = compute_cost(
        "unknown-model", input_tokens=1_000_000, output_tokens=0
    )
    assert cost == pytest.approx(3.0)  # 默认按 sonnet 价估算


# 功能：验证 CostTracker 累计调用次数/token/费用，flush 将明细追加写入 cost.jsonl
# 设计：flush 后记录清空，重复 flush 不再追加，保证落盘幂等
#       （daemon 异常恢复路径可能重复触发 flush）
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


# 功能：验证无任何记录时 flush 不创建 cost.jsonl
# 设计：未产生 LLM 调用的 run 不应留下空文件，避免 session 目录噪声
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


# 功能：验证 TrackingProvider 装饰后透传响应、记账 usage 并发布 cost.updated 事件
# 设计：fake provider 返回固定 usage，断言事件的 run_id/session_id/累计费用与 tracker 一致
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


# 功能：验证内层 provider 未返回 usage 时不记账、不发事件
# 设计：usage 缺失（异常降级响应）应静默跳过，而非记入零成本污染统计
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
