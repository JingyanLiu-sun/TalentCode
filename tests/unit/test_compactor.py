from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock

from kama_claude.core.compact.compactor import (
    _HEAD_KEEP,
    _TAIL_KEEP,
    _TOOL_RESULT_LIMIT,
    TAIL_ROUNDS,
    Compactor,
    _find_tail_boundary,
    _preprocess_for_summary,
)
from kama_claude.core.context import ExecutionContext
from kama_claude.core.events.bus import EventBus
from kama_claude.core.llm.types import LlmResponse, UsageStats


def _stub_provider(summary: str = "## 1. Original Goal\nTest\n## 2. Completed Steps\n- done") -> Any:
    provider = MagicMock()
    provider.chat = AsyncMock(return_value=LlmResponse(
        stop_reason="end_turn",
        text=summary,
        usage=UsageStats(input_tokens=100, output_tokens=30),
    ))
    return provider


def _make_messages(n: int = 5) -> list[dict[str, Any]]:
    msgs = []
    for i in range(n):
        msgs.append({"role": "user", "content": "user message " + "x" * 200})
        msgs.append({"role": "assistant", "content": "assistant reply " + "y" * 200})
    return msgs


# 功能：验证 compact_messages 成功时 provider.chat 被调用一次且不传工具 schema
# 设计：stub provider 返回非空摘要，断言 chat 调用一次，tool_schemas=[]
async def test_compact_messages_calls_provider(tmp_path: Path) -> None:
    provider = _stub_provider()
    bus = EventBus()
    compactor = Compactor(bus, tmp_path, "sess-1")
    messages = _make_messages()

    result = await compactor.compact_messages(messages, provider)

    assert result is not None
    provider.chat.assert_called_once()
    call_kwargs = provider.chat.call_args
    assert call_kwargs.kwargs.get("tool_schemas") == [] or call_kwargs.args[1] == []


# 功能：验证 compact_messages 返回的摘要文本来自 provider 响应
# 设计：stub provider 返回固定摘要字符串，断言 result.summary_text 等于该字符串
async def test_compact_messages_returns_summary(tmp_path: Path) -> None:
    expected = "## 1. Original Goal\nDo X\n## 2. Completed\n- step one"
    provider = _stub_provider(summary=expected)
    bus = EventBus()
    compactor = Compactor(bus, tmp_path, "sess-1")

    result = await compactor.compact_messages(_make_messages(), provider)

    assert result is not None
    assert result.summary_text == expected


# 功能：验证 compact() 将 context.messages 替换为 [摘要2条] + [最近N轮完整消息]
# 设计：调用 compact() 后断言前 2 条为摘要对，后续为原始 tail 消息
async def test_compact_replaces_context_messages(tmp_path: Path) -> None:
    provider = _stub_provider()
    bus = EventBus()
    compactor = Compactor(bus, tmp_path, "sess-1")
    ctx = ExecutionContext(run_id="r1", goal="test", max_steps=5)
    messages = _make_messages(5)  # 5 轮，TAIL_ROUNDS=3 → tail = 最后 3 轮 = 6 条
    ctx.messages = list(messages)

    await compactor.compact(ctx, provider)

    # 前 2 条是摘要对
    assert ctx.messages[0]["role"] == "user"
    assert ctx.messages[1]["role"] == "assistant"
    # 后续是保留的 tail（最后 3 轮 = 6 条消息）
    expected_tail = messages[4:]  # split_idx=4
    assert ctx.messages[2:] == expected_tail
    assert len(ctx.messages) == 2 + len(expected_tail)


# 功能：验证 compact() 在 session 目录写入 summary_*.md 文件
# 设计：使用 tmp_path，调用 compact() 后检查目录内是否存在 summary_ 开头的文件
async def test_compact_writes_summary_file(tmp_path: Path) -> None:
    provider = _stub_provider()
    bus = EventBus()
    compactor = Compactor(bus, tmp_path, "sess-1")
    ctx = ExecutionContext(run_id="r1", goal="test", max_steps=5)
    ctx.messages = _make_messages()

    await compactor.compact(ctx, provider)

    summary_files = list(tmp_path.glob("summary_*.md"))
    assert len(summary_files) == 1


# 功能：验证 compact() 成功后发布 ContextCompactedEvent 事件
# 设计：订阅 EventBus，收集事件，断言收到类型为 context.compacted 的事件
async def test_compact_publishes_event(tmp_path: Path) -> None:
    provider = _stub_provider()
    bus = EventBus()
    received: list[Any] = []

    async def handler(event: Any) -> None:
        received.append(event)

    bus.subscribe(handler)
    compactor = Compactor(bus, tmp_path, "sess-1")
    ctx = ExecutionContext(run_id="r1", goal="test", max_steps=5)
    ctx.messages = _make_messages()

    await compactor.compact(ctx, provider)

    types = [getattr(e, "type", None) for e in received]
    assert "context.compacted" in types


# 功能：验证 provider 抛异常时 context.messages 保持不变
# 设计：stub provider.chat 抛 RuntimeError，断言 compact() 返回 None 且 messages 未被修改
async def test_compact_failure_preserves_context(tmp_path: Path) -> None:
    provider = MagicMock()
    provider.chat = AsyncMock(side_effect=RuntimeError("LLM error"))
    bus = EventBus()
    compactor = Compactor(bus, tmp_path, "sess-1")
    ctx = ExecutionContext(run_id="r1", goal="test", max_steps=5)
    original_messages = _make_messages()
    ctx.messages = list(original_messages)

    result = await compactor.compact(ctx, provider)

    assert result is None
    assert ctx.messages == original_messages


def _tool_pair_messages(result_content: str, *, tool: str = "bash",
                        tool_input: dict[str, Any] | None = None,
                        is_error: bool = False) -> list[dict[str, Any]]:
    # 构造一对 tool_use + tool_result 消息，模拟真实会话中的工具调用历史
    result_block: dict[str, Any] = {
        "type": "tool_result",
        "tool_use_id": "tu-1",
        "content": result_content,
    }
    if is_error:
        result_block["is_error"] = True
    return [
        {"role": "assistant", "content": [
            {"type": "tool_use", "id": "tu-1", "name": tool,
             "input": tool_input or {"command": "ls"}},
        ]},
        {"role": "user", "content": [result_block]},
    ]


# 功能：验证超长 tool_result 被截断为头尾两段
# 设计：构造超过 _TOOL_RESULT_LIMIT 的 bash 输出，断言中间被省略标记替换且总长显著下降
def test_preprocess_truncates_large_tool_result() -> None:
    big = "h" * _HEAD_KEEP + "x" * 20000 + "t" * _TAIL_KEEP
    messages = _tool_pair_messages(big)

    processed = _preprocess_for_summary(messages)

    content = processed[1]["content"][0]["content"]
    assert len(content) < len(big)  # 体积显著下降
    assert len(content) < _TOOL_RESULT_LIMIT  # 截断后回到阈值以内
    assert "chars omitted" in content
    assert content.startswith("h" * _HEAD_KEEP)
    assert content.endswith("t" * _TAIL_KEEP)
    assert "x" not in content  # 中间部分被丢弃


# 功能：验证阈值以内的 tool_result 不被修改
# 设计：构造略低于 _TOOL_RESULT_LIMIT 的输出，断言预处理后内容不变
def test_preprocess_keeps_small_tool_result() -> None:
    small = "y" * (_TOOL_RESULT_LIMIT - 1)
    messages = _tool_pair_messages(small)

    processed = _preprocess_for_summary(messages)

    assert processed[1]["content"][0]["content"] == small


# 功能：验证错误结果无论多大都原样保留并加 [ERROR-VERBATIM] 标记
# 设计：构造超大 is_error=True 的结果，断言不被截断且带标记前缀
def test_preprocess_preserves_errors_verbatim() -> None:
    huge_error = "Traceback (most recent call last):\n" + "e" * 30000
    messages = _tool_pair_messages(huge_error, is_error=True)

    processed = _preprocess_for_summary(messages)

    content = processed[1]["content"][0]["content"]
    assert content.startswith("[ERROR-VERBATIM]\n")
    assert huge_error in content  # 错误内容逐字保留，未被截断


# 功能：验证 read_file 读过的代码文件被替换为索引 stub
# 设计：构造 read_file 工具对，断言结果被替换为包含路径与重取提示的短 stub
def test_preprocess_stubs_read_code_files() -> None:
    code = "def foo():\n    return 1\n" * 500  # 大段代码内容
    messages = _tool_pair_messages(
        code, tool="read_file", tool_input={"path": "src/app.py", "mode": "full"},
    )

    processed = _preprocess_for_summary(messages)

    content = processed[1]["content"][0]["content"]
    assert "src/app.py" in content
    assert "mode=structure" in content  # 提示可用 AST 结构模式低成本重取
    assert "def foo" not in content  # 代码实现细节被丢弃


# 功能：验证 read_file 读取的非代码文件（如日志）不享受 stub 待遇，走普通截断
# 设计：构造 .log 后缀的大文件结果，断言不出现 stub 文案而是头尾截断
def test_preprocess_does_not_stub_non_code_files() -> None:
    big_log = "l" * 20000
    messages = _tool_pair_messages(
        big_log, tool="read_file", tool_input={"path": "app.log"},
    )

    processed = _preprocess_for_summary(messages)

    content = processed[1]["content"][0]["content"]
    assert "File previously read" not in content
    assert "chars omitted" in content


# 功能：验证预处理不修改原始消息列表（压缩失败时上下文不受影响）
# 设计：预处理后断言原消息内容逐字不变
def test_preprocess_does_not_mutate_original() -> None:
    import copy

    big = "z" * 20000
    messages = _tool_pair_messages(big, is_error=True)
    snapshot = copy.deepcopy(messages)

    _preprocess_for_summary(messages)

    assert messages == snapshot


# 功能：验证 compact_messages 送入 LLM 的是预处理后的历史而非原始历史
# 设计：构造含超大 bash 输出的消息，断言 provider.chat 收到的 content 不含被截断的中间段
async def test_compact_messages_sends_preprocessed_history(tmp_path: Path) -> None:
    provider = _stub_provider()
    bus = EventBus()
    compactor = Compactor(bus, tmp_path, "sess-1")
    big = "h" * _HEAD_KEEP + "MIDDLE-SECRET" * 3000 + "t" * _TAIL_KEEP
    # 构造足够多轮以便有 history 部分（需要 > TAIL_ROUNDS 轮）
    messages = _make_messages(TAIL_ROUNDS + 1)  # 4 轮
    # 在第一轮 assistant 消息中嵌入大 tool_result
    messages[0] = {"role": "user", "content": "start"}
    messages[1] = {"role": "assistant", "content": [
        {"type": "tool_use", "id": "tu-1", "name": "bash", "input": {"command": "ls"}},
    ]}
    # 插入 tool_result 作为第二条消息后的 user 消息
    messages.insert(2, {"role": "user", "content": [
        {"type": "tool_result", "tool_use_id": "tu-1", "content": big},
    ]})

    await compactor.compact_messages(messages, provider)

    sent = provider.chat.call_args.kwargs["messages"][0]["content"]
    assert "MIDDLE-SECRET" not in sent  # 中间段被预处理截断，未进入 LLM 输入
    assert "chars omitted" in sent


# 功能：验证 _find_tail_boundary 正确切分历史与 tail
# 设计：构造 5 轮纯文本消息，断言切分点位于倒数第 3 轮 user 消息处
def test_find_tail_boundary_basic() -> None:
    messages = _make_messages(5)  # user_indices = [0,2,4,6,8]
    idx = _find_tail_boundary(messages, tail_rounds=3)
    assert idx == 4  # user_indices[-3] = 4


# 功能：验证轮数不足时返回 0（无可压缩历史）
# 设计：构造仅 2 轮消息，断言返回 0
def test_find_tail_boundary_insufficient_rounds() -> None:
    messages = _make_messages(2)
    idx = _find_tail_boundary(messages, tail_rounds=3)
    assert idx == 0


# 功能：验证 tool_result 类 user 消息不计入轮次
# 设计：构造含 tool_result 的消息序列，断言切分点跳过 tool_result 消息
def test_find_tail_boundary_skips_tool_results() -> None:
    messages = [
        {"role": "user", "content": "round 1"},          # idx=0, real user
        {"role": "assistant", "content": [
            {"type": "tool_use", "id": "t1", "name": "bash", "input": {}},
        ]},
        {"role": "user", "content": [                    # idx=2, tool_result → 跳过
            {"type": "tool_result", "tool_use_id": "t1", "content": "ok"},
        ]},
        {"role": "assistant", "content": "done"},
        {"role": "user", "content": "round 2"},          # idx=4, real user
        {"role": "assistant", "content": "reply 2"},
        {"role": "user", "content": "round 3"},          # idx=6, real user
        {"role": "assistant", "content": "reply 3"},
        {"role": "user", "content": "round 4"},          # idx=8, real user
        {"role": "assistant", "content": "reply 4"},
    ]
    # real user_indices = [0, 4, 6, 8], tail_rounds=3 → idx = user_indices[-3] = 4
    idx = _find_tail_boundary(messages, tail_rounds=3)
    assert idx == 4


# 功能：验证轮数不足时 compact_messages 返回 None
# 设计：构造仅 2 轮消息，断言 compact_messages 返回 None
async def test_compact_messages_skips_when_insufficient_rounds(tmp_path: Path) -> None:
    provider = _stub_provider()
    bus = EventBus()
    compactor = Compactor(bus, tmp_path, "sess-1")
    messages = _make_messages(2)  # 只有 2 轮，不足以切分

    result = await compactor.compact_messages(messages, provider)

    assert result is None
    provider.chat.assert_not_called()


# 功能：验证 compact 后 tail 消息内容完整保留（未被修改）
# 设计：对比 compact 前后 tail 部分消息是否逐字一致
async def test_compact_preserves_tail_messages_intact(tmp_path: Path) -> None:
    provider = _stub_provider()
    bus = EventBus()
    compactor = Compactor(bus, tmp_path, "sess-1")
    ctx = ExecutionContext(run_id="r1", goal="test", max_steps=5)
    messages = _make_messages(5)
    ctx.messages = list(messages)
    expected_tail = messages[4:]  # 最后 3 轮

    await compactor.compact(ctx, provider)

    assert ctx.messages[2:] == expected_tail
