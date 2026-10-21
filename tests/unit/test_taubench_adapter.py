"""taubench_adapter 的零成本单测：纯函数转换与压缩注入切分逻辑。

依赖 τ-bench（可选依赖）；未安装时整组跳过，不影响 CI 主链路。
"""
import pytest

pytest.importorskip("tau_bench")
pytest.importorskip("litellm")

from evaluation.e2e.taubench_adapter import (  # noqa: E402
    SUMMARY_ACK,
    CompactingToolCallingAgent,
    openai_history_to_kama,
)


def _make_agent(provider: object, threshold: int = 10) -> CompactingToolCallingAgent:
    return CompactingToolCallingAgent(
        tools_info=[], wiki="WIKI", model="m", provider="anthropic",
        compact_provider=provider, compact_token_threshold=threshold)


def _history(user_turns: int) -> list[dict[str, object]]:
    """system + N 个 user 轮（每轮带一次工具调用与回复）。"""
    messages: list[dict[str, object]] = [{"role": "system", "content": "WIKI"}]
    for i in range(user_turns):
        messages += [
            {"role": "user", "content": f"user turn {i}"},
            {"role": "assistant", "content": None, "tool_calls": [
                {"id": f"c{i}", "function": {"name": "lookup",
                                             "arguments": f'{{"i": {i}}}'}}]},
            {"role": "tool", "tool_call_id": f"c{i}", "name": "lookup",
             "content": f"result {i}"},
            {"role": "assistant", "content": f"reply {i}"},
        ]
    return messages


# ---- openai_history_to_kama ----

def test_conversion_alternates_and_merges_tool_chain() -> None:
    kama = openai_history_to_kama(_history(3)[1:])
    assert [m["role"] for m in kama] == ["user", "assistant"] * 3
    # 工具调用与结果并入同轮 assistant 消息
    assert "[TOOL CALL lookup]" in kama[1]["content"]
    assert "[TOOL RESULT lookup]" in kama[1]["content"]
    assert "reply 0" in kama[1]["content"]


def test_conversion_empty_history() -> None:
    assert openai_history_to_kama([]) == []


# ---- _compact_once ----

class _FakeResponse:
    text = "SUMMARY OF HISTORY"
    usage = None


class _FakeProvider:
    async def chat(self, **kwargs: object) -> _FakeResponse:
        return _FakeResponse()


class _FailingProvider:
    async def chat(self, **kwargs: object) -> None:
        raise RuntimeError("boom")


def test_compact_once_replaces_history_keeps_tail() -> None:
    agent = _make_agent(_FakeProvider())
    # 8 个 user 轮：适配层保留最后 3 轮，剩余 5 轮进压缩器（其内部再留 3 轮）
    messages = _history(8)
    result = agent._compact_once(messages, task_index=0)
    assert result is not None
    assert result[0] == messages[0]                      # system wiki 不动
    assert result[1] == {"role": "user", "content": "SUMMARY OF HISTORY"}
    assert result[2]["role"] == "assistant" and result[2]["content"] == SUMMARY_ACK
    # tail 为原 OpenAI 格式的最后 3 个 user 轮（每轮 4 条消息）
    assert result[3:] == messages[1 + 5 * 4:]
    assert len(agent.compactions) == 1
    assert agent.compactions[0]["failed"] is False
    assert agent.compactions[0]["tokens_after"] < agent.compactions[0]["tokens_before"]


def test_compact_once_skips_when_too_few_turns() -> None:
    agent = _make_agent(_FakeProvider())
    assert agent._compact_once(_history(3), task_index=0) is None  # 仅够 tail
    # 6 轮不够双段保留（适配层 3 轮 + 压缩器内部 3 轮 + 至少 1 轮可压缩）
    assert agent._compact_once(_history(6), task_index=0) is None
    assert agent.compactions == []  # 静默跳过，不计压缩失败


def test_compact_once_records_failure_without_crash() -> None:
    agent = _make_agent(_FailingProvider())
    assert agent._compact_once(_history(8), task_index=0) is None
    assert len(agent.compactions) == 1
    assert agent.compactions[0]["failed"] is True
