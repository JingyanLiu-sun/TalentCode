# Tier2（scripts/run_e2e_eval.py）回归测试：
#   ① 任务集离线校验（schema + 埋点位置）全部通过
#   ② 管道连通性：脚本化 provider 驱动完整 AgentLoop + Compactor，
#      验证"水位触发压缩 → 摘要进上下文 → 约束事实存活到最终答案"
# 注意：离线测试验证的是管道接线，摘要/续跑的真实质量由 --live 模式评测
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from kama_claude.core.bus.events import ContextCompactedEvent  # noqa: E402
from kama_claude.core.compact.compactor import Compactor  # noqa: E402
from kama_claude.core.context import ExecutionContext  # noqa: E402
from kama_claude.core.events.bus import EventBus  # noqa: E402
from kama_claude.core.llm.types import (  # noqa: E402
    LlmResponse,
    ToolCallBlock,
    UsageStats,
)
from kama_claude.core.loop import AgentLoop  # noqa: E402
from kama_claude.core.tools.base import BaseTool, ToolResult  # noqa: E402
from kama_claude.core.tools.registry import ToolRegistry  # noqa: E402

CONSTRAINT = "all functions must include type hints and a docstring"
SUMMARY = (
    "## 1. Original Goal\nRefactor helper.\n"
    f"## 3. Key Constraints & Discoveries\n- {CONSTRAINT}\n"
    "## 5. Remaining TODOs\n- finish the helper"
)


def _load_script(name: str) -> Any:
    path = ROOT / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


_E2E = _load_script("run_e2e_eval")


# --- ① 任务集离线校验 ---------------------------------------------------------


def test_all_tasks_pass_offline_validation() -> None:
    tasks = _E2E.load_tasks()
    assert len(tasks) == 4
    for task in tasks:
        row = _E2E.validate_task_offline(task)
        assert row["schema_ok"], f"{task['id']}: schema 校验失败"
        assert row["fact_in_history"], f"{task['id']}: 焦点事实落在 tail 区"


# --- ② 管道连通性：水位触发压缩且约束存活 --------------------------------------


class _ScriptedProvider:
    """按序返回预制响应：tool_use(高水位) → 摘要 → 终答。"""

    def __init__(self, responses: list[LlmResponse]) -> None:
        self._it = iter(responses)

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
        return next(self._it)


class _EchoTool(BaseTool):
    name = "echo"
    description = "Echoes msg"
    input_schema: dict[str, object] = {
        "type": "object",
        "properties": {"msg": {"type": "string"}},
        "required": ["msg"],
    }

    async def invoke(self, params: dict[str, object]) -> ToolResult:
        return ToolResult(content=str(params["msg"]))


def _prefill(rounds: int = 5) -> list[dict[str, Any]]:
    # 5 轮真实用户对话（> TAIL_ROUNDS），约束埋在第 1 轮（history 区）
    messages: list[dict[str, Any]] = []
    for i in range(rounds):
        text = f"Goal: refactor helper. Constraint: {CONSTRAINT}" \
            if i == 0 else f"Step {i + 1}: continue."
        messages.append({"role": "user", "content": text})
        messages.append({"role": "assistant", "content": "ok"})
    return messages


async def test_loop_compacts_on_water_level_and_preserves_constraint(
    tmp_path: Path,
) -> None:
    provider = _ScriptedProvider([
        # step1：工具调用，水位 0.9 >= 阈值 0.8 → 追加结果后触发自动压缩
        LlmResponse(
            stop_reason="tool_use",
            tool_calls=[ToolCallBlock(id="t1", name="echo", input={"msg": "x"})],
            usage=UsageStats(input_tokens=100, output_tokens=10, context_pct=0.9),
        ),
        # 压缩调用：返回含约束的摘要
        LlmResponse(stop_reason="end_turn", text=SUMMARY,
                    usage=UsageStats(input_tokens=50, output_tokens=40)),
        # step2：终答，约束仍被遵守
        LlmResponse(stop_reason="end_turn",
                    text=f"done — remember: {CONSTRAINT}",
                    usage=UsageStats(input_tokens=60, output_tokens=20)),
    ])
    bus = EventBus()
    events: list[Any] = []

    async def _collect(e: Any) -> None:
        events.append(e)

    bus.subscribe(_collect)

    registry = ToolRegistry()
    registry.register(_EchoTool())
    loop = AgentLoop(
        provider, registry, bus,  # type: ignore[arg-type]
        compactor=Compactor(bus, tmp_path, "eval-session"),
        compact_threshold=0.8,
    )
    ctx = ExecutionContext(
        run_id="r1", goal="refactor helper", max_steps=5,
        prefill_messages=_prefill(),
    )
    await loop.run(ctx)

    assert ctx.status == "success"
    compacted = [e for e in events if isinstance(e, ContextCompactedEvent)]
    assert len(compacted) == 1, "水位达标必须触发且仅触发一次压缩"
    # 压缩后 messages = [摘要 user, ack assistant] + tail
    assert ctx.messages[0]["content"] == SUMMARY
    assert CONSTRAINT in ctx.messages[0]["content"]
    assert CONSTRAINT in ctx.result


def test_check_runners_are_deterministic() -> None:
    import asyncio

    answer = 'def parse_timeout(raw: str) -> int:\n    """Parse 30s."""\n    return 30'
    checks = [
        {"kind": "regex",
         "pattern": r"def parse_timeout\([^)]*\)\s*->\s*\w+"},
        {"kind": "regex", "pattern": '"""'},
        {"kind": "contains", "value": "{fact}"},
    ]
    rows = asyncio.run(_run_all(checks, answer, "sentinel-id"))
    assert [r["passed"] for r in rows] == [True, True, False]


async def _run_all(checks: list[dict[str, Any]], answer: str,
                   fact: str) -> list[dict[str, Any]]:
    rows = []
    for c in checks:
        rows.append(await _E2E._run_check(c, answer, fact, judge=None))
    return rows
