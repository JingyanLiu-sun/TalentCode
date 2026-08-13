from __future__ import annotations

import copy
import logging
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from kama_claude.core.bus.events import ContextCompactedEvent
from kama_claude.core.events.bus import EventBus

if TYPE_CHECKING:
    from kama_claude.core.context import ExecutionContext
    from kama_claude.core.llm.base import LLMProvider

logger = logging.getLogger(__name__)

_COMPACT_PROMPT = """\
You are compressing an agent conversation into a handoff summary.
Another LLM instance will continue this task from your summary alone — make it complete.

Structure your response with exactly these six sections:

## 1. Original Goal
One sentence describing what the user asked the agent to accomplish.

## 2. Completed Steps
Bullet list of what has been done. Be specific (file paths, commands run, decisions made).

## 3. Key Constraints & Discoveries
Facts learned during the run that affect future decisions \
(e.g., API limitations, file formats, user preferences stated mid-conversation).

## 4. Current File State
For each file that was created or modified: path, a one-line description of its current state.

## 5. Remaining TODOs
Ordered list of what still needs to be done to complete the original goal.

## 6. Critical Data
Any values the next LLM needs verbatim: IDs, tokens, exact error messages, config values \
discovered during the run.

Be concise. Omit reasoning steps and intermediate attempts. Keep conclusions.

Special handling:
- Blocks marked [ERROR-VERBATIM] contain exact error output: copy them word-for-word \
into section 6, never paraphrase.
- Previously read code files are replaced by index stubs; note in section 4 that the \
file was read and can be re-fetched with read_file (mode=structure or mode=range).\
"""

# 预处理阈值：tool_result 超过该长度即截断，错误结果豁免
_TOOL_RESULT_LIMIT = 8192
_HEAD_KEEP = 2048
_TAIL_KEEP = 2048
# read_file 结果中被视为代码文件的扩展名：压缩时只保留索引 stub
_CODE_EXTS = frozenset({".py", ".pyi", ".ipynb"})
# compact 后保留最近 N 轮完整对话（一轮 = 真实 user 消息及其后续 assistant/tool_result）
TAIL_ROUNDS = 3


# 返回当前 UTC 时间的简短时间戳字符串（用于文件名）
def _ts_compact() -> str:
    return datetime.now(UTC).strftime("%Y%m%d_%H%M%S")


# 返回当前 UTC 时间的 ISO 8601 字符串
def _now() -> str:
    return datetime.now(UTC).isoformat()


@dataclass
class CompactionResult:
    summary_text: str
    original_token_estimate: int
    summary_tokens: int
    tail_messages: list[dict[str, Any]] | None = None


class Compactor:
    # 初始化压缩器，绑定事件总线、session 目录和 session ID
    def __init__(self, bus: EventBus, session_dir: Path, session_id: str) -> None:
        self._bus = bus
        self._session_dir = session_dir
        self._session_id = session_id

    # 压缩 ExecutionContext.messages，就地替换为 [摘要2条] + [最近N轮完整消息]
    async def compact(
        self,
        context: ExecutionContext,
        provider: LLMProvider,
        focus: str = "",
    ) -> CompactionResult | None:
        result = await self.compact_messages(context.messages, provider, focus=focus)
        if result is None:
            return None

        summary_part = [
            {"role": "user", "content": result.summary_text},
            {"role": "assistant", "content": "Understood, I'll continue from this summary."},
        ]
        tail_part = result.tail_messages or []
        context.messages = summary_part + tail_part
        self._write_summary(result.summary_text)
        await self._bus.publish(
            ContextCompactedEvent(
                session_id=self._session_id,
                run_id=context.run_id,
                original_tokens=result.original_token_estimate,
                summary_tokens=result.summary_tokens,
                ts=_now(),
            )
        )
        logger.info(
            "context compacted session=%s run=%s original≈%d summary=%d tokens",
            self._session_id, context.run_id,
            result.original_token_estimate, result.summary_tokens,
        )
        return result

    # 纯函数式压缩：只对历史部分做摘要，最近 N 轮原样保留；失败时返回 None
    async def compact_messages(
        self,
        messages: list[dict[str, Any]],
        provider: LLMProvider,
        focus: str = "",
    ) -> CompactionResult | None:
        from kama_claude.core.events.bus import EventBus as _Bus

        split_idx = _find_tail_boundary(messages, tail_rounds=TAIL_ROUNDS)
        history = messages[:split_idx]
        tail = messages[split_idx:]

        # 历史部分不足以压缩时直接跳过
        if not history:
            return None

        original_estimate = sum(
            len(str(m.get("content", ""))) for m in messages
        ) // 4  # 粗略 token 估算（字符数 / 4）

        history_text = _messages_to_text(_preprocess_for_summary(history))
        prompt = _COMPACT_PROMPT
        if focus.strip():
            prompt += f"\n\nIMPORTANT: Pay special attention to: {focus.strip()}"

        compress_request: list[dict[str, object]] = [
            {"role": "user", "content": f"{prompt}\n\n---\n\n{history_text}"}
        ]

        try:
            silent_bus = _Bus()
            response = await provider.chat(
                messages=compress_request,
                tool_schemas=[],
                bus=silent_bus,
                run_id="compact",
                step=0,
                system="You are a helpful assistant that summarizes conversations.",
            )
        except Exception:
            logger.exception("compactor: LLM call failed, skipping compaction")
            return None

        summary_text = response.text.strip()
        if not summary_text:
            logger.warning("compactor: LLM returned empty summary, skipping compaction")
            return None

        summary_tokens = response.usage.output_tokens if response.usage else len(summary_text) // 4

        return CompactionResult(
            summary_text=summary_text,
            original_token_estimate=original_estimate,
            summary_tokens=summary_tokens,
            tail_messages=tail,
        )

    # 将摘要文本写入 session 目录的 summary_<ts>.md
    def _write_summary(self, text: str) -> None:
        try:
            self._session_dir.mkdir(parents=True, exist_ok=True)
            path = self._session_dir / f"summary_{_ts_compact()}.md"
            path.write_text(text, encoding="utf-8")
        except Exception:
            logger.exception("compactor: failed to write summary file")


# 从尾部向前找 N 轮真实用户消息的起始位置，作为 history/tail 切分点。
# 真实用户轮次 = role 为 user 且 content 不是纯 tool_result 列表的消息。
# 轮数不足时返回 0（表示无可压缩历史）。
def _find_tail_boundary(messages: list[dict[str, Any]], tail_rounds: int) -> int:
    user_indices: list[int] = []
    for i, m in enumerate(messages):
        if m.get("role") != "user":
            continue
        content = m.get("content")
        # 纯 tool_result 消息不算真实用户轮次
        if isinstance(content, list) and all(
            b.get("type") == "tool_result" for b in content if isinstance(b, dict)
        ):
            continue
        user_indices.append(i)

    if len(user_indices) <= tail_rounds:
        return 0
    return user_indices[-tail_rounds]


# 确定性预处理：在送入 LLM 摘要前降低历史体积，零 LLM 成本。
# 规则：① 错误结果原样保留并加 [ERROR-VERBATIM] 标记；② read_file 读过的代码文件
# 替换为索引 stub（需要时可用 read_file mode=structure/range 低成本重取）；
# ③ 其余超长 tool_result 保留头尾、截断中间。返回深拷贝，不修改原消息。
def _preprocess_for_summary(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    processed = copy.deepcopy(messages)

    # 建立 tool_use_id → (name, input) 映射，供 tool_result 回查来源工具
    tool_uses: dict[str, tuple[str, dict[str, Any]]] = {}
    for msg in processed:
        content = msg.get("content")
        if isinstance(content, list):
            for block in content:
                if block.get("type") == "tool_use":
                    tool_uses[block.get("id", "")] = (
                        str(block.get("name", "")),
                        block.get("input") if isinstance(block.get("input"), dict) else {},
                    )

    for msg in processed:
        content = msg.get("content")
        if not isinstance(content, list):
            continue
        for block in content:
            if block.get("type") != "tool_result":
                continue
            result_content = block.get("content", "")
            if not isinstance(result_content, str):
                result_content = str(result_content)
                block["content"] = result_content

            # ① 错误结果：原样保留，摘要 prompt 会要求逐字抄入 Critical Data
            if block.get("is_error"):
                if not result_content.startswith("[ERROR-VERBATIM]"):
                    block["content"] = f"[ERROR-VERBATIM]\n{result_content}"
                continue

            tool_name, tool_input = tool_uses.get(
                block.get("tool_use_id", ""), ("", {})
            )

            # ② read_file 读过的代码文件：只留索引 stub
            file_path = str(tool_input.get("path", "")) if tool_name == "read_file" else ""
            if file_path and Path(file_path).suffix in _CODE_EXTS:
                lines = result_content.count("\n") + 1
                block["content"] = (
                    f"[File previously read: {file_path} ({lines} lines). "
                    f"Content omitted to save context — use read_file with "
                    f"mode=structure or mode=range to re-fetch what you need.]"
                )
                continue

            # ③ 其余超长结果：保留头尾，截断中间
            if len(result_content) > _TOOL_RESULT_LIMIT:
                omitted = len(result_content) - _HEAD_KEEP - _TAIL_KEEP
                block["content"] = (
                    result_content[:_HEAD_KEEP]
                    + f"\n\n[... {omitted} chars omitted ...]\n\n"
                    + result_content[-_TAIL_KEEP:]
                )

    return processed


# 将消息列表序列化为可供 LLM 阅读的纯文本
def _messages_to_text(messages: list[dict[str, Any]]) -> str:
    parts: list[str] = []
    for msg in messages:
        role = msg.get("role", "unknown").upper()
        content = msg.get("content", "")
        if isinstance(content, str):
            parts.append(f"[{role}]\n{content}")
        elif isinstance(content, list):
            blocks: list[str] = []
            for block in content:
                btype = block.get("type", "")
                if btype == "text":
                    blocks.append(block.get("text", ""))
                elif btype == "tool_use":
                    blocks.append(
                        f"<tool_call name={block.get('name')} id={block.get('id')}>\n"
                        f"{block.get('input', {})}\n</tool_call>"
                    )
                elif btype == "tool_result":
                    blocks.append(
                        f"<tool_result id={block.get('tool_use_id')}>\n"
                        f"{block.get('content', '')}\n</tool_result>"
                    )
            parts.append(f"[{role}]\n" + "\n".join(blocks))
    return "\n\n".join(parts)
