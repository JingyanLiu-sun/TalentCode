"""τ-bench 压缩注入适配器：子类化官方 ToolCallingAgent，历史超阈值时
强制调用 KamaClaude Compactor 压缩。

被测对象是压缩器本体：任务环境、工具调用、评分（数据库状态校验）全部
复用 τ-bench 官方实现，仅在消息装配处注入压缩，结果可与社区口径对照。

注意：本模块依赖 τ-bench（可选依赖，不进 pyproject，避免 git 源写入
uv.lock 破坏 `uv sync --frozen`）。未安装时请勿导入本模块，
改用 scripts/run_taubench_eval.py（内部有 ImportError 优雅降级）。
"""
from __future__ import annotations

import asyncio
import time
from pathlib import Path
from typing import Any

from litellm import completion
from tau_bench.agents.tool_calling_agent import ToolCallingAgent, message_to_action
from tau_bench.envs.base import Env
from tau_bench.types import RESPOND_ACTION_NAME, SolveResult

from kama_claude.core.compact.compactor import TAIL_ROUNDS, Compactor
from kama_claude.core.events.bus import EventBus
from kama_claude.core.llm.tokens import count_messages_tokens

SUMMARY_ACK = "Understood, I'll continue from this summary."


# 将 τ-bench 的 OpenAI 格式历史（user/assistant/tool）转为 KamaClaude
# user/assistant 交替格式：两个 user 消息之间的 assistant 工具调用与工具
# 返回合并为一条 assistant 消息，保证角色严格交替（与 Compactor 按
# role==user 计轮的口径一致）。纯函数，可零成本单测。
def openai_history_to_kama(history: list[dict[str, Any]],
                           ) -> list[dict[str, str]]:
    messages: list[dict[str, str]] = []
    user_turn: dict[str, str] | None = None
    assist_parts: list[str] = []

    def _flush() -> None:
        nonlocal user_turn, assist_parts
        if user_turn is not None:
            messages.append(user_turn)
            if assist_parts:
                messages.append({"role": "assistant",
                                 "content": "\n".join(assist_parts)})
        user_turn = None
        assist_parts = []

    for msg in history:
        role = msg.get("role")
        if role == "user":
            _flush()
            user_turn = {"role": "user",
                         "content": str(msg.get("content") or "")}
        elif role == "assistant":
            parts: list[str] = []
            if msg.get("content"):
                parts.append(str(msg["content"]))
            for tc in msg.get("tool_calls") or []:
                fn = tc.get("function") or {}
                parts.append(
                    f"[TOOL CALL {fn.get('name', '?')}] "
                    f"{fn.get('arguments', '')}")
            if parts:
                assist_parts.append("\n".join(parts))
        elif role == "tool":
            assist_parts.append(
                f"[TOOL RESULT {msg.get('name', '?')}]\n"
                f"{msg.get('content', '')}")
    _flush()
    return messages


class CompactingToolCallingAgent(ToolCallingAgent):
    """官方 Agent + 压缩注入：历史 token 超阈值时强制压缩。

    压缩策略与生产 Compactor.compact 同口径：最近 ``tail_turns`` 个
    user 轮次原样保留（原 OpenAI 格式不动，工具调用链不受影响），
    更早的历史转为 KamaClaude 格式做摘要，替换为 [摘要, 确认] 消息对。
    压缩失败（返回 None）时记录失败并冷却若干步，不中断任务。
    """

    def __init__(
        self,
        tools_info: list[dict[str, Any]],
        wiki: str,
        model: str,
        provider: str,
        compact_provider: Any,
        *,
        temperature: float = 0.0,
        compact_token_threshold: int = 6000,
        tail_turns: int = 3,
        min_steps_between_compactions: int = 4,
        compact_dir: Path | None = None,
    ) -> None:
        super().__init__(tools_info, wiki, model, provider,
                         temperature=temperature)
        self.compact_provider = compact_provider
        self.compact_token_threshold = compact_token_threshold
        self.tail_turns = tail_turns
        self.min_steps_between_compactions = min_steps_between_compactions
        self.compact_dir = compact_dir or Path("output/evaluation/taubench-summaries")
        # 每次压缩一条记录（成功/失败均记），供报告统计压缩失败率与延迟
        self.compactions: list[dict[str, Any]] = []

    def _compact_once(self, messages: list[dict[str, Any]], task_index: int,
                      ) -> list[dict[str, Any]] | None:
        """尝试压缩一次；不满足条件或失败时返回 None。"""
        if count_messages_tokens(messages) < self.compact_token_threshold:
            return None  # 历史还不够长，不值得花一次压缩调用
        history = messages[1:]  # messages[0] 是 system wiki，不参与压缩
        user_idxs = [i for i, m in enumerate(history)
                     if m.get("role") == "user"]
        # 适配层保留 tail_turns 轮、压缩器内部还要再保留 TAIL_ROUNDS 轮；
        # 轮数不足以双段保留时静默跳过（不算压缩失败，不进失败率门禁）
        if len(user_idxs) < self.tail_turns + TAIL_ROUNDS + 1:
            return None
        split = user_idxs[-self.tail_turns]
        kama_history = openai_history_to_kama(history[:split])
        if len(kama_history) < 2:
            return None

        compactor = Compactor(
            EventBus(), self.compact_dir, f"taubench-task{task_index}")
        started = time.monotonic()
        error = ""
        try:
            result = asyncio.run(
                compactor.compact_messages(kama_history, self.compact_provider))
        except Exception as exc:  # 失败原因进报告，便于诊断环境问题
            result = None
            error = f"{type(exc).__name__}: {exc}"[:300]
        latency = time.monotonic() - started
        tokens_before = count_messages_tokens(messages)

        if result is None:
            self.compactions.append({
                "failed": True, "task_index": task_index,
                "tokens_before": tokens_before,
                "latency_s": round(latency, 2),
                "error": error or "compactor returned None (LLM 调用失败或空摘要)",
            })
            return None

        compacted = (
            [messages[0],
             {"role": "user", "content": result.summary_text},
             {"role": "assistant", "content": SUMMARY_ACK}]
            + history[split:]
        )
        self.compactions.append({
            "failed": False, "task_index": task_index,
            "tokens_before": tokens_before,
            "tokens_after": count_messages_tokens(compacted),
            "latency_s": round(latency, 2),
            "compacted_turns": len(kama_history),
        })
        return compacted

    def solve(  # type: ignore[override]
        self, env: Env, task_index: int | None = None,
        max_num_steps: int = 30,
    ) -> SolveResult:
        """官方 solve 循环 + 每步装配消息前的压缩注入。"""
        self.compactions = []
        steps_since_compaction = self.min_steps_between_compactions
        total_cost = 0.0
        env_reset_res = env.reset(task_index=task_index)
        obs = env_reset_res.observation
        info = env_reset_res.info.model_dump()
        reward = 0.0
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": self.wiki},
            {"role": "user", "content": obs},
        ]
        for _ in range(max_num_steps):
            steps_since_compaction += 1
            if steps_since_compaction >= self.min_steps_between_compactions:
                compacted = self._compact_once(messages, task_index or 0)
                if compacted is not None:
                    messages = compacted
                    steps_since_compaction = 0
            res = completion(
                messages=messages,
                model=self.model,
                custom_llm_provider=self.provider,
                tools=self.tools_info,
                temperature=self.temperature,
            )
            next_message = res.choices[0].message.model_dump()
            total_cost += res._hidden_params["response_cost"] or 0
            action = message_to_action(next_message)
            env_response = env.step(action)
            reward = env_response.reward
            info = {**info, **env_response.info.model_dump()}
            if action.name != RESPOND_ACTION_NAME:
                next_message["tool_calls"] = next_message["tool_calls"][:1]
                messages.extend(
                    [
                        next_message,
                        {
                            "role": "tool",
                            "tool_call_id": next_message["tool_calls"][0]["id"],
                            "name": next_message["tool_calls"][0]["function"]["name"],
                            "content": env_response.observation,
                        },
                    ]
                )
            else:
                messages.extend(
                    [
                        next_message,
                        {"role": "user",
                         "content": env_response.observation},
                    ]
                )
            if env_response.done:
                break
        return SolveResult(
            reward=reward,
            info=info,
            messages=messages,
            total_cost=total_cost,
        )
