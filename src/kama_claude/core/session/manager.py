from __future__ import annotations

import asyncio
import uuid
from collections.abc import Callable
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

from kama_claude.core.bus.envelope import HandlerError
from kama_claude.core.bus.events import (
    SessionClosedEvent,
    SessionCreatedEvent,
    SessionMessageQueuedEvent,
    SessionMessageReceivedEvent,
    SessionResumedEvent,
    SessionWaitingForInputEvent,
    SkillInvokedEvent,
)
from kama_claude.core.events.bus import EventBus
from kama_claude.core.runs import new_run_id
from kama_claude.core.session.model import Session, SessionMode
from kama_claude.core.session.store import SessionStore
from kama_claude.core.skills.loader import SkillLoader

if TYPE_CHECKING:
    from kama_claude.core.llm.base import LLMProvider
    from kama_claude.core.runner import AgentRunner

SESSION_NOT_FOUND = -32010
SESSION_CLOSED = -32011
SESSION_BUSY = -32012


# 返回当前 UTC 时间的 ISO 8601 字符串
def _now() -> str:
    return datetime.now(UTC).isoformat()


class SessionManager:
    # 初始化会话管理器，接入文件存储、runner 工厂、事件总线和可选的 LLM provider（用于手动压缩）
    def __init__(
        self,
        store: SessionStore,
        runner_factory: Callable[[], AgentRunner],
        bus: EventBus,
        provider: LLMProvider | None = None,
    ) -> None:
        self._store = store
        self._runner_factory = runner_factory
        self._bus = bus
        self._provider = provider
        self._sessions: dict[str, Session] = {}
        self._locks: dict[str, asyncio.Lock] = {}
        # 忙时排队待执行的消息：sid -> (content, run_id, 完成信号, 已持久化标志) 队列
        self._queues: dict[
            str, asyncio.Queue[tuple[str, str, asyncio.Future[None], bool]]
        ] = {}
        self._skill_loader = SkillLoader()

    # 创建新 session 并写入 meta.json
    async def create(self, mode: SessionMode, title: str = "") -> Session:
        sid = f"sess-{uuid.uuid4().hex[:12]}"
        ts = _now()
        session = Session(
            id=sid,
            mode=mode,
            status="active",
            title=title,
            created_at=ts,
            updated_at=ts,
            run_ids=[],
        )
        self._sessions[sid] = session
        self._locks[sid] = asyncio.Lock()
        self._queues[sid] = asyncio.Queue()
        self._store.write_meta(session)
        await self._bus.publish(SessionCreatedEvent(session_id=sid, mode=mode, ts=ts))
        return session

    # session 已关闭时抛 HandlerError；供入口检查与持锁后的二次检查复用
    def _ensure_open(self, session: Session) -> None:
        if session.status == "closed":
            raise HandlerError(SESSION_CLOSED, "session already closed")

    # 处理用户消息，追加 thread 并启动一次 agent run；
    # session 忙时不再拒绝：消息立即持久化并入队，待当前 run 结束后自动依次执行
    async def send_message(self, sid: str, content: str, *, run_id: str | None = None) -> str:
        session = self._get_session(sid)
        self._ensure_open(session)
        lock = self._locks[sid]
        run_id = run_id or new_run_id()

        if lock.locked():
            # 忙时排队：先持久化并发布，使客户端立刻看到消息，再登记到队列等待执行
            self._store.append_message(sid, "user", content)
            await self._bus.publish(
                SessionMessageReceivedEvent(session_id=sid, content=content, ts=_now())
            )
            if not session.title:
                session.title = content[:40]
                self._store.write_meta(session)
            queue = self._queues[sid]
            done: asyncio.Future[None] = asyncio.get_running_loop().create_future()
            # 调用方可能提前取消，避免该 future 的异常无人认领触发告警
            done.add_done_callback(lambda f: f.exception() if not f.cancelled() else None)
            # 消息已在上方落盘，标志已持久化，drain 执行时不再重复写入/发布
            queue.put_nowait((content, run_id, done, True))
            await self._bus.publish(
                SessionMessageQueuedEvent(
                    session_id=sid,
                    run_id=run_id,
                    position=queue.qsize(),
                    ts=_now(),
                )
            )
            asyncio.create_task(self._drain_queue(sid))
            await done
            return run_id

        async with lock:
            self._ensure_open(session)

            if session.status == "waiting_for_input":
                await self._bus.publish(SessionResumedEvent(session_id=sid, ts=_now()))

            await self._execute_message(sid, session, content, run_id)
            await self._transition(sid, session)
            return run_id

    # 依次消费队列中的排队消息，持锁串行执行，保证同一 session 同时只有一个 run；
    # 并发产生的多个 drain task 会因锁与空队列检查自然串行化
    async def _drain_queue(self, sid: str) -> None:
        session = self._sessions.get(sid)
        if session is None:
            return
        async with self._locks[sid]:
            queue = self._queues[sid]
            while not queue.empty():
                content, run_id, done, persisted = queue.get_nowait()
                try:
                    if session.status == "closed":
                        raise HandlerError(SESSION_CLOSED, "session already closed")
                    if session.status == "waiting_for_input":
                        await self._bus.publish(SessionResumedEvent(session_id=sid, ts=_now()))
                    await self._execute_message(
                        sid, session, content, run_id, persist=not persisted
                    )
                    await self._transition(sid, session)
                except BaseException as exc:
                    if not done.done():
                        done.set_exception(exc)
                else:
                    if not done.done():
                        done.set_result(None)

    # run 结束后的状态流转：队列还有待执行消息时保持 active，否则进入 waiting_for_input/closed
    async def _transition(self, sid: str, session: Session) -> None:
        session.updated_at = _now()
        if self._queues[sid].empty():
            if session.mode == "one_shot":
                session.status = "closed"
                await self._bus.publish(SessionClosedEvent(session_id=sid, ts=session.updated_at))
            else:
                session.status = "waiting_for_input"
                await self._bus.publish(
                    SessionWaitingForInputEvent(
                        session_id=sid,
                        last_run_id=session.run_ids[-1],
                        ts=session.updated_at,
                    )
                )
        self._store.write_meta(session)

    # 执行单条消息对应的 agent run：追加 user 消息、解析 Skill 并调用 runner；
    # persist=False 用于忙时已落盘的排队消息，避免 thread.jsonl 与事件重复写入
    async def _execute_message(
        self, sid: str, session: Session, content: str, run_id: str, *, persist: bool = True
    ) -> None:
        if persist:
            self._store.append_message(sid, "user", content)
            await self._bus.publish(
                SessionMessageReceivedEvent(session_id=sid, content=content, ts=_now())
            )

        if not session.title:
            session.title = content[:40]

        session.run_ids.append(run_id)
        session.updated_at = _now()
        self._store.write_meta(session)

        # Skill 解析：检测 "/" 前缀，展开为系统提示覆盖和工具白名单
        goal = content
        system_prompt_override: str | None = None
        tool_whitelist: list[str] | None = None
        if content.startswith("/"):
            parts = content[1:].split(None, 1)
            skill_name = parts[0]
            arguments = parts[1] if len(parts) > 1 else ""
            skill = self._skill_loader.resolve(skill_name)
            if skill is not None:
                goal = self._skill_loader.render_prompt(skill, arguments)
                system_prompt_override = skill.system_prompt_template
                tool_whitelist = skill.allowed_tools or None
                await self._bus.publish(
                    SkillInvokedEvent(
                        skill_name=skill_name,
                        arguments=arguments,
                        run_id=run_id,
                        ts=_now(),
                    )
                )

        runner = self._runner_factory()
        await runner.run_and_capture(
            goal,
            run_id=run_id,
            session=session,
            store=self._store,
            system_prompt_override=system_prompt_override,
            tool_whitelist=tool_whitelist,
        )

    # 关闭指定 session 并更新 meta.json
    async def close(self, sid: str) -> None:
        session = self._get_session(sid)
        lock = self._locks[sid]
        if lock.locked():
            raise HandlerError(SESSION_BUSY, "session busy")
        async with lock:
            session.status = "closed"
            session.updated_at = _now()
            self._store.write_meta(session)
            await self._bus.publish(SessionClosedEvent(session_id=sid, ts=session.updated_at))

    # 手动压缩指定 session 的 thread，将摘要持久化写入 thread.jsonl
    async def compact(self, sid: str, focus: str = "") -> Any:
        self._get_session(sid)
        lock = self._locks[sid]
        if lock.locked():
            raise HandlerError(SESSION_BUSY, "session busy")
        if self._provider is None:
            raise HandlerError(-32020, "provider not available for compaction")
        async with lock:
            from kama_claude.core.bus.commands import SessionCompactResult
            from kama_claude.core.compact.compactor import Compactor
            messages = self._store.read_messages(sid)
            session_dir = self._store.session_dir(sid)
            compactor = Compactor(self._bus, session_dir, sid)
            result = await compactor.compact_messages(messages, self._provider, focus=focus)
            if result is None:
                raise HandlerError(-32021, "compaction failed or not beneficial")
            compacted = [
                {"role": "user", "content": result.summary_text},
                {"role": "assistant", "content": "Understood, I'll continue from this summary."},
            ] + (result.tail_messages or [])
            self._store.write_compacted(sid, compacted)
            return SessionCompactResult(
                summary_tokens=result.summary_tokens,
                saved_tokens=max(0, result.original_token_estimate - result.summary_tokens),
            )

    # 读取指定 session 的完整 thread 历史
    async def get_history(self, sid: str) -> list[dict[str, Any]]:
        self._get_session(sid)
        return self._store.read_messages(sid)

    # 从内存索引取 session，不存在时抛 JSON-RPC 结构化错误
    def _get_session(self, sid: str) -> Session:
        session = self._sessions.get(sid)
        if session is None:
            raise HandlerError(SESSION_NOT_FOUND, "session not found")
        return session
