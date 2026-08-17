from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from kama_claude.core.bus.envelope import HandlerError
from kama_claude.core.events.bus import EventBus
from kama_claude.core.runner import RunOutcome
from kama_claude.core.session.manager import SESSION_CLOSED, SESSION_NOT_FOUND, SessionManager
from kama_claude.core.session.model import Session
from kama_claude.core.session.store import SessionStore


class _Runner:
    # 模拟 AgentRunner，将 run 新消息写入 thread 后返回成功
    async def run_and_capture(
        self,
        goal: str,
        *,
        run_id: str | None = None,
        session: Session | None = None,
        store: SessionStore | None = None,
        system_prompt_override: str | None = None,
        tool_whitelist: list[str] | None = None,
    ) -> RunOutcome:
        assert run_id is not None
        assert session is not None
        assert store is not None
        store.append_messages(
            session.id,
            [{"role": "assistant", "content": [{"type": "text", "text": f"done {goal}"}]}],
            run_id,
        )
        return RunOutcome(status="success", result="done", reason=None)


# 功能：验证 create 会创建 active session、写入 meta 并发布 session.created 事件
# 设计：用真实 SessionStore + EventBus 收集事件，覆盖 manager 与 store/bus 的协作边界
async def test_create_session_writes_meta_and_event(tmp_path: Path) -> None:
    events: list[object] = []
    bus = EventBus()

    async def collect(event: object) -> None:
        events.append(event)

    bus.subscribe(collect)
    store = SessionStore(tmp_path)
    manager = SessionManager(store, lambda: _Runner(), bus)  # type: ignore[arg-type]

    session = await manager.create("chat", "title")

    assert session.status == "active"
    assert store.read_meta(session.id).title == "title"
    assert [e.type for e in events] == ["session.created"]  # type: ignore[attr-defined]


# 功能：验证 chat session 处理一条消息后进入 waiting_for_input，并保留 user/assistant thread
# 设计：mock runner 主动追加 assistant 消息，确认 send_message 负责 user 消息、状态流转和 run_id 记录
async def test_send_message_chat_enters_waiting_and_writes_thread(tmp_path: Path) -> None:
    store = SessionStore(tmp_path)
    manager = SessionManager(store, lambda: _Runner(), EventBus())  # type: ignore[arg-type]
    session = await manager.create("chat")

    run_id = await manager.send_message(session.id, "hello")

    loaded = store.read_meta(session.id)
    assert loaded.status == "waiting_for_input"
    assert loaded.run_ids == [run_id]
    messages = store.read_messages(session.id)
    assert messages[0] == {"role": "user", "content": "hello"}
    assert messages[1]["role"] == "assistant"


# 功能：验证 one_shot session 在单次消息完成后自动 closed
# 设计：复用 mock runner 的成功路径，聚焦 mode 对最终状态的影响，保证 kama run 的统一路径正确
async def test_one_shot_auto_closes(tmp_path: Path) -> None:
    store = SessionStore(tmp_path)
    manager = SessionManager(store, lambda: _Runner(), EventBus())  # type: ignore[arg-type]
    session = await manager.create("one_shot")

    await manager.send_message(session.id, "hello")

    assert store.read_meta(session.id).status == "closed"


# 功能：验证不存在的 session_id 返回 session_not_found 错误码
# 设计：直接调用 get_history 的查找路径，断言 HandlerError code，覆盖 IPC handler 可结构化返回错误
async def test_missing_session_raises_handler_error(tmp_path: Path) -> None:
    manager = SessionManager(SessionStore(tmp_path), lambda: _Runner(), EventBus())  # type: ignore[arg-type]
    with pytest.raises(HandlerError) as exc:
        await manager.get_history("missing")
    assert exc.value.code == SESSION_NOT_FOUND


# 功能：验证 closed session 不能继续 send_message
# 设计：先显式 close，再发送消息，断言 session_closed 错误码，覆盖状态机拒绝路径
async def test_closed_session_rejects_message(tmp_path: Path) -> None:
    store = SessionStore(tmp_path)
    manager = SessionManager(store, lambda: _Runner(), EventBus())  # type: ignore[arg-type]
    session = await manager.create("chat")
    await manager.close(session.id)

    with pytest.raises(HandlerError) as exc:
        await manager.send_message(session.id, "again")
    assert exc.value.code == SESSION_CLOSED


# 功能：验证 session 忙时新消息不再报 session_busy，而是持久化入队并在当前 run 后顺序执行
# 设计：首个 run 用 Event 阻塞占住锁，期间发送第二条消息；断言排队消息立即落盘并发布
#       session.message_queued 事件，释放后两个 run 按序完成、状态回到 waiting_for_input
async def test_send_message_while_busy_queues_and_runs_in_order(tmp_path: Path) -> None:
    events: list[object] = []
    bus = EventBus()

    async def collect(event: object) -> None:
        events.append(event)

    bus.subscribe(collect)

    release = asyncio.Event()
    calls: list[str] = []

    class _BlockingRunner:
        first_call = True

        async def run_and_capture(
            self,
            goal: str,
            *,
            run_id: str | None = None,
            session: Session | None = None,
            store: SessionStore | None = None,
            system_prompt_override: str | None = None,
            tool_whitelist: list[str] | None = None,
        ) -> RunOutcome:
            assert run_id is not None and session is not None and store is not None
            calls.append(goal)
            if _BlockingRunner.first_call:
                _BlockingRunner.first_call = False
                await release.wait()
            store.append_messages(
                session.id,
                [{"role": "assistant", "content": [{"type": "text", "text": f"done {goal}"}]}],
                run_id,
            )
            return RunOutcome(status="success", result="done", reason=None)

    store = SessionStore(tmp_path)
    manager = SessionManager(store, _BlockingRunner, bus)  # type: ignore[arg-type]
    session = await manager.create("chat")

    first_task = asyncio.create_task(manager.send_message(session.id, "first"))
    await asyncio.sleep(0.01)  # 等首个 run 拿锁

    queued_task = asyncio.create_task(manager.send_message(session.id, "second"))
    await asyncio.sleep(0.01)

    # 排队消息不报错、不阻塞，且立即持久化并发布引导事件
    assert not queued_task.done()
    messages = store.read_messages(session.id)
    assert [m["content"] for m in messages if m["role"] == "user"] == ["first", "second"]
    types = [e.type for e in events]  # type: ignore[attr-defined]
    assert "session.message_queued" in types

    release.set()
    run1 = await asyncio.wait_for(first_task, timeout=5.0)
    run2 = await asyncio.wait_for(queued_task, timeout=5.0)

    assert calls == ["first", "second"]
    # 排队消息只在入队时落盘一次：drain 执行后 thread 中不出现重复用户消息/重复事件
    final_user = [m["content"] for m in store.read_messages(session.id) if m["role"] == "user"]
    assert final_user == ["first", "second"]
    received = [
        e
        for e in events  # type: ignore[attr-defined]
        if e.type == "session.message_received"
    ]
    assert [e.content for e in received] == ["first", "second"]
    loaded = store.read_meta(session.id)
    assert loaded.status == "waiting_for_input"
    assert loaded.run_ids == [run1, run2]
