from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from kama_claude.core.bus.events import CostUpdatedEvent
from kama_claude.core.events.bus import EventBus
from kama_claude.core.llm.tokens import compute_cost

logger = logging.getLogger(__name__)


# 返回当前 UTC 时间的 ISO 8601 字符串
def _now() -> str:
    return datetime.now(UTC).isoformat()


# 累计一次 run（或一个 session）的 token 与费用
@dataclass
class CostTracker:
    model: str
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_cache_read_tokens: int = 0
    total_cache_creation_tokens: int = 0
    total_cost_usd: float = 0.0
    call_count: int = 0
    _records: list[dict[str, Any]] = field(default_factory=list)

    # 记账一次 LLM 调用，返回该次调用成本（USD）
    def record(
        self,
        *,
        run_id: str,
        input_tokens: int,
        output_tokens: int,
        cache_read_input_tokens: int = 0,
        cache_creation_input_tokens: int = 0,
    ) -> float:
        cost = compute_cost(
            self.model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cache_read_input_tokens=cache_read_input_tokens,
            cache_creation_input_tokens=cache_creation_input_tokens,
        )
        self.total_input_tokens += input_tokens
        self.total_output_tokens += output_tokens
        self.total_cache_read_tokens += cache_read_input_tokens
        self.total_cache_creation_tokens += cache_creation_input_tokens
        self.total_cost_usd += cost
        self.call_count += 1
        self._records.append({
            "ts": _now(),
            "run_id": run_id,
            "model": self.model,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "cache_read_input_tokens": cache_read_input_tokens,
            "cache_creation_input_tokens": cache_creation_input_tokens,
            "cost_usd": round(cost, 8),
        })
        return cost

    # 将本 tracker 的所有调用记录追加写入 cost.jsonl
    def flush(self, path: Path) -> None:
        if not self._records:
            return
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("a", encoding="utf-8") as f:
                for rec in self._records:
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            self._records.clear()
        except OSError:
            logger.warning("cost tracker: failed to write %s", path)


# 包装任意 LLMProvider：每次调用后记账并发布 CostUpdatedEvent；
# 装饰器不改变被包装 provider 的任何行为与返回值
class TrackingProvider:
    def __init__(
        self,
        inner: Any,
        tracker: CostTracker,
        bus: EventBus,
        *,
        run_id: str,
        session_id: str = "",
    ) -> None:
        self._inner = inner
        self._tracker = tracker
        self._bus = bus
        self._run_id = run_id
        self._session_id = session_id

    async def chat(
        self,
        messages: list[dict[str, object]],
        tool_schemas: list[dict[str, object]],
        bus: EventBus,
        run_id: str,
        *,
        step: int = 0,
        system: str | None = None,
    ) -> Any:
        response = await self._inner.chat(
            messages, tool_schemas, bus, run_id, step=step, system=system
        )
        usage = getattr(response, "usage", None)
        if usage is not None:
            cost = self._tracker.record(
                run_id=run_id,
                input_tokens=usage.input_tokens,
                output_tokens=usage.output_tokens,
                cache_read_input_tokens=getattr(usage, "cache_read_input_tokens", 0),
                cache_creation_input_tokens=getattr(
                    usage, "cache_creation_input_tokens", 0
                ),
            )
            await self._bus.publish(
                CostUpdatedEvent(
                    run_id=run_id,
                    session_id=self._session_id,
                    call_cost_usd=round(cost, 8),
                    cumulative_cost_usd=round(self._tracker.total_cost_usd, 8),
                    total_input_tokens=self._tracker.total_input_tokens,
                    total_output_tokens=self._tracker.total_output_tokens,
                    ts=_now(),
                )
            )
        return response
