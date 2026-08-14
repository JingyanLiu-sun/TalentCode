from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

# tiktoken 是 OpenAI 分词器；Claude 无官方开源 tokenizer，这里用 cl100k_base
# 做离线近似计数（误差通常 <15%），精确值以 API usage 返回为准。
# 加载失败（缺依赖/离线无 BPE 文件）时回退到 字符数/4 的粗略估算。
try:
    import tiktoken

    _ENCODING: tiktoken.Encoding | None = tiktoken.get_encoding("cl100k_base")
    TOKENIZER_NAME = "cl100k_base"
except Exception:  # noqa: BLE001 — 任何加载失败都降级为粗略估算
    _ENCODING = None
    TOKENIZER_NAME = "char_div_4_fallback"
    logger.warning("tiktoken unavailable, falling back to chars/4 token estimate")


# 用真实 tokenizer 统计文本 token 数；不可用时回退 字符数/4
def count_tokens(text: str) -> int:
    if _ENCODING is not None:
        return len(_ENCODING.encode(text))
    return len(text) // 4


# 用真实 tokenizer 统计消息列表 token 数（含 role 与结构开销的粗略补偿：每消息 +4）
def count_messages_tokens(messages: list[dict[str, Any]]) -> int:
    total = 0
    for msg in messages:
        content = msg.get("content", "")
        if isinstance(content, str):
            total += count_tokens(content)
        elif isinstance(content, list):
            for block in content:
                if not isinstance(block, dict):
                    continue
                btype = block.get("type", "")
                if btype == "text":
                    total += count_tokens(str(block.get("text", "")))
                elif btype == "tool_use":
                    total += count_tokens(str(block.get("name", "")))
                    total += count_tokens(str(block.get("input", "")))
                elif btype == "tool_result":
                    total += count_tokens(str(block.get("content", "")))
        total += 4  # role/分隔符等结构开销近似
    return total


# 每百万 token 单价（USD）：(普通输入, 输出, 缓存读, 缓存写)
# 缓存读按输入价 10%、缓存写按输入价 125%（Anthropic 定价模型）
_MODEL_PRICING: dict[str, tuple[float, float]] = {
    "claude-sonnet-4-6": (3.0, 15.0),
    "claude-haiku-4-5-20251001": (1.0, 5.0),
    "claude-opus-4-7": (15.0, 75.0),
}
_DEFAULT_PRICING = (3.0, 15.0)
_CACHE_READ_RATIO = 0.1
_CACHE_WRITE_RATIO = 1.25


# 按 Anthropic 定价模型计算单次调用成本（USD）：
# input 按普通输入价，cache_read 按 10%，cache_creation 按 125%
def compute_cost(
    model: str,
    *,
    input_tokens: int,
    output_tokens: int,
    cache_read_input_tokens: int = 0,
    cache_creation_input_tokens: int = 0,
) -> float:
    base = _MODEL_PRICING.get(model, _DEFAULT_PRICING)
    in_price, out_price = base[0] / 1_000_000, base[1] / 1_000_000
    cost = input_tokens * in_price + output_tokens * out_price
    cost += cache_read_input_tokens * in_price * _CACHE_READ_RATIO
    cost += cache_creation_input_tokens * in_price * _CACHE_WRITE_RATIO
    return cost


# 返回模型是否已配置精确价目（未知模型按默认价估算）
def has_pricing(model: str) -> bool:
    return model in _MODEL_PRICING
