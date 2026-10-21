"""LoCoMo → KamaClaude 消息格式适配（纯函数，零成本，单测覆盖）。

LoCoMo（Snap Research, ACL 2024, snap-research/locomo）：10 段人工校对的长对话
+ 带金标答案的 QA。本模块负责：
- 把 `conversation`（session 字典，含 session_N / session_N_date_time 键）
  展平为 KamaClaude 消息列表（严格 user/assistant 交替）；
- 建立 `dia_id`（如 "D2:8"）→ 消息索引的映射，用于埋点位置校验；
- 从 `qa` 中切出固定子集（单跳/时序/多跳三类可答题），并校验证据全部
  落在 history 区（最近 TAIL_ROUNDS 轮之外）——与合成集埋点校验同口径。

不采纳 QA 类别 4（开放域，依赖外部常识）与 5（对抗，答案不在对话中）。
"""
from __future__ import annotations

import re
from typing import Any

from kama_claude.core.compact.compactor import TAIL_ROUNDS, _find_tail_boundary

# 可答题类别：1 单跳 / 2 时序 / 3 多跳（答案可由对话内容推出）
ANSWERABLE_CATEGORIES = (1, 2, 3)
CATEGORY_NAMES = {
    1: "single-hop",
    2: "temporal",
    3: "multi-hop",
    4: "open-domain",
    5: "adversarial",
}

DEFAULT_MAX_PER_CATEGORY = 15
DEFAULT_CONV_INDICES = (0, 3)


# 按编号升序返回对话中的 session 键（排除 session_N_date_time 元数据键）
def session_keys(conv: dict[str, Any]) -> list[str]:
    keys = [k for k in conv
            if k.startswith("session_") and not k.endswith("_date_time")]
    return sorted(keys, key=lambda k: int(k.split("_")[1]))


# 展平为 KamaClaude 消息列表。返回 (messages, dia_id→消息索引映射)。
# 规则：首个发言者固定映射为 user、另一人映射为 assistant；连续同发言者
# 轮次合并为一条消息（Anthropic 要求严格角色交替）；每个 session 的时间
# 戳作为标记行写进该 session 第一条消息（供时序类问题使用）。
def build_messages(conv: dict[str, Any]) -> tuple[list[dict[str, Any]],
                                                   dict[str, int]]:
    messages: list[dict[str, Any]] = []
    dia_index: dict[str, int] = {}
    role_of: dict[str, str] = {}

    for key in session_keys(conv):
        session = conv[key]
        date = conv.get(f"{key}_date_time", "")
        first_of_session = True
        for turn in session:
            speaker = str(turn.get("speaker", ""))
            text = str(turn.get("text", "")).strip()
            if not speaker or not text:
                continue
            if speaker not in role_of:
                role_of[speaker] = "user" if len(role_of) == 0 else "assistant"
            role = role_of[speaker]
            if first_of_session and date:
                text = f"[Session {key.removeprefix('session_')} | {date}]\n{text}"
                first_of_session = False

            dia_id = str(turn.get("dia_id", ""))
            if messages and messages[-1]["role"] == role:
                messages[-1]["content"] += f"\n{text}"
            else:
                messages.append({"role": role, "content": text})
            if dia_id:
                dia_index[dia_id] = len(messages) - 1

    return messages, dia_index


# 校验证据是否全部落在 history 区（必须经过摘要才能幸存，与合成集同口径）
def evidence_in_history(messages: list[dict[str, Any]],
                        dia_index: dict[str, int],
                        evidence: list[str],
                        tail_rounds: int = TAIL_ROUNDS) -> bool:
    split = _find_tail_boundary(messages, tail_rounds=tail_rounds)
    indices = [dia_index[e] for e in evidence if e in dia_index]
    return bool(indices) and all(idx < split for idx in indices)


# 从 qa 列表切出固定子集：仅可答类别、按原始顺序取前 max_per_category 条
# （确定性切片，配合数据集哈希锁死，防止针对子集调优）
def select_qa(qa_items: list[dict[str, Any]],
              max_per_category: int = DEFAULT_MAX_PER_CATEGORY,
              ) -> list[dict[str, Any]]:
    taken: dict[int, int] = {}
    subset: list[dict[str, Any]] = []
    for q in qa_items:
        cat = q.get("category")
        if cat not in ANSWERABLE_CATEGORIES:
            continue
        if taken.get(cat, 0) >= max_per_category:
            continue
        if not q.get("answer") or not q.get("evidence"):
            continue
        taken[cat] = taken.get(cat, 0) + 1
        subset.append(q)
    return subset


# 由完整数据集构造固定子集（供 prepare_datasets 与单测共用，纯函数）。
# 返回 (conv_records, qa_records)：前者每段对话一条（含消息与 dia 映射），
# 后者每道题一条，仅保留证据全部落在 history 区的题目。
def build_locomo_subset(raw: list[dict[str, Any]],
                        conv_indices: tuple[int, ...] = DEFAULT_CONV_INDICES,
                        max_per_category: int = DEFAULT_MAX_PER_CATEGORY,
                        ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    conv_records: list[dict[str, Any]] = []
    qa_records: list[dict[str, Any]] = []
    for idx in conv_indices:
        record = raw[idx]
        conv = record["conversation"]
        messages, dia_index = build_messages(conv)
        conv_id = f"locomo-c{idx}"
        conv_records.append({
            "conv_id": conv_id,
            "sample_id": record.get("sample_id", idx),
            "messages": messages,
            "dia_index": dia_index,
        })
        # 注意：qa 挂在对话记录顶层，不在 conversation 字典内
        for i, q in enumerate(select_qa(record.get("qa", []), max_per_category)):
            if not evidence_in_history(messages, dia_index, q["evidence"]):
                continue  # 证据落在 tail 区，未经压缩也能答——评测无效，剔除
            qa_records.append({
                "id": f"{conv_id}-q{i:03d}",
                "conv_id": conv_id,
                "question": q["question"],
                "answer": q["answer"],
                "category": q["category"],
                "category_name": CATEGORY_NAMES[q["category"]],
                "evidence": q["evidence"],
            })
    return conv_records, qa_records


# ---- 确定性评分：SQuAD 式归一化 + EM / token F1 ----

_PUNCT_RE = re.compile(r"[^a-z0-9\s]")
_ARTICLES_RE = re.compile(r"\b(a|an|the)\b")
_WS_RE = re.compile(r"\s+")


def normalize_answer(text: str) -> str:
    text = text.lower()
    text = _PUNCT_RE.sub(" ", text)
    text = _ARTICLES_RE.sub(" ", text)
    return _WS_RE.sub(" ", text).strip()


def exact_match(pred: str, gold: str) -> bool:
    return normalize_answer(pred) == normalize_answer(gold)


# token 级 F1；金标或预测为空时记 0
def token_f1(pred: str, gold: str) -> float:
    pred_toks, gold_toks = normalize_answer(pred).split(), normalize_answer(gold).split()
    if not pred_toks or not gold_toks:
        return 0.0
    common = sum(min(pred_toks.count(t), gold_toks.count(t)) for t in set(pred_toks))
    if common == 0:
        return 0.0
    precision = common / len(pred_toks)
    recall = common / len(gold_toks)
    return 2 * precision * recall / (precision + recall)


# 确定性判定：EM 命中，或 F1 >= 阈值（容忍日期格式差异等表面变体）
def deterministic_pass(pred: str, gold: str, f1_threshold: float = 0.5) -> bool:
    return exact_match(pred, gold) or token_f1(pred, gold) >= f1_threshold


# ---- Cohen's kappa（二分类，裁判校准用）----

# 返回两组二分类判定间的 Cohen's kappa；样本不足或期望一致率退化时返回 0.0
def cohens_kappa(labels_a: list[bool], labels_b: list[bool]) -> float:
    if len(labels_a) != len(labels_b) or len(labels_a) < 10:
        return 0.0
    n = len(labels_a)
    agree = sum(1 for a, b in zip(labels_a, labels_b) if a == b)
    po = agree / n
    pa_true = sum(labels_a) / n
    pb_true = sum(labels_b) / n
    pe = pa_true * pb_true + (1 - pa_true) * (1 - pb_true)
    if pe >= 1.0:
        return 0.0
    return (po - pe) / (1 - pe)
