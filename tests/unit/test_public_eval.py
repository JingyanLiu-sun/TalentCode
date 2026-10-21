"""Tier0 公开数据集评测的零成本单测：适配转换、埋点校验、评分口径、子集确定性。

不触网、不调 LLM：只验证 evaluation/public 下的纯函数与
scripts/run_public_eval.py 的离线校验逻辑。
"""
from __future__ import annotations

import json
from typing import Any

from evaluation.public.locomo_adapter import (
    ANSWERABLE_CATEGORIES,
    build_locomo_subset,
    build_messages,
    cohens_kappa,
    deterministic_pass,
    evidence_in_history,
    exact_match,
    normalize_answer,
    select_qa,
    token_f1,
)


# 构造 LoCoMo 风格的合成对话：n 轮交替 + 尾部连续同发言者（测合并规则）
def _fake_conv(n_turns: int = 20) -> dict[str, Any]:
    turns = []
    for i in range(1, n_turns + 1):
        speaker = "Alice" if i % 2 == 1 else "Bob"
        turns.append({"speaker": speaker, "dia_id": f"D1:{i}",
                      "text": f"turn {i} content"})
    # 末尾追加一条与上一轮同发言者的消息，验证合并
    turns.append({"speaker": "Bob", "dia_id": f"D1:{n_turns + 1}",
                  "text": "extra tail turn"})
    return {
        "session_1": turns,
        "session_1_date_time": "10:00 am on 1 January, 2023",
        "session_2": [{"speaker": "Alice", "dia_id": "D2:1", "text": "later talk"}],
        "session_2_date_time": "10:00 am on 2 January, 2023",
    }


def _fake_qa() -> list[dict[str, Any]]:
    return [
        # 单跳，证据在早期（history 区）
        {"question": "q1", "answer": "a1", "evidence": ["D1:2"], "category": 1},
        # 时序，证据也在早期
        {"question": "q2", "answer": "a2", "evidence": ["D1:4"], "category": 2},
        # 多跳，跨两个证据点
        {"question": "q3", "answer": "a3", "evidence": ["D1:2", "D1:6"],
         "category": 3},
        # 证据在尾部（应被剔除）
        {"question": "q4", "answer": "a4", "evidence": ["D1:21"], "category": 1},
        # 不可答类别（应被过滤）
        {"question": "q5", "answer": "a5", "evidence": ["D1:2"], "category": 4},
        {"question": "q6", "adversarial_answer": "x", "evidence": ["D1:2"],
         "category": 5},
        # 缺答案（应被过滤）
        {"question": "q7", "answer": "", "evidence": ["D1:2"], "category": 1},
    ]


# ---- 格式转换 ----

def test_build_messages_alternates_and_merges() -> None:
    messages, dia_index = build_messages(_fake_conv())
    assert messages[0]["role"] == "user"
    assert all(messages[i]["role"] != messages[i + 1]["role"]
               for i in range(len(messages) - 1))
    # 尾部两条 Bob 消息合并为一条（位于 session_2 之前）
    merged = [m for m in messages if "extra tail turn" in m["content"]]
    assert len(merged) == 1
    assert merged[0]["role"] == "assistant"
    assert "turn 20" in merged[0]["content"]
    # session 时间戳标记进入该 session 首条消息；末条为 session_2 的 Alice 消息
    assert "[Session 1 |" in messages[0]["content"]
    assert messages[-1]["role"] == "user"
    assert "[Session 2 |" in messages[-1]["content"]
    # dia_id 映射全覆盖且指向合法索引
    assert dia_index["D1:1"] == 0
    assert all(0 <= idx < len(messages) for idx in dia_index.values())


def test_build_messages_skips_empty_turns() -> None:
    conv = {"session_1": [
        {"speaker": "Alice", "dia_id": "D1:1", "text": "hello"},
        {"speaker": "Bob", "dia_id": "D1:2", "text": ""},
        {"speaker": "Bob", "dia_id": "D1:3", "text": "world"},
    ]}
    messages, dia_index = build_messages(conv)
    assert len(messages) == 2
    assert "D1:2" not in dia_index


# ---- 埋点位置校验 ----

def test_evidence_in_history_boundary() -> None:
    messages, dia_index = build_messages(_fake_conv())
    assert evidence_in_history(messages, dia_index, ["D1:2"])
    # 最后一条消息必然在 tail 区
    assert not evidence_in_history(messages, dia_index, ["D1:21"])
    # 未知 dia_id 视为无效（不放行）
    assert not evidence_in_history(messages, dia_index, ["D9:99"])


# ---- 子集切片 ----

def test_select_qa_filters_and_caps() -> None:
    qa = _fake_qa()
    subset = select_qa(qa, max_per_category=1)
    # 每类最多 1 条，且只含可答类别
    assert len(subset) == 3
    assert all(q["category"] in ANSWERABLE_CATEGORIES for q in subset)
    # 无答案条目被过滤：单跳只剩 q1（q7 空答案、q4 因上限被跳过前先判定）
    assert all(q.get("answer") for q in subset)


def test_build_locomo_subset_drops_tail_evidence() -> None:
    raw = [{"sample_id": "s0", "conversation": _fake_conv(), "qa": _fake_qa()},
           {"sample_id": "s3", "conversation": _fake_conv(), "qa": _fake_qa()}]
    convs, qa = build_locomo_subset(raw, conv_indices=(0, 1), max_per_category=15)
    assert [c["conv_id"] for c in convs] == ["locomo-c0", "locomo-c1"]
    ids = {q["id"] for q in qa}
    # 证据在尾部的题被剔除；不可答类别与空答案被过滤
    assert len(ids) == len(qa) == 6  # 2 段对话 × 3 道可答题
    for q in qa:
        assert q["category"] in ANSWERABLE_CATEGORIES
    # 切片逐字节确定性：同样输入必然产生同样输出
    convs2, qa2 = build_locomo_subset(raw, conv_indices=(0, 1), max_per_category=15)
    assert json.dumps(qa, sort_keys=True) == json.dumps(qa2, sort_keys=True)
    assert json.dumps(convs, sort_keys=True) == json.dumps(convs2, sort_keys=True)


# ---- 确定性评分口径 ----

def test_normalize_and_exact_match() -> None:
    assert normalize_answer("The 7 May, 2023!") == "7 may 2023"
    assert exact_match("7 May 2023", "the 7 may, 2023.")
    assert not exact_match("7 May 2023", "8 May 2023")


def test_token_f1_and_deterministic_pass() -> None:
    assert token_f1("adoption agencies", "Adoption agencies") == 1.0
    assert 0.0 < token_f1("adoption something", "adoption agencies") < 1.0
    assert token_f1("", "x") == 0.0
    # F1 阈值容忍表面变体
    assert deterministic_pass("She researched adoption agencies", "Adoption agencies")
    assert not deterministic_pass("I don't know", "Adoption agencies")


# ---- Cohen's kappa ----

def test_cohens_kappa_perfect_and_degenerate() -> None:
    a = [True, True, False, False, True, False, True, True, False, False]
    assert cohens_kappa(a, list(a)) == 1.0
    # 样本不足返回 0.0（避免小样本假高一致）
    assert cohens_kappa([True, False], [True, False]) == 0.0
    # 完全相反判定为负值
    b = [not x for x in a]
    assert cohens_kappa(a, b) < 0
