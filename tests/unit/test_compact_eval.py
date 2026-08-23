# Tier1（scripts/run_compact_eval.py + evaluation/compact/generate_cases.py）
# 的回归测试：固化数据集结构与离线门禁口径
#   - 生成器完全确定性（无随机源）
#   - 所有预埋事实必须落在 history 区（最近 3 轮之外）
#   - 字面事实经过确定性预处理后保留率 = 100%（离线门禁）
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluation.compact.generate_cases import (  # noqa: E402
    DEFAULT_OUT,
    FACT_TYPES,
    build_case_messages,
    depth_bucket,
    generate_cases,
    load_cases,
)

from kama_claude.core.compact.compactor import (  # noqa: E402
    TAIL_ROUNDS,
    _find_tail_boundary,
)


# 以文件路径加载 scripts 下的 harness 模块（scripts 不是包）
def _load_harness() -> Any:
    path = ROOT / "scripts" / "run_compact_eval.py"
    spec = importlib.util.spec_from_file_location("run_compact_eval", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["run_compact_eval"] = module
    spec.loader.exec_module(module)
    return module


_HARNESS = _load_harness()
_CASES = generate_cases()


def test_generator_is_deterministic() -> None:
    assert generate_cases() == _CASES


def test_committed_dataset_in_sync_with_generator() -> None:
    # 入库数据集必须由当前生成器产出（sha 口径一致），否则报告不可复现
    assert DEFAULT_OUT.exists(), "先运行 uv run python -m evaluation.compact.generate_cases"
    assert load_cases(DEFAULT_OUT) == _CASES


def test_dataset_shape() -> None:
    assert len(_CASES) == 24
    for case in _CASES:
        assert len(case["planted_facts"]) == len(FACT_TYPES)
        rounds = [f["round"] for f in case["planted_facts"]]
        assert len(set(rounds)) == len(rounds), "六类事实轮次必须互异"
        literals = [f for f in case["planted_facts"] if f["check"] == "literal"]
        semantics = [f for f in case["planted_facts"] if f["check"] == "semantic"]
        assert len(literals) == 3 and len(semantics) == 3


def test_all_facts_planted_in_history_region() -> None:
    for case in _CASES:
        messages = build_case_messages(case)
        split = _find_tail_boundary(messages, tail_rounds=TAIL_ROUNDS)
        assert split > 0, f"{case['id']}: 历史不足，无法压缩"
        for f in case["planted_facts"]:
            # 每轮固定 3 条消息，第 r 轮真实用户消息索引 = 3*(r-1)
            assert 3 * (f["round"] - 1) < split, (
                f"{case['id']}: 事实 {f['key']} 落在 tail 区，评测无效"
            )


def test_depth_buckets_covered_across_dataset() -> None:
    buckets = {f["depth"] for c in _CASES for f in c["planted_facts"]}
    assert buckets == {"early", "mid", "deep"}
    assert depth_bucket(0.1) == "early"
    assert depth_bucket(0.5) == "mid"
    assert depth_bucket(0.9) == "deep"


def test_offline_literal_retention_gate_is_100_percent() -> None:
    # 离线门禁口径：确定性预处理不允许丢任何字面事实
    for case in _CASES:
        row = _HARNESS.evaluate_case_offline(case)
        assert row["facts_in_history"]
        assert row["literal_hits"] == row["literal_total"], (
            f"{case['id']}: 预处理丢了字面事实"
        )
        assert 0.0 < row["preprocess_ratio"] < 0.5, (
            f"{case['id']}: 预处理压缩比异常 {row['preprocess_ratio']}"
        )
