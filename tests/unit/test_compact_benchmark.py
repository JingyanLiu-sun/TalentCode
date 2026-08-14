# scripts/benchmark_compact.py 离线指标的回归测试：
# 保证确定性预处理的质量底线（错误保留、stub 替换、tail 边界）不被改坏
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any


# 以文件路径加载 scripts 下的 benchmark 模块（scripts 不是包）
def _load_benchmark() -> Any:
    root = Path(__file__).resolve().parents[2]
    path = root / "scripts" / "benchmark_compact.py"
    spec = importlib.util.spec_from_file_location("benchmark_compact", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["benchmark_compact"] = module
    spec.loader.exec_module(module)
    return module


_BENCH = _load_benchmark()


def test_fixture_has_enough_history() -> None:
    messages = _BENCH.build_fixture()
    off = _BENCH.measure_offline(messages)
    # 4 轮真实对话 > TAIL_ROUNDS(3)，存在可压缩历史
    assert off["real_user_rounds"] == 4
    assert off["tail_boundary_valid"]


def test_offline_metrics_quality_floor() -> None:
    messages = _BENCH.build_fixture()
    off = _BENCH.measure_offline(messages)
    # 错误块必须逐字保留；两个 read_file 代码文件都被 stub 替换
    assert off["error_preserved"] is True
    assert off["stub_replaced"] == off["stub_expected"]
    # 预处理后体积显著下降（超长 tool_result 被截断/stub 化）
    assert off["preprocess_ratio"] < 0.5


def test_planted_facts_present_in_fixture() -> None:
    text = str(_BENCH.build_fixture()).lower()
    for fact in _BENCH.PLANTED_FACTS.values():
        assert fact.lower() in text
