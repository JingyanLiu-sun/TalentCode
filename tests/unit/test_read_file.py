from __future__ import annotations

from pathlib import Path

import pytest

from kama_claude.core.tools.builtin.read_file import ReadFileTool


# 功能：验证读取存在的文件时返回完整内容且 is_error 为 False
# 设计：写临时文件后读取，断言 content 和 is_error，覆盖正常路径（happy path）
async def test_read_existing_file(tmp_path: Path) -> None:
    f = tmp_path / "hello.txt"
    f.write_text("hello world", encoding="utf-8")
    result = await ReadFileTool().invoke({"path": str(f)})
    assert not result.is_error
    assert result.content == "hello world"


# 功能：验证文件不存在时抛出 FileNotFoundError 而非返回错误 ToolResult
# 设计：传入不存在的路径，确认 ReadFileTool 不吞掉异常，让调用方（invoke_tool）负责错误分类和事件发布
async def test_file_not_found_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        await ReadFileTool().invoke({"path": str(tmp_path / "missing.txt")})


# 功能：验证包含 `..` 的路径被拒绝并抛出 PermissionError
# 设计：传入 `"../secret.txt"` 这种最典型的目录遍历形式，确认安全边界第一道防线有效
async def test_path_traversal_dotdot_raises() -> None:
    with pytest.raises(PermissionError):
        await ReadFileTool().invoke({"path": "../secret.txt"})


# 功能：验证多级路径中嵌入的 `..` 经过路径规范化后也被正确检测
# 设计：使用 `"subdir/../../etc/passwd"` 测试路径 resolve 后的深度遍历，确认单层 `..` 过滤不足以覆盖此情况
async def test_path_traversal_nested_raises() -> None:
    with pytest.raises(PermissionError):
        await ReadFileTool().invoke({"path": "subdir/../../etc/passwd"})


# 功能：验证超过 512KB 的文件被截断并在末尾追加 [truncated] 标记
# 设计：写 600KB 文件，断言内容以 x×512KB 开头、以 [truncated] 结尾，确认截断不破坏前缀内容
async def test_truncation_over_512kb(tmp_path: Path) -> None:
    f = tmp_path / "big.txt"
    f.write_bytes(b"x" * (600 * 1024))
    result = await ReadFileTool().invoke({"path": str(f)})
    assert not result.is_error
    assert result.content.endswith("[truncated]")
    # Actual text content is exactly 512KB worth of 'x' chars
    assert result.content.startswith("x" * (512 * 1024))


# 功能：验证恰好等于 512KB 的文件不被截断（边界值：超过而非大于等于）
# 设计：boundary check，确认截断阈值为"严格超过 512KB"，防止 off-by-one 错误
async def test_exact_512kb_is_not_truncated(tmp_path: Path) -> None:
    f = tmp_path / "exact.txt"
    f.write_bytes(b"y" * (512 * 1024))
    result = await ReadFileTool().invoke({"path": str(f)})
    assert not result.is_error
    assert not result.content.endswith("[truncated]")
    assert len(result.content) == 512 * 1024


# 功能：验证空文件返回空字符串而非 None 或错误
# 设计：零字节文件确认 content="" 的正常返回，避免调用方（LLM prompt 组装）对空内容做额外 None 判断
async def test_empty_file_returns_empty_content(tmp_path: Path) -> None:
    f = tmp_path / "empty.txt"
    f.write_text("", encoding="utf-8")
    result = await ReadFileTool().invoke({"path": str(f)})
    assert not result.is_error
    assert result.content == ""


# ---- mode=structure（AST 结构提取） ----

# 功能：验证 structure 模式能提取 Python 文件的顶层类/函数签名及行号
# 设计：写一个含类、方法、顶层函数、装饰器的样例文件，断言签名和行号均出现在输出中，且不含函数体实现
async def test_structure_mode_extracts_signatures(tmp_path: Path) -> None:
    f = tmp_path / "sample.py"
    f.write_text(
        "import os\n"
        "\n"
        "\n"
        "class Foo:\n"
        "    def bar(self, x: int) -> str:\n"
        "        return str(x)\n"
        "\n"
        "    @staticmethod\n"
        "    def baz():\n"
        "        pass\n"
        "\n"
        "\n"
        "async def top(a, b=1):\n"
        "    return a + b\n",
        encoding="utf-8",
    )
    result = await ReadFileTool().invoke({"path": str(f), "mode": "structure"})
    assert not result.is_error
    assert "class Foo" in result.content
    assert "def bar(self, x: int) -> str" in result.content
    assert "@staticmethod def baz()" in result.content
    assert "async def top(a, b=1)" in result.content
    # 实现细节不应出现在结构输出中
    assert "return str(x)" not in result.content


# 功能：验证 structure 模式对非 Python 文件返回 schema_error 并提示改用 full 模式
# 设计：error_type 用 schema_error（不可重试），确认确定性失败不会触发 invoke_tool 的指数退避重试
async def test_structure_mode_rejects_non_python(tmp_path: Path) -> None:
    f = tmp_path / "data.txt"
    f.write_text("plain text", encoding="utf-8")
    result = await ReadFileTool().invoke({"path": str(f), "mode": "structure"})
    assert result.is_error
    assert result.error_type == "schema_error"
    assert "mode=full" in result.content


# 功能：验证 structure 模式对语法错误的 Python 文件返回 schema_error 而非抛出异常
# 设计：破损源码确认 ast.parse 异常被捕获并转为可操作的错误提示
async def test_structure_mode_syntax_error(tmp_path: Path) -> None:
    f = tmp_path / "broken.py"
    f.write_text("def broken(:\n", encoding="utf-8")
    result = await ReadFileTool().invoke({"path": str(f), "mode": "structure"})
    assert result.is_error
    assert result.error_type == "schema_error"
    assert "AST parse failed" in result.content


# ---- mode=range（行范围读取） ----

# 功能：验证 range 模式返回指定行范围并带行号前缀（1-based，两端包含）
# 设计：读 5 行文件的第 2-4 行，断言行号与内容，确认 off-by-one 边界正确
async def test_range_mode_returns_numbered_lines(tmp_path: Path) -> None:
    f = tmp_path / "lines.txt"
    f.write_text("a\nb\nc\nd\ne\n", encoding="utf-8")
    result = await ReadFileTool().invoke(
        {"path": str(f), "mode": "range", "start_line": 2, "end_line": 4}
    )
    assert not result.is_error
    assert result.content == "2→b\n3→c\n4→d"


# 功能：验证 range 模式省略 start/end 时默认读取全文范围
# 设计：只传 start_line 时 end 取末行，确认缺省值逻辑不越界
async def test_range_mode_defaults(tmp_path: Path) -> None:
    f = tmp_path / "lines.txt"
    f.write_text("a\nb\nc\n", encoding="utf-8")
    result = await ReadFileTool().invoke({"path": str(f), "mode": "range", "start_line": 2})
    assert not result.is_error
    assert result.content == "2→b\n3→c"


# 功能：验证 range 模式的非法范围（start > end 或 start 越界）返回 schema_error
# 设计：确定性参数错误不抛异常，返回带文件总行数提示的错误信息供 LLM 修正
async def test_range_mode_invalid_range(tmp_path: Path) -> None:
    f = tmp_path / "lines.txt"
    f.write_text("a\nb\n", encoding="utf-8")
    result = await ReadFileTool().invoke(
        {"path": str(f), "mode": "range", "start_line": 5, "end_line": 3}
    )
    assert result.is_error
    assert result.error_type == "schema_error"
    assert "invalid range" in result.content
