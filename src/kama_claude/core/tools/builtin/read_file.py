from __future__ import annotations

import ast
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict

from kama_claude.core.tools.base import BaseTool, ToolResult

_MAX_BYTES = 512 * 1024  # 512 KB


class ReadFileParams(BaseModel):
    model_config = ConfigDict(extra="ignore")
    path: str
    mode: Literal["full", "structure", "range"] = "full"
    start_line: int | None = None
    end_line: int | None = None


class ReadFileTool(BaseTool):
    params_model = ReadFileParams
    name = "read_file"
    description = (
        "Read a file. Path must be relative to the current working directory. "
        "mode=full (default) returns the entire text; files larger than 512 KB are truncated. "
        "mode=structure returns top-level class/function signatures extracted via AST "
        "(Python files only) — use it to survey a large file before reading details. "
        "mode=range returns only lines start_line..end_line (1-based, inclusive)."
    )
    input_schema: dict[str, object] = {
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "Relative path to the file (relative to current working directory).",
            },
            "mode": {
                "type": "string",
                "enum": ["full", "structure", "range"],
                "description": (
                    "full: entire file content (default); "
                    "structure: AST-extracted class/function signatures (Python only); "
                    "range: specific line range via start_line/end_line."
                ),
            },
            "start_line": {
                "type": "integer",
                "description": "First line to read, 1-based (only used with mode=range).",
            },
            "end_line": {
                "type": "integer",
                "description": "Last line to read, 1-based inclusive (only used with mode=range).",
            },
        },
        "required": ["path"],
    }

    # 按 mode 分发：full 读全文，structure 提取 AST 结构，range 读取行范围；
    # 安全边界：禁止 .. 路径遍历与绝对路径（与 description 的相对路径契约一致）
    async def invoke(self, params: dict[str, object]) -> ToolResult:
        p = ReadFileParams.model_validate(params)

        if ".." in Path(p.path).parts:
            raise PermissionError(f"path traversal not allowed: {p.path}")

        if Path(p.path).is_absolute():
            raise PermissionError(f"absolute path is not allowed: {p.path}")

        path = Path(p.path)
        if p.mode == "structure":
            return self._read_structure(path)
        if p.mode == "range":
            return self._read_range(path, p.start_line, p.end_line)
        return self._read_full(path)

    # 读取完整文件内容；超 512KB 截断
    def _read_full(self, path: Path) -> ToolResult:
        raw = path.read_bytes()  # raises FileNotFoundError if absent
        truncated = len(raw) > _MAX_BYTES
        text = raw[:_MAX_BYTES].decode("utf-8", errors="replace")
        if truncated:
            text += "\n[truncated]"
        return ToolResult(content=text)

    # 读取指定行范围（1-based，两端包含）；参数缺省或越界时返回 schema_error 提示修正
    def _read_range(self, path: Path, start_line: int | None, end_line: int | None) -> ToolResult:
        text = path.read_bytes()[:_MAX_BYTES].decode("utf-8", errors="replace")
        lines = text.splitlines()
        total = len(lines)

        start = start_line if start_line is not None else 1
        end = end_line if end_line is not None else total
        if start < 1 or end < start or (total > 0 and start > total):
            return ToolResult(
                content=(
                    f"invalid range: start_line={start_line} end_line={end_line}, "
                    f"file has {total} lines. Use 1-based start_line <= end_line."
                ),
                is_error=True,
                error_type="schema_error",
            )
        end = min(end, total)

        width = len(str(end))
        numbered = [
            f"{n:>{width}}\u2192{lines[n - 1]}" for n in range(start, end + 1)
        ]
        return ToolResult(content="\n".join(numbered))

    # 用 AST 提取 Python 文件的顶层类/函数签名；非 Python 或解析失败时返回 schema_error
    def _read_structure(self, path: Path) -> ToolResult:
        if path.suffix not in (".py", ".pyi"):
            return ToolResult(
                content=(
                    f"mode=structure supports Python files only, got '{path.name}'. "
                    "Use mode=full instead."
                ),
                is_error=True,
                error_type="schema_error",
            )
        source = path.read_bytes()[:_MAX_BYTES].decode("utf-8", errors="replace")
        try:
            tree = ast.parse(source)
        except (SyntaxError, ValueError) as exc:
            return ToolResult(
                content=f"AST parse failed for '{path.name}': {exc}. Use mode=full instead.",
                is_error=True,
                error_type="schema_error",
            )

        total_lines = source.count("\n") + 1
        out: list[str] = [f"[structure] {path} ({total_lines} lines)", ""]
        found = False
        for node in tree.body:
            if isinstance(node, ast.ClassDef):
                found = True
                bases = f"({', '.join(ast.unparse(b) for b in node.bases)})" if node.bases else ""
                out.append(f"class {node.name}{bases}  # line {node.lineno}")
                for item in node.body:
                    if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        out.append(f"  {self._func_sig(item)}")
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                found = True
                out.append(self._func_sig(node))
        if not found:
            out.append("(no top-level class or function definitions)")
        return ToolResult(content="\n".join(out))

    # 格式化单个函数/方法签名：装饰器 + def 行 + 行号
    @staticmethod
    def _func_sig(node: ast.FunctionDef | ast.AsyncFunctionDef) -> str:
        prefix = "async def" if isinstance(node, ast.AsyncFunctionDef) else "def"
        args = ast.unparse(node.args)
        ret = f" -> {ast.unparse(node.returns)}" if node.returns is not None else ""
        decorators = ""
        for dec in node.decorator_list:
            decorators += f"@{ast.unparse(dec)} "
        return f"{decorators}{prefix} {node.name}({args}){ret}  # line {node.lineno}"
