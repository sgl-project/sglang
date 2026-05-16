"""Multi-format tool-call detector that dispatches on a per-request tool_format.

Ported from vLLM's MultiFormatToolParser (v0.12.0-ifm_xllm-fix branch).

Dialects:
  - "default" : delegate to HermesDetector
  - "qwen3"   : delegate to Qwen3CoderDetector (XML form)
  - "minimax" : embedded XML extractor
  - "dsv32"   : embedded XML extractor with string-type flag
  - "glm"     : embedded <arg_key>/<arg_value> extractor
  - "gptoss"  : embedded "<tool_call>...to=functions.fn json\\n{...}\\n</tool_call>"
  - "python"  : embedded Python-AST literal extractor

Streaming is only supported for the delegating dialects (default, qwen3).
Embedded dialects buffer and emit nothing during streaming; their parse
runs on the final non-stream call. This mirrors the vLLM source behavior.
"""

from __future__ import annotations

import ast
import json
import logging
import re
from typing import Any, List, Optional

from sglang.srt.entrypoints.openai.protocol import Tool
from sglang.srt.function_call.base_format_detector import BaseFormatDetector
from sglang.srt.function_call.core_types import (
    StreamingParseResult,
    ToolCallItem,
    _GetInfoFunc,
)

logger = logging.getLogger(__name__)

_EMBEDDED_DIALECTS = {"minimax", "dsv32", "glm", "gptoss", "python"}
_DELEGATING_DIALECTS = {"default", "qwen3"}
_SUPPORTED_DIALECTS = _EMBEDDED_DIALECTS | _DELEGATING_DIALECTS


class MultiFormatDetector(BaseFormatDetector):
    """Dispatcher detector. Selects extraction strategy from tool_format."""

    def __init__(
        self,
        tool_format: Optional[str] = None,
        chat_template_kwargs: Optional[dict] = None,
    ):
        super().__init__()
        if tool_format is None and chat_template_kwargs:
            tool_format = chat_template_kwargs.get("tool_format")
        self.tool_format = tool_format or "default"

        self._delegate: Optional[BaseFormatDetector] = None

        if self.tool_format not in _SUPPORTED_DIALECTS:
            raise ValueError(
                f"Unsupported tool_format for multi_format parser: "
                f"{self.tool_format!r}. Supported formats: "
                f"{', '.join(sorted(_SUPPORTED_DIALECTS))}."
            )

        if self.tool_format == "default":
            from sglang.srt.function_call.hermes_detector import HermesDetector

            self._delegate = HermesDetector()
        elif self.tool_format == "qwen3":
            # Task 1 confirms this is the right SGLang detector for vLLM's qwen3 XML.
            from sglang.srt.function_call.qwen3_coder_detector import (
                Qwen3CoderDetector,
            )

            self._delegate = Qwen3CoderDetector()

    # BaseFormatDetector contract -------------------------------------

    def has_tool_call(self, text: str) -> bool:
        if self._delegate is not None:
            return self._delegate.has_tool_call(text)
        if self.tool_format in ("minimax", "dsv32"):
            return "<tool_calls>" in text
        if self.tool_format in ("glm", "python"):
            return "<tool_call>" in text
        if self.tool_format == "gptoss":
            return "<tool_call>" in text and "to=functions." in text
        return False

    def detect_and_parse(
        self, text: str, tools: List[Tool]
    ) -> StreamingParseResult:
        if self._delegate is not None:
            return self._delegate.detect_and_parse(text, tools)

        try:
            if self.tool_format == "minimax":
                return self._extract_minimax(text, tools, type_aware=False)
            if self.tool_format == "dsv32":
                return self._extract_minimax(text, tools, type_aware=True)
            if self.tool_format == "glm":
                return self._extract_glm(text, tools)
            if self.tool_format == "gptoss":
                return self._extract_gptoss(text, tools)
            if self.tool_format == "python":
                return self._extract_python(text, tools)
        except Exception:
            logger.exception(
                "MultiFormatDetector failed to extract for tool_format=%s",
                self.tool_format,
            )
        return StreamingParseResult(normal_text=text, calls=[])

    def parse_streaming_increment(
        self, new_text: str, tools: List[Tool]
    ) -> StreamingParseResult:
        if self._delegate is not None:
            return self._delegate.parse_streaming_increment(new_text, tools)
        # Embedded dialects: vLLM source does not support streaming either.
        # NOTE: self._buffer here is intentionally write-only — it is not
        # read by detect_and_parse (which receives the complete text via its
        # `text` parameter). Embedded extractors in Tasks 5/7/8/9 must use
        # their `text` argument, not self._buffer, as their input source.
        self._buffer += new_text
        return StreamingParseResult()

    def supports_structural_tag(self) -> bool:
        if self._delegate is not None:
            return self._delegate.supports_structural_tag()
        return False

    def structure_info(self) -> _GetInfoFunc:
        if self._delegate is not None:
            return self._delegate.structure_info()
        raise NotImplementedError(
            f"structure_info is not implemented for tool_format={self.tool_format!r}"
        )

    # Embedded extractors (stubs filled in Tasks 5/7/8/9) -------------

    def _extract_minimax(
        self, text: str, tools: List[Tool], type_aware: bool
    ) -> StreamingParseResult:
        if self._MINIMAX_START not in text:
            return StreamingParseResult(normal_text=text, calls=[])

        tool_indices = self._get_tool_indices(tools)
        calls: List[ToolCallItem] = []
        for block in self._MINIMAX_BLOCK.findall(text):
            for func_name, body in self._MINIMAX_INVOKE.findall(block):
                args: dict[str, Any] = {}
                for pname, string_flag, pvalue in self._MINIMAX_PARAM.findall(body):
                    # string_flag is None (not "") when the attribute is absent
                    # in the source — Python's re returns None for non-participating
                    # optional capture groups. None == "true" is False, so the
                    # else branch is taken for both unset and string="false" cases.
                    if type_aware and string_flag == "true":
                        args[pname] = pvalue
                    else:
                        args[pname] = self._json_or_string(pvalue)
                calls.append(
                    ToolCallItem(
                        tool_index=tool_indices.get(func_name, -1),
                        name=func_name,
                        parameters=json.dumps(args, ensure_ascii=False),
                    )
                )

        if not calls:
            return StreamingParseResult(normal_text=text, calls=[])

        prefix = text[: text.find(self._MINIMAX_START)]
        return StreamingParseResult(normal_text=prefix, calls=calls)

    def _extract_glm(self, text: str, tools: List[Tool]) -> StreamingParseResult:
        matches = list(self._GLM_BLOCK.finditer(text))
        if not matches:
            return StreamingParseResult(normal_text=text, calls=[])

        tool_indices = self._get_tool_indices(tools)
        calls: List[ToolCallItem] = []
        for match in matches:
            block = match.group(1)
            first_arg_idx = block.find("<arg_key>")
            if first_arg_idx == -1:
                func_name = block.strip()
                args: dict[str, Any] = {}
            else:
                func_name = block[:first_arg_idx].strip()
                args = {}
                arg_section = block[first_arg_idx:]
                for k, v in self._GLM_ARG.findall(arg_section):
                    key = k.strip()
                    raw = v.strip()
                    if self._glm_param_is_string(func_name, key, tools):
                        args[key] = raw
                    else:
                        args[key] = self._deserialize_glm_value(raw)
            if not func_name:
                continue
            calls.append(
                ToolCallItem(
                    tool_index=tool_indices.get(func_name, -1),
                    name=func_name,
                    parameters=json.dumps(args, ensure_ascii=False),
                )
            )

        if not calls:
            return StreamingParseResult(normal_text=text, calls=[])
        prefix = text[: matches[0].start()]
        return StreamingParseResult(normal_text=prefix, calls=calls)

    def _extract_gptoss(
        self, text: str, tools: List[Tool]
    ) -> StreamingParseResult:
        matches = list(self._GPTOSS_BLOCK.finditer(text))
        if not matches:
            return StreamingParseResult(normal_text=text, calls=[])

        tool_indices = self._get_tool_indices(tools)
        calls: List[ToolCallItem] = []
        for m in matches:
            func_name = m.group(1)
            args = json.loads(m.group(2).strip())
            calls.append(
                ToolCallItem(
                    tool_index=tool_indices.get(func_name, -1),
                    name=func_name,
                    parameters=json.dumps(args, ensure_ascii=False),
                )
            )
        if not calls:
            return StreamingParseResult(normal_text=text, calls=[])
        prefix = text[: matches[0].start()]
        return StreamingParseResult(normal_text=prefix, calls=calls)

    def _extract_python(
        self, text: str, tools: List[Tool]
    ) -> StreamingParseResult:
        matches = list(self._PYTHON_BLOCK.finditer(text))
        if not matches:
            return StreamingParseResult(normal_text=text, calls=[])

        tool_indices = self._get_tool_indices(tools)
        calls: List[ToolCallItem] = []
        for match in matches:
            block = match.group(1).strip()
            module = ast.parse(block)
            for stmt in module.body:
                if not isinstance(stmt, ast.Expr) or not isinstance(
                    stmt.value, ast.Call
                ):
                    raise ValueError(
                        "Expected Python function call(s) inside <tool_call>."
                    )
                call = stmt.value
                if not isinstance(call.func, ast.Name):
                    raise ValueError("Invalid tool-call name")
                func_name = call.func.id
                args: dict[str, Any] = {}
                for kw in call.keywords:
                    args[kw.arg] = self._python_literal(kw.value)
                calls.append(
                    ToolCallItem(
                        tool_index=tool_indices.get(func_name, -1),
                        name=func_name,
                        parameters=json.dumps(args, ensure_ascii=False),
                    )
                )
        if not calls:
            return StreamingParseResult(normal_text=text, calls=[])
        prefix = text[: matches[0].start()]
        return StreamingParseResult(normal_text=prefix, calls=calls)

    # Class-level regex constants and helpers for minimax/dsv32 dialects ------

    _MINIMAX_START = "<tool_calls>"
    _MINIMAX_BLOCK = re.compile(r"<tool_calls>(.*?)</tool_calls>", re.DOTALL)
    _MINIMAX_INVOKE = re.compile(
        r'<invoke\s+name="([^"]+)"\s*>(.*?)</invoke>', re.DOTALL
    )
    _MINIMAX_PARAM = re.compile(
        r'<parameter\s+name="([^"]+)"(?:\s+string="(true|false)")?\s*>(.*?)</parameter>',
        re.DOTALL,
    )

    @staticmethod
    def _json_or_string(value: str) -> Any:
        value = value.strip()
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return value

    # Class-level regex constants and helpers for glm dialect ------------------

    _GLM_BLOCK = re.compile(r"<tool_call>(.*?)</tool_call>", re.DOTALL)
    _GLM_ARG = re.compile(
        r"<arg_key>(.*?)</arg_key>\s*<arg_value>(.*?)</arg_value>", re.DOTALL
    )

    @staticmethod
    def _deserialize_glm_value(value: str) -> Any:
        value = value.strip()
        try:
            return json.loads(value)
        except Exception:
            pass
        try:
            return ast.literal_eval(value)
        except Exception:
            pass
        return value

    @staticmethod
    def _glm_param_is_string(
        tool_name: str, arg_name: str, tools: List[Tool]
    ) -> bool:
        for tool in tools:
            if tool.function.name != tool_name or tool.function.parameters is None:
                continue
            arg_type = (
                tool.function.parameters.get("properties", {})
                .get(arg_name, {})
                .get("type")
            )
            return arg_type == "string"
        return False

    # Class-level regex constants for gptoss dialect ---------------------------

    _GPTOSS_BLOCK = re.compile(
        r"<tool_call>\s*(?:assistant\s+)?to=functions\.(\S+?)"
        r"(?:\s+json)?\s*\n(.*?)\n?\s*</tool_call>",
        re.DOTALL,
    )

    # Class-level regex constants and helpers for python dialect ---------------

    _PYTHON_BLOCK = re.compile(r"<tool_call>(.*?)</tool_call>", re.DOTALL)

    @staticmethod
    def _python_literal(node: ast.expr) -> Any:
        """Custom AST walker accepting only literal subtrees.

        This intentionally does NOT call ast.literal_eval — we walk the AST
        ourselves so the accepted subset is fully explicit, and any non-literal
        (BinOp, Call, Name except true/false/null, etc.) raises ValueError.
        """
        if isinstance(node, ast.Constant):
            return node.value
        if isinstance(node, ast.Name):
            if node.id in {"true", "True"}:
                return True
            if node.id in {"false", "False"}:
                return False
            if node.id in {"null", "None"}:
                return None
        if isinstance(node, ast.Dict):
            if not all(isinstance(k, ast.Constant) for k in node.keys):
                raise ValueError("dict keys must be literals")
            return {
                k.value: MultiFormatDetector._python_literal(v)
                for k, v in zip(node.keys, node.values)
            }
        if isinstance(node, ast.List):
            return [MultiFormatDetector._python_literal(v) for v in node.elts]
        if isinstance(node, ast.Tuple):
            return [MultiFormatDetector._python_literal(v) for v in node.elts]
        if (
            isinstance(node, ast.UnaryOp)
            and isinstance(node.op, (ast.USub, ast.UAdd))
            and isinstance(node.operand, ast.Constant)
            and isinstance(node.operand.value, (int, float))
        ):
            return (
                -node.operand.value
                if isinstance(node.op, ast.USub)
                else node.operand.value
            )
        raise ValueError("tool-call arguments must be literals")
