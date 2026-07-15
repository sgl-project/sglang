import json
import logging
import re
from typing import List, Tuple

logger = logging.getLogger(__name__)


class StreamingToolCallParser:
    """
    流式 ToolCall 解析器，支持 Qwen3-Coder 的 XML 风格格式：
    <tool_call>
    <function=func_name></function>
    <parameter=key1>value1</parameter>
    <parameter=key2>value2</parameter>
    </function>
    </tool_call>
    """
    _FUNC_OPEN_RE = re.compile(r"<function=([^>]+)>")
    _FUNC_CLOSE_TAG = "</function>"
    _PARAM_RE = re.compile(r"<parameter=([^>]+)>(.*?)(</parameter>)", re.DOTALL)

    def __init__(self, start_token="<tool_call>", end_token="</tool_call>"):
        self.start_token = start_token
        self.end_token = end_token
        self._buffer = ""
        self._in_tool_call = False

    def _calculate_safe_prefix_length(self, text: str, token: str) -> int:
        """防止标签前缀被当作正常文本输出"""
        if chr(60) not in text:
            return len(text)
        for i in range(len(token) - 1, 0, -1):
            if text.endswith(token[:i]):
                return len(text) - i
        return len(text)

    def _parse_function_block(self, block: str) -> dict:
        """解析单个 <function=name></function> 块"""
        m = self._FUNC_OPEN_RE.search(block)
        if not m:
            return None
        func_name = m.group(1).strip()
        arguments = {}
        for pm in self._PARAM_RE.finditer(block):
            key = pm.group(1).strip()
            value = pm.group(2).strip()
            arguments[key] = value
        return {
            "name": func_name,
            "arguments": json.dumps(arguments, ensure_ascii=False),
        }

    def _split_function_blocks(self, text: str) -> List[str]:
        """把 <tool_call> 内部的内容按 <function=...></function> 拆分"""
        blocks = []
        cursor = 0
        while True:
            start = text.find("<function=", cursor)
            if start == -1:
                break
            end = text.find(self._FUNC_CLOSE_TAG, start)
            if end == -1:
                break
            blocks.append(text[start:end + len(self._FUNC_CLOSE_TAG)])
            cursor = end + len(self._FUNC_CLOSE_TAG)
        return blocks

    def parse_chunk(self, delta: str) -> Tuple[str, List[dict]]:
        """喂入流式 chunk，返回 (normal_text, tool_calls增量)"""
        if not self._in_tool_call and chr(60) not in delta and chr(60) not in self._buffer:
            return delta, []

        self._buffer += delta
        normal_text_output = ""
        current_tool_calls = []

        while True:
            if not self._in_tool_call:
                start_idx = self._buffer.find(self.start_token)
                if start_idx != -1:
                    self._in_tool_call = True
                    normal_text_output += self._buffer[:start_idx]
                    self._buffer = self._buffer[start_idx + len(self.start_token):]
                else:
                    safe_len = self._calculate_safe_prefix_length(
                        self._buffer, self.start_token
                    )
                    normal_text_output += self._buffer[:safe_len]
                    self._buffer = self._buffer[safe_len:]
                    break
            if self._in_tool_call:
                end_idx = self._buffer.find(self.end_token)
                if end_idx != -1:
                    block = self._buffer[:end_idx]
                    self._buffer = self._buffer[end_idx + len(self.end_token):]
                    self._in_tool_call = False
                    func_blocks = self._split_function_blocks(block)
                    for fb in func_blocks:
                        parsed = self._parse_function_block(fb)
                        if parsed:
                            current_tool_calls.append(parsed)
                else:
                    break
        return normal_text_output, current_tool_calls

    def flush(self) -> Tuple[str, List[dict]]:
        """流结束时调用，处理未闭合的 tool_call"""
        normal_text_output = ""
        current_tool_calls = []
        if not self._in_tool_call:
            normal_text_output = self._buffer
        else:
            block = self._buffer
            func_blocks = self._split_function_blocks(block)
            for fb in func_blocks:
                parsed = self._parse_function_block(fb)
                if parsed:
                    current_tool_calls.append(parsed)
            if not func_blocks:
                normal_text_output = self._buffer
        self._buffer = ""
        self._in_tool_call = False
        return normal_text_output, current_tool_calls
