import orjson

from sglang.srt.entrypoints.openai.protocol import ToolChoice
from sglang.srt.function_call.core_types import StreamingParseResult, ToolCallItem
from sglang.srt.function_call.glm47_moe_detector import (
    Glm47MoeDetector,
    get_argument_type,
    iter_arg_pair_matches,
)
from sglang.srt.function_call.json_array_parser import JsonArrayParser
from sglang.srt.function_call.utils import get_json_schema_constraint


class IQuestQ1Detector(Glm47MoeDetector):
    def __init__(self):
        super().__init__()
        self.bot_token = "<iquest_tool_call>"
        self.eot_token = "</iquest_tool_call>"
        self._inside_call = False

    def _parse_body(self, body, tools):
        cursor = body.find("<arg_key>")
        name = body[:cursor].strip() if cursor >= 0 else body.strip()
        if not name:
            return None
        raw = body[cursor:] if cursor >= 0 else ""
        pairs = []
        leftover = []
        prev_end = 0
        for start, end, key, value in iter_arg_pair_matches(raw):
            pairs.append((key, value))
            leftover.append(raw[prev_end:start])
            prev_end = end
        leftover.append(raw[prev_end:])
        if "".join(leftover).strip() or any(not key.strip() for key, _ in pairs):
            return None
        arguments = self._parse_argument_pairs(pairs, name, tools)
        for key, value in pairs:
            key = key.strip()
            if get_argument_type(name, key, tools, arguments) == "string":
                arguments[key] = value
        return {"name": name, "arguments": arguments}

    def detect_and_parse(self, text, tools):
        calls = []
        normal = []
        content_start = 0
        search = 0
        while (start := text.find(self.bot_token, search)) >= 0:
            begin = start + len(self.bot_token)
            end = text.find(self.eot_token, begin)
            if end < 0:
                break
            action = self._parse_body(text[begin:end], tools)
            search = end + len(self.eot_token)
            if action is not None:
                for call in self.parse_base_json(action, tools):
                    call.tool_index = len(calls)
                    calls.append(call)
                normal.append(text[content_start:start])
                content_start = search
        normal.append(text[content_start:])
        return StreamingParseResult(
            normal_text="".join(normal),
            calls=calls,
        )

    def parse_streaming_increment(self, new_text, tools):
        self._buffer += new_text
        normal, calls = [], []
        while True:
            if not self._inside_call:
                start = self._buffer.find(self.bot_token)
                if start < 0:
                    keep = self._ends_with_partial_token(self._buffer, self.bot_token)
                    end = len(self._buffer) - keep
                    normal.append(self._buffer[:end])
                    self._buffer = self._buffer[end:]
                    break
                normal.append(self._buffer[:start])
                self._buffer = self._buffer[start + len(self.bot_token) :]
                self._inside_call = True
            end = self._buffer.find(self.eot_token)
            if end < 0:
                break
            body = self._buffer[:end]
            self._buffer = self._buffer[end + len(self.eot_token) :]
            self._inside_call = False
            action = self._parse_body(body, tools)
            if action is None:
                normal.append(self.bot_token + body + self.eot_token)
                continue
            for call in self.parse_base_json(action, tools):
                call.tool_index = len(self.prev_tool_call_arr)
                calls.append(call)
                self.prev_tool_call_arr.append(action)
                self.streamed_args_for_tool.append(call.parameters)
                self.current_tool_id = call.tool_index
                self.current_tool_name_sent = True
        return StreamingParseResult(normal_text="".join(normal), calls=calls)

    def finish(self, tools):
        text = (self.bot_token if self._inside_call else "") + self._buffer
        self._buffer = ""
        self._inside_call = False
        return StreamingParseResult(normal_text=text)

    def parses_required_natively(self):
        return False

    def get_required_tool_parser(self, tool_choice):
        if tool_choice != "required" and not isinstance(tool_choice, ToolChoice):
            return None
        name = (
            tool_choice.function.name if isinstance(tool_choice, ToolChoice) else None
        )
        return IQuestQ1JsonToolParser(name)

    def supports_structural_tag(self):
        return False

    def structure_info(self):
        raise NotImplementedError("IQuest Q1 uses XML argument pairs")


class IQuestQ1JsonToolParser(JsonArrayParser):
    def __init__(self, tool_name=None):
        super().__init__()
        self.tool_name = tool_name

    def supports_structural_tag(self):
        return False

    def get_required_tool_parser(self, tool_choice):
        return self

    def get_json_schema_constraint(self, tools, tool_choice, parallel_tool_calls=True):
        if self.tool_name is not None:
            return next(
                tool.function.parameters or {}
                for tool in tools
                if tool.function.name == self.tool_name
            )
        return get_json_schema_constraint(
            tools, tool_choice, parallel_tool_calls=parallel_tool_calls
        )

    def has_tool_call(self, text):
        return True

    def detect_and_parse(self, text, tools):
        value = orjson.loads(text)
        if self.tool_name is not None:
            if not isinstance(value, dict):
                raise ValueError("Named tool arguments must be a JSON object.")
            return StreamingParseResult(
                calls=[ToolCallItem(tool_index=0, name=self.tool_name, parameters=text)]
            )
        if not isinstance(value, list) or not all(
            isinstance(call, dict) and "name" in call for call in value
        ):
            raise ValueError("Expected a JSON array of named tool calls.")
        return StreamingParseResult(calls=self.parse_base_json(value, tools))

    def parse_streaming_increment(self, new_text, tools):
        if self.tool_name is not None:
            if not new_text:
                return StreamingParseResult()
            name = None if self.current_tool_name_sent else self.tool_name
            self.current_tool_name_sent = True
            return StreamingParseResult(
                calls=[ToolCallItem(tool_index=0, name=name, parameters=new_text)]
            )

        calls = []
        while True:
            result = super().parse_streaming_increment(new_text, tools)
            calls.extend(result.calls)
            if not result.calls:
                break
            new_text = ""
        return StreamingParseResult(calls=calls)
