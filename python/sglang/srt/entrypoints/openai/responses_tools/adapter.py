# SPDX-License-Identifier: Apache-2.0
"""Request-local tool translation; ordinary function tools pass through."""

from sglang.srt.entrypoints.openai.responses_adapters import (
    ResponseToolState,
    flatten_response_tools,
    harmony_custom_tool,
    response_tool_choice,
)


class ResponseToolAdapter:
    def __init__(self, tools, choice="auto"):
        if any(tool["type"] == "shell" for tool in tools):
            raise ValueError("Shell tools are not supported by this model")
        self.flat_tools = tools
        self.public_choice = choice
        self.builtins = {}

    @classmethod
    def from_request(cls, request, *, use_harmony, tool_call_parser):
        """Translate tool definitions while retaining their public representation."""
        from sglang.srt.entrypoints.openai.protocol import ResponseTool

        flat_tools, names = flatten_response_tools(request.tools)
        adapter = cls(flat_tools, request.tool_choice)
        flat_tools = adapter.flat_tools
        adapted_choice = adapter.choice()
        choice = response_tool_choice(adapted_choice, flat_tools, names)
        state = ResponseToolState(
            public_tools=request.tools,
            public_choice=(
                request.tool_choice
                if adapted_choice != request.tool_choice
                else request.effective_tool_choice()
            ),
            names=names,
            custom_names={t["name"] for t in flat_tools if t["type"] == "custom"},
        )
        if use_harmony:
            flat_tools = [harmony_custom_tool(t) for t in flat_tools]
        translated = request.model_copy(
            update={
                "tools": [ResponseTool.model_validate(t) for t in flat_tools],
                "tool_choice": choice,
                "chat_template_kwargs": adapter.prepare(
                    request, tool_call_parser=tool_call_parser
                ),
            }
        )
        translated._tool_state = state
        return translated, adapter

    def prepare(self, request, *, tool_call_parser):
        """Return the model's template options without changing the API request."""
        return request.chat_template_kwargs

    def render(self, request):
        """Return template kwargs and tools to render; None keeps request.tools."""
        return request.chat_template_kwargs, None

    def choice(self):
        return self.public_choice

    def input_item(self, item):
        kind = item.get("type") if isinstance(item, dict) else item.type
        if kind in ("shell_call", "shell_call_output"):
            raise ValueError("Shell replay is not supported by this model")
        return item

    def response(self, response):
        return response

    def event(self, event):
        return event


def get_tool_adapter(model_type: str) -> type[ResponseToolAdapter]:
    if model_type == "beam":
        from sglang.srt.entrypoints.openai.responses_tools.beam import BeamToolAdapter

        return BeamToolAdapter
    return ResponseToolAdapter
