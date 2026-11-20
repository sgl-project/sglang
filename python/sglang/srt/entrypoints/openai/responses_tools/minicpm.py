# SPDX-License-Identifier: Apache-2.0
"""MiniCPM5's chat template renders flat function tools, as Beam's does, so its
Responses tools take Beam's translation. Codex also replays tool_search items,
which the template cannot render: the tools a search loaded are declared with
the request's own, and the search items are left out of the prompt."""

import msgspec

from sglang.srt.entrypoints.openai.responses_tools.beam import BeamToolAdapter


def tool_search_tools(request):
    """Tools loaded by ``tool_search_output`` input items that the request does
    not declare itself."""
    from sglang.srt.entrypoints.openai.protocol import ResponseTool

    if not isinstance(request.input, list):
        return []
    seen = {(tool.type, tool.name) for tool in request.tools}
    loaded = []
    for item in request.input:
        if not isinstance(item, dict) or item.get("type") != "tool_search_output":
            continue
        for tool in item.get("tools") or []:
            if not isinstance(tool, dict):
                continue
            key = (tool.get("type"), tool.get("name"))
            if key[0] not in ("function", "custom", "namespace") or key in seen:
                continue
            seen.add(key)
            loaded.append(ResponseTool.model_validate(tool))
    return loaded


class MiniCPMToolAdapter(BeamToolAdapter):
    @classmethod
    def from_request(cls, request, *, tool_call_parser):
        translated, adapter = super().from_request(
            request.model_copy(
                update={"tools": [*request.tools, *tool_search_tools(request)]}
            ),
            tool_call_parser=tool_call_parser,
        )
        # The response echoes the tools the client declared, not the loaded ones.
        translated._tool_state = msgspec.structs.replace(
            translated._tool_state, public_tools=request.tools
        )
        return translated, adapter

    def input_item(self, item):
        if isinstance(item, dict) and item.get("type") in (
            "tool_search_call",
            "tool_search_output",
        ):
            return None
        return super().input_item(item)
