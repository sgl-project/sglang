# SPDX-License-Identifier: Apache-2.0
"""Translate MiniCPM Responses tools using Beam's flat function format.

Merge tools loaded by searches across the effective input history. Keep search
items out of the prompt because MiniCPM's chat template cannot render them.
"""

import msgspec

from sglang.srt.entrypoints.openai.responses_tools.beam import BeamToolAdapter


def tools_with_search_results(request, input_items):
    """Keep request definitions first, then add new tools and namespace members."""
    from sglang.srt.entrypoints.openai.protocol import ResponseTool

    tools = list(request.tools)
    indices = {(tool.type, tool.name): i for i, tool in enumerate(tools)}
    for item in input_items:
        if not isinstance(item, dict) or item.get("type") != "tool_search_output":
            continue
        for raw_tool in item.get("tools") or []:
            if not isinstance(raw_tool, dict) or raw_tool.get("type") not in (
                "function",
                "custom",
                "namespace",
            ):
                continue
            tool = ResponseTool.model_validate(raw_tool)
            key = (tool.type, tool.name)
            index = indices.get(key)
            if index is None:
                indices[key] = len(tools)
                tools.append(tool)
                continue
            if tool.type == "namespace":
                existing = tools[index]
                members = list(existing.tools or [])
                seen = {(member.get("type"), member.get("name")) for member in members}
                for member in tool.tools or []:
                    member_key = (member.get("type"), member.get("name"))
                    if member_key not in seen:
                        members.append(member)
                        seen.add(member_key)
                tools[index] = existing.model_copy(update={"tools": members})
    return tools


class MiniCPMToolAdapter(BeamToolAdapter):
    @classmethod
    def from_request(cls, request, *, tool_call_parser, input_items=None):
        if input_items is None:
            input_items = request.input if isinstance(request.input, list) else []
        translated, adapter = super().from_request(
            request.model_copy(
                update={"tools": tools_with_search_results(request, input_items)}
            ),
            tool_call_parser=tool_call_parser,
        )
        # Public responses echo the tools declared by the client.
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
