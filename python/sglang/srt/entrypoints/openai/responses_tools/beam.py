# SPDX-License-Identifier: Apache-2.0
"""Translate Responses tools to the flat function tools Beam's template renders:
namespaced tools under joined names, and shell as Beam's trained bash."""

import json
from typing import Union

from openai.types.responses import ResponseFunctionShellToolCall
from openai.types.responses.response_input_item_param import ShellCall, ShellCallOutput
from pydantic import BaseModel, TypeAdapter

from sglang.srt.entrypoints.openai.responses_adapters import response_tool_name

_SHELL_INPUT = TypeAdapter(Union[ShellCall, ShellCallOutput])


class ResponseToolChoiceError(ValueError):
    """A tool-choice translation error, reported against tool_choice."""


def flat_namespace_tool_name(namespace: str | None, name: str) -> str:
    return f"{namespace}__{name}" if namespace is not None else name


def flatten_response_tools(tools):
    """Return flat tools and a declared-name map for existing function parsers."""
    flat_tools, names = [], {}
    namespaces = set()
    for tool in tools:
        tool = tool.model_dump(exclude_none=True)
        namespace = tool.get("name") if tool["type"] == "namespace" else None
        if tool["type"] == "namespace":
            if not isinstance(namespace, str) or not namespace:
                raise ValueError("Namespaces require a name")
            if namespace in namespaces:
                raise ValueError(f"Duplicate namespace {namespace!r}")
            namespaces.add(namespace)
            children = tool.get("tools")
            if not isinstance(children, list) or not children:
                raise ValueError(f"Namespace {namespace!r} requires tools")
        else:
            children = [tool]
        for child in children:
            if namespace is not None and (
                not isinstance(child, dict)
                or child.get("type") not in ("function", "custom")
            ):
                raise ValueError("Namespaces support only function and custom tools")
            if child["type"] not in ("function", "custom"):
                flat_tools.append(child)
                continue
            name = child.get("name")
            if not isinstance(name, str) or not name:
                raise ValueError("Function and custom tools require a name")
            alias = flat_namespace_tool_name(namespace, name)
            if alias in names:
                raise ValueError(
                    f"Duplicate or colliding tool name: {namespace!r}/{name!r}"
                )
            names[alias] = {"name": name}
            if namespace is not None:
                names[alias]["namespace"] = namespace
            flat = {**child, "name": alias}
            if namespace is not None and tool.get("description"):
                flat["description"] = (
                    f"Namespace {namespace}: {tool['description']}\n\n{child.get('description') or ''}"
                )
            flat_tools.append(flat)
    return flat_tools, names


def response_tool_choice(choice, flat_tools, names):
    if not isinstance(choice, dict) or choice.get("type") not in ("function", "custom"):
        return choice
    name = choice.get("name") or (choice.get("function") or {}).get("name")
    alias = flat_namespace_tool_name(choice.get("namespace"), name)
    declared = response_tool_name(alias, names)
    if declared["name"] != name or declared.get("namespace") != choice.get("namespace"):
        raise ResponseToolChoiceError(f"Tool choice {name!r} is not declared in tools")
    if not any(
        t["name"] == alias and t["type"] == choice["type"]
        for t in flat_tools
        if "name" in t
    ):
        raise ResponseToolChoiceError(f"Tool choice {name!r} is not declared in tools")
    return {"type": "function", "name": alias}


def flatten_response_input(item):
    if isinstance(item, BaseModel):
        item = item.model_dump(exclude_none=True)
    if isinstance(item, dict) and item.get("type") in (
        "function_call",
        "custom_tool_call",
    ):
        item = dict(item)
        name = item.get("name")
        if not isinstance(name, str) or not name:
            raise ValueError(f"{item['type']} input items require a name")
        item["name"] = flat_namespace_tool_name(item.pop("namespace", None), name)
    return item


class BeamToolAdapter:
    def __init__(self, tools, choice="auto"):
        self.public_choice = choice
        self.flat_tools = []
        self.names = set()
        self.builtins = {}
        self.builtin_streams = {}
        for tool in tools:
            if tool["type"] == "shell":
                self._options(tool, {"type", "environment", "strict"})
                if tool.get("environment") != {"type": "local"}:
                    raise ValueError(
                        "Shell supports only environment={type:local}, without environment options"
                    )
                self._builtin(
                    "bash",
                    "shell",
                    "Execute a bash command. The output contains the command's stdout and stderr.",
                    {
                        "type": "object",
                        "properties": {
                            "command": {
                                "type": "string",
                                "description": "The bash command to execute.",
                            },
                        },
                        "required": ["command"],
                    },
                )
            else:
                self._add(tool)

    @classmethod
    def from_request(cls, request, *, tool_call_parser):
        """Translate tool definitions while retaining their public representation."""
        from sglang.srt.entrypoints.openai.protocol import (
            ResponseTool,
            ResponseToolState,
        )

        flat_tools, names = flatten_response_tools(request.tools)
        adapter = cls(flat_tools, request.tool_choice)
        if adapter.builtins and tool_call_parser not in ("beam", "response_template"):
            raise ValueError(
                "Beam built-in tools require the beam or response_template tool-call parser"
            )
        adapted_choice = adapter.choice()
        choice = response_tool_choice(adapted_choice, adapter.flat_tools, names)
        state = ResponseToolState(
            public_tools=request.tools,
            public_choice=(
                request.tool_choice
                if adapted_choice != request.tool_choice
                else request.effective_tool_choice()
            ),
            names=names,
        )
        translated = request.model_copy(
            update={
                "tools": [ResponseTool.model_validate(t) for t in adapter.flat_tools],
                "tool_choice": choice,
            }
        )
        translated._tool_state = state
        return translated, adapter

    def _builtin(self, name, kind, description, parameters):
        self._add(
            {
                "type": "function",
                "name": name,
                "description": description,
                "parameters": parameters,
            }
        )
        self.builtins[name] = kind

    @staticmethod
    def _options(tool, allowed):
        unsupported = set(tool) - allowed
        if unsupported:
            raise ValueError(
                f"Unsupported Responses tool options: {', '.join(sorted(unsupported))}"
            )

    def _add(self, tool):
        name = tool.get("name")
        if name is not None:
            if name in self.names:
                raise ValueError(f"Duplicate or colliding tool name: {name!r}")
            self.names.add(name)
        self.flat_tools.append(tool)

    def choice(self):
        if (
            self.public_choice == {"type": "shell"}
            and self.builtins.get("bash") == "shell"
        ):
            return {"type": "function", "name": "bash"}
        return self.public_choice

    def input_item(self, item):
        if isinstance(item, BaseModel):
            item = item.model_dump(exclude_none=True)
        if not isinstance(item, dict):
            return item
        if item.get("type") in ("shell_call", "shell_call_output"):
            if "bash" in self.names and self.builtins.get("bash") != "shell":
                raise ValueError("Shell replay conflicts with a user-defined bash tool")
        if item.get("type") == "shell_call":
            call = _SHELL_INPUT.validate_python(item)
            self._options(
                item["action"], {"commands", "timeout_ms", "max_output_length"}
            )
            action = call["action"]
            # Cutoffs have no command to replay; keep them only in public history.
            if call.get("status") == "incomplete" and not action["commands"]:
                return None
            if (
                len(action["commands"]) != 1
                or action.get("timeout_ms") not in (None, 600000)
                or action.get("max_output_length") is not None
            ):
                raise ValueError(
                    "Beam shell replay supports one command, the default timeout, and no max_output_length"
                )
            return {
                "type": "function_call",
                "id": call.get("id"),
                "call_id": call["call_id"],
                "name": "bash",
                "arguments": json.dumps({"command": action["commands"][0]}),
                "status": call.get("status") or "completed",
            }
        if item.get("type") == "shell_call_output":
            output = _SHELL_INPUT.validate_python(item)
            output["output"] = list(output["output"])
            return {
                "type": "function_call_output",
                "call_id": output["call_id"],
                "output": _SHELL_INPUT.dump_json(output, exclude_none=True).decode(),
            }
        return flatten_response_input(item)

    def output_item(self, item):
        if not isinstance(item, dict) or item.get("type") not in (
            "function_call",
            "custom_tool_call",
        ):
            return item
        if item.get("namespace") is not None:
            return item
        if self.builtins.get(item.get("name")) == "shell":
            return self._shell_call(item)
        return item

    @staticmethod
    def _shell_call(item):
        try:
            args = json.loads(item.get("arguments") or "{}")
        except ValueError:
            args = {}
        valid = (
            isinstance(args, dict)
            and set(args) == {"command"}
            and isinstance(args["command"], str)
        )
        status = item.get("status", "completed")
        if not valid and status != "in_progress":
            status = "incomplete"
        return ResponseFunctionShellToolCall(
            type="shell_call",
            id=item["id"],
            call_id=item["call_id"],
            status=status,
            action={
                "commands": [args["command"]]
                if valid and status == "completed"
                else [],
                "timeout_ms": 600000,
                "max_output_length": None,
            },
        ).model_dump()

    def response(self, response):
        result = {
            **response,
            "output": [self.output_item(i) for i in response.get("output", [])],
        }
        if response.get("status") == "completed" and any(
            i.get("type") == "shell_call" and i.get("status") == "incomplete"
            for i in result["output"]
        ):
            result.update(
                status="failed",
                error={
                    "code": "server_error",
                    "message": "Model returned an invalid bash(command:string) call",
                },
            )
        return result

    def event(self, event):
        if "item" in event:
            item = event["item"]
            if item.get("namespace") is None and item.get("name") in self.builtins:
                self.builtin_streams[item["id"]] = self.builtins[item["name"]]
            event = {**event, "item": self.output_item(event["item"])}
        item_id = event.get("item_id")
        if self.builtin_streams.get(item_id) == "shell" and event["type"].startswith(
            "response.function_call_arguments."
        ):
            return None
        if "response" in event:
            event = {**event, "response": self.response(event["response"])}
            if (
                event["type"] == "response.completed"
                and event["response"]["status"] == "failed"
            ):
                event["type"] = "response.failed"
        return event
