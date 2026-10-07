# SPDX-License-Identifier: Apache-2.0
"""Adapt Beam's trained built-in tools independently of namespace mapping."""

import json
from typing import Union

from openai.types.responses import ResponseFunctionShellToolCall
from openai.types.responses.response_input_item_param import ShellCall, ShellCallOutput
from pydantic import TypeAdapter

from sglang.srt.entrypoints.openai.responses_tools.adapter import ResponseToolAdapter

_SHELL_INPUT = TypeAdapter(Union[ShellCall, ShellCallOutput])


class BeamToolAdapter(ResponseToolAdapter):
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
            elif tool["type"] == "local_shell":
                raise ValueError(
                    "Legacy local_shell argv/env/cwd/user semantics are not implemented; use shell with environment={type:local}"
                )
            else:
                self._add(tool)

    def prepare(self, request, *, tool_call_parser):
        if self.builtins:
            if tool_call_parser not in ("beam", "response_template"):
                raise ValueError(
                    "Beam built-in tools require the beam or response_template tool-call parser"
                )
        return request.chat_template_kwargs

    def template_kwargs(self, request):
        """Pass built-ins separately for the template's system tool block."""
        if not self.builtins:
            return request.chat_template_kwargs
        tools = request.tools or []
        if request.tool_choice == "none":
            tools = []
        elif not isinstance(request.tool_choice, str):
            tools = [
                t for t in tools if t.function.name == request.tool_choice.function.name
            ]
        functions, builtins = [], []
        for tool in tools:
            definition = tool.model_dump(exclude_none=True)
            if tool.function.name in self.builtins:
                builtins.append(definition)
            else:
                functions.append(definition)
        kwargs = dict(request.chat_template_kwargs or {})
        for key, value in (("tools", functions), ("builtin_tools", builtins)):
            if key in kwargs and kwargs[key] != value:
                raise ValueError(f"{key} is managed by Responses built-in tools")
            kwargs[key] = value
        return kwargs

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
        if hasattr(item, "model_dump"):
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
        return item

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
