from typing import Optional

from jsonschema import Draft202012Validator, SchemaError

from sglang.srt.entrypoints.chat_input.schema import ChatCompletionMessageGenericParam
from sglang.srt.entrypoints.chat_input.types import ChatInput
from sglang.srt.function_call.utils import normalize_json_schema_types


def validate_chat_input(request: ChatInput) -> Optional[str]:
    if not request.messages:
        return "Messages cannot be empty."

    effective_tools = request.effective_tools()
    has_message_tools = any(
        isinstance(message, ChatCompletionMessageGenericParam)
        and message.role in ("system", "developer")
        and message.tools
        for message in request.messages
    )
    if (
        isinstance(request.tool_choice, str)
        and request.tool_choice.lower() == "required"
        and not effective_tools
    ):
        return "Tools cannot be empty if tool choice is set to required."

    if request.tool_choice is not None and not isinstance(request.tool_choice, str):
        if not effective_tools:
            return "Tools cannot be empty if tool choice is set to a specific tool."
        tool_name = request.tool_choice.function.name
        tool_exists = any(tool.function.name == tool_name for tool in effective_tools)
        if not tool_exists:
            return f"Tool '{tool_name}' not found in tools list."

    if has_message_tools:
        names = [tool.function.name for tool in effective_tools]
        if len(names) != len(set(names)):
            return "Tool names must be unique across request and message tools."

    # Validate tool definitions
    for i, tool in enumerate(effective_tools):
        if tool.function.parameters is None:
            continue
        try:
            # Rewrite DB/ORM-style aliases (e.g. "varchar", "enum", "int")
            # to standard JSON Schema types before validation. RecursionError
            # guards against hand-crafted cyclic schemas so the request gets
            # a 400 instead of crashing into a 500.
            normalize_json_schema_types(tool.function.parameters)
            Draft202012Validator.check_schema(tool.function.parameters)
        except SchemaError as e:
            return f"Tool {i} function has invalid 'parameters' schema: {str(e)}"
        except RecursionError:
            return (
                f"Tool {i} function 'parameters' schema is too deeply nested "
                "or contains a cycle."
            )

    if request.response_format and request.response_format.type == "json_schema":
        schema = getattr(request.response_format.json_schema, "schema_", None)
        if schema is None:
            return "schema_ is required for json_schema response format request."

    return None


class MediaInputError(ValueError):
    pass


def validate_media_content(request: ChatInput, is_multimodal: bool) -> Optional[str]:
    if is_multimodal:
        return None
    for message in request.messages:
        if not isinstance(message.content, list):
            continue
        for part in message.content:
            if part.type in ("image_url", "video_url", "audio_url"):
                return (
                    "Model only supports text input; "
                    + f"received unsupported content type '{part.type}'."
                )
    return None
