# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Adapted from vLLM: https://github.com/vllm-project/vllm/blob/1b9902806915040ac9b3029f2ab7522ec505afc3/vllm/entrypoints/harmony_utils.py
# Slight differences in processing chat messages
import datetime
import logging
import re
from collections.abc import Collection, Iterable
from typing import Optional, Union

import orjson
from openai.types.responses import (
    ResponseCodeInterpreterToolCall,
    ResponseOutputItem,
)
from openai.types.responses import ResponseOutputMessage as OpenAIResponseOutputMessage
from openai.types.responses import (
    ResponseOutputText,
    ResponseReasoningItem,
)
from openai.types.responses.response_function_tool_call import ResponseFunctionToolCall
from openai.types.responses.response_function_web_search import (
    ActionFind,
    ActionOpenPage,
    ActionSearch,
    ResponseFunctionWebSearch,
)
from openai.types.responses.response_reasoning_item import (
    Content as ResponseReasoningTextContent,
)
from openai.types.responses.tool import Tool
from openai_harmony import (
    Author,
    Conversation,
    DeveloperContent,
    HarmonyEncodingName,
    HarmonyError,
    Message,
    ReasoningEffort,
    Role,
    StreamableParser,
    StreamState,
    SystemContent,
    TextContent,
    ToolDescription,
    load_harmony_encoding,
)

from sglang.srt.entrypoints.openai.protocol import (
    ReasoningEffortTier,
    ResponseInputOutputItem,
    ResponseOutputMessage,
)
from sglang.srt.entrypoints.openai.responses_adapters import (
    decode_reasoning_state,
    encode_custom_tool_input,
)
from sglang.srt.utils import random_uuid

logger = logging.getLogger(__name__)

REASONING_EFFORT = {
    "high": ReasoningEffort.HIGH,
    "medium": ReasoningEffort.MEDIUM,
    "low": ReasoningEffort.LOW,
    # harmony has only these three, so outer tiers clamp inward. "none" is left
    # out so it falls through to the harmony default.
    "minimal": ReasoningEffort.LOW,
    "xhigh": ReasoningEffort.HIGH,
    "max": ReasoningEffort.HIGH,
}

_harmony_encoding = None


def get_encoding():
    global _harmony_encoding
    if _harmony_encoding is None:
        _harmony_encoding = load_harmony_encoding(HarmonyEncodingName.HARMONY_GPT_OSS)
    return _harmony_encoding


def get_system_message(
    model_identity: Optional[str] = None,
    reasoning_effort: Optional[ReasoningEffortTier] = None,
    start_date: Optional[str] = None,
    browser_description: Optional[str] = None,
    python_description: Optional[str] = None,
) -> Message:
    sys_msg_content = SystemContent.new()
    if model_identity is not None:
        sys_msg_content = sys_msg_content.with_model_identity(model_identity)
    if reasoning_effort is not None:
        effort = REASONING_EFFORT.get(reasoning_effort)
        if effort is not None:
            sys_msg_content = sys_msg_content.with_reasoning_effort(effort)
    if start_date is None:
        start_date = datetime.datetime.now().strftime("%Y-%m-%d")
    sys_msg_content = sys_msg_content.with_conversation_start_date(start_date)
    if browser_description is not None:
        sys_msg_content = sys_msg_content.with_tools(browser_description)
    if python_description is not None:
        sys_msg_content = sys_msg_content.with_tools(python_description)
    sys_msg = Message.from_role_and_content(Role.SYSTEM, sys_msg_content)
    return sys_msg


def get_developer_message(
    instructions: Optional[str] = None, tools: Optional[list[Tool]] = None
) -> Message:
    dev_msg_content = DeveloperContent.new()
    if instructions is not None:
        dev_msg_content = dev_msg_content.with_instructions(instructions)
    if tools is not None:
        function_tools = []
        for tool in tools:
            if tool.type in (
                "web_search",
                "web_search_preview",
                "code_interpreter",
            ):
                # These are built-in tools that are added to the system message.
                pass
            elif tool.type == "function":
                function_tools.append(tool)
            else:
                # No harmony prompt template for the remaining built-ins;
                # drop them so the request still runs.
                logger.debug(
                    "harmony: ignoring unsupported response tool type %r",
                    tool.type,
                )
        if function_tools:
            function_tool_descriptions = [
                ToolDescription.new(
                    name=tool.name,
                    description=tool.description,
                    parameters=tool.parameters,
                )
                for tool in function_tools
            ]
            dev_msg_content = dev_msg_content.with_function_tools(
                function_tool_descriptions
            )
    dev_msg = Message.from_role_and_content(Role.DEVELOPER, dev_msg_content)
    return dev_msg


def get_user_message(content: str) -> Message:
    return Message.from_role_and_content(Role.USER, content)


def parse_response_input(
    response_msg: ResponseInputOutputItem,
    prev_responses: list[Union[ResponseOutputItem, ResponseReasoningItem]],
) -> Message:
    if not isinstance(response_msg, dict):
        response_msg = response_msg.model_dump()
    if "type" not in response_msg or response_msg["type"] == "message":
        role = response_msg["role"]
        content = response_msg["content"]
        if role == "system":
            # User is trying to set a system message. Change it to:
            # <|start|>developer<|message|># Instructions
            # {instructions}<|end|>
            role = "developer"
            text_prefix = "Instructions:\n"
        else:
            text_prefix = ""
        if isinstance(content, str):
            msg = Message.from_role_and_content(role, text_prefix + content)
        else:
            # Filter to text parts first, then enumerate, so the surviving first
            # text chunk always carries the system→developer text_prefix even if
            # earlier parts were non-text (image/audio) and got dropped.
            text_chunks = [
                c
                for c in content
                if c.get("type") in ("text", "input_text", "output_text")
            ]
            contents = [
                TextContent(text=(text_prefix if i == 0 else "") + c.get("text", ""))
                for i, c in enumerate(text_chunks)
            ]
            msg = Message.from_role_and_contents(role, contents)
        if role == "assistant" and response_msg.get("phase") is not None:
            msg = msg.with_channel(
                "final" if response_msg["phase"] == "final_answer" else "commentary"
            )
    elif response_msg["type"] in ("function_call_output", "custom_tool_call_output"):
        call_id = response_msg["call_id"]
        call_response = None
        for previous in reversed(prev_responses):
            if not isinstance(previous, dict):
                previous = previous.model_dump()
            if (
                previous.get("type") in ("function_call", "custom_tool_call")
                and previous.get("call_id") == call_id
            ):
                call_response = previous
                break
        if call_response is None:
            raise ValueError(f"No call message found for {call_id}")
        output = response_msg.get("output", "")
        if isinstance(output, list):
            output = "".join(
                part.get("text", "") for part in output if isinstance(part, dict)
            )
        msg = Message.from_author_and_content(
            Author.new(Role.TOOL, f"functions.{call_response['name']}"),
            output,
        )
    elif response_msg["type"] == "reasoning":
        text = ""
        for field in ("summary", "content"):
            text = "\n".join(
                part.get("text", "")
                for part in response_msg.get(field) or []
                if isinstance(part, dict)
            )
            if text:
                break
        if not text:
            text = decode_reasoning_state(response_msg.get("encrypted_content")) or ""
        msg = Message.from_role_and_content(Role.ASSISTANT, text).with_channel(
            "analysis"
        )
    elif response_msg["type"] in ("function_call", "custom_tool_call"):
        arguments = (
            encode_custom_tool_input(response_msg.get("input") or "")
            if response_msg["type"] == "custom_tool_call"
            else response_msg.get("arguments") or "{}"
        )
        if isinstance(arguments, dict):
            arguments = orjson.dumps(arguments).decode()
        msg = Message.from_role_and_content(Role.ASSISTANT, arguments)
        msg = msg.with_channel("commentary")
        msg = msg.with_recipient(f"functions.{response_msg['name']}")
        msg = msg.with_content_type("json")
    else:
        raise ValueError(f"Unknown input type: {response_msg['type']}")
    return msg


def parse_response_output(output: ResponseOutputItem) -> Message:
    if isinstance(output, OpenAIResponseOutputMessage):
        role = output.role
        contents = [TextContent(text=c.text) for c in output.content]
        msg = Message.from_role_and_contents(role, contents)
        phase = getattr(output, "phase", None)
        if phase is not None:
            msg = msg.with_channel("final" if phase == "final_answer" else "commentary")
        return msg
    elif isinstance(output, ResponseFunctionToolCall):
        msg = Message.from_role_and_content(Role.ASSISTANT, output.arguments)
        msg = msg.with_channel("commentary")
        msg = msg.with_recipient(output.name)
        msg = msg.with_content_type("json")
        return msg
    else:
        raise ValueError(f"Unknown output type: {type(output)}")


def parse_chat_input(chat_msg) -> Message:
    role = chat_msg.role
    content = chat_msg.content
    if isinstance(content, str):
        contents = [TextContent(text=content)]
    else:
        # TODO: Support refusal.
        contents = [TextContent(text=c.text) for c in content]
    msg = Message.from_role_and_contents(role, contents)
    return msg


def render_for_completion(messages: list[Message]) -> list[int]:
    conversation = Conversation.from_messages(messages)
    token_ids = get_encoding().render_conversation_for_completion(
        conversation, Role.ASSISTANT
    )
    return token_ids


def get_stop_tokens_for_assistant_actions() -> list[int]:
    return get_encoding().stop_tokens_for_assistant_actions()


def get_streamable_parser_for_assistant() -> StreamableParser:
    return StreamableParser(get_encoding(), role=Role.ASSISTANT)


# Narrow output recovery adapted from vLLM PR #59387, commit
# 756ee24f369475638e4bb887828fa45333425864. Prompt rendering is unchanged.
def _harmony_token_id(text: str) -> int:
    return get_encoding().encode(text, allowed_special={text}, disallowed_special=())[0]


def _current_harmony_header(parser: StreamableParser):
    # Native history includes the token that raised. Only inspect it on recovery.
    tokens = parser.tokens
    start = _harmony_token_id("<|start|>")
    offset = next((i for i in range(len(tokens) - 1, -1, -1) if tokens[i] == start), 0)
    return tokens[:offset], tokens[offset:]


def _is_assistant_harmony_prefix(tokens: list[int]) -> bool:
    if tokens and tokens[0] == _harmony_token_id("<|start|>"):
        prefix = get_encoding().decode(tokens[1:])
        pattern = r"assistant(?: to=[A-Za-z0-9_.-]+)?"
    else:
        # Responses contexts prime the native parser with the assistant role.
        prefix = get_encoding().decode(tokens)
        pattern = r"(?: to=[A-Za-z0-9_.-]+)?"
    return re.fullmatch(pattern, prefix) is not None


def _replay_harmony_tokens(tokens: list[int]) -> StreamableParser | None:
    if tokens and tokens[0] == _harmony_token_id("<|start|>"):
        parser = StreamableParser(get_encoding(), role=None)
    else:
        parser = get_streamable_parser_for_assistant()
    try:
        for token in tokens:
            parser.process(token)
    except HarmonyError:
        return None
    return parser


def recover_harmony_header(parser: StreamableParser) -> StreamableParser | None:
    """Repair recognized spliced metadata after a native header parse failure."""
    encoding = get_encoding()
    prefix, tokens = _current_harmony_header(parser)
    message = _harmony_token_id("<|message|>")
    if not tokens or tokens[-1] != message:
        return None
    tokens = tokens[:-1]
    channel_marker = _harmony_token_id("<|channel|>")
    constrain_marker = _harmony_token_id("<|constrain|>")
    try:
        channel = tokens.index(channel_marker)
        constrain = tokens.index(constrain_marker, channel + 1)
    except ValueError:
        return None
    if not _is_assistant_harmony_prefix(tokens[:channel]):
        return None
    header = (
        encoding.decode(tokens[:channel])
        + " "
        + encoding.decode(tokens[channel + 1 : constrain])
    )
    recipients = re.findall(r"(?:^|\s)to=([A-Za-z0-9_.-]+)(?=\s|$)", header)
    if len(recipients) != 1:
        return None
    parts, plain = [], []
    duplicate_marker = False
    for token in tokens[constrain + 1 :]:
        if token in (channel_marker, constrain_marker):
            parts.extend(encoding.decode(plain).split())
            plain.clear()
            duplicate_marker = True
        elif encoding.is_special_token(token):
            return None
        else:
            plain.append(token)
    parts.extend(encoding.decode(plain).split())
    if not parts or (len(parts) < 2 and not duplicate_marker):
        return None
    if parts[-1] not in ("code", "json"):
        return None
    if any(
        part not in ("analysis", "commentary", "final", recipients[0])
        and re.fullmatch(r"to=[A-Za-z0-9_.-]+", part) is None
        for part in parts[:-1]
    ):
        return None
    corrected = (
        prefix
        + tokens[: constrain + 1]
        + encoding.encode(parts[-1], disallowed_special=())
        + [message]
    )
    # Replay preserves earlier messages without keeping a second message store.
    return _replay_harmony_tokens(corrected)


def _harmony_argument_tokens(payload: list[int], boundary: int) -> list[int]:
    # Preserve every argument ID except a token straddling the header/body boundary.
    encoding = get_encoding()
    for index in range(1, len(payload) + 1):
        try:
            prefix = encoding.decode_utf8(payload[:index])
        except HarmonyError:
            continue
        if len(prefix) >= boundary:
            return (
                encoding.encode(prefix[boundary:], disallowed_special=())
                + payload[index:]
            )
    return []


def recover_harmony_missing_message(
    parser: StreamableParser, function_names: Collection[str] | None
) -> StreamableParser | None:
    """Recover a complete declared function call still in HEADER at generation end."""
    if parser.state != StreamState.HEADER:
        return None
    encoding = get_encoding()
    prefix, tokens = _current_harmony_header(parser)
    call = _harmony_token_id("<|call|>")
    if not tokens or tokens[-1] != call:
        return None
    try:
        channel_index = tokens.index(_harmony_token_id("<|channel|>"))
        role_prefix = tokens[:channel_index]
        # Only channel-first assistant function headers are recognized.
        if role_prefix and role_prefix != encoding.encode(
            "<|start|>assistant", allowed_special={"<|start|>"}
        ):
            return None
        payload = tokens[channel_index + 1 : -1]
        constrain = _harmony_token_id("<|constrain|>")
        argument_tokens = None
        if constrain in payload:
            index = payload.index(constrain)
            header = encoding.decode_utf8(payload[:index]).rstrip(" \t")
            if (
                re.fullmatch(
                    r"(?:analysis|commentary) to=functions\.[A-Za-z0-9_-]+", header
                )
                is None
            ):
                return None
            argument_tokens = payload[index + 1 :]
            payload = encoding.encode(header, disallowed_special=()) + argument_tokens
        if any(encoding.is_special_token(token) for token in payload):
            return None
        match = re.fullmatch(
            r"(analysis|commentary) to=(functions\.[A-Za-z0-9_-]+)"
            r"([ \t\r\n]*\{.*\}[ \t\r\n]*)",
            encoding.decode_utf8(payload),
            re.DOTALL,
        )
        if match is None:
            return None
        channel, recipient, arguments = match.groups()
        if (
            function_names is not None
            and recipient.removeprefix("functions.") not in function_names
        ):
            return None
        if not isinstance(orjson.loads(arguments), dict):
            return None
        if argument_tokens is None:
            argument_tokens = _harmony_argument_tokens(payload, match.start(3))
    except (HarmonyError, ValueError, RecursionError):
        return None
    header = encoding.encode(
        f"<|channel|>{channel} to={recipient} <|constrain|>json<|message|>",
        allowed_special={"<|channel|>", "<|constrain|>", "<|message|>"},
    )
    return _replay_harmony_tokens(
        prefix + role_prefix + header + argument_tokens + [call]
    )


def parse_output_message(message: Message):
    if message.author.role != "assistant":
        # This is a message from a tool to the assistant (e.g., search result).
        # Don't include it in the final output for now. This aligns with
        # OpenAI's behavior on models like o4-mini.
        return []

    output_items = []
    recipient = message.recipient
    if recipient is not None and recipient.startswith("browser."):
        if len(message.content) != 1:
            raise ValueError("Invalid number of contents in browser message")
        content = message.content[0]
        browser_call = orjson.loads(content.text)
        # TODO: translate to url properly!
        if recipient == "browser.search":
            action = ActionSearch(
                query=f"cursor:{browser_call.get('query', '')}", type="search"
            )
        elif recipient == "browser.open":
            action = ActionOpenPage(
                url=f"cursor:{browser_call.get('url', '')}", type="open_page"
            )
        elif recipient == "browser.find":
            action = ActionFind(
                pattern=browser_call["pattern"],
                url=f"cursor:{browser_call.get('url', '')}",
                type="find",
            )
        else:
            raise ValueError(f"Unknown browser action: {recipient}")
        web_search_item = ResponseFunctionWebSearch(
            id=f"ws_{random_uuid()}",
            action=action,
            status="completed",
            type="web_search_call",
        )
        output_items.append(web_search_item)
    elif recipient is not None and recipient.startswith("python"):
        output_items.append(
            ResponseCodeInterpreterToolCall(
                id=f"ci_{random_uuid()}",
                type="code_interpreter_call",
                code="".join(content.text for content in message.content),
                container_id="auto",
                outputs=[],
                status="completed",
            )
        )
    elif (
        recipient is not None
        and recipient.startswith("functions.")
        and message.channel in ("analysis", "commentary")
    ):
        function_name = recipient.removeprefix("functions.")
        for content in message.content:
            random_id = random_uuid()
            response_item = ResponseFunctionToolCall(
                arguments=content.text,
                call_id=f"call_{random_id}",
                type="function_call",
                name=function_name,
                id=f"ft_{random_id}",
            )
            output_items.append(response_item)
    elif message.channel == "analysis":
        for content in message.content:
            reasoning_item = ResponseReasoningItem(
                id=f"rs_{random_uuid()}",
                type="reasoning",
                summary=[],
                content=[
                    ResponseReasoningTextContent(
                        text=content.text, type="reasoning_text"
                    )
                ],
                status=None,
            )
            output_items.append(reasoning_item)
    elif message.channel == "commentary" and message.recipient is not None:
        if message.recipient.startswith("python") or message.recipient.startswith(
            "browser"
        ):
            for content in message.content:
                reasoning_item = ResponseReasoningItem(
                    id=f"rs_{random_uuid()}",
                    type="reasoning",
                    summary=[],
                    content=[
                        ResponseReasoningTextContent(
                            text=content.text, type="reasoning_text"
                        )
                    ],
                    status=None,
                )
                output_items.append(reasoning_item)
        else:
            raise ValueError(f"Unknown recipient: {message.recipient}")
    elif message.channel == "final" or (
        message.channel == "commentary" and message.recipient is None
    ):
        contents = []
        for content in message.content:
            output_text = ResponseOutputText(
                text=content.text,
                annotations=[],  # TODO
                type="output_text",
                logprobs=None,  # TODO
            )
            contents.append(output_text)
        text_item = ResponseOutputMessage(
            phase="final_answer" if message.channel == "final" else "commentary",
            id=f"msg_{random_uuid()}",
            content=contents,
            role=message.author.role,
            status="completed",
            type="message",
        )
        output_items.append(text_item)
    else:
        raise ValueError(f"Unknown channel: {message.channel}")
    return output_items


def parse_remaining_state(parser: StreamableParser):
    if not parser.current_content:
        return []
    if parser.current_role != Role.ASSISTANT:
        return []
    current_recipient = parser.current_recipient
    if current_recipient is not None and current_recipient.startswith("browser."):
        return []

    if current_recipient is not None and current_recipient.startswith("python"):
        return [
            ResponseCodeInterpreterToolCall(
                id=f"ci_{random_uuid()}",
                type="code_interpreter_call",
                code=parser.current_content,
                container_id="auto",
                outputs=[],
                status="in_progress",
            )
        ]
    if current_recipient is not None and current_recipient.startswith("functions."):
        random_id = random_uuid()
        return [
            ResponseFunctionToolCall(
                id=f"ft_{random_id}",
                call_id=f"call_{random_id}",
                type="function_call",
                name=current_recipient.removeprefix("functions."),
                arguments=parser.current_content,
                status="in_progress",
            )
        ]
    elif parser.current_channel == "analysis":
        reasoning_item = ResponseReasoningItem(
            id=f"rs_{random_uuid()}",
            type="reasoning",
            summary=[],
            content=[
                ResponseReasoningTextContent(
                    text=parser.current_content, type="reasoning_text"
                )
            ],
            status=None,
        )
        return [reasoning_item]
    elif parser.current_channel == "final" or (
        parser.current_channel == "commentary" and current_recipient is None
    ):
        output_text = ResponseOutputText(
            text=parser.current_content,
            annotations=[],  # TODO
            type="output_text",
            logprobs=None,  # TODO
        )
        text_item = ResponseOutputMessage(
            phase="final_answer" if parser.current_channel == "final" else "commentary",
            id=f"msg_{random_uuid()}",
            content=[output_text],
            role="assistant",
            status="completed",
            type="message",
        )
        return [text_item]
    return []


def parse_output_into_messages(token_ids: Iterable[int]):
    parser = get_streamable_parser_for_assistant()
    for token_id in token_ids:
        parser.process(token_id)
    return parser
