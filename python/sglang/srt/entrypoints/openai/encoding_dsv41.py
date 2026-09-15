# Adapted from the DeepSeek-V4.1 release reference implementation.
"""Encode DeepSeek-V4.1 chat messages.

Supports tool calls, reasoning effort, quick instruction tasks and image content.
Mid-conversation system messages trigger the assistant generation header;
image records are returned alongside the encoded prompt.
"""

import copy
import json
from typing import Any, Dict, List, Optional, Tuple, Union

# ============================================================
# Special Tokens
# ============================================================

bos_token: str = "<｜begin▁of▁sentence｜>"
eos_token: str = "<｜end▁of▁sentence｜>"
thinking_start_token: str = "<think>"
thinking_end_token: str = "</think>"
dsml_token: str = "｜DSML｜"

USER_SP_TOKEN = "<｜User｜>"
ASSISTANT_SP_TOKEN = "<｜Assistant｜>"
SYSTEM_SP_TOKEN = "<｜System｜>"
LATEST_REMINDER_SP_TOKEN = "<｜latest_reminder｜>"

IMAGE_PLACEHOLDER = "<｜deepseek_image｜>"

# Task special tokens for internal classification tasks
DS_TASK_SP_TOKENS = {
    "action": "<｜action｜>",
    "query": "<｜query｜>",
    "authority": "<｜authority｜>",
    "domain": "<｜domain｜>",
    "title": "<｜title｜>",
    "read_url": "<｜read_url｜>",
}
VALID_TASKS = set(DS_TASK_SP_TOKENS.keys())

# ============================================================
# Templates
# ============================================================

system_msg_template: str = "{content}"
user_msg_template: str = "{content}"
latest_reminder_msg_template: str = "{content}"
assistant_msg_template: str = "{reasoning}{content}{tool_calls}" + eos_token
assistant_msg_wo_eos_template: str = "{reasoning}{content}{tool_calls}"
thinking_template: str = "{reasoning_content}"

response_format_template: str = "## Response Format:\n\nYou MUST strictly adhere to the following schema to reply:\n{schema}"

tool_calls_block_name: str = " calls"
tool_call_tag_name: str = " invoke"
tool_parameter_tag_name: str = " parameter"

tool_call_template: str = '<{dsml_token}{tool_call_tag_name} name="{name}">\n{arguments}\n</{dsml_token}{tool_call_tag_name}>'
tool_calls_template = (
    "<{dsml_token}{tc_block_name}>\n{tool_calls}\n</{dsml_token}{tc_block_name}>"
)

tool_output_template: str = "<tool_result>{content}</tool_result>"

REASONING_EFFORT_TEMPLATE = (
    "Reasoning Effort: {budget} "
    "(range 1-100, the higher the value, the more thorough the reasoning)\n\n"
)

REASONING_EFFORT_MAPPINGS: Dict[str, int] = {
    "low": 25,
    "high": 50,
    "xhigh": 75,
    "max": 100,
}
DEFAULT_REASONING_EFFORT = "high"

TOOLS_TEMPLATE = """## Tools

You have access to a set of tools to help answer the user's question. You can invoke tools by writing a "<{dsml_token}{tc_block_name}>" block like the following:

<{dsml_token}{tc_block_name}>
<{dsml_token}{tool_call_tag_name} name="$TOOL_NAME">
<{dsml_token}{tool_parameter_tag_name} name="$PARAMETER_NAME" string="true|false">$PARAMETER_VALUE</{dsml_token}{tool_parameter_tag_name}>
...
</{dsml_token}{tool_call_tag_name}>
<{dsml_token}{tool_call_tag_name} name="$TOOL_NAME2">
...
</{dsml_token}{tool_call_tag_name}>
</{dsml_token}{tc_block_name}>

String parameters should be specified as is and set `string="true"`. For all other types (numbers, booleans, arrays, objects), pass the value in JSON format and set `string="false"`.

If thinking_mode is enabled (triggered by {thinking_start_token}), you MUST output your complete reasoning inside {thinking_start_token}...{thinking_end_token} BEFORE any tool calls or final response.

Otherwise, output directly after {thinking_end_token} with tool calls or final response.

### Available Tool Schemas

{tool_schemas}

You MUST strictly follow the above defined tool name and parameter schemas to invoke tool calls.
"""

# ============================================================
# Utility Functions
# ============================================================


def to_json(value: Any) -> str:
    """Serialize a value to JSON string."""
    try:
        return json.dumps(value, ensure_ascii=False)
    except:
        return json.dumps(value, ensure_ascii=True)


def tools_from_openai_format(tools):
    """Extract function definitions from OpenAI-format tool list."""
    return [tool["function"] for tool in tools]


def tool_calls_from_openai_format(tool_calls):
    """Convert OpenAI-format tool calls to internal format."""
    return [
        {
            "name": tool_call["function"]["name"],
            "arguments": tool_call["function"]["arguments"],
        }
        for tool_call in tool_calls
    ]


def encode_arguments_to_dsml(tool_call: Dict[str, Any]) -> str:
    """Encode tool call arguments into V4.1 DSML parameter format."""
    p_dsml_template = (
        '<{dsml_token}{tool_parameter_tag_name} name="{key}" string="{is_str}">'
        "{value}</{dsml_token}{tool_parameter_tag_name}>"
    )
    P_dsml_strs = []

    raw_arguments = tool_call["arguments"]
    arguments = (
        json.loads(raw_arguments) if isinstance(raw_arguments, str) else raw_arguments
    )
    if not isinstance(arguments, dict):
        raise ValueError(
            "Assistant tool call function.arguments must be a JSON object."
        )

    for k, v in arguments.items():
        P_dsml_strs.append(
            p_dsml_template.format(
                dsml_token=dsml_token,
                tool_parameter_tag_name=tool_parameter_tag_name,
                key=k,
                is_str="true" if isinstance(v, str) else "false",
                value=v if isinstance(v, str) else to_json(v),
            )
        )

    return "\n".join(P_dsml_strs)


def render_tools(tools: List[Dict[str, Union[str, Dict[str, Any]]]]) -> str:
    """Render tool schemas into the V4.1 system prompt format."""
    tools_json = [to_json(t) for t in tools]

    return TOOLS_TEMPLATE.format(
        tool_schemas="\n".join(tools_json),
        dsml_token=dsml_token,
        tc_block_name=tool_calls_block_name,
        tool_call_tag_name=tool_call_tag_name,
        tool_parameter_tag_name=tool_parameter_tag_name,
        thinking_start_token=thinking_start_token,
        thinking_end_token=thinking_end_token,
    )


def render_reasoning_effort(
    index: int,
    thinking_mode: str,
    effort: Union[str, int, None],
) -> str:
    """Render the numeric reasoning effort prefix (thinking mode, index 0 only)."""
    if effort is None:
        effort = DEFAULT_REASONING_EFFORT
    if not (
        (type(effort) is int and 1 <= effort <= 100)
        or effort in REASONING_EFFORT_MAPPINGS
    ):
        raise ValueError(
            f"Invalid reasoning effort for deepseek_v41: {effort!r}, should be "
            f"int within [1,100] or {list(REASONING_EFFORT_MAPPINGS)}"
        )
    if type(effort) is str:
        effort = REASONING_EFFORT_MAPPINGS[effort]
    if index == 0 and thinking_mode == "thinking":
        return REASONING_EFFORT_TEMPLATE.format(budget=effort)
    return ""


def find_last_user_index(messages: List[Dict[str, Any]]) -> int:
    """Mid-conversation system messages also count as user messages here;
    they trigger the assistant generation header.
    """
    last_user_index = -1
    for idx in range(len(messages) - 1, -1, -1):
        role = messages[idx].get("role")
        if role in ["user", "developer"] or (role == "system" and idx > 0):
            last_user_index = idx
            break
    return last_user_index


def attach_task_to_last_user_message(messages: List[Dict[str, Any]], task: str) -> None:
    """Set `task` on the most recent user/developer message; raise if none exists."""
    idx = find_last_user_index(messages)
    if idx == -1:
        raise ValueError(
            "`task` requires at least one message with role='user' or 'developer'."
        )
    messages[idx]["task"] = task


# ============================================================
# Message Rendering
# ============================================================


def render_message(
    index: int,
    messages: List[Dict[str, Any]],
    thinking_mode: str,
    drop_thinking: bool = True,
    reasoning_effort: Union[str, int, None] = None,
) -> str:
    """Render the message at `index` into its DeepSeek-V4.1 encoded string form."""
    assert 0 <= index < len(messages)
    assert thinking_mode in [
        "chat",
        "thinking",
    ], f"Invalid thinking_mode `{thinking_mode}`"

    msg = messages[index]
    last_user_idx = find_last_user_index(messages)

    role = msg.get("role")
    content = msg.get("content")
    tools = msg.get("tools")
    response_format = msg.get("response_format")
    tool_calls = msg.get("tool_calls")
    reasoning_content = msg.get("reasoning_content")
    wo_eos = msg.get("wo_eos", False)

    if tools:
        tools = tools_from_openai_format(tools)
    if tool_calls:
        tool_calls = tool_calls_from_openai_format(tool_calls)

    reasoning_effort_prompt = render_reasoning_effort(
        index, thinking_mode, reasoning_effort
    )
    # The leading system token is emitted whenever there is something to host
    # at index 0: a system message or the effort prompt (even before a user).
    prompt = (
        SYSTEM_SP_TOKEN
        if index == 0 and (reasoning_effort_prompt or role == "system")
        else ""
    )
    prompt += reasoning_effort_prompt

    if role == "system":
        if index > 0:
            prompt += SYSTEM_SP_TOKEN
        prompt += system_msg_template.format(content=content or "")
        if tools:
            prompt += "\n\n" + render_tools(tools)
        if response_format:
            prompt += "\n\n" + response_format_template.format(
                schema=to_json(response_format)
            )

    elif role == "developer":
        assert content, f"Invalid message for role `{role}`: {msg}"

        content_developer = USER_SP_TOKEN
        content_developer += content

        if tools:
            content_developer += "\n\n" + render_tools(tools)
        if response_format:
            content_developer += "\n\n" + response_format_template.format(
                schema=to_json(response_format)
            )

        prompt += user_msg_template.format(content=content_developer)

    elif role == "user":
        prompt += USER_SP_TOKEN

        # Handle content blocks (tool results mixed with text)
        content_blocks = msg.get("content_blocks")
        if content_blocks:
            parts = []
            for block in content_blocks:
                block_type = block.get("type")
                if block_type == "text":
                    parts.append(block.get("text", ""))
                elif block_type == "tool_result":
                    tool_content = block.get("content", "")
                    if isinstance(tool_content, list):
                        text_parts = []
                        for b in tool_content:
                            if b.get("type") == "text":
                                text_parts.append(b.get("text", ""))
                            else:
                                text_parts.append(f"[Unsupported {b.get('type')}]")
                        tool_content = "\n\n".join(text_parts)
                    parts.append(tool_output_template.format(content=tool_content))
                else:
                    parts.append(f"[Unsupported {block_type}]")
            prompt += "\n\n".join(parts)
        else:
            prompt += content or ""

    elif role == "latest_reminder":
        prompt += LATEST_REMINDER_SP_TOKEN + latest_reminder_msg_template.format(
            content=content
        )

    elif role == "tool":
        raise NotImplementedError(
            "deepseek_v41 merges tool messages into user; please preprocess with merge_tool_messages()"
        )

    elif role == "assistant":
        thinking_part = ""
        tc_content = ""

        if tool_calls:
            tc_list = [
                tool_call_template.format(
                    dsml_token=dsml_token,
                    tool_call_tag_name=tool_call_tag_name,
                    name=tc.get("name"),
                    arguments=encode_arguments_to_dsml(tc),
                )
                for tc in tool_calls
            ]
            tc_content += "\n\n" + tool_calls_template.format(
                dsml_token=dsml_token,
                tool_calls="\n".join(tc_list),
                tc_block_name=tool_calls_block_name,
            )

        summary_content = content or ""
        rc = reasoning_content or ""

        # Check if previous message has a task - if so, this is a task output (no thinking)
        prev_has_task = index - 1 >= 0 and messages[index - 1].get("task") is not None

        if thinking_mode == "thinking" and not prev_has_task:
            if not drop_thinking or index > last_user_idx:
                thinking_part = (
                    thinking_template.format(reasoning_content=rc) + thinking_end_token
                )
            else:
                thinking_part = ""

        if wo_eos:
            prompt += assistant_msg_wo_eos_template.format(
                reasoning=thinking_part,
                content=summary_content,
                tool_calls=tc_content,
            )
        else:
            prompt += assistant_msg_template.format(
                reasoning=thinking_part,
                content=summary_content,
                tool_calls=tc_content,
            )
    else:
        raise NotImplementedError(f"Unknown role: {role}")

    # Append transition tokens based on what follows
    if index + 1 < len(messages) and messages[index + 1].get("role") not in [
        "assistant",
        "latest_reminder",
    ]:
        return prompt

    task = messages[index].get("task")
    if task is not None:
        # Task special token for internal classification tasks
        assert task in VALID_TASKS, (
            f"Invalid task: '{task}'. Valid tasks are: {list(VALID_TASKS)}"
        )
        task_sp_token = DS_TASK_SP_TOKENS[task]

        if task != "action":
            # Non-action tasks: append task sp token directly after the message
            prompt += task_sp_token
        else:
            # Action task: append Assistant + thinking token + action sp token
            prompt += ASSISTANT_SP_TOKEN
            prompt += (
                thinking_end_token
                if thinking_mode != "thinking"
                else thinking_start_token
            )
            prompt += task_sp_token

    elif messages[index].get("role") in ["user", "developer"] or (
        messages[index].get("role") == "system" and index > 0
    ):
        # Normal generation: append Assistant + thinking token
        prompt += ASSISTANT_SP_TOKEN
        if not drop_thinking and thinking_mode == "thinking":
            prompt += thinking_start_token
        elif drop_thinking and thinking_mode == "thinking" and index >= last_user_idx:
            prompt += thinking_start_token
        else:
            prompt += thinking_end_token

    return prompt


# ============================================================
# Preprocessing
# ============================================================


def merge_tool_messages(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Tool results are encoded within user messages;
    DeepSeek-V4.1 has no standalone tool role.
    """
    merged: List[Dict[str, Any]] = []

    for msg in messages:
        msg = copy.deepcopy(msg)
        role = msg.get("role")

        if role == "tool":
            # Convert tool message to a user message with tool_result block
            tool_block = {
                "type": "tool_result",
                "tool_use_id": msg.get("tool_call_id", ""),
                "content": msg.get("content", ""),
            }
            # Merge into previous message if it's already a user (merged tool)
            if (
                merged
                and merged[-1].get("role") == "user"
                and "content_blocks" in merged[-1]
            ):
                merged[-1]["content_blocks"].append(tool_block)
            else:
                merged.append(
                    {
                        "role": "user",
                        "content_blocks": [tool_block],
                    }
                )
        elif role == "user":
            content_blocks = msg.get("content_blocks")
            if content_blocks is None:
                content_blocks = [{"type": "text", "text": msg.get("content", "")}]
            if (
                merged
                and merged[-1].get("role") == "user"
                and "content_blocks" in merged[-1]
                and merged[-1].get("task") is None
            ):
                merged[-1]["content_blocks"].extend(content_blocks)
            else:
                # Keeps structured content and every message-level field.
                new_msg = msg
                new_msg["content_blocks"] = content_blocks
                merged.append(new_msg)
        else:
            merged.append(msg)

    return merged


def sort_tool_results_by_call_order(
    messages: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """
    Sort tool_result blocks within user messages by the order of tool_calls
    in the preceding assistant message.
    """
    last_tool_call_order: Dict[str, int] = {}

    for msg in messages:
        role = msg.get("role")
        if role == "assistant" and msg.get("tool_calls"):
            last_tool_call_order = {}
            for idx, tc in enumerate(msg["tool_calls"]):
                tc_id = tc.get("id") or tc.get("function", {}).get("id", "")
                if tc_id:
                    last_tool_call_order[tc_id] = idx

        elif role == "user" and msg.get("content_blocks"):
            tool_blocks = [
                b for b in msg["content_blocks"] if b.get("type") == "tool_result"
            ]
            if len(tool_blocks) > 1 and last_tool_call_order:
                sorted_blocks = sorted(
                    tool_blocks,
                    key=lambda b: last_tool_call_order.get(b.get("tool_use_id", ""), 0),
                )
                sorted_idx = 0
                new_blocks = []
                for block in msg["content_blocks"]:
                    if block.get("type") == "tool_result":
                        new_blocks.append(sorted_blocks[sorted_idx])
                        sorted_idx += 1
                    else:
                        new_blocks.append(block)
                msg["content_blocks"] = new_blocks

    return messages


# ============================================================
# Vision Message Preprocessing
# ============================================================


def _is_image_block(block: Dict[str, Any]) -> bool:
    return isinstance(block, dict) and block.get("type") in ("image", "image_url")


def _extract_image(block: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize a supported image block into an internal image record."""
    record: Dict[str, Any] = {"type": "image"}
    if block.get("type") == "image_url":
        image_url = block.get("image_url")
        if isinstance(image_url, str):
            record["url"] = image_url
        else:
            record["url"] = (image_url or {}).get("url", "")
    else:
        for key in ("source", "url", "data"):
            if key in block:
                record[key] = block[key]
    if not any(record.get(key) for key in ("source", "url", "data")):
        raise ValueError("Image block does not contain a valid source")
    return record


def _process_image_blocks(
    blocks: List[Any], image_placeholder: str = IMAGE_PLACEHOLDER
) -> Tuple[List[Any], List[Dict[str, Any]]]:
    """Replace image blocks with placeholders and collect their records in order."""
    new_blocks: List[Any] = []
    images: List[Dict[str, Any]] = []
    for block in blocks:
        if not isinstance(block, dict):
            new_blocks.append(block)
            continue
        if _is_image_block(block):
            new_blocks.append({"type": "text", "text": image_placeholder})
            images.append(_extract_image(block))
        elif block.get("type") == "tool_result" and isinstance(
            block.get("content"), list
        ):
            block = copy.copy(block)
            block["content"], nested_images = _process_image_blocks(
                block["content"], image_placeholder
            )
            new_blocks.append(block)
            images.extend(nested_images)
        elif block.get("type") == "text":
            text = block.get("text") or ""
            if IMAGE_PLACEHOLDER in text:
                raise ValueError(
                    f"Text block contains image placeholder '{IMAGE_PLACEHOLDER}': "
                    f"'{text[:100]}'. Images should be separate content blocks."
                )
            new_blocks.append(block)
        else:
            new_blocks.append(block)
    return new_blocks, images


def _validate_no_image_sp_tokens(msg: Dict[str, Any]) -> None:
    """Reject user-supplied image placeholder tokens in textual fields."""
    content = msg.get("content")
    if isinstance(content, str) and IMAGE_PLACEHOLDER in content:
        raise ValueError(
            f"Message content contains image special token '{IMAGE_PLACEHOLDER}'. "
            "Images should be provided as image content blocks."
        )
    reasoning_content = msg.get("reasoning_content")
    if isinstance(reasoning_content, str) and IMAGE_PLACEHOLDER in reasoning_content:
        raise ValueError(
            f"reasoning_content contains image special token '{IMAGE_PLACEHOLDER}'"
        )


def process_image_messages(
    messages: List[Dict[str, Any]],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Normalize image blocks and return their records in prompt order."""
    processed: List[Dict[str, Any]] = []
    images: List[Dict[str, Any]] = []
    for msg in messages:
        msg = copy.deepcopy(msg)
        _validate_no_image_sp_tokens(msg)

        if isinstance(msg.get("content"), list) and "content_blocks" not in msg:
            msg["content_blocks"] = msg.pop("content")

        if msg.get("content_blocks"):
            msg["content_blocks"], message_images = _process_image_blocks(
                msg["content_blocks"]
            )
            images.extend(message_images)
            if not isinstance(msg.get("content"), str):
                texts = [
                    block.get("text", "")
                    for block in msg["content_blocks"]
                    if isinstance(block, dict) and block.get("type") == "text"
                ]
                msg["content"] = "\n\n".join(texts)

        processed.append(msg)
    return processed, images


# ============================================================
# Main Encoding Function
# ============================================================


def _drop_thinking_messages(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    last_user_idx = find_last_user_index(messages)
    result = []
    keep_roles = {"user", "system", "tool", "latest_reminder", "direct_search_results"}

    for idx, msg in enumerate(messages):
        role = msg.get("role")
        if role in keep_roles or idx >= last_user_idx:
            result.append(msg)
        elif role == "assistant":
            msg = copy.copy(msg)
            msg.pop("reasoning_content", None)
            result.append(msg)
        # developer and other roles before last_user_idx are dropped

    return result


def _encode_messages_text(
    messages: List[Dict[str, Any]],
    thinking_mode: str,
    context: Optional[List[Dict[str, Any]]] = None,
    drop_thinking: bool = True,
    add_default_bos_token: bool = True,
    reasoning_effort: Union[str, int, None] = None,
) -> str:
    """Encode preprocessed (text-only) messages into the V4.1 prompt format."""
    context = context if context else []

    # Preprocess: merge tool messages and sort tool results
    messages = merge_tool_messages(messages)
    messages = sort_tool_results_by_call_order(context + messages)[len(context) :]
    if context:
        context = merge_tool_messages(context)
        context = sort_tool_results_by_call_order(context)

    full_messages = context + messages

    prompt = bos_token if add_default_bos_token and len(context) == 0 else ""

    # Resolve drop_thinking: if any message has tools defined, don't drop thinking
    effective_drop_thinking = drop_thinking
    if any(m.get("tools") for m in full_messages):
        effective_drop_thinking = False

    if thinking_mode == "thinking" and effective_drop_thinking:
        full_messages = _drop_thinking_messages(full_messages)
        num_to_render = len(full_messages) - len(_drop_thinking_messages(context))
        context_len = len(full_messages) - num_to_render
    else:
        num_to_render = len(messages)
        context_len = len(context)

    for idx in range(num_to_render):
        prompt += render_message(
            idx + context_len,
            full_messages,
            thinking_mode=thinking_mode,
            drop_thinking=effective_drop_thinking,
            reasoning_effort=reasoning_effort,
        )

    return prompt


def encode_messages(
    messages: List[Dict[str, Any]],
    thinking_mode: str,
    context: Optional[List[Dict[str, Any]]] = None,
    drop_thinking: bool = True,
    add_default_bos_token: bool = True,
    reasoning_effort: Union[str, int, None] = None,
    return_multi_modal_data: bool = False,
) -> Any:
    """
    Encode a list of messages into the DeepSeek-V4.1 prompt format.

    Handles BOS insertion, thinking mode with optional reasoning dropping, tool
    message merging, multi-turn context, and image content blocks. Returns the
    prompt string, or ``(prompt, {"images": [...]})`` when
    ``return_multi_modal_data`` is set; the image records are in prompt order.
    """
    context = context or []
    processed_context, _ = process_image_messages(context) if context else ([], [])
    processed_messages, images = process_image_messages(messages)
    prompt = _encode_messages_text(
        processed_messages,
        thinking_mode=thinking_mode,
        context=processed_context if processed_context else None,
        drop_thinking=drop_thinking,
        add_default_bos_token=add_default_bos_token,
        reasoning_effort=reasoning_effort,
    )
    if return_multi_modal_data:
        return prompt, {"images": images}
    return prompt
