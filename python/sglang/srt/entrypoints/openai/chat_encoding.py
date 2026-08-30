"""Single home for the chat-encoding dispatch.

Which encoder turns chat messages into prompt tokens is a property of the
model, so the serving path and offline tools (benchmarks, evals) must resolve
it here instead of re-deriving it from model architectures themselves.
"""

from __future__ import annotations

import ast
import hashlib
import logging
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Dict, List, Optional

from sglang.srt.entrypoints.openai import encoding_dsv4

logger = logging.getLogger(__name__)

DSV4_REASONING_EFFORT_PROFILE_OVERRIDE = "dsv4_reasoning_effort_profile"
_DSV4_REASONING_EFFORT_ENCODER = "encoding/encoding_dsv4.py"
_MAX_DSV4_ENCODER_BYTES = 1 << 20

_GLM_TOOL_RESULT_SORT_START = "    {%- set ns_a = namespace(tool_calls=none) -%}"
_GLM_TOOL_RESULT_SORT_END = "\n{% endif -%}\n{%- elif m.role == 'system' -%}"
# sha256 of the excised region in the zai-org/GLM-5.3-Flash template at HF
# revision 04c4e9e9; order_glm_tool_results replicates exactly that region.
_GLM_TOOL_RESULT_SORT_SHA256 = (
    "f585e1f2937c781d8ce1234622eb032d99268e5f876f06c752599f2dd29c821a"
)
_GLM_LINEAR_TOOL_RESULT_RENDER = """    {%- for k in range(block_start, ns_blk.end + 1) -%}
        {{- render_tool_response(messages[k]) -}}
    {%- endfor -%}"""


def _tool_call_id(item: Any) -> Optional[str]:
    if not isinstance(item, Mapping):
        return None
    value = item.get("tool_call_id") or item.get("id")
    return str(value) if value else None


def _canonical_tool_output(item: Any) -> Any:
    # The template's sorted path renders only entry.output; other keys (e.g. a
    # "type") would send the split-off entry down a different template branch.
    if not isinstance(item, Mapping):
        return item
    return {key: item[key] for key in ("tool_call_id", "id", "output") if key in item}


def _is_list_of_tool_outputs(message: Mapping[str, Any]) -> bool:
    content = message.get("content")
    return bool(
        isinstance(content, list)
        and content
        and isinstance(content[0], Mapping)
        and "output" in content[0]
    )


def _order_tool_result_block(
    tool_calls: List[Dict[str, Any]],
    tool_results: List[Dict[str, Any]],
) -> Optional[List[Dict[str, Any]]]:
    call_ids = []
    seen_call_ids = set()
    for tool_call in tool_calls:
        call_id = _tool_call_id(tool_call)
        if call_id is None or call_id in seen_call_ids:
            return None
        call_ids.append(call_id)
        seen_call_ids.add(call_id)

    results_by_id = {}
    for message in tool_results:
        if _is_list_of_tool_outputs(message):
            units = [
                ({"role": "tool", "content": [_canonical_tool_output(item)]}, item)
                for item in message["content"]
            ]
        else:
            units = [(message, message)]

        for unit, item in units:
            result_id = _tool_call_id(item)
            if (
                result_id is None
                or result_id in results_by_id
                or result_id not in seen_call_ids
            ):
                return None
            results_by_id[result_id] = unit

    return [results_by_id[call_id] for call_id in call_ids if call_id in results_by_id]


def order_glm_tool_results(
    messages: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Order valid GLM tool-result blocks by their declared tool calls."""
    ordered_messages = list(messages)
    index = 0
    while index + 1 < len(ordered_messages):
        message = ordered_messages[index]
        tool_calls = message.get("tool_calls")
        if message.get("role") != "assistant" or not isinstance(tool_calls, list):
            index += 1
            continue

        block_start = index + 1
        block_end = block_start
        while (
            block_end < len(ordered_messages)
            and ordered_messages[block_end].get("role") == "tool"
        ):
            block_end += 1
        if block_end == block_start:
            index += 1
            continue

        ordered_block = _order_tool_result_block(
            tool_calls, ordered_messages[block_start:block_end]
        )
        if ordered_block is not None:
            ordered_messages[block_start:block_end] = ordered_block
            block_end = block_start + len(ordered_block)
        index = block_end

    return ordered_messages


def resolve_glm_tool_result_template(
    *, hf_config: Any, tokenizer: Any
) -> Optional[str]:
    """Replace GLM's quadratic Jinja result ordering with a linear render."""
    architectures = hf_config.architectures
    if not any(
        arch in ("Glm5NextForConditionalGeneration", "GlmMoeDsaForCausalLM")
        for arch in architectures or []
    ):
        return None

    template = getattr(tokenizer, "chat_template", None)
    if (
        not isinstance(template, str)
        or template.count(_GLM_TOOL_RESULT_SORT_START) != 1
    ):
        return None

    start = template.index(_GLM_TOOL_RESULT_SORT_START)
    end = template.find(_GLM_TOOL_RESULT_SORT_END, start)
    if (
        end < 0
        or hashlib.sha256(template[start:end].encode("utf-8")).hexdigest()
        != _GLM_TOOL_RESULT_SORT_SHA256
    ):
        logger.info(
            "GLM chat template does not match the known quadratic tool-result "
            "ordering region; keeping the stock template."
        )
        return None
    logger.info("Replacing quadratic GLM tool-result ordering with a linear render.")
    return template[:start] + _GLM_LINEAR_TOOL_RESULT_RENDER + template[end:]


def _detect_dsv4_reasoning_effort_profile(
    model_path: str, revision: Optional[str] = None
) -> Optional[str]:
    encoder_path = Path(model_path) / _DSV4_REASONING_EFFORT_ENCODER
    try:
        if not encoder_path.is_file():
            from huggingface_hub import hf_hub_download

            encoder_path = Path(
                hf_hub_download(
                    model_path,
                    _DSV4_REASONING_EFFORT_ENCODER,
                    revision=revision,
                )
            )
        if encoder_path.stat().st_size > _MAX_DSV4_ENCODER_BYTES:
            return None
        tree = ast.parse(encoder_path.read_text(encoding="utf-8"))
    except Exception as error:
        logger.debug(
            "Could not inspect DeepSeek-V4 checkpoint encoder at %s: %s",
            encoder_path,
            error,
        )
        return None

    assignments = {}
    for node in tree.body:
        if isinstance(node, ast.Assign):
            targets = node.targets
            value = node.value
        elif isinstance(node, ast.AnnAssign):
            targets = [node.target]
            value = node.value
        else:
            continue

        for target in targets:
            if not isinstance(target, ast.Name):
                continue
            try:
                assignments[target.id] = ast.literal_eval(value)
            except (TypeError, ValueError):
                continue

    prompts = assignments.get("REASONING_EFFORT_PROMPTS")
    if (
        assignments.get("DEFAULT_REASONING_EFFORT") == "low"
        and isinstance(prompts, dict)
        and {"low", "high", "max"} <= prompts.keys()
    ):
        return "official"
    if "REASONING_EFFORT_MAX" in assignments:
        return "preview"
    return None


def _validate_dsv4_reasoning_effort_profile(profile: str) -> str:
    if profile not in encoding_dsv4.REASONING_EFFORT_PROFILES:
        raise ValueError(
            f"Invalid {DSV4_REASONING_EFFORT_PROFILE_OVERRIDE}: {profile!r}; "
            f"expected one of {list(encoding_dsv4.REASONING_EFFORT_PROFILES)}"
        )
    return profile


def resolve_dsv4_reasoning_effort_profile(
    *,
    model_path: str,
    revision: Optional[str] = None,
    override: Optional[str] = None,
) -> str:
    if override is not None:
        return _validate_dsv4_reasoning_effort_profile(override)

    return (
        _detect_dsv4_reasoning_effort_profile(
            model_path=model_path,
            revision=revision,
        )
        or "preview"
    )


def resolve_chat_encoding_spec(
    *,
    hf_config: Any,
    tokenizer: Any,
    tool_call_parser: Optional[str] = None,
) -> Optional[str]:
    """Return the chat encoding spec for a model.

    None means the default path (HF chat template); any non-None spec also owns
    reasoning-history rendering (:func:`spec_owns_reasoning_history`).
    """
    if tool_call_parser == "deepseekv4":
        return "dsv4"
    if tool_call_parser == "deepseekv32":
        return "dsv32"
    if tool_call_parser == "kimi_k3":
        return "kimi_k3"

    architectures = hf_config.architectures
    arch = architectures[0] if architectures else ""

    if "DeepseekV4" in arch:
        return "dsv4"
    if "KimiK3" in arch:
        return "kimi_k3"

    # Inkling has no Jinja chat_template and uses a tiktoken base + a special-token
    # overlay + negative MM placeholders, so it can't go through apply_chat_template;
    # render input_ids directly via the Inkling renderer (serving_chat._encode_messages).
    if "InklingForConditionalGeneration" in arch:
        return "inkling"

    has_chat_template = tokenizer is not None and tokenizer.chat_template is not None
    if "DeepseekV3" in arch and not has_chat_template:
        return "dsv32"
    return None


def spec_owns_reasoning_history(spec: Optional[str]) -> bool:
    """Whether the encoder for ``spec`` renders assistant reasoning history itself.

    Custom encoders frame the reasoning and content channels, so history must be
    passed as assistant ``reasoning_content``. Splicing a detector's markers into
    content instead nests a reasoning block inside the content channel and leaves
    the real one empty, teaching the model to emit raw markers as visible text.

    Answered for the whole family rather than a list of specs, so a new spec gets
    the safe default: worst case is dropped history, not a leak.
    """
    return spec is not None


def encode_simple_chat(
    *,
    tokenizer: Any,
    spec: Optional[str],
    messages: List[Dict[str, Any]],
    thinking_mode: str = "chat",
) -> List[int]:
    """Encode a plain-text chat conversation into prompt token ids.

    Minimal encode for offline tools: no tools, no multimodal content, no
    continue_final_message; the serving path keeps its full request-level
    pipeline in ``serving_chat``. Like
    ``serving_chat``, an empty system message is prepended when the
    conversation does not start with one (for the dsv4/dsv32 encoders this
    currently renders to zero tokens, but keeping the insertion explicit ties
    this helper to the serving semantics rather than to that coincidence).
    """
    if spec == "inkling":
        from sglang.srt.parser.inkling_renderer import render_inkling_messages
        from sglang.srt.parser.inkling_tokenizer import InklingTokenizer

        return render_inkling_messages(
            messages,
            InklingTokenizer(tokenizer=tokenizer),
            add_generation_prompt=False,
        )

    if spec in ("dsv4", "dsv32"):
        if messages and messages[0]["role"] != "system":
            messages = [{"role": "system", "content": ""}] + list(messages)
        if spec == "dsv4":
            from sglang.srt.entrypoints.openai import encoding_dsv4

            real_input = encoding_dsv4.encode_messages(
                messages, thinking_mode=thinking_mode
            )
        else:
            from sglang.srt.entrypoints.openai import encoding_dsv32

            real_input = encoding_dsv32.encode_messages(
                messages, thinking_mode=thinking_mode
            )
        return tokenizer.encode(real_input)

    if getattr(tokenizer, "chat_template", None) is None:
        raise ValueError(
            "This model has no HF chat template and no custom chat encoder; "
            f"cannot encode chat messages with {getattr(tokenizer, 'name_or_path', tokenizer)!r}."
        )
    return tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=True
    )
