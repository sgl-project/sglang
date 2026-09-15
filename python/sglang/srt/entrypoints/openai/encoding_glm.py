"""GLM tool-result encoding: linearize the quadratic template ordering.

The GLM-5.3-Flash chat template validates and reorders contiguous tool-result
blocks in O(n^2) Jinja. resolve_glm_tool_result_template splices that region
out of the recognized template and order_glm_tool_results replicates it in
O(n) Python before rendering.
"""

from __future__ import annotations

import hashlib
import logging
from collections.abc import Mapping
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

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
    return {"output": item["output"]} if "output" in item else {}


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
            if result_id in results_by_id or result_id not in seen_call_ids:
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


def glm_template_for_request(
    cached_template: Optional[str],
    chat_template_kwargs: Optional[Dict[str, Any]],
) -> Optional[str]:
    """A request-supplied chat_template wins; the reorder is only valid for the
    patched stock template, so the two are enabled or disabled together."""
    if cached_template is None or "chat_template" in (chat_template_kwargs or {}):
        return None
    return cached_template


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

    template = tokenizer.chat_template
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
