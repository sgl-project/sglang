"""Tokenize-time multimodal processing shared by both frontend paths.

`run_mm_processor_for_request` is the multimodal block of
`TokenizerManager._tokenize_one_request`, extracted verbatim so the Python
TokenizerManager and the embedded Rust server's MM path
(`sglang.srt.managers.rust_server.MmProcessorHost`) stay behavior-identical by
construction: both run the same model-specific `mm_processor`, the same
input_ids/token_type_ids overrides, and the same mm_hashes / precompute-hash
handling.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, List, Optional, Tuple, Union

from sglang.srt.environ import envs
from sglang.srt.managers.schedule_batch import (
    MultimodalDataItem,
    MultimodalProcessorOutput,
)

if TYPE_CHECKING:
    from sglang.srt.managers.io_struct import EmbeddingReqInput, GenerateReqInput
    from sglang.srt.server_args import ServerArgs

logger = logging.getLogger(__name__)


def validate_mm_limits(
    obj: Union["GenerateReqInput", "EmbeddingReqInput"],
    limit_mm_data_per_request: Optional[dict],
) -> None:
    """Enforce ``--limit-mm-data-per-request`` (moved from TokenizerManager)."""
    if not limit_mm_data_per_request:
        return

    for modality, limit in limit_mm_data_per_request.items():
        data = getattr(obj, f"{modality}_data", None)
        if data:
            count = len(data) if isinstance(data, list) else 1
            if count > limit:
                raise ValueError(
                    f"{modality.capitalize()} count {count} exceeds limit {limit} per request."
                )


async def run_mm_processor_for_request(
    *,
    obj: Union["GenerateReqInput", "EmbeddingReqInput"],
    input_text: Optional[str],
    input_ids: Optional[List[int]],
    token_type_ids: Optional[List[int]],
    mm_processor: Any,
    hf_architectures: List[str],
    server_args: "ServerArgs",
    max_req_input_len: Optional[int],
    mm_receiver: Any = None,
) -> Tuple[
    Optional[List[int]], Optional[MultimodalProcessorOutput], Optional[List[int]]
]:
    """Run the model's multimodal processor for one (already-normalized) request.

    Returns ``(input_ids, mm_inputs, token_type_ids)`` where ``input_ids`` /
    ``token_type_ids`` are the (possibly processor-overridden) values and
    ``mm_inputs`` is the ``MultimodalProcessorOutput`` to ship to the scheduler
    (``None`` when the request carries no multimodal input or the model has no
    mm processor). ``obj``'s ``image_data``/``video_data``/``audio_data`` are
    normalized to lists in place, as before.
    """
    contains_mm_input = obj.contains_mm_input()
    is_mossvl = "MossVLForConditionalGeneration" in hf_architectures
    should_run_mm_processor = mm_processor is not None and (
        contains_mm_input or is_mossvl
    )
    if not should_run_mm_processor:
        return input_ids, None, token_type_ids

    if obj.image_data is not None and not isinstance(obj.image_data, list):
        obj.image_data = [obj.image_data]
    if obj.video_data is not None and not isinstance(obj.video_data, list):
        obj.video_data = [obj.video_data]
    if obj.audio_data is not None and not isinstance(obj.audio_data, list):
        obj.audio_data = [obj.audio_data]
    if contains_mm_input:
        validate_mm_limits(obj, server_args.limit_mm_data_per_request)

    mm_inputs = None

    if (
        not server_args.language_only
        or server_args.encoder_transfer_backend == "zmq_to_tokenizer"
    ):
        if server_args.language_only:
            mm_inputs = await mm_receiver.recv_mm_data(
                request_obj=obj,
                mm_processor=mm_processor,
                prompt=(input_text or input_ids),
                need_wait_for_mm_inputs=obj.need_wait_for_mm_inputs,
            )
        if mm_inputs is None:
            if server_args.language_only:
                logger.warning(
                    "Encoder embedding not available, "
                    "falling back to local mm processing"
                )
            mm_inputs = await mm_processor.process_mm_data_async(
                image_data=obj.image_data,
                audio_data=obj.audio_data,
                input_text=(input_text or input_ids),
                request_obj=obj,
                max_req_input_len=max_req_input_len,
            )
    elif (
        server_args.language_only
        and server_args.encoder_transfer_backend in ["zmq_to_scheduler", "mooncake"]
        and not obj.need_wait_for_mm_inputs
    ):
        # In language_only mode with zmq_to_scheduler/mooncake, if we didn't dispatch
        # to encoder (e.g., only one image), process locally like non-language_only mode
        mm_inputs = await mm_processor.process_mm_data_async(
            image_data=obj.image_data,
            audio_data=obj.audio_data,
            input_text=(input_text or input_ids),
            request_obj=obj,
            max_req_input_len=max_req_input_len,
        )

    if mm_inputs and mm_inputs.input_ids is not None:
        input_ids = mm_inputs.input_ids
    if mm_inputs and mm_inputs.token_type_ids is not None:
        token_type_ids = mm_inputs.token_type_ids
        if not isinstance(token_type_ids, list):
            token_type_ids = token_type_ids.flatten().tolist()
    # Caller-supplied per-image hashes (external KV routers, e.g.
    # routing-aware orchestrators that compute a content-addressed
    # hash before dispatch). Setting MultimodalDataItem.hash here
    # short-circuits the internal hash_feature() recompute inside
    # set_pad_value(), making the derived pad_value deterministic
    # from the caller's hash. That alignment lets the router's
    # routing decision agree with sglang's prefix-cache key for
    # the same image. On any per-item parse error or list-length
    # mismatch we fall back to the internal recompute so a
    # malformed mm_hashes never blocks a request.
    caller_mm_hashes = getattr(obj, "mm_hashes", None)
    if caller_mm_hashes and mm_inputs and mm_inputs.mm_items:
        if len(caller_mm_hashes) != len(mm_inputs.mm_items):
            logger.warning(
                "mm_hashes length (%d) != mm_items length (%d); "
                "ignoring caller hashes for this request.",
                len(caller_mm_hashes),
                len(mm_inputs.mm_items),
            )
        else:
            for item, hex_hash in zip(mm_inputs.mm_items, caller_mm_hashes):
                if not isinstance(item, MultimodalDataItem):
                    continue
                try:
                    item.hash = int(hex_hash, 16)
                except (TypeError, ValueError):
                    logger.warning(
                        "Ignoring malformed mm_hashes entry %r; "
                        "this item will fall back to hash_feature().",
                        hex_hash,
                    )
    if envs.SGLANG_MM_PRECOMPUTE_HASH.get() and mm_inputs and mm_inputs.mm_items:
        for item in mm_inputs.mm_items:
            if isinstance(item, MultimodalDataItem):
                item.set_pad_value()

    return input_ids, mm_inputs, token_type_ids
