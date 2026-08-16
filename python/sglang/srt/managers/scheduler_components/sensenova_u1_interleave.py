# SPDX-License-Identifier: Apache-2.0
"""Scheduler-owned SenseNova U1 text-image continuation."""

from __future__ import annotations

import base64
import copy
import logging
from array import array
from dataclasses import dataclass
from enum import Enum, auto
from http import HTTPStatus
from typing import TYPE_CHECKING

import torch
from sglang.srt.disaggregation.utils import DisaggregationMode
from sglang.srt.managers.io_struct import AbortReq
from sglang.srt.managers.schedule_batch import (
    FINISH_ABORT,
    FINISH_MATCHED_TOKEN,
    Modality,
    MultimodalDataItem,
    MultimodalInputs,
    Req,
)
from sglang.srt.mem_cache.common import release_kv_cache
from sglang.srt.models.neo_chat_limits import (
    U1_FLOW_BATCH_ISOLATION_PARAM,
    U1_FLOW_CUSTOM_PARAM,
    U1_FLOW_PREFILL_GRAPH_VARIANT_PARAM,
    U1_FLOW_RADIX_PREFIX_LIMIT_PARAM,
    U1_INTERLEAVE_CUSTOM_PARAM,
)

if TYPE_CHECKING:
    from sglang.srt.managers.scheduler import Scheduler

logger = logging.getLogger(__name__)


class U1InterleavePhase(Enum):
    DECODING = auto()
    FLOW = auto()
    RESUMING = auto()


@dataclass(slots=True)
class U1InterleaveState:
    parent: Req
    spec: dict
    original_max_new_tokens: int
    phase: U1InterleavePhase = U1InterleavePhase.DECODING
    image_count: int = 0
    inserted_span_tokens: int = 0
    child: Req | None = None


def _next_u1_t_index(
    input_ids: list[int],
    *,
    image_start_token_id: int,
    image_context_token_id: int,
) -> int:
    image_start_shift = [0]
    image_start_shift.extend(
        int(token_id == image_start_token_id) for token_id in input_ids[:-1]
    )
    current = -1
    maximum = -1
    for token_id, shifted_start in zip(
        input_ids,
        image_start_shift,
        strict=True,
    ):
        current += shifted_start + int(token_id != image_context_token_id)
        maximum = max(maximum, current)
    return maximum + 1


def _clone_multimodal_inputs(
    multimodal_inputs: MultimodalInputs | None,
) -> MultimodalInputs | None:
    if multimodal_inputs is None:
        return None
    cloned = copy.copy(multimodal_inputs)
    cloned.mm_items = []
    for item in multimodal_inputs.mm_items:
        cloned_item = copy.copy(item)
        cloned_item.model_specific_data = dict(item.model_specific_data)
        cloned.mm_items.append(cloned_item)
    return cloned


class SenseNovaU1InterleaveController:
    def __init__(self, scheduler: Scheduler) -> None:
        self.scheduler = scheduler
        self._parents: dict[str, U1InterleaveState] = {}
        self._children: dict[str, U1InterleaveState] = {}

    @staticmethod
    def request_spec(req: Req) -> dict | None:
        custom_params = req.sampling_params.custom_params
        if not isinstance(custom_params, dict):
            return None
        spec = custom_params.get(U1_INTERLEAVE_CUSTOM_PARAM)
        return spec if isinstance(spec, dict) else None

    @staticmethod
    def is_internal_child(req: Req) -> bool:
        return bool(getattr(req, "_sensenova_u1_internal_flow_child", False))

    @staticmethod
    def is_parked(req: Req) -> bool:
        return bool(getattr(req, "_sensenova_u1_interleave_parked", False))

    def register_parent(self, req: Req) -> str | None:
        spec = self.request_spec(req)
        if spec is None:
            return None

        unsupported = self._unsupported_reason(req)
        if unsupported is not None:
            return unsupported

        namespace = f"sensenova_u1_interleave:{req.rid}"
        req.extra_key = f"{req.extra_key}|{namespace}" if req.extra_key else namespace
        self._parents[req.rid] = U1InterleaveState(
            parent=req,
            spec=spec,
            original_max_new_tokens=int(req.sampling_params.max_new_tokens),
        )
        return None

    def discard_parent(self, req: Req) -> None:
        state = self._parents.pop(req.rid, None)
        if state is not None and state.child is not None:
            self._children.pop(state.child.rid, None)

    def _unsupported_reason(self, req: Req) -> str | None:
        scheduler = self.scheduler
        if scheduler.enable_overlap:
            return (
                "SenseNova U1 interleave requires --disable-overlap-schedule "
                "until per-request lookahead rollback is supported"
            )
        if not scheduler.spec_algorithm.is_none():
            return "SenseNova U1 interleave does not support speculative decoding"
        if scheduler.disaggregation_mode != DisaggregationMode.NULL:
            return "SenseNova U1 interleave does not support disaggregated serving"
        if scheduler.ps.pp_size != 1:
            return "SenseNova U1 interleave currently requires pipeline parallel size 1"
        if scheduler.enable_hisparse:
            return "SenseNova U1 interleave does not support HiSparse"
        if scheduler.tree_cache.supports_mamba():
            return "SenseNova U1 interleave does not support Mamba cache layouts"
        if req.session is not None or req.session_id is not None:
            return "SenseNova U1 interleave does not support sessions"
        if req.input_embeds is not None or req.positional_embed_overrides is not None:
            return "SenseNova U1 interleave requires token input"
        if (
            req.return_logprob
            or req.return_sampling_mask
            or req.return_hidden_states
            or req.return_routed_experts
            or req.return_indexer_topk
        ):
            return (
                "SenseNova U1 interleave does not support logprob, sampling-mask, "
                "or hidden-state capture outputs"
            )
        sampling = req.sampling_params
        if any(
            value is not None
            for value in (
                sampling.json_schema,
                sampling.regex,
                sampling.ebnf,
                sampling.structural_tag,
            )
        ):
            return "SenseNova U1 interleave does not support constrained decoding"
        if int(sampling.max_new_tokens) <= 0:
            return "SenseNova U1 interleave requires max_new_tokens > 0"
        return None

    def maybe_park_parent(self, req: Req) -> bool:
        state = self._parents.get(req.rid)
        if state is None or state.parent is not req:
            return False
        reason = req.finished_reason
        image_start_token_id = int(state.spec["img_start_token_id"])
        if not (
            isinstance(reason, FINISH_MATCHED_TOKEN)
            and int(reason.matched) == image_start_token_id
        ):
            return False
        if state.image_count >= int(state.spec["max_images"]):
            return False

        error = self._validate_boundary_capacity(state)
        if error is not None:
            req.finished_reason = FINISH_ABORT(
                error,
                HTTPStatus.BAD_REQUEST,
                "BadRequestError",
            )
            req.finished_len = len(req.output_ids)
            return False
        if req.req_pool_idx is None or req.kv is None:
            req.finished_reason = FINISH_ABORT(
                "SenseNova U1 interleave lost the parent KV allocation",
                HTTPStatus.INTERNAL_SERVER_ERROR,
            )
            req.finished_len = len(req.output_ids)
            return False

        try:
            child = self._build_flow_child(state)
            self._cache_parent_committed_prefix(req)
        except Exception as error:
            logger.exception(
                "SenseNova U1 interleave failed to park parent %s",
                req.rid,
            )
            req.finished_reason = FINISH_ABORT(
                f"SenseNova U1 interleave could not start image flow: {error}",
                HTTPStatus.INTERNAL_SERVER_ERROR,
            )
            req.finished_len = len(req.output_ids)
            return False
        if not self.scheduler._add_internal_request_to_queue(child):
            req.finished_reason = FINISH_ABORT(
                "SenseNova U1 interleave could not admit the internal flow request",
                HTTPStatus.SERVICE_UNAVAILABLE,
            )
            req.finished_len = len(req.output_ids)
            return False

        req.finished_reason = None
        req.finished_len = None
        req.to_finish = None
        req._sensenova_u1_interleave_parked = True
        state.phase = U1InterleavePhase.FLOW
        state.child = child
        self._children[child.rid] = state
        logger.info(
            "SenseNova U1 interleave parked parent %s for image turn %d",
            req.rid,
            state.image_count,
        )
        return True

    def _validate_boundary_capacity(self, state: U1InterleaveState) -> str | None:
        parent = state.parent
        future_len = parent.seqlen + int(state.spec["image_span_tokens"])
        if future_len >= int(self.scheduler.max_req_len):
            return (
                "SenseNova U1 interleave image continuation exceeds the context "
                f"window: {future_len} >= {self.scheduler.max_req_len}"
            )
        return None

    def _cache_parent_committed_prefix(self, parent: Req) -> None:
        parent._refresh_fill_ids()
        parent.set_extend_range(0, parent.kv_committed_len)
        self.scheduler.tree_cache.cache_unfinished_req(parent)

    def _build_flow_child(self, state: U1InterleaveState) -> Req:
        parent = state.parent
        spec = state.spec
        image_start = parent.seqlen
        image_tokens = int(spec["image_tokens"])
        placeholder_id = parent.tokenizer.eos_token_id
        if placeholder_id is None:
            raise ValueError("SenseNova U1 requires a tokenizer EOS token")

        unpadded_prefix = [
            *[int(token_id) for token_id in parent.origin_input_ids_unpadded],
            *[int(token_id) for token_id in parent.output_ids],
        ]
        flow_spec = {
            "width": int(spec["width"]),
            "height": int(spec["height"]),
            "num_steps": int(spec["num_steps"]),
            "seed": (int(spec["seed"]) + state.image_count) % (2**63),
            "image_start": image_start,
            "image_tokens": image_tokens,
            "image_t_index": _next_u1_t_index(
                unpadded_prefix,
                image_start_token_id=int(spec["img_start_token_id"]),
                image_context_token_id=int(spec["img_context_token_id"]),
            ),
            "token_height": int(spec["token_height"]),
            "token_width": int(spec["token_width"]),
            "timestep_shift": float(spec["timestep_shift"]),
            "enable_timestep_shift": bool(spec["enable_timestep_shift"]),
            "return_image_tensor": False,
            "return_image_tensor_raw": True,
        }
        sampling_params = copy.copy(parent.sampling_params)
        custom_params = dict(sampling_params.custom_params or {})
        custom_params.pop("__req__", None)
        custom_params.pop(U1_INTERLEAVE_CUSTOM_PARAM, None)
        custom_params[U1_FLOW_CUSTOM_PARAM] = flow_spec
        custom_params[U1_FLOW_BATCH_ISOLATION_PARAM] = (
            f"sensenova_u1_interleave_flow:{parent.rid}:{state.image_count}"
        )
        custom_params[U1_FLOW_RADIX_PREFIX_LIMIT_PARAM] = image_start
        custom_params[U1_FLOW_PREFILL_GRAPH_VARIANT_PARAM] = "sensenova_u1_flow"
        sampling_params.custom_params = custom_params
        sampling_params.max_new_tokens = 1
        sampling_params.min_new_tokens = 0
        sampling_params.stop_token_ids = None
        sampling_params.stop_strs = []
        sampling_params.stop_str_max_len = 0
        sampling_params.stop_regex_strs = []
        sampling_params.stop_regex_max_len = 0
        sampling_params.ignore_eos = True
        sampling_params.logit_bias = None

        child = Req(
            rid=f"{parent.rid}::sensenova_u1_flow:{state.image_count}",
            origin_input_text=None,
            origin_input_ids=array(
                "q",
                [
                    *[int(token_id) for token_id in parent.origin_input_ids],
                    *[int(token_id) for token_id in parent.output_ids],
                    *([int(placeholder_id)] * image_tokens),
                ],
            ),
            sampling_params=sampling_params,
            stream=False,
            eos_token_ids=parent.eos_token_ids,
            vocab_size=parent.vocab_size,
            priority=parent.priority,
            extra_key=parent.extra_key,
        )
        child.tokenizer = parent.tokenizer
        child.multimodal_inputs = _clone_multimodal_inputs(parent.multimodal_inputs)
        child._sensenova_u1_internal_flow_child = True
        child.skip_radix_cache_insert = True
        return child

    def complete_child(self, child: Req) -> None:
        state = self._children.pop(child.rid, None)
        if state is None:
            return
        state.child = None
        parent = state.parent

        if isinstance(child.finished_reason, FINISH_ABORT):
            self._fail_parked_parent(
                state,
                f"SenseNova U1 internal flow failed: {child.finished_reason.message}",
            )
            return

        try:
            image_tensor = self._extract_child_image(child)
            self._attach_generated_image(state, image_tensor)
        except Exception as error:
            logger.exception(
                "SenseNova U1 interleave failed to attach generated image for %s",
                parent.rid,
            )
            self._fail_parked_parent(
                state,
                f"SenseNova U1 image continuation failed: {error}",
            )
            return

        parent._sensenova_u1_interleave_parked = False
        parent._sensenova_u1_live_prefix_len = parent.kv_committed_len
        parent._sensenova_u1_reuse_prefix_lock_once = True
        parent.already_computed = max(
            int(parent.already_computed),
            int(parent.kv_committed_len),
        )
        state.phase = U1InterleavePhase.RESUMING
        self.scheduler._resume_interleave_parent(parent)
        logger.info(
            "SenseNova U1 interleave resumed parent %s after image turn %d",
            parent.rid,
            state.image_count - 1,
        )

    @staticmethod
    def _extract_child_image(child: Req) -> torch.Tensor:
        customized_info = child.customized_info or {}
        values = customized_info.get("sensenova_u1_flow_image_tensor") or []
        for value in values:
            if isinstance(value, torch.Tensor):
                return value
        raise RuntimeError("internal flow returned no generated image tensor")

    def _attach_generated_image(
        self,
        state: U1InterleaveState,
        image_tensor: torch.Tensor,
    ) -> None:
        from sglang.srt.multimodal.processors.neo_chat import (
            build_u1_mrope_positions,
            generated_u1_image_to_native_feature,
        )

        parent = state.parent
        spec = state.spec
        expected_shape = (
            1,
            3,
            int(spec["height"]),
            int(spec["width"]),
        )
        if tuple(image_tensor.shape) != expected_shape:
            raise ValueError(
                f"generated image shape {tuple(image_tensor.shape)} does not match "
                f"{expected_shape}"
            )

        boundary_output_index = len(parent.output_ids) - 1
        context_start = parent.seqlen
        image_tokens = int(spec["image_tokens"])
        context_end = context_start + image_tokens - 1
        feature, grid_hw = generated_u1_image_to_native_feature(
            image_tensor,
            patch_size=int(self.scheduler.model_config.hf_config.patch_size),
        )
        item = MultimodalDataItem(
            modality=Modality.IMAGE,
            feature=feature,
            offsets=[(context_start, context_end)],
            model_specific_data={"grid_hw": grid_hw},
        )
        item.set_pad_value()
        # Generated spans stay visible in output_ids. The parent has a unique
        # Radix namespace, so the real context token is also a safe scatter key.
        item.pad_value = int(spec["img_context_token_id"])

        multimodal_inputs = parent.multimodal_inputs
        if multimodal_inputs is None:
            multimodal_inputs = MultimodalInputs(
                mm_items=[],
                im_token_id=int(spec["img_context_token_id"]),
                im_start_id=int(spec["img_start_token_id"]),
                im_end_id=int(spec["img_end_token_id"]),
            )
            parent.multimodal_inputs = multimodal_inputs
        multimodal_inputs.mm_items.append(item)

        parent.output_ids.extend([int(spec["img_context_token_id"])] * image_tokens)
        parent.output_ids.append(int(spec["img_end_token_id"]))
        state.image_count += 1
        state.inserted_span_tokens += int(spec["image_span_tokens"])
        parent.sampling_params.max_new_tokens = (
            state.original_max_new_tokens + state.inserted_span_tokens
        )
        parent.mm_image_tokens += image_tokens

        ordered_image_items = sorted(
            (mm_item for mm_item in multimodal_inputs.mm_items if mm_item.is_image()),
            key=lambda mm_item: mm_item.offsets[0][0],
        )
        combined_grid_hw = torch.cat(
            [mm_item.grid_hw.to(device="cpu") for mm_item in ordered_image_items],
            dim=0,
        )
        unpadded_ids = torch.tensor(
            [
                *[int(token_id) for token_id in parent.origin_input_ids_unpadded],
                *[int(token_id) for token_id in parent.output_ids],
            ],
            dtype=torch.long,
        )
        future_decode_tokens = max(
            state.original_max_new_tokens
            - (len(parent.output_ids) - state.inserted_span_tokens),
            0,
        )
        (
            multimodal_inputs.mrope_positions,
            multimodal_inputs.mrope_position_delta,
        ) = build_u1_mrope_positions(
            unpadded_ids,
            img_start_token_id=int(spec["img_start_token_id"]),
            img_context_token_id=int(spec["img_context_token_id"]),
            grid_hw=combined_grid_hw,
            downsample_ratio=float(
                self.scheduler.model_config.hf_config.downsample_ratio
            ),
            future_decode_tokens=future_decode_tokens + 1,
        )

        if bool(spec["return_images"]):
            final_image = image_tensor.detach().to(torch.float16).cpu().contiguous()
            image_b64 = base64.b64encode(final_image.numpy().tobytes()).decode("ascii")
            self._record_parent_custom_info(
                parent,
                "sensenova_u1_interleave_image_b64",
                image_b64,
                boundary_output_index,
            )
            self._record_parent_custom_info(
                parent,
                "sensenova_u1_interleave_image_shape",
                list(final_image.shape),
                boundary_output_index,
            )
            self._record_parent_custom_info(
                parent,
                "sensenova_u1_interleave_image_index",
                state.image_count - 1,
                boundary_output_index,
            )

        if state.image_count >= int(spec["max_images"]):
            logit_bias = dict(parent.sampling_params.logit_bias or {})
            logit_bias[str(spec["img_start_token_id"])] = -100.0
            parent.sampling_params.logit_bias = logit_bias

    @staticmethod
    def _record_parent_custom_info(
        parent: Req,
        key: str,
        value,
        output_index: int,
    ) -> None:
        if parent.customized_info is None:
            parent.customized_info = {}
        for values in parent.customized_info.values():
            if len(values) < len(parent.output_ids):
                values.extend([None] * (len(parent.output_ids) - len(values)))
        values = parent.customized_info.setdefault(
            key,
            [None] * len(parent.output_ids),
        )
        if len(values) < len(parent.output_ids):
            values.extend([None] * (len(parent.output_ids) - len(values)))
        values[output_index] = value

    def _fail_parked_parent(
        self,
        state: U1InterleaveState,
        message: str,
    ) -> None:
        parent = state.parent
        self._parents.pop(parent.rid, None)
        if state.child is not None:
            self._children.pop(state.child.rid, None)
            state.child = None
        parent._sensenova_u1_interleave_parked = False
        parent.finished_reason = FINISH_ABORT(
            message,
            HTTPStatus.INTERNAL_SERVER_ERROR,
        )
        parent.finished_len = len(parent.output_ids)
        if parent.req_pool_idx is not None:
            release_kv_cache(parent, self.scheduler.tree_cache, is_insert=False)
        parent.time_stats.set_completion_time()
        self.scheduler.output_streamer.stream_output(
            [parent],
            parent.return_logprob,
        )
        if parent.multimodal_inputs is not None and parent.session is None:
            parent.multimodal_inputs.release_features()

    def fail_internal_child(self, child: Req, message: str) -> None:
        state = self._children.pop(child.rid, None)
        if state is None:
            return
        state.child = None
        self._fail_parked_parent(state, message)

    def cleanup_finished(self, reqs: list[Req]) -> None:
        for req in reqs:
            state = self._parents.get(req.rid)
            if state is None or state.parent is not req or not req.finished():
                continue
            self._parents.pop(req.rid, None)
            if state.child is not None:
                self._children.pop(state.child.rid, None)
            req._sensenova_u1_interleave_parked = False

    def before_abort(self, recv_req: AbortReq) -> list[Req]:
        direct_abort: list[Req] = []
        for rid, state in list(self._parents.items()):
            parent = state.parent
            if not (recv_req.abort_all or rid.startswith(recv_req.rid)):
                continue
            self._parents.pop(rid, None)
            if state.child is not None:
                self._children.pop(state.child.rid, None)

            if state.phase == U1InterleavePhase.FLOW:
                if parent.req_pool_idx is not None:
                    release_kv_cache(
                        parent,
                        self.scheduler.tree_cache,
                        is_insert=False,
                    )
                # The just-completed decode batch may still reference the parent
                # until the next scheduler pass filters parked requests.
                parent._sensenova_u1_interleave_parked = True
                direct_abort.append(parent)
            elif parent in self.scheduler.waiting_queue:
                if parent.req_pool_idx is not None:
                    release_kv_cache(
                        parent,
                        self.scheduler.tree_cache,
                        is_insert=False,
                    )
                parent._sensenova_u1_interleave_parked = False
        return direct_abort


__all__ = [
    "SenseNovaU1InterleaveController",
    "U1InterleavePhase",
    "U1InterleaveState",
    "_next_u1_t_index",
]
