# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass
from typing import Any

import numpy as np
from PIL import Image

from sglang.multimodal_gen.configs.sample.sensenova_u1 import (
    resolve_sensenova_u1_pixel_flow_cfg,
)
from sglang.multimodal_gen.runtime.disaggregation.roles import RoleType
from sglang.multimodal_gen.runtime.pipelines_core.schedule_batch import Req
from sglang.multimodal_gen.runtime.pipelines_core.stages.base import PipelineStage
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.sensenova_u1_decode import (
    SenseNovaU1PixelFlowDecoder,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.sensenova_u1_denoise import (
    SenseNovaU1PixelFlowDenoiser,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.sensenova_u1_prepare import (
    SenseNovaU1PixelFlowPreparer,
)
from sglang.multimodal_gen.runtime.server_args import ServerArgs

_U1_T2I_CFG_UNCONDITION_ROLE = "u1_t2i_cfg_uncondition"
_U1_INTERLEAVE_TEXT_UNCONDITION_ROLE = "u1_interleave_text_uncondition"
_U1_EDIT_IMG_CONDITION_ROLE = "u1_edit_img_condition"
_U1_EDIT_UNCONDITION_ROLE = "u1_edit_uncondition"


@dataclass(frozen=True, slots=True)
class _SenseNovaU1GContext:
    session_id: str
    position_count: int
    sidecar_role: str | None = None


class SenseNovaU1PixelFlowGSegmentStage(PipelineStage):
    """SenseNova U1 image segment stage for standalone diffusion execution."""

    def __init__(
        self,
        executor: Any,
        *,
        context_ops_key: str = "sensenova_u1_context_ops",
        output_extra_key: str = "sensenova_u1_generated_segment",
    ) -> None:
        super().__init__()
        self.executor = executor
        self.context_ops_key = context_ops_key
        self.output_extra_key = output_extra_key

    @property
    def role_affinity(self):
        return RoleType.DENOISER

    def forward(
        self,
        batch: Req,
        server_args: ServerArgs,
    ) -> Req:
        context_ops = batch.extra.get(self.context_ops_key)
        if context_ops is None:
            raise RuntimeError(
                "SenseNova U1 pixel-flow stage requires batch.extra"
                f"[{self.context_ops_key!r}]"
            )

        segment = self.executor(
            context_ops=context_ops,
            batch=batch,
            server_args=server_args,
        )
        if getattr(segment, "type", None) != "image":
            raise ValueError(
                "SenseNova U1 pixel-flow expected an image segment, "
                f"got {getattr(segment, 'type', None)!r}"
            )

        batch.extra[self.output_extra_key] = segment
        batch.output = _image_to_numpy_batch(segment.image)
        return batch


class SenseNovaU1PixelFlowGSegmentExecutor:
    """Run SenseNova U1 pixel-flow G through the model-specific diffusion stages."""

    required_g_kind: str = "pixel_flow"

    def __call__(
        self,
        *,
        context_ops: Any,
        batch: Req,
        server_args: ServerArgs,
    ) -> Any:
        if getattr(context_ops, "g_kind", None) != self.required_g_kind:
            raise ValueError(
                "SenseNova U1 pixel-flow executor requires g_kind='pixel_flow', got "
                f"{getattr(context_ops, 'g_kind', None)!r}"
            )
        get_model = getattr(context_ops, "get_model", None)
        if not callable(get_model):
            raise RuntimeError("SenseNova U1 pixel-flow requires model access")
        forward_batch_provider = getattr(
            context_ops,
            "build_temporary_forward_batch",
            None,
        )
        if not callable(forward_batch_provider):
            raise RuntimeError(
                "SenseNova U1 pixel-flow requires temporary query forward batches"
            )
        (
            u1_context,
            cfg_img_condition_u1_context,
            cfg_uncondition_u1_context,
        ) = _resolve_u1_contexts(
            context_ops=context_ops,
            batch=batch,
        )
        native_runner = _SenseNovaU1NativePixelFlowRunner(
            get_model(),
            forward_batch_provider=forward_batch_provider,
        )
        return native_runner.generate(
            context_metadata=dict(getattr(context_ops, "metadata", {}) or {}),
            batch=batch,
            server_args=server_args,
            u1_context=u1_context,
            cfg_img_condition_u1_context=cfg_img_condition_u1_context,
            cfg_uncondition_u1_context=cfg_uncondition_u1_context,
        )


def _resolve_u1_contexts(
    *,
    context_ops: Any,
    batch: Req,
) -> tuple[
    _SenseNovaU1GContext,
    _SenseNovaU1GContext | None,
    _SenseNovaU1GContext | None,
]:
    get_position_count = getattr(context_ops, "get_position_count", None)
    if not callable(get_position_count):
        raise RuntimeError(
            "SenseNova U1 pixel-flow requires latest context position count"
        )

    u1_context = _require_context(
        context_ops,
        "SenseNova U1 pixel-flow has no context position count",
    )
    cfg_img_condition_context = None
    cfg_uncondition_context = None
    sampling_params = batch.sampling_params
    mode = getattr(sampling_params, "ug_generation_mode", None)
    cfg = resolve_sensenova_u1_pixel_flow_cfg(sampling_params)

    t2i_uncondition_role = context_ops.get_role(
        "t2i_cfg_uncondition_role",
        _U1_T2I_CFG_UNCONDITION_ROLE,
    )
    interleave_text_uncondition_role = context_ops.get_role(
        "interleave_text_uncondition_role",
        _U1_INTERLEAVE_TEXT_UNCONDITION_ROLE,
    )
    edit_img_condition_role = context_ops.get_role(
        "edit_img_condition_role",
        _U1_EDIT_IMG_CONDITION_ROLE,
    )
    edit_uncondition_role = context_ops.get_role(
        "edit_uncondition_role",
        _U1_EDIT_UNCONDITION_ROLE,
    )

    if mode == "edit":
        if cfg.needs_img_condition:
            cfg_img_condition_context = _require_context(
                context_ops,
                "SenseNova U1 edit image CFG requires sidecar context position count",
                edit_img_condition_role,
            )
        if cfg.needs_uncondition:
            cfg_uncondition_context = _require_context(
                context_ops,
                "SenseNova U1 edit uncondition CFG requires sidecar context position count",
                edit_uncondition_role,
            )
    elif mode == "interleave":
        if cfg.needs_img_condition:
            cfg_img_condition_context = _require_context(
                context_ops,
                "SenseNova U1 interleave text CFG requires sidecar context position count",
                interleave_text_uncondition_role,
            )
        if cfg.needs_uncondition:
            cfg_uncondition_context = _require_context(
                context_ops,
                "SenseNova U1 interleave image CFG requires sidecar context position count",
                t2i_uncondition_role,
            )
    elif cfg.text_scale > 1.0:
        cfg_img_condition_context = _require_context(
            context_ops,
            "SenseNova U1 pixel-flow CFG requires sidecar context position count",
            t2i_uncondition_role,
        )
    return u1_context, cfg_img_condition_context, cfg_uncondition_context


def _require_context(
    context_ops: Any,
    message: str,
    sidecar_role: str | None = None,
) -> _SenseNovaU1GContext:
    position_count = context_ops.get_position_count(sidecar_role=sidecar_role)
    if position_count is None:
        suffix = f" sidecar {sidecar_role}" if sidecar_role is not None else ""
        raise RuntimeError(f"{message} for context {context_ops.session_id}{suffix}")
    return _SenseNovaU1GContext(
        session_id=context_ops.session_id,
        sidecar_role=sidecar_role,
        position_count=int(position_count),
    )


class _SenseNovaU1NativePixelFlowRunner:
    """Model-specific pixel-flow runner fed by generic context ops."""

    def __init__(
        self,
        model: Any,
        *,
        forward_batch_provider: Any,
    ) -> None:
        self.preparer = SenseNovaU1PixelFlowPreparer(model)
        self.denoiser = SenseNovaU1PixelFlowDenoiser(
            model,
            forward_batch_provider=forward_batch_provider,
        )
        self.decoder = SenseNovaU1PixelFlowDecoder()

    def generate(
        self,
        *,
        context_metadata: dict[str, Any],
        batch: Any,
        server_args: Any,
        u1_context: _SenseNovaU1GContext,
        cfg_img_condition_u1_context: _SenseNovaU1GContext | None = None,
        cfg_uncondition_u1_context: _SenseNovaU1GContext | None = None,
    ) -> Any:
        import torch

        del server_args
        with torch.inference_mode():
            prepared = self.preparer.forward(
                context_metadata=context_metadata,
                batch=batch,
                u1_context=u1_context,
                cfg_img_condition_u1_context=cfg_img_condition_u1_context,
                cfg_uncondition_u1_context=cfg_uncondition_u1_context,
            )
            image_prediction = self.denoiser.forward(prepared)
            return self.decoder.forward(prepared, image_prediction)


def _image_to_numpy_batch(image: Any) -> np.ndarray:
    if isinstance(image, Image.Image):
        array = np.asarray(image.convert("RGB"))
    else:
        array = np.asarray(image)
    if array.ndim == 3:
        array = array[None, ...]
    if array.dtype != np.uint8:
        array = np.clip(array, 0, 255).astype(np.uint8)
    return array
