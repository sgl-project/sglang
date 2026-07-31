import os
from typing import Any, Optional

import numpy as np
import pybase64
import torch

from sglang.srt.configs.model_config import ModelConfig
from sglang.srt.layers.dp_attention import (
    attn_tp_all_gather_into_tensor,
    get_dp_local_slice_cpu,
    is_dp_attention_enabled,
)
from sglang.srt.layers.moe import get_moe_a2a_backend
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.runtime_context import get_parallel, get_server_args
from sglang.srt.state_capturer.base import BaseTopkCapturer

_ROUTED_EXPERTS_DTYPE_ENV = "SGLANG_ROUTED_EXPERTS_DTYPE"
_ROUTED_EXPERTS_WIRE_DTYPES = {
    "int32": np.dtype(np.int32),
    "uint16": np.dtype(np.uint16),
    "uint8": torch.uint8,
}


def get_routed_experts_wire_dtype():
    dtype_name = get_routed_experts_wire_dtype_name()
    return _ROUTED_EXPERTS_WIRE_DTYPES[dtype_name]


def get_routed_experts_wire_dtype_name():
    dtype_name = os.environ.get(_ROUTED_EXPERTS_DTYPE_ENV, "int32").strip().lower()
    if dtype_name not in _ROUTED_EXPERTS_WIRE_DTYPES:
        raise ValueError(
            f"Unsupported {_ROUTED_EXPERTS_DTYPE_ENV}={dtype_name!r}. "
            f"Supported values are: {', '.join(_ROUTED_EXPERTS_WIRE_DTYPES)}."
        )
    return dtype_name


def _get_routed_experts_wire_dtype_from_meta_info(meta_info: dict):
    dtype_name = meta_info.get("routed_experts_dtype")
    if dtype_name is None:
        return get_routed_experts_wire_dtype()

    if not isinstance(dtype_name, str):
        raise ValueError("routed_experts_dtype must be a string when provided.")

    dtype_name = dtype_name.strip().lower()
    if dtype_name not in _ROUTED_EXPERTS_WIRE_DTYPES:
        raise ValueError(
            f"Unsupported routed_experts_dtype={dtype_name!r}. "
            f"Supported values are: {', '.join(_ROUTED_EXPERTS_WIRE_DTYPES)}."
        )
    return _ROUTED_EXPERTS_WIRE_DTYPES[dtype_name]


def encode_routed_experts_for_wire(routed_experts: torch.Tensor) -> str:
    """Encode row-major routings as a flat base64 wire payload."""
    routed_experts_np = routed_experts.numpy().reshape(-1)
    wire_dtype = get_routed_experts_wire_dtype()
    dtype_name = get_routed_experts_wire_dtype_name()

    if dtype_name in ("uint16", "uint8") and routed_experts_np.size:
        np_dtype = np.dtype(np.uint16) if dtype_name == "uint16" else np.dtype(np.uint8)
        info = np.iinfo(np_dtype)
        if routed_experts_np.min() < info.min or routed_experts_np.max() > info.max:
            raise ValueError(
                f"Cannot encode routed experts as {dtype_name} because at least one "
                f"expert id is outside [{info.min}, {info.max}]."
            )

    routed_experts_np = routed_experts_np.astype(wire_dtype, copy=False)
    return pybase64.b64encode(routed_experts_np.tobytes()).decode("utf-8")


class RoutedExpertsCapturer(BaseTopkCapturer):
    """Capturer for routed experts with host buffer.

    Routed experts share a global device buffer across DP ranks (indexed by
    dp_rank), so `_get_local_slice` overrides the default to apply DP-rank-aware
    slicing. The device cache also holds extra columns for any fused shared
    experts; the host cache and user-facing return drop them via the
    [:topk_size] truncation.
    """

    @staticmethod
    def create(
        *,
        model: torch.nn.Module,
        model_config: ModelConfig,
        num_tokens: int,
        max_running_requests: int,
        device: str,
    ) -> Optional["RoutedExpertsCapturer"]:
        server_args = get_server_args()
        if not server_args.enable_return_routed_experts:
            return None
        if not server_args.disable_shared_experts_fusion and hasattr(
            model, "num_fused_shared_experts"
        ):
            num_fused_shared_experts = model.num_fused_shared_experts
        else:
            num_fused_shared_experts = 0
        return RoutedExpertsCapturer(
            model_config,
            num_tokens=num_tokens,
            max_running_requests=max_running_requests,
            num_fused_shared_experts=num_fused_shared_experts,
            device=device,
        )

    def __init__(
        self,
        model_config: ModelConfig,
        num_tokens: int,
        max_running_requests: int,
        num_fused_shared_experts: int,
        device: str,
    ):
        self.num_fused_shared_experts = num_fused_shared_experts
        topk_size = model_config.hf_text_config.num_experts_per_tok
        num_layers = model_config.hf_text_config.num_hidden_layers

        server_args = get_server_args()
        # Scale by dp_size so the buffer covers the full DP-concatenated batch.
        # _get_local_slice indexes into [attention_dp_rank * cuda_graph_batch, ...)
        # and otherwise overflows on dp_rank > 0 when max_running_requests >
        # chunked_prefill_size.
        # FIXME: spec decoding's num_verify_tokens is still not accounted for.
        max_batch_size = max(
            server_args.chunked_prefill_size * server_args.dp_size,
            max_running_requests * server_args.dp_size,
        )

        super().__init__(
            num_tokens=num_tokens,
            max_batch_size=max_batch_size,
            num_layers=num_layers,
            topk_size=topk_size,
            device=device,
            name="routed_experts",
            device_topk_size=topk_size + num_fused_shared_experts,
        )

        # DeepEP a2a path: each attn-TP rank only sees its scattered slice of
        # topk_ids. All-gather across attn-TP at capture time so device_cache
        # holds the full batch and the existing _get_local_slice / D2H sync
        # paths work unchanged. Pre-allocate the gather target.
        if get_moe_a2a_backend().is_deepep():
            attn_tp_size = (
                get_parallel().attn_tp_size if is_dp_attention_enabled() else 1
            )
            self.gather_buffer = torch.empty(
                (
                    self.device_cache.buffer.shape[0] * attn_tp_size,
                    self.device_cache.buffer.shape[2],
                ),
                dtype=torch.int32,
                device=device,
            )

    def capture(self, layer_id: int, topk_indices: torch.Tensor):
        if get_moe_a2a_backend().is_deepep():
            local_topk = topk_indices
            topk_indices = self.gather_buffer[
                : local_topk.size(0) * get_parallel().attn_tp_size
            ]
            attn_tp_all_gather_into_tensor(topk_indices, local_topk)
        super().capture(layer_id, topk_indices)

    def _get_local_slice(
        self,
        forward_batch: ForwardBatch,
        can_run_graph: bool,
        cuda_graph_batch: Optional[int],
    ) -> torch.Tensor:
        # Under DeepEP, capture() already attn_tp_all_gathered into the head of
        # the per-rank buffer, so the local DP rank's data lives at [0:N_local]
        # rather than at the global [start_pos:end_pos] offset.
        if is_dp_attention_enabled() and not get_moe_a2a_backend().is_deepep():
            # GPU->CPU sync would break overlap; operate on CPU directly.
            local_start_pos, local_num_tokens = get_dp_local_slice_cpu(
                forward_batch, can_run_graph, cuda_graph_batch
            )
            local_end_pos = local_start_pos + local_num_tokens
        else:
            local_start_pos, local_end_pos = 0, forward_batch.out_cache_loc.shape[0]
        return self.device_cache.buffer[
            local_start_pos:local_end_pos, :, : self.topk_size
        ]


def get_global_experts_capturer() -> Optional[RoutedExpertsCapturer]:
    from sglang.srt.runtime_context import get_resources

    return get_resources().experts_capturer


def set_global_experts_capturer(capturer: Optional[RoutedExpertsCapturer]):
    from sglang.srt.runtime_context import get_resources

    get_resources().experts_capturer = capturer


def extract_routed_experts_from_meta_info(
    data,
    num_layers: Optional[int] = None,
    topk: Optional[int] = None,
):
    """Decode routed experts from response metadata.

    SGLang returns a flat row-major buffer where row p contains the routing that
    produced token p + 1. When num_layers and topk are provided, the return value
    is reshaped to (positions, num_layers * topk) using the advertised wire dtype.
    """
    meta_info = data["meta_info"]
    routed_experts_base64 = meta_info.get("routed_experts", None)
    wire_dtype = _get_routed_experts_wire_dtype_from_meta_info(meta_info)
    routed_experts = np.frombuffer(
        pybase64.b64decode(routed_experts_base64.encode("utf-8")), dtype=wire_dtype
    )

    if num_layers is None and topk is None:
        return routed_experts
    if num_layers is None or topk is None:
        raise ValueError("num_layers and topk must be provided together.")

    row_width = num_layers * topk
    if row_width <= 0:
        raise ValueError(
            f"Expected positive routed experts row width, got {row_width}."
        )

    return routed_experts.reshape(-1, row_width)


def disable_routed_experts_capture_for_draft(model: Any) -> None:
    """Opt every draft MoE ``TopK`` out of routed-experts (R3) capture.

    Capture is target-only; a draft ``TopK`` must never write the target's
    process-global buffer. ``HashTopK`` has no ``topk_config`` and never
    captures, so it is left untouched.
    """
    # Lazy import: ``layers.moe.topk`` imports ``get_global_experts_capturer``
    # from this module, so a top-level import here would be circular.
    from sglang.srt.layers.moe.topk import TopK

    for module in model.modules():
        if isinstance(module, TopK):
            module.topk_config.allow_routed_experts_capture = False
