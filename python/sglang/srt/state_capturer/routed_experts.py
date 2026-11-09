from typing import Any, Optional

import numpy as np
import pybase64
import torch
import torch.nn.functional as F

from sglang.srt.configs.model_config import ModelConfig
from sglang.srt.layers.dp_attention import (
    attn_tp_all_gather_into_tensor,
    dp_slot_in,
    get_dp_global_num_tokens,
    get_dp_local_slice_cpu,
    is_dp_attention_enabled,
)
from sglang.srt.layers.moe import get_moe_a2a_backend
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.model_executor.forward_context import (
    get_forward_context,
    has_forward_context,
)
from sglang.srt.runtime_context import (
    get_exec,
    get_parallel,
    get_resources,
    get_schedule,
    is_scheduler_io_rank,
)
from sglang.srt.state_capturer.base import BaseDeviceCache, BaseTopkCapturer
from sglang.srt.utils import ceil_align


def _is_scattered_a2a_backend() -> bool:
    """Return whether routed tokens are scattered across attention-TP ranks."""
    backend = get_moe_a2a_backend()
    return backend.is_deepep() or backend.is_deepep_v2() or backend.is_ascend_ops()


def _routed_layer_ids(model: torch.nn.Module) -> list[int]:
    # Lazy import: ``layers.moe.topk`` imports this module.
    from sglang.srt.layers.moe.topk import TopK

    return sorted(
        module.layer_id
        for module in model.modules()
        if isinstance(module, TopK) and module.topk_config.allow_routed_experts_capture
    )


class RoutedExpertsCapturer(BaseTopkCapturer):
    """Capturer for routed experts with host buffer.

    Routed experts share a global device buffer across DP ranks (indexed by
    dp_rank), so `_get_local_slice` overrides the default to apply DP-rank-aware
    slicing. The device cache also holds extra columns for any fused shared
    experts; the host cache and user-facing return drop them via the
    [:topk_size] truncation.

    Only the rank that sends outputs keeps a host cache.
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
        features = get_exec().features
        if not features.enable_return_routed_experts:
            return None
        # The loader attaches this to every entry model, including wrappers.
        # Size from the owner, independent of which runner is currently active.
        num_fused_shared_experts = model.shared_experts_config.num_fused_shared_experts
        return RoutedExpertsCapturer(
            model_config,
            num_tokens=num_tokens,
            max_running_requests=max_running_requests,
            num_fused_shared_experts=num_fused_shared_experts,
            device=device,
            # Bounded replay runs late layers on fewer rows; keep per-layer gathers.
            routed_layer_ids=(
                None
                if features.enable_decoder_swa_bounded_replay
                else _routed_layer_ids(model)
            ),
        )

    def __init__(
        self,
        model_config: ModelConfig,
        num_tokens: int,
        max_running_requests: int,
        num_fused_shared_experts: int,
        device: str,
        routed_layer_ids: Optional[list[int]] = None,
    ):
        self.num_fused_shared_experts = num_fused_shared_experts
        topk_size = model_config.hf_text_config.num_experts_per_tok
        num_layers = model_config.hf_text_config.num_hidden_layers

        # Scale by num_dp_ranks so the buffer covers the full DP-concatenated batch.
        # _get_local_slice indexes into [attention_dp_rank * cuda_graph_batch, ...)
        # and otherwise overflows on dp_rank > 0 when max_running_requests >
        # chunked_prefill_size.
        # FIXME: spec decoding's num_verify_tokens is still not accounted for.
        max_batch_size = max(
            get_schedule().chunked_prefill_size * get_parallel().num_dp_ranks,
            max_running_requests * get_parallel().num_dp_ranks,
        )
        if _is_scattered_a2a_backend():
            # Shares are padded to equal size, so whole shares must fit.
            max_batch_size = ceil_align(max_batch_size, get_parallel().attn_tp_size)

        super().__init__(
            num_tokens=num_tokens,
            max_batch_size=max_batch_size,
            num_layers=num_layers,
            topk_size=topk_size,
            device=device,
            name="routed_experts",
            device_topk_size=topk_size + num_fused_shared_experts,
            # Other ranks compute the same ids (or join the gather below) and
            # send nothing, so they keep no rows.
            allocate_caches=is_scheduler_io_rank(),
        )

        # Scattered backends split tokens across attention-TP ranks; only the
        # output rank's group (pp0, attn-cp0) rebuilds them for readback.
        parallel = get_parallel()
        self.first_routed_layer_id: Optional[int] = None
        self.gather_layer_id: Optional[int] = None
        self._first_layer_rows: Optional[int] = None
        self.local_cache: Optional[BaseDeviceCache] = None
        self.all_gather_scratch: Optional[torch.Tensor] = None
        self._gathers_per_layer = False
        if _is_scattered_a2a_backend():
            if routed_layer_ids and parallel.attn_tp_size > 1:
                if parallel.pp_rank == 0 and parallel.attn_cp_rank == 0:
                    self.first_routed_layer_id = routed_layer_ids[0]
                    self.gather_layer_id = routed_layer_ids[-1]
                    self.local_cache = BaseDeviceCache(
                        max_batch_size=max_batch_size // parallel.attn_tp_size,
                        num_layers=num_layers,
                        topk_size=self.device_topk_size,
                        device=device,
                        name="routed_experts_local",
                    )
                    if self.device_cache is None:
                        # The all-gather fills every rank; only the output
                        # rank keeps the rows.
                        self.all_gather_scratch = torch.empty(
                            (max_batch_size, num_layers, self.device_topk_size),
                            dtype=torch.int32,
                            device=device,
                        )
            else:
                self._gathers_per_layer = True
                attn_tp_size = parallel.attn_tp_size if is_dp_attention_enabled() else 1
                self.gather_buffer = torch.empty(
                    (max_batch_size * attn_tp_size, self.device_topk_size),
                    dtype=torch.int32,
                    device=device,
                )

    def capture(self, layer_id: int, topk_indices: torch.Tensor):
        if (
            self.device_cache is None
            and self.local_cache is None
            and not self._gathers_per_layer
        ):
            return
        device_width = self.device_topk_size
        if topk_indices.shape[1] != device_width:
            # Routes whose shared columns are appended after capture pass routed ids only.
            if topk_indices.shape[1] != device_width - self.num_fused_shared_experts:
                raise ValueError(
                    f"Captured top-k width {topk_indices.shape[1]} is neither the "
                    f"device width {device_width} nor its routed part "
                    f"{device_width - self.num_fused_shared_experts}"
                )
            topk_indices = F.pad(
                topk_indices, (0, self.num_fused_shared_experts), value=-1
            )
        if self.local_cache is not None:
            self._capture_local_rows(layer_id, topk_indices)
            return
        if self._gathers_per_layer:
            local_topk = topk_indices
            topk_indices = self.gather_buffer[
                : local_topk.size(0) * get_parallel().attn_tp_size
            ]
            attn_tp_all_gather_into_tensor(topk_indices, local_topk)
        if self.device_cache is None:
            # Joined the all-gather; the output rank keeps the rows.
            return
        context = get_forward_context() if has_forward_context() else None
        if context is None or context.tbo_parent_token_range is None:
            super().capture(layer_id, topk_indices)
            return
        # TBO children share this buffer and are padded to a common length.
        start, end = context.tbo_parent_token_range
        source, target = 0, 0
        if context.tbo_parent_dp_offset is not None:
            # Non-EP MoE sees every DP rank's padded rows; readback uses parent offsets.
            child_sizes = get_dp_global_num_tokens()
            source = sum(child_sizes[: dp_slot_in(child_sizes)])
            target = context.tbo_parent_dp_offset
        self.device_cache.buffer[target + start : target + end, layer_id] = (
            topk_indices[source : source + end - start]
        )

    def _capture_local_rows(self, layer_id: int, local_topk: torch.Tensor):
        """Keep this rank's equal, rank-ordered share; the last routed layer
        all-gathers all layers, sized by the captured rows so graphs replay it."""
        if layer_id > self.gather_layer_id:
            raise RuntimeError(
                f"Layer {layer_id} captured routed experts after the all-gather "
                f"layer {self.gather_layer_id}."
            )
        num_rows = local_topk.shape[0]
        if layer_id == self.first_routed_layer_id:
            self._first_layer_rows = num_rows
        elif num_rows != self._first_layer_rows:
            raise RuntimeError(
                f"Layer {layer_id} captured {num_rows} routed-expert rows, but layer "
                f"{self.first_routed_layer_id} captured {self._first_layer_rows}."
            )
        self.local_cache.capture(layer_id, local_topk)
        if layer_id != self.gather_layer_id:
            return
        out = (
            self.device_cache.buffer
            if self.device_cache is not None
            else self.all_gather_scratch
        )
        attn_tp_all_gather_into_tensor(
            out[: num_rows * get_parallel().attn_tp_size],
            self.local_cache.buffer[:num_rows],
        )

    def _get_local_slice(
        self,
        forward_batch: ForwardBatch,
        can_run_graph: bool,
        cuda_graph_batch: Optional[int],
    ) -> torch.Tensor:
        # Gathered rows start at buffer offset zero on every DP rank.
        if is_dp_attention_enabled() and not _is_scattered_a2a_backend():
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

    return get_resources().experts_capturer


def set_global_experts_capturer(capturer: Optional[RoutedExpertsCapturer]):

    get_resources().experts_capturer = capturer


def destroy_global_experts_capturer():
    if (capturer := get_resources().experts_capturer) is not None:
        capturer.destroy()
    get_resources().experts_capturer = None


def extract_routed_experts_from_meta_info(data):
    # To solve the performance issue, we return the experts_ids in base64
    # We left this function for user to change it back to normal int32
    # See detokenizer_manager::_extract_routed_experts
    routed_experts_base64 = data["meta_info"].get("routed_experts", None)
    routed_experts = np.frombuffer(
        pybase64.b64decode(routed_experts_base64.encode("utf-8")), dtype=np.int32
    )
    return routed_experts


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
