import logging
from typing import Any, Optional

import numpy as np
import pybase64
import torch
import torch.nn.functional as F

from sglang.srt.configs.model_config import ModelConfig
from sglang.srt.layers.dp_attention import (
    attn_tp_all_gather_into_tensor,
    get_dp_local_slice_cpu,
    get_local_dp_buffer_len,
    is_dp_attention_enabled,
)
from sglang.srt.layers.moe import get_moe_a2a_backend
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.runtime_context import (
    get_exec,
    get_parallel,
    get_resources,
    get_schedule,
)
from sglang.srt.state_capturer.base import (
    BaseDeviceCache,
    BaseTopkCapturer,
    TopkCaptureOutput,
)

logger = logging.getLogger(__name__)


def _is_scattered_a2a_backend() -> bool:
    """Return whether routed tokens are scattered across attention-TP ranks."""
    backend = get_moe_a2a_backend()
    return backend.is_deepep() or backend.is_deepep_v2() or backend.is_ascend_ops()


def _is_output_rank() -> bool:
    # Must match the scheduler's is_rank_zero: only that rank sends outputs.
    parallel = get_parallel()
    return (
        parallel.pp_rank == 0
        and parallel.attn_tp_rank == 0
        and parallel.attn_cp_rank == 0
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
        if not get_exec().features.enable_return_routed_experts:
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

        # Scale by num_dp_ranks so the buffer covers the full DP-concatenated batch.
        # _get_local_slice indexes into [attention_dp_rank * cuda_graph_batch, ...)
        # and otherwise overflows on dp_rank > 0 when max_running_requests >
        # chunked_prefill_size.
        # FIXME: spec decoding's num_verify_tokens is still not accounted for.
        max_batch_size = max(
            get_schedule().chunked_prefill_size * get_parallel().num_dp_ranks,
            max_running_requests * get_parallel().num_dp_ranks,
        )

        # Other ranks compute the same ids (or join the gather below) and send
        # nothing, so they keep no rows.
        output_rank = _is_output_rank()
        super().__init__(
            num_tokens=num_tokens,
            max_batch_size=max_batch_size,
            num_layers=num_layers,
            topk_size=topk_size,
            device=device,
            name="routed_experts",
            device_topk_size=topk_size + num_fused_shared_experts,
            allocate_host_cache=output_rank,
            allocate_device_cache=output_rank,
        )

        # Rebuild the full token batch before routed-expert readback. With DP
        # attention each attention-TP rank keeps its own tokens' ids, and the
        # output rank's group all-gathers every layer once in on_forward_end;
        # the other groups (pp > 0, attn_cp > 0) send nothing.
        parallel = get_parallel()
        self.local_cache: Optional[BaseDeviceCache] = None
        self.all_gather_scratch: Optional[torch.Tensor] = None
        self._local_rows: Optional[int] = None
        self._gathers_per_layer = False
        if _is_scattered_a2a_backend():
            if is_dp_attention_enabled() and parallel.attn_tp_size > 1:
                if parallel.pp_rank == 0 and parallel.attn_cp_rank == 0:
                    local_rows = -(-max_batch_size // parallel.attn_tp_size)
                    self.local_cache = BaseDeviceCache(
                        local_rows,
                        num_layers,
                        self.device_topk_size,
                        device,
                        name="routed_experts_local",
                    )
                    if self.device_cache is None:
                        # The all-gather fills every rank; only the output
                        # rank keeps the rows.
                        self.all_gather_scratch = torch.empty(
                            (
                                local_rows * parallel.attn_tp_size,
                                num_layers,
                                self.device_topk_size,
                            ),
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
            self._local_rows = topk_indices.shape[0]
            self.local_cache.capture(layer_id, topk_indices)
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
        super().capture(layer_id, topk_indices)

    def on_forward_end(
        self,
        forward_batch: ForwardBatch,
        can_run_graph: bool,
        cuda_graph_batch: Optional[int],
        no_copy_to_cpu: bool = False,
    ) -> Optional[TopkCaptureOutput]:
        if self.local_cache is not None:
            self._all_gather_local_rows(can_run_graph)
        return super().on_forward_end(
            forward_batch, can_run_graph, cuda_graph_batch, no_copy_to_cpu
        )

    def _all_gather_local_rows(self, can_run_graph: bool):
        """Collect every layer's ids for all of the group's tokens on the output rank.

        Each attention-TP rank holds an equal, rank-ordered share of the local
        DP buffer, so concatenating the shares rebuilds the row order the
        per-layer all-gather produced. Every rank of the group must call this
        once per forward. It is an all-gather so it reuses the group's NCCL
        buffers; a gather would run as send/recv with buffers of its own.
        """
        parallel = get_parallel()
        # Graph replay runs no Python capture(); the graph runners publish the
        # replayed graph's padded size as the local DP buffer length.
        local_rows = get_local_dp_buffer_len() // parallel.attn_tp_size
        if not can_run_graph and self._local_rows is not None:
            if self._local_rows != local_rows:
                logger.warning(
                    "Routed-experts capture saw %d rows per attention-TP rank, but "
                    "the local DP buffer implies %d.",
                    self._local_rows,
                    local_rows,
                )
            local_rows = self._local_rows
        self._local_rows = None
        out = (
            self.device_cache.buffer
            if self.device_cache is not None
            else self.all_gather_scratch
        )
        attn_tp_all_gather_into_tensor(
            out[: local_rows * parallel.attn_tp_size],
            self.local_cache.buffer[:local_rows],
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
