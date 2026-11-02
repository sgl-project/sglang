"""NPU DWDP with resident expert shards and two remote-only VMM page slots.

All ranks keep their own tokens. Peer shards are immutable for the manager's
lifetime; only local stream events coordinate slot readers and writers.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass

import torch

from sglang.srt.hardware_backend.npu.dwdp.transport import NpuWeightTransport
from sglang.srt.hardware_backend.npu.dwdp.weight_manager import NpuDWDPWeightManager
from sglang.srt.layers.moe.dwdp.dwdp_manager import DwdpManager
from sglang.srt.layers.moe.dwdp.layout import DwdpExpertLayout
from sglang.srt.runtime_context import get_parallel

logger = logging.getLogger(__name__)
_WEIGHTS = ("w13_weight", "w2_weight")
_DTYPES = (torch.bfloat16, torch.float16, torch.int8)


@dataclass(frozen=True)
class _WeightSpec:
    shape: tuple[int, ...]
    dtype: torch.dtype
    element_size: int

    @property
    def nbytes(self) -> int:
        return math.prod(self.shape) * self.element_size


class NpuDwdpManager(DwdpManager):
    def __init__(self, server_args):
        super().__init__(server_args, device_module=torch.npu)
        parallel = get_parallel()
        self._group = parallel.tp_group
        self.device = torch.device("npu", self.device_id)
        self._transport = None
        self._weight_buffer = None
        self._moe_layers = []
        self._specs = {}
        self._setup_complete = False
        self._imports_started = False

    def _collect_weight_specs(self, model):
        layers = self._collect_moe_layers(model)
        if not layers:
            raise ValueError("NPU DWDP found no FusedMoE layers")
        counts = {experts.num_global_routed_experts for _, experts in layers}
        if len(counts) != 1 or next(iter(counts)) % self.dwdp_size:
            raise ValueError("NPU DWDP requires uniform experts divisible by dwdp_size")
        self.num_experts = counts.pop()
        self._local_experts = self.num_experts // self.dwdp_size
        self.layout = DwdpExpertLayout(self.num_experts, self.dwdp_size, self.dwdp_rank)
        specs = {}
        for li, experts in layers:
            specs[li] = {}
            for name in _WEIGHTS:
                weight = getattr(experts, name)
                if weight.dtype not in _DTYPES:
                    raise ValueError(
                        f"NPU DWDP unsupported {name} dtype: {weight.dtype}"
                    )
                if (
                    weight.device != self.device
                    or weight.ndim != 3
                    or weight.shape[0] != self._local_experts
                ):
                    raise ValueError(
                        f"NPU DWDP invalid {name} shard: {weight.shape} on {weight.device}"
                    )
                specs[li][name] = _WeightSpec(
                    (self.num_experts, *weight.shape[1:]),
                    weight.dtype,
                    weight.element_size(),
                )
        self._moe_layers, self._specs = layers, specs
        self._moe_layer_indices = [li for li, _ in layers]

    def _setup_stage(self, operation, function):
        value, error = None, None
        try:
            value = function()
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
        errors = self._group.all_gather_object(error)
        if any(errors):
            raise RuntimeError(f"NPU DWDP {operation} failed: {errors}")
        return value

    def setup(self, model) -> None:
        if self._setup_complete:
            return
        from sglang.srt.model_executor.cuda_graph_config import (
            cuda_graph_fully_disabled,
        )

        # Explicit phase configuration can override the automatic graph-off
        # flag. DCP would introduce collectives between independent DWDP ranks.
        if not cuda_graph_fully_disabled():
            raise ValueError("NPU DWDP requires graphs disabled")
        if get_parallel().attn_dcp_size != 1:
            raise ValueError("NPU DWDP requires DCP1")
        self._setup_stage("weight metadata", lambda: self._collect_weight_specs(model))
        try:
            self._transport = self._setup_stage(
                "transport initialization", lambda: NpuWeightTransport(self.device)
            )
            identity = self._setup_stage("identity", self._transport.identity)
            identities = self._group.all_gather_object(identity)
            peer_pids = [
                pid for r, (_, _, pid) in enumerate(identities) if r != self.dwdp_rank
            ]
            self._setup_stage(
                "weight publication", lambda: self._publish_weights(peer_pids)
            )
            self._setup_stage("composite buffer mapping", self._bind_weights)
            metadata = self._group.all_gather_object(self._transport.export_metadata())
            self._imports_started = True
            self._setup_stage(
                "peer import",
                lambda: self._transport.import_peers(
                    metadata, self.dwdp_rank, identities
                ),
            )
            self._allgather_small_params(self._moe_layers, self._group)
            self._weight_manager = NpuDWDPWeightManager(
                self._weight_buffer,
                self._transport,
                self._specs,
                self.layout,
                self.device,
            )
            torch.npu.synchronize(self.device)
            self._group.barrier()
            self._setup_complete = True
            torch.npu.empty_cache()
            logger.info(
                "NPU DWDP ready: %d ranks, %d MoE layers, native ACL prefetch, two remote-only VMM slots",
                self.dwdp_size,
                len(self._moe_layers),
            )
        except Exception:
            self.cleanup()
            raise

    def _publish_weights(self, peer_pids) -> None:
        for li, experts in self._moe_layers:
            for name in _WEIGHTS:
                weight = getattr(experts, name)
                # Preserve logical values after loader transforms. Only BF16,
                # FP16 and INT8 are accepted; MXFP transpose/scale contracts
                # require a separate, format-aware implementation.
                data = torch.ops.npu.npu_format_cast(weight.data, 2).contiguous()
                torch.npu.current_stream(self.device).synchronize()
                shard_bytes = data.numel() * data.element_size()
                offset = self.dwdp_rank * shard_bytes % self._transport.granularity
                self._transport.publish((li, name), data, peer_pids, offset)
                # Drop loader storage one weight at a time. This temporary
                # view aliases the resident shard until full views are bound.
                experts.replace_expert_tensor(
                    name, self._transport.local_tensor((li, name), data)
                )

    def _bind_weights(self):
        from sglang.srt.hardware_backend.npu.dwdp.buffer import NpuWeightBuffer

        self._weight_buffer = NpuWeightBuffer(
            self._specs, self._transport, self.dwdp_size, self.dwdp_rank
        )
        for li, experts in self._moe_layers:
            experts.bind_full_expert_weights(
                {
                    name: self._weight_buffer.get_full_tensor(li, name)
                    for name in _WEIGHTS
                }
            )

    def _prepare_small_param(self, name, data, experts):
        data = torch.ops.npu.npu_format_cast(data, 2).contiguous()
        if (
            name in ("w13_weight_bias", "w2_weight_bias")
            and experts.w13_weight.dtype == torch.float16
        ):
            data = data.to(torch.float16)
        return data

    def cleanup(self) -> None:
        if self._transport is None:
            return
        torch.npu.synchronize(self.device)
        self._weight_manager = None
        if self._weight_buffer is not None:
            self._weight_buffer.release()
            self._weight_buffer = None
        self._transport.close_imports()
        # Every consumer must drop imported mappings before any owner releases
        # immutable source weights. This is teardown, never a forward barrier.
        if self._imports_started:
            self._group.barrier()
        self._transport.close()
        self._transport = None
        self._setup_complete = False
