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
        self._parallel = parallel
        self._server_args = server_args
        self.device = torch.device("npu", self.device_id)
        self._transport = None
        self._weight_buffer = None
        self._moe_layers = []
        self._specs = {}
        self._setup_complete = False
        self._imports_started = False

    def _validate(self, model):
        from sglang.srt.model_executor.cuda_graph_config import (
            cuda_graph_fully_disabled,
        )

        parallel = self._parallel
        if (
            self.dwdp_size < 2
            or self.dwdp_size != parallel.tp_size
            or self._group.world_size != self.dwdp_size
            or parallel.attn_tp_size != 1
            or parallel.pp_size != 1
        ):
            raise ValueError("NPU DWDP requires DWDP=TP, attention TP1 and PP1")
        if parallel.attn_cp_size != 1 or parallel.attn_dcp_size != 1:
            raise ValueError("NPU DWDP requires attention CP1 and DCP1")
        if not cuda_graph_fully_disabled():
            raise ValueError(
                "NPU DWDP requires both prefill and decode graphs disabled"
            )
        if (
            self._server_args.enable_lora
            or (self._server_args.enable_lora is None and self._server_args.lora_paths)
            or self._server_args.cpu_offload_gb > 0
            or self._server_args.offload_group_size > 0
            or self._server_args.enable_memory_saver
            or self._server_args.enable_pd_role_switch
        ):
            raise ValueError(
                "NPU DWDP does not support LoRA, CPU offload, memory saver or PD role switching"
            )
        from sglang.srt.layers.moe.fused_moe_triton.layer import FusedMoE

        decoder = model.model if hasattr(model, "model") else model
        if any(
            sum(isinstance(m, FusedMoE) for m in layer.modules()) > 1
            for layer in decoder.layers
        ):
            raise ValueError("NPU DWDP requires at most one FusedMoE per decoder layer")
        layers = self._collect_moe_layers(model)
        if not layers:
            raise ValueError("NPU DWDP found no FusedMoE layers")
        counts = {experts.num_global_routed_experts for _, experts in layers}
        if len(counts) != 1 or next(iter(counts)) % self.dwdp_size:
            raise ValueError("NPU DWDP requires uniform experts divisible by dwdp_size")
        self.num_experts = counts.pop()
        self._local_experts = self.num_experts // self.dwdp_size
        self.layout = DwdpExpertLayout(self.num_experts, self.dwdp_size, self.dwdp_rank)
        from sglang.srt.distributed.utils import get_group_rank_size
        from sglang.srt.hardware_backend.npu.quantization.moe_methods import (
            NPUUnquantMoEMethod,
            NPUW8A8Int8MoEMethod,
        )
        from sglang.srt.layers.linear import ColumnParallelLinear, RowParallelLinear
        from sglang.srt.layers.vocab_parallel_embedding import VocabParallelEmbedding

        # Independent ranks cannot execute residual TP collectives. Models must
        # construct attention, dense/shared projections and vocabulary locally.
        for name, module in model.named_modules():
            if isinstance(module, (ColumnParallelLinear, RowParallelLinear)):
                size = get_group_rank_size(module.tp_group)[1]
            elif isinstance(module, VocabParallelEmbedding):
                size = module.tp_size
            else:
                continue
            if size != 1:
                raise ValueError(
                    f"NPU DWDP requires local non-expert weights; {name} has TP{size}"
                )
        specs = {}
        signature = []
        for li, experts in layers:
            if experts.layer_id != li:
                raise ValueError("NPU DWDP MoE layer IDs must match decoder indices")
            if experts.num_fused_shared_experts:
                raise ValueError("NPU DWDP requires --disable-shared-experts-fusion")
            if experts.moe_tp_size != 1 or experts.moe_ep_size != self.dwdp_size:
                raise ValueError("NPU DWDP requires MoE TP1 and EP equal to DWDP")
            specs[li] = {}
            for name in _WEIGHTS:
                kernel = getattr(experts, name.replace("_weight", "_kernel"), None)
                if not isinstance(kernel, (NPUUnquantMoEMethod, NPUW8A8Int8MoEMethod)):
                    raise ValueError(
                        f"NPU DWDP unsupported kernel for {name}: {type(kernel).__name__}"
                    )
                weight = getattr(experts, name)
                if weight.dtype not in _DTYPES:
                    raise ValueError(
                        f"NPU DWDP unsupported {name} dtype: {weight.dtype}"
                    )
                if weight.ndim != 3 or weight.shape[0] != self._local_experts:
                    raise ValueError(
                        f"NPU DWDP invalid expert shard {name}: {weight.shape}"
                    )
                if weight.device != self.device:
                    raise ValueError(
                        "NPU DWDP weights must already be on the local NPU"
                    )
                shape = (self.num_experts, *weight.shape[1:])
                specs[li][name] = _WeightSpec(
                    shape, weight.dtype, weight.element_size()
                )
                signature.append((li, name, shape, str(weight.dtype)))
            for name, value in experts.named_per_expert_tensors(self._local_experts):
                signature.append((li, name, tuple(value.shape), str(value.dtype)))
        self._moe_layers, self._specs = layers, specs
        self._moe_layer_indices = [li for li, _ in layers]
        return signature

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
        signature = self._setup_stage("validation", lambda: self._validate(model))
        signatures = self._group.all_gather_object(signature)
        if any(peer != signature for peer in signatures):
            raise ValueError("NPU DWDP weight shapes/dtypes differ across ranks")
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
