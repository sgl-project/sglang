# SPDX-License-Identifier: Apache-2.0
"""Single-host Ascend DWDP: resident local weights and remote-only prefetch.

Only setup/cleanup use collectives. Two remote page slots are reused in layer
order; pages intersecting a local shard remain private to that layer.
"""

from __future__ import annotations

import logging
import socket

import torch
import torch.distributed as dist

from sglang.srt.layers.moe.dwdp.layer_utils import (
    collect_moe_layers,
    gather_per_expert_tensors,
)
from sglang.srt.layers.moe.dwdp.layout import WeightSpec, build_remote_slices
from sglang.srt.layers.moe.dwdp.weight_manager import (
    DwdpPrefetch,
    copy_remote_expert_slices,
)
from sglang.srt.runtime_context import get_parallel

from .vmm import NPUWeightVMM, _check, tensor_from_pointer

logger = logging.getLogger(__name__)
_WEIGHTS = ("w13_weight", "w2_weight")


def validate_dwdp_args(cfg):
    if cfg.nnodes != 1:
        raise ValueError("NPU DWDP requires a single host (--nnodes 1) for ACL IPC")
    # Validate loaded MoE kernels/layouts in setup, not the model-wide quant label.
    if cfg.enable_lora:
        raise ValueError("NPU DWDP does not support LoRA")
    if cfg.enable_pdmux:
        raise ValueError("NPU DWDP does not support PDMux layer-split execution")


class NPUDwdpManager(DwdpPrefetch):
    def __init__(self, server_args):
        import acl

        self.acl = acl
        parallel = get_parallel()
        self.dwdp_size, self.dwdp_rank = parallel.dwdp_size, parallel.tp_rank
        self.group = parallel.tp_group
        self.device = torch.device("npu", torch.npu.current_device())
        self._moe_layer_indices, self._position = [], {}
        self._specs, self._weights, self._exports = {}, {}, {}
        self._peer_views, self._peer_ranges, self._remote_slices = {}, {}, {}
        self._vmm = None
        self._state = "new"

    def _gather(self, value):
        values = [None] * self.dwdp_size
        dist.all_gather_object(values, value, group=self.group.cpu_group)
        return values

    _collect_moe_layers = staticmethod(collect_moe_layers)

    def _validate(self, layers):
        from sglang.srt.hardware_backend.npu.quantization.moe_methods import (
            NPUUnquantMoEMethod,
            NPUW8A8Int8MoEMethod,
        )

        supported = {
            NPUUnquantMoEMethod: (torch.bfloat16, torch.float16),
            NPUW8A8Int8MoEMethod: (torch.int8,),
        }
        if not layers:
            raise ValueError("NPU DWDP requires at least one FusedMoE layer")
        for li, layer in layers:
            if layer.moe_ep_size != self.dwdp_size or layer.moe_tp_size != 1:
                raise ValueError("NPU DWDP requires EP=DWDP=TP and MoE TP=1")
            if layer.num_global_routed_experts % self.dwdp_size:
                raise ValueError("NPU DWDP expert count must be divisible by dwdp_size")
            for name in _WEIGHTS:
                kernel = type(getattr(layer, name.replace("_weight", "_kernel"), None))
                if kernel not in supported:
                    raise ValueError(
                        "NPU DWDP supports unquantized BF16/FP16 or W8A8 INT8 MoE only"
                    )
                weight = getattr(layer, name, None)
                if (
                    not isinstance(weight, torch.Tensor)
                    or weight.device != self.device
                    or weight.dtype not in supported[kernel]
                    or weight.ndim != 3
                ):
                    raise ValueError(
                        f"NPU DWDP layer {li}: incompatible device/dtype/shape for {name}"
                    )
                if weight.shape[0] * self.dwdp_size != layer.num_global_routed_experts:
                    raise ValueError(
                        f"NPU DWDP layer {li}: unexpected expert weight layout"
                    )

    def setup(self, model):
        if self._state == "ready":
            return
        if self._state != "new":
            raise RuntimeError(
                "DWDP setup cannot be retried; clean up and reload weights"
            )
        layers = self._collect_moe_layers(model)
        error = None
        try:
            self._validate(layers)
        except ValueError as exc:
            error = str(exc)
        errors = self._gather(error)
        if any(errors):
            raise ValueError(f"NPU DWDP model validation failed: {errors}")
        self._specs.clear()
        for li, layer in layers:
            for name in _WEIGHTS:
                weight = getattr(layer, name)
                count = layer.num_global_routed_experts
                self._specs[li, name] = WeightSpec(
                    count, tuple(weight.shape), (count, *weight.shape[1:]), weight.dtype
                )
        metadata = {key: vars(spec) for key, spec in self._specs.items()}
        if any(peer != metadata for peer in self._gather(metadata)):
            raise ValueError("NPU DWDP ranks have different expert weight layouts")
        pid, ret = self.acl.rt.device_get_bare_tgid()
        _check(ret, "device_get_bare_tgid")
        peers = self._gather((socket.gethostname(), pid))
        if len({host for host, _ in peers}) != 1:
            raise ValueError("NPU DWDP IPC is supported only within one host")
        peer_pids = [
            pid for rank, (_, pid) in enumerate(peers) if rank != self.dwdp_rank
        ]

        self._state = "initializing"
        self._moe_layer_indices = [li for li, _ in layers]
        self._position = {li: pos for pos, li in enumerate(self._moe_layer_indices)}
        self._vmm = NPUWeightVMM(self.acl, self.device)
        self._init_prefetch(torch.npu, self.device)
        local_handles = {}
        for li, layer in layers:
            for name in _WEIGHTS:
                local_handles[li, name] = self._migrate_weight(
                    (li, name), getattr(layer, name), peer_pids
                )
        self._weights = self._vmm.create_weights(
            self._specs,
            self.dwdp_rank,
            local_handles,
            {li: self._buffer_index_for_layer(li) for li in self._moe_layer_indices},
        )
        for li, layer in layers:
            self._replicate_small_params(layer)
            layer.bind_full_expert_weights(
                {name: self._weights[li, name] for name in _WEIGHTS}
            )

        for rank, handles in enumerate(self._gather(self._exports)):
            if rank == self.dwdp_rank:
                continue
            for key, handle in handles.items():
                spec = self._specs[key]
                page = self._vmm.layout(spec, rank)
                ptr = self._vmm.import_shard(handle, page.mnnvl_size, page.data_offset)
                self._peer_views[rank, *key] = tensor_from_pointer(
                    ptr,
                    spec.chunk_bytes,
                    shape=spec.chunk_shape,
                    dtype=spec.dtype,
                    device=self.device,
                )
        self._prepare_copy_views()
        torch.npu.synchronize(self.device)
        dist.barrier(group=self.group.cpu_group)
        torch.npu.empty_cache()
        self._state = "ready"
        logger.info(
            "NPU DWDP ready: %d ranks, %d MoE layers, local VMM + remote-only double buffering",
            self.dwdp_size,
            len(layers),
        )

    def _migrate_weight(self, key, param, peer_pids):
        # NZ has padding and cannot resize its storage. Rebind to ND before
        # releasing the source; the unquantized NPU kernel accepts ND weights.
        data = torch.ops.npu.npu_format_cast(param.data, 2).contiguous()
        torch.npu.current_stream(self.device).synchronize()
        param.data = data
        spec = self._specs[key]
        ptr, handle = self._vmm.create_local(spec, self.dwdp_rank)
        _check(
            self.acl.rt.memcpy(
                ptr, spec.chunk_bytes, param.data_ptr(), spec.chunk_bytes, 3
            ),
            "copy local shard",
        )
        torch.npu.synchronize(self.device)
        share, ret = self.acl.rt.mem_export_to_shareable_handle(handle, 0, 0)
        _check(ret, "mem_export_to_shareable_handle")
        self._exports[key] = share
        _check(
            self.acl.rt.mem_set_pid_to_shareable_handle(share, peer_pids),
            "mem_set_pid_to_shareable_handle",
        )
        param.untyped_storage().resize_(0)
        # Return cached storage before the next driver-owned allocation.
        torch.npu.empty_cache()
        return handle

    def _replicate_small_params(self, layer):
        count = layer.num_global_routed_experts // self.dwdp_size
        for name, full in gather_per_expert_tensors(
            layer, count, layer.num_global_routed_experts, self.dwdp_size, self.group
        ):
            if (
                name in ("w13_weight_bias", "w2_weight_bias")
                and layer.w13_weight.dtype == torch.float16
            ):
                full = full.to(torch.float16)
            layer.replace_expert_tensor(name, full)

    def _prepare_copy_views(self):
        for key, full in self._weights.items():
            li, _ = key
            spec = self._specs[key]
            count = spec.local_experts
            start, end = self.dwdp_rank * count, (self.dwdp_rank + 1) * count
            self._peer_ranges[li] = [
                (rank * count, (rank + 1) * count) for rank in range(self.dwdp_size)
            ]
            self._remote_slices[key] = build_remote_slices(
                full, start, end, spec.num_experts
            )

    def _get_remote_slices(self, layer_idx, name):
        return self._remote_slices[layer_idx, name]

    def _buffer_index_for_layer(self, layer_idx):
        return self._position[layer_idx] % 2

    def _prefetch_layer_per_slice(self, layer_idx):
        copy_remote_expert_slices(
            layer_idx,
            _WEIGHTS,
            self._get_remote_slices,
            self._peer_views,
            self._peer_ranges[layer_idx],
        )

    def prefetch_first_layers(self):
        if self._state != "ready":
            raise RuntimeError("NPU DWDP manager has not been initialized")
        super().prefetch_first_layers()

    def cleanup(self):
        if self._state == "closed":
            return
        if self._state != "new":
            torch.npu.synchronize(self.device)
            dist.barrier(group=self.group.cpu_group)
            self._peer_views.clear()
            self._remote_slices.clear()
            if self._vmm is not None:
                self._vmm.close_imports()
            dist.barrier(group=self.group.cpu_group)
            if self._vmm is not None:
                self._vmm.close()
        for data in (
            self._moe_layer_indices,
            self._position,
            self._specs,
            self._weights,
            self._exports,
            self._peer_views,
            self._peer_ranges,
            self._remote_slices,
        ):
            data.clear()
        self._state = "closed"
