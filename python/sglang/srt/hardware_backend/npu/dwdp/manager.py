# SPDX-License-Identifier: Apache-2.0
"""Ascend DWDP: resident local weights and remote-only prefetch.

Only setup/cleanup use collectives. Two remote page slots are reused in layer
order; pages intersecting a local shard remain private to that layer.
"""

from __future__ import annotations

import logging

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
            for name in _WEIGHTS:
                kernel = type(getattr(layer, name.replace("_weight", "_kernel"), None))
                weight = getattr(layer, name, None)
                if (
                    not isinstance(weight, torch.Tensor)
                    or weight.device != self.device
                    or weight.dtype not in supported.get(kernel, ())
                    or weight.ndim != 3
                    or weight.shape[0] * self.dwdp_size
                    != layer.num_global_routed_experts
                ):
                    raise ValueError(
                        f"NPU DWDP layer {li}: incompatible kernel/device/dtype/shape for {name}"
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
        peer_pids = [
            peer_pid
            for rank, peer_pid in enumerate(self._gather(pid))
            if rank != self.dwdp_rank
        ]

        self._state = "initializing"
        self._moe_layer_indices = [li for li, _ in layers]
        self._position = {li: pos for pos, li in enumerate(self._moe_layer_indices)}
        self._init_prefetch(torch.npu, self.device)
        local_handles = {}
        error = None
        try:
            self._vmm = NPUWeightVMM(self.acl, self.device)
            for li, layer in layers:
                for name in _WEIGHTS:
                    local_handles[li, name] = self._migrate_weight(
                        (li, name), getattr(layer, name), peer_pids
                    )
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
        errors = self._gather(error)
        if any(errors):
            raise RuntimeError(f"NPU DWDP weight migration/export failed: {errors}")
        logger.info("DWDP handle exchange via FABRIC")
        self._weights = self._vmm.create_weights(
            self._specs,
            self.dwdp_rank,
            local_handles,
            {li: self._buffer_index_for_layer(li) for li in self._moe_layer_indices},
        )
        for li, layer in layers:
            self._replicate_small_params(layer)
            weights = {name: self._weights[li, name] for name in _WEIGHTS}
            count = layer.num_global_routed_experts // self.dwdp_size
            self._peer_ranges[li] = [
                (rank * count, (rank + 1) * count) for rank in range(self.dwdp_size)
            ]
            for name, full in weights.items():
                self._remote_slices[li, name] = build_remote_slices(
                    full,
                    self.dwdp_rank * count,
                    (self.dwdp_rank + 1) * count,
                    layer.num_global_routed_experts,
                )
            layer.bind_full_expert_weights(weights)

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
        self._exports[key] = self._vmm.export_shard(handle, peer_pids)
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
            self._peer_ranges,
        ):
            data.clear()
        self._state = "closed"
