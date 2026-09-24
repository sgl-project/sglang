# SPDX-License-Identifier: Apache-2.0
"""Single-host Ascend DWDP with ACL IPC and two reusable weight slots.

Only setup/cleanup use collectives. Forward passes use one-sided device copies,
so an idle DP rank need not enter a matching forward. Resident expert shards are
immutable until collective cleanup; weight updates/offload are not supported.
Unlike CUDA VMM, each slot also contains a copy of the local experts.
"""

from __future__ import annotations

import logging
import math
import socket

import torch
import torch.distributed as dist

from sglang.srt.runtime_context import get_parallel

logger = logging.getLogger(__name__)
_WEIGHTS = ("w13_weight", "w2_weight")


def _check(ret, operation):
    if ret != 0:
        raise RuntimeError(f"NPU DWDP: {operation} failed with ACL error {ret}")


class NPUDwdpManager:
    def __init__(self, server_args):
        import acl

        self.acl = acl
        self.dwdp_size = get_parallel().dwdp_size
        self.dwdp_rank = get_parallel().tp_rank
        self.device = torch.device("npu", torch.npu.current_device())
        self.group = get_parallel().tp_group.cpu_group
        self._layers = []
        self._position = {}
        self._weights = {}
        self._local = {}
        self._exports = {}
        self._imports = {}
        self._peer_ptrs = {}
        self._slots = {}
        self._ready = False
        self._copy_stream = torch.npu.Stream(device=self.device)
        self._prefetched = [torch.npu.Event() for _ in range(2)]
        self._consumed = [torch.npu.Event() for _ in range(2)]
        for event in self._consumed:
            event.record(torch.npu.current_stream(self.device))

    def _gather(self, value):
        values = [None] * self.dwdp_size
        dist.all_gather_object(values, value, group=self.group)
        return values

    @staticmethod
    def _collect_moe_layers(model):
        from sglang.srt.layers.moe.fused_moe_triton.layer import FusedMoE

        return sorted(
            (
                (layer.layer_id, layer)
                for layer in model.modules()
                if isinstance(layer, FusedMoE)
            ),
            key=lambda item: item[0],
        )

    def _validate(self, layers):
        from sglang.srt.hardware_backend.npu.quantization.moe_methods import (
            NPUUnquantMoEMethod,
        )

        if not layers:
            raise ValueError("NPU DWDP requires at least one FusedMoE layer")
        if len({li for li, _ in layers}) != len(layers):
            raise ValueError("NPU DWDP requires unique MoE layer IDs")
        for li, layer in layers:
            if layer.num_fused_shared_experts:
                raise ValueError("NPU DWDP requires --disable-shared-experts-fusion")
            if layer.moe_ep_size != self.dwdp_size or layer.moe_tp_size != 1:
                raise ValueError("NPU DWDP requires EP=DWDP=TP and MoE TP=1")
            if layer.num_global_routed_experts % self.dwdp_size:
                raise ValueError("NPU DWDP expert count must be divisible by dwdp_size")
            for prefix in ("w13", "w2"):
                if (
                    type(getattr(layer, f"{prefix}_kernel", None))
                    is not NPUUnquantMoEMethod
                ):
                    raise ValueError(
                        "NPU DWDP currently supports NPUUnquantMoEMethod only"
                    )
            for name in _WEIGHTS:
                weight = getattr(layer, name)
                if weight.device != self.device or weight.dtype not in (
                    torch.bfloat16,
                    torch.float16,
                ):
                    raise ValueError(
                        f"NPU DWDP layer {li}: expected device-resident BF16/FP16 {name}"
                    )
                if weight.shape[0] * self.dwdp_size != layer.num_global_routed_experts:
                    raise ValueError(
                        f"NPU DWDP layer {li}: unexpected expert weight layout"
                    )

    def setup(self, model):
        if self._ready:
            return
        layers = self._collect_moe_layers(model)
        error = None
        try:
            self._validate(layers)
        except ValueError as exc:
            error = str(exc)
        errors = self._gather(error)
        if any(errors):
            raise ValueError(f"NPU DWDP model validation failed: {errors}")

        pid, ret = self.acl.rt.device_get_bare_tgid()
        _check(ret, "device_get_bare_tgid")
        peers = self._gather((socket.gethostname(), pid))
        if len({host for host, _ in peers}) != 1:
            raise ValueError("NPU DWDP IPC is supported only within one host")
        peer_pids = [
            pid for rank, (_, pid) in enumerate(peers) if rank != self.dwdp_rank
        ]

        specs = {}
        capacity = {}
        for li, layer in layers:
            for name in _WEIGHTS:
                weight = getattr(layer, name)
                shape = (layer.num_global_routed_experts, *weight.shape[1:])
                specs[li, name] = (shape, weight.dtype)
                capacity[name, weight.dtype] = max(
                    capacity.get((name, weight.dtype), 0), math.prod(shape)
                )
        if any(peer != specs for peer in self._gather(specs)):
            raise ValueError("NPU DWDP ranks have different expert weight layouts")

        for (name, dtype), size in capacity.items():
            for slot in range(min(2, len(layers))):
                self._slots[slot, name, dtype] = torch.empty(
                    size, dtype=dtype, device=self.device
                )
        self._layers = [li for li, _ in layers]
        self._position = {li: i for i, li in enumerate(self._layers)}

        for li, layer in layers:
            full_weights = {}
            for name in _WEIGHTS:
                # NZ storage includes padding and cannot be copied as logical
                # tensor bytes. Normalize once; the unquantized GMM accepts ND.
                data = torch.ops.npu.npu_format_cast(
                    getattr(layer, name).data, 2
                ).contiguous()
                torch.npu.current_stream(self.device).synchronize()
                nbytes = data.numel() * data.element_size()
                # Ascend A3 can export ordinary device allocations through IPC.
                # HUGE_FIRST_P2P uses a separate, capacity-limited pool (8 GiB
                # on the validation host), too small for full-model shards.
                ptr, ret = self.acl.rt.malloc(nbytes, 0)  # HUGE_FIRST
                _check(ret, f"malloc(layer={li}, weight={name}, bytes={nbytes})")
                self._local[li, name] = (ptr, nbytes)
                _check(
                    self.acl.rt.memcpy(ptr, nbytes, data.data_ptr(), nbytes, 3),
                    "copy local shard",
                )
                key, ret = self.acl.rt.ipc_mem_get_export_key(ptr, nbytes, 65, 0)
                _check(ret, "ipc_mem_get_export_key")
                self._exports[li, name] = key
                _check(
                    self.acl.rt.ipc_mem_set_import_pid(key, peer_pids),
                    "ipc_mem_set_import_pid",
                )
                shape, dtype = specs[li, name]
                full = self._slots[self._position[li] % 2, name, dtype][
                    : math.prod(shape)
                ].view(shape)
                self._weights[li, name] = full
                full_weights[name] = full
                del data

            # Small per-expert tensors (e.g. bias) remain fully replicated.
            for name, data in layer.named_per_expert_tensors(layer.num_local_experts):
                shards = self._gather(data.detach().cpu())
                full = torch.cat(shards).to(self.device)
                # Ascend GMM requires FP16 bias for FP16 inputs (BF16 GMM
                # uses FP32 bias). FusedMoE initially allocates FP32 biases.
                if (
                    name in ("w13_weight_bias", "w2_weight_bias")
                    and layer.w13_weight.dtype == torch.float16
                ):
                    full = full.to(torch.float16)
                layer.replace_expert_tensor(name, full)
            layer.bind_full_expert_weights(full_weights)
            # ACL allocations bypass PyTorch's caching allocator. Return each
            # replaced shard to the driver before allocating the next layer's
            # IPC shard, otherwise startup temporarily retains two model shards.
            torch.npu.current_stream(self.device).synchronize()
            torch.npu.empty_cache()

        # Publish only after all immutable source shards have been initialized.
        exports = self._gather(self._exports)
        for rank, handles in enumerate(exports):
            for key, handle in handles.items():
                if rank == self.dwdp_rank:
                    ptr = self._local[key][0]
                else:
                    ptr, ret = self.acl.rt.ipc_mem_import_by_key(handle, 1)
                    _check(ret, "ipc_mem_import_by_key(enable_peer_access)")
                    self._imports[handle] = ptr
                self._peer_ptrs[rank, *key] = ptr
        torch.npu.synchronize(self.device)
        dist.barrier(group=self.group)
        torch.npu.empty_cache()
        self._ready = True
        logger.info(
            "NPU DWDP ready: %d ranks, %d MoE layers, IPC shards + two ND weight slots",
            self.dwdp_size,
            len(layers),
        )

    def _prefetch(self, li):
        slot = self._position[li] % 2
        with torch.npu.stream(self._copy_stream):
            self._copy_stream.wait_event(self._consumed[slot])
            for name in _WEIGHTS:
                full = self._weights[li, name]
                shard_bytes = self._local[li, name][1]
                for rank in range(self.dwdp_size):
                    _check(
                        self.acl.rt.memcpy_async(
                            full.data_ptr() + rank * shard_bytes,
                            shard_bytes,
                            self._peer_ptrs[rank, li, name],
                            shard_bytes,
                            3,
                            self._copy_stream.npu_stream,
                        ),
                        "prefetch expert shard",
                    )
            self._prefetched[slot].record(self._copy_stream)

    def prefetch_first_layers(self):
        if not self._ready:
            raise RuntimeError("NPU DWDP manager has not been initialized")
        for li in self._layers[:2]:
            self._prefetch(li)

    def wait_prefetch(self, layer_idx):
        torch.npu.current_stream(self.device).wait_event(
            self._prefetched[self._position[layer_idx] % 2]
        )

    def record_compute_and_prefetch_next(self, layer_idx):
        pos = self._position[layer_idx]
        self._consumed[pos % 2].record(torch.npu.current_stream(self.device))
        if pos + 2 < len(self._layers):
            self._prefetch(self._layers[pos + 2])

    def cleanup(self):
        if not self._ready:
            return
        torch.npu.synchronize(self.device)
        # Every consumer must finish before any producer releases its exports.
        dist.barrier(group=self.group)
        for key in self._imports:
            _check(self.acl.rt.ipc_mem_close(key), "close imported IPC memory")
        dist.barrier(group=self.group)
        for key in self._exports.values():
            _check(self.acl.rt.ipc_mem_close(key), "close exported IPC memory")
        for ptr, _ in self._local.values():
            _check(self.acl.rt.free(ptr), "free resident expert shard")
        self._imports.clear()
        self._exports.clear()
        self._local.clear()
        self._peer_ptrs.clear()
        self._weights.clear()
        self._slots.clear()
        self._ready = False
