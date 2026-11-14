# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""L2 weight prefetch for the specialized decode.

Adapted from vLLM runtime/prefetch.py at 30044971a864d132e9a47f135e9ed11ece5c34d3.
Compared with PR #61867 final head 063c0c6d25658ed0702670b2fd07f02f21cadf10:
https://github.com/vllm-project/vllm/pull/61867
The existing SGLang plugin owns the CUDA prefetch kernel.

Small-batch decode is bound by HBM bandwidth, and each kernel that starts on weights
not yet in L2 stalls on DRAM. Some kernels leave bandwidth idle, such as the attention
core and the all-reduces. At fixed points of each layer, this module launches a small
kernel (``cutedsl/common/prefetch.py``) on a side stream. It pulls the byte ranges of
the weights the next kernels read into L2, so that they start warm.

Prefetch points (``fork(point, layer, batch)``), each with its batches per TP:

- ``oproj_down``: after the o-projection (B3-8), the MoE front's packed
  router + latent-down rows.
- ``attention_moe``: at TP8 only, after the KDA input projection (B5-8) or the MLA
  core (B8), the same front rows plus the shared expert's weights.
- ``next_attention``: once the routed experts are launched, the next layer's
  attention weights, capped in MiB per batch (``NEXT_ATTENTION``). Not above B8,
  where the HBM-bound experts lose more bandwidth than the attention gains.

The front and shared-expert ranges are on the critical path. They launch with the
highest stream priority and the maximum shared-memory carveout, which needs a driver
launch because CuTe drops both attributes.

Each fork records an event on the main stream that the side stream waits on, so a
prefetch starts only after the work it overlaps. ``join()`` at the end of the step
makes the main stream wait for the side stream, so a captured graph always rejoins
it. This lives outside ``cutedsl/`` because it is host-side
orchestration across layers: a stream, events, and per-(point, layer, batch) tables
built from the model's layers.
"""

import ctypes
from functools import partial

import torch

from sglang.kernels.jit.kimi_k3.policy import batch_table

# Per-TP batch -> MiB cap (or cap.carveout) of the next layer's attention weights,
# prefetched once the routed experts finish.
# Not at B>8: the HBM-bound experts lose more bandwidth than the attention gains.
NEXT_ATTENTION = {8: "1:96.100,2-3:96,4-7:64,8:32", 16: "1-8:1024"}


def attention_weights(provider, module):
    """The prepared provider's tensors, in consumption order.

    Use the coalesced SGLang projection and the core's normalized weight views,
    including owned MLA W_UK layout copies, rather than stale module aliases.
    """
    weights = [provider.projection_weight]
    weights.extend(provider.core.weights)
    weights.append(module.o_proj.weight)
    return tuple(weights)


def weight_ranges(weights, byte_cap=0):
    """Build bounded positive-stride address ranges without CUDA work."""
    if byte_cap < 0 or (byte_cap and byte_cap < 16):
        raise ValueError("L2 prefetch byte cap must be zero or at least 16")
    ranges, remaining = [], byte_cap
    for weight in weights:
        if weight.numel() == 0:
            continue
        if weight.data_ptr() % 16 or any(stride < 0 for stride in weight.stride()):
            raise ValueError("L2 prefetch requires aligned, positive-stride weights")
        span = 1 + sum(
            (n - 1) * stride for n, stride in zip(weight.shape, weight.stride())
        )
        size = span * weight.element_size()
        if byte_cap:
            size = min(size, remaining)
        size = size // 16 * 16
        if size:
            ranges.append((weight.data_ptr(), size))
            if byte_cap:
                remaining -= size
        if byte_cap and remaining < 16:
            break
    return ranges


def _stream_handle(stream):
    from cuda.bindings.driver import CUstream

    return CUstream(stream.cuda_stream)


class PreparedWeightRanges:
    """One owner retains weights, fixed-size range table, executable and launch ABI."""

    def __init__(self, weights, *, byte_cap=0, grid=32, carveout=None, priority=False):
        import cuda.bindings.driver as cuda

        from sglang.kernels.jit.kimi_k3.host import _checked, resolve_kernel
        from sglang.kernels.jit.kimi_k3.tensor_adapter import to_cute
        from sglang.kernels.ops.memory.kimi_k3_specialized.prefetch import (
            compile_kernel,
        )

        self.weights = weights = tuple(weights)
        self.byte_cap = byte_cap
        self.weight_layouts = tuple(
            (tuple(weight.shape), tuple(weight.stride()), weight.dtype)
            for weight in weights
        )
        if not weights or any(weight.device != weights[0].device for weight in weights):
            raise ValueError(
                "L2 prefetch needs nonempty, same-device immutable weights"
            )
        device = weights[0].device
        ranges = weight_ranges(weights, byte_cap)
        if not ranges:
            raise ValueError("L2 prefetch has no aligned weight range")
        self.range_sizes = tuple(size for _, size in ranges)
        self.table = torch.tensor(ranges, dtype=torch.int64, device=device)
        self.compiled = compile_kernel(len(ranges), grid, carveout, device.index)
        self.config = None
        if not priority and carveout is None:
            self.views = to_cute(self.table, 16)
            return
        # Launch priority / carveout need a driver launch (CuTe drops both).
        self.executor, self.function = resolve_kernel(self.compiled, device.index)
        ids, self.attributes = cuda.CUlaunchAttributeID, []
        if priority:  # the greatest stream priority
            top = cuda.CUlaunchAttribute()
            top.id = ids.CU_LAUNCH_ATTRIBUTE_PRIORITY
            top.value.priority = _checked(cuda.cuCtxGetStreamPriorityRange())[1]
            self.attributes.append(top)
        if carveout is not None:
            smem = cuda.CUlaunchAttribute()
            smem.id = ids.CU_LAUNCH_ATTRIBUTE_PREFERRED_SHARED_MEMORY_CARVEOUT
            smem.value.sharedMemCarveout = carveout
            self.attributes.append(smem)
        config = self.config = cuda.CUlaunchConfig()
        config.gridDimX, config.gridDimY, config.gridDimZ = grid, 1, 1
        config.blockDimX, config.blockDimY, config.blockDimZ = 32, 1, 1
        config.sharedMemBytes, config.attrs = 0, self.attributes
        config.numAttrs = len(self.attributes)
        self.pointer = ctypes.c_void_p(self.table.data_ptr())
        self.arguments = (ctypes.c_void_p * 1)(ctypes.addressof(self.pointer))
        self.address = ctypes.addressof(self.arguments)

    def bind(self):
        """Refresh final weight addresses without reallocating captured table storage."""
        if self.table.is_cuda and torch.cuda.is_current_stream_capturing():
            raise RuntimeError("L2 prefetch addresses must be bound before capture")
        if any(weight.device != self.table.device for weight in self.weights):
            raise ValueError("L2 prefetch weight device changed before binding")
        if (
            tuple(
                (tuple(weight.shape), tuple(weight.stride()), weight.dtype)
                for weight in self.weights
            )
            != self.weight_layouts
        ):
            raise ValueError("L2 prefetch weight layout changed before binding")
        ranges = weight_ranges(self.weights, self.byte_cap)
        if (
            tuple(self.table.shape) != (len(ranges), 2)
            or tuple(size for _, size in ranges) != self.range_sizes
        ):
            raise ValueError("L2 prefetch range geometry changed before binding")
        # Keep both the CuTe view and driver argument pointing at the allocated
        # device table. Only this temporary CPU tensor is created after KV sizing.
        self.table.copy_(torch.tensor(ranges, dtype=torch.int64, device="cpu"))

    def launch(self, stream):
        if self.config is None:
            self.compiled(self.views, stream)
        else:
            self.config.hStream = stream
            import cuda.bindings.driver as cuda

            from sglang.kernels.jit.kimi_k3.host import _checked

            _checked(cuda.cuLaunchKernelEx(self.config, self.function, self.address, 0))


class PrefetchController:
    """Prefetch launches per (point, layer, batch) on one side stream."""

    def __init__(self, runtime, layers, packed, providers):
        self.device, tp = runtime.device, runtime.tp
        self.stream = torch.cuda.Stream(device=self.device)
        self.events, self.pending, self.tables = {}, False, {}
        urgent = partial(PreparedWeightRanges, carveout=100, priority=True)
        following = batch_table(NEXT_ATTENTION[tp])
        for position, layer in enumerate(layers):
            if not layer._is_moe_layer:
                continue
            index, moe = layer.layer_idx, layer.mlp
            # The front GEMM reads [router shard; down shard] from the packed copy.
            ranges = urgent((packed[index],), grid=8)
            self._add("oproj_down", index, range(3, 9), ranges)
            if tp == 8:
                shared = moe.shared_experts
                shared = shared.gate_up_proj.weight, shared.down_proj.weight
                ranges = urgent((packed[index], *shared), grid=32)
                batches = range(5, 9) if providers[index].kind == "kda" else (8,)
                self._add("attention_moe", index, batches, ranges)
            if position + 1 < len(layers):
                next_layer = layers[position + 1]
                weights = attention_weights(
                    providers[next_layer.layer_idx], next_layer.self_attn
                )
                self._add_capped("next_attention", index, weights, following)

    def _add_capped(self, point, layer, weights, table):
        groups: dict[tuple, list[int]] = {}
        for batch, spec in table.items():
            cap, carveout = spec if isinstance(spec, tuple) else (spec, None)
            if cap:
                groups.setdefault((cap, carveout), []).append(batch)
        for (cap, carveout), batches in groups.items():
            ranges = PreparedWeightRanges(
                weights, byte_cap=cap * 1024**2, grid=32, carveout=carveout
            )
            self._add(point, layer, batches, ranges)

    def _add(self, point, layer, batches, ranges):
        key = point, layer
        if key not in self.events:
            event = torch.cuda.Event()
            event.record(torch.cuda.current_stream(self.device))
            self.events[key] = event
        for batch in batches:
            self.tables[point, layer, batch] = ranges

    def bind(self):
        """Finalize each shared range table once, after native weight preparation."""
        for ranges in dict.fromkeys(self.tables.values()):
            ranges.bind()

    def fork(self, point, layer, batch):
        ranges = self.tables.get((point, layer, batch))
        if ranges is None:
            return False
        event = self.events[point, layer]
        event.record(torch.cuda.current_stream(self.device))
        self.stream.wait_event(event)
        self.pending = True
        ranges.launch(_stream_handle(self.stream))
        return True

    def join(self):
        if self.pending:
            torch.cuda.current_stream(self.device).wait_stream(self.stream)
            self.pending = False
