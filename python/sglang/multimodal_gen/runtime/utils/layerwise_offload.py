import re
import time
from contextlib import contextmanager
from itertools import chain
from typing import Any, Dict, List, Set, Tuple

import torch

from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger

logger = init_logger(__name__)


# Adapted from skywork AI Infra diffusion optimize
class LayerwiseOffloadManager:
    """A lightweight layerwise CPU offload manager.

    This utility offloads per-layer parameters/buffers from GPU to CPU, and
    supports async H2D prefetch using a dedicated CUDA stream.

    Typical usage:
    - Construct the manager with the target model and the list-like module
      attribute that represents transformer blocks (e.g. ``blocks``).
    - Call :meth:`initialize` once to offload weights and prefetch layer 0.
    - During forward, call :meth:`prefetch_layer` for the next layer and
      :meth:`release_layer` for the finished layer.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        *,
        module_list_attr: str,
        num_layers: int,
        enabled: bool,
        pin_cpu_memory: bool = True,
        auto_initialize: bool = False,
    ) -> None:
        self.model = model
        self.module_list_attr = module_list_attr
        self.num_layers = num_layers
        self.pin_cpu_memory = pin_cpu_memory

        self.enabled = bool(enabled and torch.cuda.is_available())
        self.device = (
            torch.device("cuda", torch.cuda.current_device()) if self.enabled else None
        )
        self.copy_stream = torch.cuda.Stream() if self.enabled else None

        self._layer_name_re = re.compile(
            rf"(^|\.){re.escape(module_list_attr)}\.(\d+)(\.|$)"
        )

        # layer_idx -> {dtype: consolidated_pinned_cpu_tensor}
        # stores the consolidated weight from a same layer, of same dtype
        self._consolidated_cpu_weights: Dict[int, Dict[torch.dtype, torch.Tensor]] = {}
        # layer_idx -> {name: {dtype, offset, numel, shape}}
        # stores the offset and numel of each weight from a same layer, of same dtype
        self._weight_metadata: Dict[int, Dict[str, Dict[str, Any]]] = {}
        # layer indices that are already in gpu
        self._gpu_layers: Set[int] = set()

        self._named_parameters: Dict[str, torch.nn.Parameter] = {}
        self._named_buffers: Dict[str, torch.Tensor] = {}

        # --- Static Buffer Pool ---
        # A pool of pre-allocated GPU buffers to avoid torch.empty overhead.
        # {dtype: [buffer_1, buffer_2, ...]}
        self._gpu_buffer_pool: Dict[torch.dtype, List[torch.Tensor]] = {}
        # Track which buffer is assigned to which layer: layer_idx -> {dtype: buffer_index}
        self._layer_buffer_indices: Dict[int, Dict[torch.dtype, int]] = {}
        # Max number of buffers to keep in pool (e.g., 2 or 3 for double/triple buffering)
        self._buffer_pool_size = 2
        # Max buffer size needed per dtype (calculated during init)
        self._max_buffer_size_per_dtype: Dict[torch.dtype, int] = {}

        if auto_initialize:
            self._initialize()

    def _match_layer_idx(self, name: str) -> int | None:
        m = self._layer_name_re.search(name)
        if not m:
            return None
        try:
            return int(m.group(2))
        except Exception:
            return None

    @torch.compiler.disable
    def _initialize(self) -> None:
        if not self.enabled:
            return

        self._named_parameters = dict(self.model.named_parameters())
        self._named_buffers = dict(self.model.named_buffers())

        # 1. collect and group tensors by layer and dtype
        layer_groups: Dict[int, Dict[torch.dtype, List[Tuple[str, torch.Tensor]]]] = {}
        all_tensors = chain(self._named_parameters.items(), self._named_buffers.items())
        for name, tensor in all_tensors:
            layer_idx = self._match_layer_idx(name)
            if layer_idx is None or layer_idx >= self.num_layers:
                continue
            layer_groups.setdefault(layer_idx, {}).setdefault(tensor.dtype, []).append(
                (name, tensor)
            )

        # 2. concat and offload (in pinned memory)
        for layer_idx, dtype_to_params in layer_groups.items():
            self._consolidated_cpu_weights[layer_idx] = {}
            self._weight_metadata[layer_idx] = {}

            for dtype, weights in dtype_to_params.items():
                total_numel = sum(t.numel() for _, t in weights)

                # Track max size needed for static buffer
                self._max_buffer_size_per_dtype[dtype] = max(
                    self._max_buffer_size_per_dtype.get(dtype, 0), total_numel
                )

                # concatenated CPU buffer (in pinned memory)
                cpu_buffer = torch.empty(
                    total_numel, dtype=dtype, pin_memory=self.pin_cpu_memory
                )

                # offload weights to the buffer
                current_offset = 0
                for name, weight in weights:
                    numel = weight.numel()
                    cpu_buffer[current_offset : current_offset + numel].copy_(
                        weight.flatten()
                    )
                    self._weight_metadata[layer_idx][name] = {
                        "dtype": dtype,
                        "offset": current_offset,
                        "numel": numel,
                        "shape": weight.shape,
                    }

                    weight.data = torch.empty((1,), device=self.device, dtype=dtype)

                    current_offset += numel

                self._consolidated_cpu_weights[layer_idx][dtype] = cpu_buffer

        # 3. Pre-allocate GPU static buffers
        if self.device is not None:
            for dtype, max_size in self._max_buffer_size_per_dtype.items():
                self._gpu_buffer_pool[dtype] = []
                for _ in range(self._buffer_pool_size):
                    # Allocate max needed size
                    self._gpu_buffer_pool[dtype].append(
                        torch.empty(max_size, dtype=dtype, device=self.device)
                    )

        # prefetch the first layer for warm-up
        self.prepare_for_next_denoise(non_blocking=False)

    def prepare_for_next_denoise(self, non_blocking=True):
        self.prefetch_layer(0, non_blocking=non_blocking)
        if not non_blocking and self.copy_stream is not None:
            torch.cuda.current_stream().wait_stream(self.copy_stream)

    def get_target_with_name(self, name: str) -> torch.Tensor:
        """get the target model weight/buffer to be replaced"""
        if name in self._named_parameters:
            target = self._named_parameters[name]
        else:
            target = self._named_buffers[name]
        return target

    @torch.compiler.disable
    def prefetch_layer(self, layer_idx: int, non_blocking: bool = True) -> None:
        if not self.enabled or self.device is None or self.copy_stream is None:
            return
        if layer_idx < 0 or layer_idx >= self.num_layers:
            return
        if layer_idx in self._gpu_layers:
            return
        if layer_idx not in self._consolidated_cpu_weights:
            return

        self.copy_stream.wait_stream(torch.cuda.current_stream())

        # Determine which buffer index to use from the pool.
        # We use a simple round-robin or based on layer_idx to assign a slot.
        # Since we usually prefetch i+1 while i is active, layer_idx % pool_size works well
        # to avoid overwriting the currently active layer.
        pool_idx = layer_idx % self._buffer_pool_size
        self._layer_buffer_indices[layer_idx] = {}

        # load from CPU buffer to Static GPU buffer
        gpu_buffers: Dict[torch.dtype, torch.Tensor] = {}
        with torch.cuda.stream(self.copy_stream):
            for dtype, cpu_buffer in self._consolidated_cpu_weights[layer_idx].items():
                if dtype not in self._gpu_buffer_pool:
                    # Fallback if dtype wasn't pre-allocated (shouldn't happen if init is correct)
                    gpu_buffer = torch.empty(
                        cpu_buffer.shape, dtype=dtype, device=self.device
                    )
                    gpu_buffer.copy_(cpu_buffer, non_blocking=non_blocking)
                    gpu_buffers[dtype] = gpu_buffer
                else:
                    # Use static buffer
                    static_buffer = self._gpu_buffer_pool[dtype][pool_idx]
                    # Slice the static buffer to match the needed size
                    # NOTE: We must use a slice to ensure copy_ size matches cpu_buffer
                    target_slice = static_buffer[: cpu_buffer.numel()].view(
                        cpu_buffer.shape
                    )
                    target_slice.copy_(cpu_buffer, non_blocking=non_blocking)
                    gpu_buffers[dtype] = target_slice
                    self._layer_buffer_indices[layer_idx][dtype] = pool_idx

        # restore model's weights by their metadata using gpu buffer
        for name, meta in self._weight_metadata[layer_idx].items():
            dtype = meta["dtype"]
            gpu_buffer = gpu_buffers[dtype]

            # map the parameter's data to the correct slice of the GPU buffer
            target = self.get_target_with_name(name)
            # Direct pointer assignment into the static buffer slicetarget.data = gpu_buffer[
                meta["offset"] : meta["offset"] + meta["numel"]
            ].view(meta["shape"])

        self._gpu_layers.add(layer_idx)

    @contextmanager
    def layer_scope(
        self,
        *,
        prefetch_layer_idx: int | None,
        release_layer_idx: int | None,
        non_blocking: bool = True,
    ):
        """A helper context manager to improve readability at call sites.

        It optionally prefetches ``prefetch_layer_idx`` before entering the
        context, and waits for the copy stream then releases
        ``release_layer_idx`` on exit.
        """
        if self.enabled and prefetch_layer_idx is not None:
            # 记录预取开始时间
            prefetch_start_time = time.perf_counter()
            self.prefetch_layer(prefetch_layer_idx, non_blocking=non_blocking)

        # 记录计算开始时间
        comp_start_time = time.perf_counter()
        try:
            yield
        finally:
            # print(f"finally")
            # # 计算结束时间
            # comp_end_time = time.perf_counter()
            # comp_duration = (comp_end_time - comp_start_time) * 1000 # ms
            # if self.enabled and self.copy_stream is not None and prefetch_layer_idx is not None:
            #     print(f"entering")
            #     # 核心分析逻辑：
            #     # 我们通过同步 copy_stream 来检测搬运是否已经完成。
            #     # 如果搬运已经完成，synchronize 会立即返回。
            #     wait_start = time.perf_counter()
            #     self.copy_stream.synchronize()
            #     wait_duration = (time.perf_counter() - wait_start) * 1000 # ms
            #
            #     if wait_duration > 0.1: # 超过 0.1ms 的等待被视为搬运未完全隐藏
            #         print(
            #             f"[Offload Log] Layer {prefetch_layer_idx} prefetch NOT fully hidden! "
            #             f"Comp: {comp_duration:.2f}ms, EXTRA Wait: {wait_duration:.2f}ms"
            #         )
            #     else:
            #         print(
            #             f"[Offload Log] Layer {prefetch_layer_idx} prefetch fully hidden. "
            #             f"Comp: {comp_duration:.2f}ms"
            #         )
            #
            #     # 维持原有的流水线同步逻辑
            #     torch.cuda.current_stream().wait_stream(self.copy_stream)

            if self.enabled and self.copy_stream is not None:
                torch.cuda.current_stream().wait_stream(self.copy_stream)
            if self.enabled and release_layer_idx is not None:
                self.release_layer(release_layer_idx)

    @torch.compiler.disable
    def release_layer(self, layer_idx: int) -> None:
        if not self.enabled or self.device is None:
            return
        if layer_idx <= 0:
            return

        if layer_idx not in self._gpu_layers:
            return

        # Release GPU memory by pointing to dummy tensors
        # Note: We don't "free" the static buffer, we just let it be overwritten by the next layer
        # that claims this pool_idx.
        for name, meta in self._tensor_metadata.get(layer_idx, {}).items():
            target = self.get_target_with_name(name)
            if target is not None:
                target.data = torch.empty((1,), device=self.device, dtype=meta["dtype"])

        # Clean up tracking info
        if layer_idx in self._layer_buffer_indices:
            del self._layer_buffer_indices[layer_idx]

        self._gpu_layers.discard(layer_idx)

    @torch.compiler.disable
    def release_all(self) -> None:
        if not self.enabled or self.device is None:
            return
        if self.copy_stream is not None:
            torch.cuda.current_stream().wait_stream(self.copy_stream)

        for layer_idx in list(self._gpu_layers):
            self.release_layer(layer_idx)
