import re
from contextlib import contextmanager
from itertools import chain
from typing import Any, Dict, List, Set, Tuple

import torch


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

                # create concatenated CPU buffer (in pinned memory)
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

        # create gpu buffer and load from CPU buffer
        gpu_buffers: Dict[torch.dtype, torch.Tensor] = {}
        with torch.cuda.stream(self.copy_stream):
            for dtype, cpu_buffer in self._consolidated_cpu_weights[layer_idx].items():
                gpu_buffer = torch.empty(
                    cpu_buffer.shape, dtype=dtype, device=self.device
                )
                gpu_buffer.copy_(cpu_buffer, non_blocking=non_blocking)
                gpu_buffers[dtype] = gpu_buffer

        # restore model's weights by their metadata using gpu buffer
        for name, meta in self._weight_metadata[layer_idx].items():
            dtype = meta["dtype"]
            gpu_buffer = gpu_buffers[dtype]

            # map the parameter's data to the correct slice of the GPU buffer
            target = self.get_target_with_name(name)
            target.data = gpu_buffer[
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
            self.prefetch_layer(prefetch_layer_idx, non_blocking=non_blocking)
        try:
            yield
        finally:
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

        for name, meta in self._weight_metadata.get(layer_idx, {}).items():
            target = self.get_target_with_name(name)
            target.data = torch.empty((1,), device=self.device, dtype=meta["dtype"])

        self._gpu_layers.discard(layer_idx)

    @torch.compiler.disable
    def release_all(self) -> None:
        if not self.enabled or self.device is None:
            return
        if self.copy_stream is not None:
            torch.cuda.current_stream().wait_stream(self.copy_stream)

        for layer_idx in list(self._gpu_layers):
            self.release_layer(layer_idx)
