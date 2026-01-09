import re
from contextlib import contextmanager
from typing import Dict, Set

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

        self._cpu_weights: Dict[int, Dict[str, torch.Tensor]] = {}
        self._cpu_dtypes: Dict[int, Dict[str, torch.dtype]] = {}
        self._gpu_layers: Dict[int, Set[str]] = {}

        self._named_parameters: Dict[str, torch.nn.Parameter] = {}
        self._named_buffers: Dict[str, torch.Tensor] = {}

        if auto_initialize:
            self.initialize()

    def _match_layer_idx(self, name: str) -> int | None:
        m = self._layer_name_re.search(name)
        if not m:
            return None
        try:
            return int(m.group(2))
        except Exception:
            return None

    def _offload_tensor(self, name: str, tensor: torch.Tensor, layer_idx: int) -> None:
        if layer_idx not in self._cpu_weights:
            self._cpu_weights[layer_idx] = {}
            self._cpu_dtypes[layer_idx] = {}

        cpu_weight = tensor.detach().to("cpu")
        if self.pin_cpu_memory:
            cpu_weight = cpu_weight.pin_memory()
        self._cpu_weights[layer_idx][name] = cpu_weight
        self._cpu_dtypes[layer_idx][name] = tensor.dtype

        if self.device is not None:
            tensor.data = torch.empty((1,), device=self.device, dtype=tensor.dtype)

    @torch.compiler.disable
    def initialize(self) -> None:
        if not self.enabled:
            return

        self._named_parameters = dict(self.model.named_parameters())
        self._named_buffers = dict(self.model.named_buffers())

        for name, param in self._named_parameters.items():
            layer_idx = self._match_layer_idx(name)
            if layer_idx is None or layer_idx >= self.num_layers:
                continue
            self._offload_tensor(name, param, layer_idx)

        for name, buf in self._named_buffers.items():
            layer_idx = self._match_layer_idx(name)
            if layer_idx is None or layer_idx >= self.num_layers:
                continue
            self._offload_tensor(name, buf, layer_idx)

        self.prefetch_layer(0, non_blocking=False)
        if self.copy_stream is not None:
            torch.cuda.current_stream().wait_stream(self.copy_stream)

    @torch.compiler.disable
    def prefetch_layer(self, layer_idx: int, non_blocking: bool = True) -> None:
        if not self.enabled or self.device is None or self.copy_stream is None:
            return
        if layer_idx < 0 or layer_idx >= self.num_layers:
            return
        if layer_idx in self._gpu_layers:
            return
        if layer_idx not in self._cpu_weights:
            return

        self.copy_stream.wait_stream(torch.cuda.current_stream())

        param_names: Set[str] = set()

        for name, cpu_weight in self._cpu_weights[layer_idx].items():
            if name in self._named_parameters:
                target = self._named_parameters[name]
            else:
                target = self._named_buffers[name]

            gpu_weight = torch.empty(
                cpu_weight.shape,
                dtype=self._cpu_dtypes[layer_idx][name],
                device=self.device,
            )
            with torch.cuda.stream(self.copy_stream):
                gpu_weight.copy_(cpu_weight, non_blocking=non_blocking)
            target.data = gpu_weight
            param_names.add(name)

        self._gpu_layers[layer_idx] = param_names

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

        param_names = self._gpu_layers.pop(layer_idx, None)
        if not param_names:
            return

        for name in param_names:
            if name in self._named_parameters:
                target = self._named_parameters[name]
            else:
                target = self._named_buffers[name]
            target.data = torch.empty((1,), device=self.device, dtype=target.dtype)

    @torch.compiler.disable
    def release_all(self) -> None:
        if not self.enabled or self.device is None:
            return
        if self.copy_stream is not None:
            torch.cuda.current_stream().wait_stream(self.copy_stream)

        layer_indices = list(self._gpu_layers.keys())
        for layer_idx in layer_indices:
            param_names = self._gpu_layers.pop(layer_idx, None)
            if not param_names:
                continue
            for name in param_names:
                if name in self._named_parameters:
                    target = self._named_parameters[name]
                else:
                    target = self._named_buffers[name]
                target.data = torch.empty((1,), device=self.device, dtype=target.dtype)
