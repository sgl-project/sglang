"""Online MXFP4 loading via native shard loaders; adapted from sglang#29048."""

import threading
from typing import Callable

import torch
from torch.nn import Parameter

from sglang.srt.layers.parameter import ModelWeightParameter
from sglang.srt.layers.quantization.dequantization import copy_missing_attrs
from sglang.srt.layers.quantization.online_quantization import CopyNumelCounter


class _NPUOnlineWeightLoader:
    """Load one logical weight, convert on completion, and validate before reload."""

    def __init__(self, layer, weight_name, params_dtype, load_weight, on_complete):
        if params_dtype not in (torch.float16, torch.bfloat16):
            raise ValueError(
                f"Online MXFP4 loading requires FP16/BF16 weights, got {params_dtype}."
            )
        self.layer = layer
        self.weight_name = weight_name
        self.params_dtype = params_dtype
        self.load_device = torch.get_default_device()
        self.load_weight = load_weight
        self.on_complete = on_complete
        self.lock = threading.Lock()
        self.state = "loading"
        self.loaded_numel = 0
        self.loaded_shards = set()

    def register_source(self, weight: Parameter) -> None:
        self.source_shape = tuple(weight.shape)
        self.target_numel = weight.numel()
        if not self.target_numel:
            raise ValueError("Online MXFP4 requires a nonempty local weight.")

    def _materialize_source(self) -> Parameter:
        current = getattr(self.layer, self.weight_name)
        data = torch.empty(
            self.source_shape, dtype=self.params_dtype, device=self.load_device
        )
        if self.weight_name == "weight":
            source = ModelWeightParameter(
                data=data, input_dim=1, output_dim=0, weight_loader=self.weight_loader
            )
        else:
            source = Parameter(data, requires_grad=False)
        copy_missing_attrs(current, source)
        setattr(self.layer, self.weight_name, source)
        return source

    def _load(self, loaded_weight, shard_keys, *args, **kwargs):
        # Copy and conversion share the lock: checkpoint workers may overlap.
        with self.lock:
            reloading = self.state == "ready_reload"
            if reloading:
                self.loaded_numel = 0
                self.loaded_shards.clear()
                self.state = "loading"
            if self.state != "loading":
                raise RuntimeError(
                    f"Cannot load {self.weight_name} in state {self.state}: duplicate "
                    "load before post-load processing, or a previous load failed."
                )
            if self.loaded_shards.intersection(shard_keys) or (
                self.loaded_shards
                and (None in shard_keys or None in self.loaded_shards)
            ):
                raise RuntimeError(f"Duplicate shard for online {self.weight_name}.")
            source = getattr(self.layer, self.weight_name)
            if source.is_meta or reloading:
                source = self._materialize_source()
            if (
                tuple(source.shape) != self.source_shape
                or source.dtype != self.params_dtype
            ):
                raise RuntimeError(
                    f"Unexpected source layout for online {self.weight_name}."
                )
            try:
                counter = CopyNumelCounter()
                with counter:
                    self.load_weight(source, loaded_weight, *args, **kwargs)
                copied = self.loaded_numel + counter.copied_numel
                if counter.copied_numel == 0 or copied > self.target_numel:
                    raise RuntimeError(
                        f"Invalid online {self.weight_name} load: copied {copied} "
                        f"elements, expected {self.target_numel}."
                    )
                self.loaded_numel = copied
                self.loaded_shards.update(shard_keys)
                if copied == self.target_numel:
                    self.on_complete()
                    prepared = getattr(self.layer, self.weight_name)
                    if prepared.is_meta or prepared.dtype == self.params_dtype:
                        raise RuntimeError(
                            f"Online {self.weight_name} was not converted."
                        )
                    copy_missing_attrs(source, prepared)
                    self.prepared_layout = (tuple(prepared.shape), prepared.dtype)
                    # Cached Parameter handles must not retain the source storage.
                    if prepared is not source:
                        source.data = torch.empty(
                            0, device=source.device, dtype=source.dtype
                        )
                    self.state = "converted"
            except Exception:
                self.state = "failed"
                raise

    def finish_post_load(self) -> None:
        with self.lock:
            if self.state not in ("converted", "ready_reload"):
                raise RuntimeError(
                    f"Online {self.weight_name} was not completely loaded through its "
                    f"completion-tracked loader ({self.loaded_numel}/{self.target_numel}, "
                    f"state={self.state})."
                )
            current = getattr(self.layer, self.weight_name)
            if (
                current.is_meta
                or (tuple(current.shape), current.dtype) != self.prepared_layout
            ):
                raise RuntimeError(
                    f"Online {self.weight_name} reload must use the registered weight loader."
                )
            self.state = "ready_reload"


class NPUOnlineDenseWeightLoader(_NPUOnlineWeightLoader):
    def __init__(
        self, layer, params_dtype, original_weight_loader: Callable, on_complete
    ):
        super().__init__(
            layer, "weight", params_dtype, original_weight_loader, on_complete
        )

    def weight_loader(self, param, loaded_weight, loaded_shard_id=None):
        # Ignore cached handles; _load resolves the current source from the layer.
        keys = (
            set(loaded_shard_id)
            if isinstance(loaded_shard_id, tuple)
            else {loaded_shard_id}
        )
        kwargs = {} if loaded_shard_id is None else {"loaded_shard_id": loaded_shard_id}
        self._load(loaded_weight, keys, **kwargs)


class NPUOnlineMoEWeightLoader:
    def __init__(self, layer, params_dtype, original_weight_loader: Callable):
        self.layer = layer
        self.weights = {
            prefix: _NPUOnlineWeightLoader(
                layer,
                f"{prefix}_weight",
                params_dtype,
                original_weight_loader,
                lambda prefix=prefix: getattr(
                    layer, f"{prefix}_kernel"
                ).process_weights_after_loading(layer, prefix),
            )
            for prefix in ("w13", "w2")
        }

    def register_sources(self):
        for prefix, loader in self.weights.items():
            loader.register_source(getattr(self.layer, f"{prefix}_weight"))

    def weight_loader(
        self, param, loaded_weight, weight_name, shard_id, expert_id=None
    ):
        if shard_id not in ("w1", "w3", "w13", "w2"):
            raise ValueError(f"Unexpected online MoE shard {shard_id!r}.")
        prefix = "w2" if shard_id == "w2" else "w13"
        if f"{prefix}_weight" not in weight_name:
            raise ValueError(f"Unexpected online MoE weight {weight_name!r}.")
        if expert_id is None:
            if shard_id not in ("w13", "w2"):
                raise ValueError("Per-expert online weights require an expert_id.")
            args = (weight_name, shard_id)
            keys = {None}  # Fused loader writes the entire logical tensor.
        else:
            current = getattr(self.layer, f"{prefix}_weight")
            destinations = tuple(
                self.layer._weight_loader_expert_ids(current, expert_id)
            )
            if not destinations:
                # Nonlocal experts must not allocate or start a reload.
                return
            args = (weight_name, shard_id, expert_id)
            keys = {(destination, shard_id) for destination in destinations}
        self.weights[prefix]._load(loaded_weight, keys, *args)

    def finish_post_load(self):
        for loader in self.weights.values():
            loader.finish_post_load()
