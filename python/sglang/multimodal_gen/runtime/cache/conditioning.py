# SPDX-License-Identifier: Apache-2.0
"""Exact, bounded host caching at deterministic conditioning boundaries."""

import copy
import hashlib
import pickle
import weakref
from collections import OrderedDict
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, fields, is_dataclass
from functools import wraps
from typing import Callable

import numpy as np
import torch
import torch.distributed as dist
from diffusers.models.autoencoders.vae import DiagonalGaussianDistribution
from PIL import Image

from sglang.multimodal_gen.runtime.distributed import (
    get_tp_group,
    get_world_size,
    model_parallel_is_initialized,
)
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger

logger = init_logger(__name__)
_active_cache: ContextVar["ConditioningCache | None"] = ContextVar(
    "conditioning_cache", default=None
)
_refresh_cache: ContextVar[bool] = ContextVar(
    "refresh_conditioning_cache", default=False
)
_container_types: set[type] = {DiagonalGaussianDistribution}
_live_caches = weakref.WeakSet()
_weights_epoch = 0


def invalidate_conditioning_caches():
    """Invalidate before weight mutations, including partially failed updates."""
    global _weights_epoch
    _weights_epoch += 1
    for cache in _live_caches:
        cache.clear()


def conditioning_weights_epoch():
    return _weights_epoch


def register_conditioning_container(cls):
    """Opt in a tensor-only posterior container whose fields can be copied."""
    _container_types.add(cls)
    return cls


class Uncacheable(TypeError):
    pass


def _fingerprint(value):
    if isinstance(value, torch.Tensor):
        if (
            type(value) is not torch.Tensor
            or value.layout != torch.strided
            or value.requires_grad
        ):
            raise Uncacheable("only dense inference tensors can be cached")
        tensor = value.detach().cpu().contiguous()
        data = tensor.reshape(-1).view(torch.uint8).numpy()
        return (
            "tensor",
            str(value.dtype),
            str(value.device),
            tuple(value.shape),
            tuple(value.stride()),
            hashlib.sha256(data).digest(),
        )
    if isinstance(value, Image.Image):
        return (
            "image",
            value.mode,
            value.size,
            hashlib.sha256(value.tobytes()).digest(),
            _fingerprint(value.getpalette()),
            _fingerprint(value.info.get("transparency")),
        )
    if isinstance(value, np.ndarray):
        if value.dtype.hasobject:
            raise Uncacheable("object arrays are not conditioning values")
        return (
            "array",
            value.dtype.str,
            value.shape,
            hashlib.sha256(value.tobytes()).digest(),
        )
    if type(value) in (str, bytes, int, float, bool, type(None)):
        return (type(value).__name__, value)
    if isinstance(value, (torch.dtype, torch.device)):
        return (type(value).__name__, str(value))
    if isinstance(value, dict):
        return (
            "dict",
            tuple((_fingerprint(k), _fingerprint(v)) for k, v in value.items()),
        )
    if isinstance(value, (tuple, list)):
        return (type(value).__name__, tuple(_fingerprint(v) for v in value))
    raise Uncacheable(f"unsupported conditioning input: {type(value).__name__}")


@dataclass
class _HostTensor:
    data: torch.Tensor
    device: torch.device


def _map_output(value, tensor_fn, *, restore=False):
    if isinstance(value, _HostTensor if restore else torch.Tensor):
        return tensor_fn(value)
    if type(value) in (str, bytes, int, float, bool, type(None)):
        return value
    if isinstance(value, (torch.dtype, torch.device)):
        return value
    # ModelOutput is both a dataclass and a dict. Reconstruct it as a dataclass
    # so attribute access and tuple indexing retain their original semantics.
    if is_dataclass(value) and not isinstance(value, type):
        return type(value)(
            **{
                field.name: _map_output(
                    vars(value)[field.name], tensor_fn, restore=restore
                )
                for field in fields(value)
            }
        )
    if type(value) in _container_types:
        result = copy.copy(value)
        for name, item in vars(value).items():
            setattr(result, name, _map_output(item, tensor_fn, restore=restore))
        return result
    if type(value) is dict:
        return {k: _map_output(v, tensor_fn, restore=restore) for k, v in value.items()}
    if type(value) is list:
        return [_map_output(v, tensor_fn, restore=restore) for v in value]
    if isinstance(value, tuple):
        items = [_map_output(v, tensor_fn, restore=restore) for v in value]
        return type(value)(*items) if type(value) is not tuple else tuple(items)
    raise Uncacheable(f"unsupported conditioning output: {type(value).__name__}")


class ConditioningCache:
    """One cache per executor/rank. Stored tensors never alias request tensors."""

    def __init__(self, max_bytes: int):
        if max_bytes < 0:
            raise ValueError("conditioning cache capacity must be nonnegative")
        _live_caches.add(self)
        self.max_bytes = max_bytes
        self.bytes = 0
        self.hits = self.misses = self.evictions = self.bypasses = 0
        self._entries = OrderedDict()
        self._models = weakref.WeakKeyDictionary()
        self._next_model = 0

    def clear(self):
        self._entries.clear()
        self.bytes = 0

    def stats(self):
        return dict(
            hits=self.hits,
            misses=self.misses,
            evictions=self.evictions,
            bypasses=self.bypasses,
            entries=len(self._entries),
            bytes=self.bytes,
        )

    @contextmanager
    def scope(self, enabled=True, *, refresh=False):
        # Zero-capacity ranks still participate in encoder hit consensus.
        token = _active_cache.set(self if enabled else None)
        refresh_token = _refresh_cache.set(refresh or _refresh_cache.get())
        try:
            yield
        finally:
            _refresh_cache.reset(refresh_token)
            _active_cache.reset(token)

    def run(
        self,
        model,
        method,
        args,
        kwargs,
        compute: Callable,
        group=None,
        *,
        nested=False,
    ):
        if model not in self._models:
            self._models[model] = self._next_model
            self._next_model += 1
        try:
            if not self.max_bytes:
                raise Uncacheable("cache disabled")
            if kwargs.get("use_cache") or kwargs.get("past_key_values") is not None:
                raise Uncacheable("stateful autoregressive encoding")
            parameter = next(model.parameters(), None)
            precision = str(parameter.dtype) if parameter is not None else None
            key = (
                self._models[model],
                method,
                precision,
                torch.is_autocast_enabled("cuda"),
                torch.get_autocast_dtype("cuda"),
                torch.is_autocast_enabled("cpu"),
                torch.get_autocast_dtype("cpu"),
                _fingerprint(args),
                _fingerprint(kwargs),
            )
            key = hashlib.sha256(pickle.dumps(key)).digest()
        except Uncacheable:
            key = None
        entry = self._entries.get(key)
        hit = entry is not None and not _refresh_cache.get()
        if group is not None and group.world_size > 1:
            # Encoders may issue TP/folding collectives. A rank-local eviction
            # must never leave another rank returning early from the encoder.
            flag = torch.tensor(int(hit), dtype=torch.int32)
            dist.all_reduce(flag, op=dist.ReduceOp.MIN, group=group.cpu_group)
            hit = bool(flag.item())
        if hit:
            self.hits += 1
            self._entries.move_to_end(key)
            logger.debug("Conditioning cache hit: %s.%s", type(model).__name__, method)
            restored = {}

            def restore(t):
                if id(t) not in restored:
                    restored[id(t)] = t.data.to(t.device, copy=True)
                return restored[id(t)]

            return _map_output(entry[0], restore, restore=True)
        if key is None:
            self.bypasses += 1
            return compute()
        self.misses += 1
        # A VLM may reuse image features even when its joint text/image key
        # misses. VAE delegates share one outer posterior entry instead.
        with self.scope(enabled=nested):
            output = compute()
        try:
            size = 0
            seen = set()

            def count(t):
                nonlocal size
                if id(t) not in seen:
                    size += t.numel() * t.element_size()
                    seen.add(id(t))
                if type(t) is not torch.Tensor or t.requires_grad:
                    raise Uncacheable("autograd output")
                return t

            _map_output(output, count)
            if not size or size > self.max_bytes:
                self.bypasses += 1
                return output
        except Uncacheable:
            self.bypasses += 1
            return output
        old = self._entries.pop(key, None)
        if old is not None:
            self.bytes -= old[1]
        while self.bytes + size > self.max_bytes or len(self._entries) >= 128:
            _, (_, removed_size) = self._entries.popitem(last=False)
            self.bytes -= removed_size
            self.evictions += 1
        tensors = {}

        def snapshot(t):
            if id(t) not in tensors:
                tensors[id(t)] = _HostTensor(t.detach().to("cpu", copy=True), t.device)
            return tensors[id(t)]

        stored = _map_output(output, snapshot)
        self._entries[key] = (stored, size)
        self.bytes += size
        logger.debug(
            "Conditioning cache store: %s.%s, %d bytes",
            type(model).__name__,
            method,
            size,
        )
        return output


def _inference_cache(model):
    if torch.compiler.is_compiling():
        return None
    cache = _active_cache.get()
    if cache is None or model.training or torch.is_grad_enabled():
        return None
    # graph capture cannot hash or copy CUDA values through host memory
    if torch.cuda.is_available() and torch.cuda.is_current_stream_capturing():
        return None
    return cache


def cached_encoder_call(model, args, kwargs, compute, group=None):
    cache = _inference_cache(model)
    if (
        cache is None
        or kwargs.get("use_cache")
        or kwargs.get("past_key_values") is not None
    ):
        return compute()
    return cache.run(model, "forward", args, kwargs, compute, group, nested=True)


def cached_conditioning(fn):
    """Cache a deterministic conditioning method on the current encoder TP group."""

    @wraps(fn)
    def wrapped(self, *args, **kwargs):
        cache = _inference_cache(self)
        if cache is None:
            return fn(self, *args, **kwargs)
        group = get_tp_group() if model_parallel_is_initialized() else None
        return cache.run(
            self, fn.__name__, args, kwargs, lambda: fn(self, *args, **kwargs), group
        )

    return wrapped


def cached_vae_encode(fn):
    """Cache the posterior, before sample()/mode() and latent normalization.

    Distributed VAEs currently bypass this cache: their encoding methods can
    run on an owner rank or a model-specific subgroup, unlike encoder TP.
    """

    @wraps(fn)
    def wrapped(self, *args, **kwargs):
        cache = _inference_cache(self)
        if cache is None or (model_parallel_is_initialized() and get_world_size() > 1):
            return fn(self, *args, **kwargs)
        # Tiling and slicing can change numerical results without changing x.
        settings = {
            name: value
            for name, value in vars(self).items()
            if name.startswith(
                (
                    "use_",
                    "tile_",
                    "blend_",
                    "parallel_",
                    "spatial_compression",
                    "temporal_compression",
                )
            )
            and type(value) in (int, float, bool, str, type(None))
        }
        return cache.run(
            self,
            fn.__name__,
            (args, settings),
            kwargs,
            lambda: fn(self, *args, **kwargs),
        )

    return wrapped


class ConditioningEncoderMixin:
    """Cache replicated encoders without adding a module or altering state_dict."""

    def __call__(self, *args, **kwargs):
        forward = super().__call__
        return cached_encoder_call(self, args, kwargs, lambda: forward(*args, **kwargs))
