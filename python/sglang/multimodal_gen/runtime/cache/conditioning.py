# SPDX-License-Identifier: Apache-2.0
"""Exact conditioning reuse within grouped stages and across requests."""

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
    get_replica_group,
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
_cross_request_cache: ContextVar[bool] = ContextVar(
    "cross_request_conditioning_cache", default=True
)
_prefer_cache: ContextVar[bool] = ContextVar("prefer_conditioning_cache", default=False)
_stage_encoder: ContextVar[object | None] = ContextVar("stage_encoder", default=None)
_container_types: set[type] = {DiagonalGaussianDistribution}
_live_caches = weakref.WeakSet()
_weights_epoch = 0


def invalidate_conditioning_caches(modules=None):
    """Invalidate before weight mutations, including partially failed updates."""
    global _weights_epoch
    _weights_epoch += 1
    parameters = (
        None
        if modules is None
        else {id(p) for module in modules for p in module.parameters()}
    )
    for cache in _live_caches:
        cache.invalidate(parameters)


def conditioning_weights_epoch():
    return _weights_epoch


def register_conditioning_container(cls):
    """Opt in a tensor-only posterior container whose fields can be copied."""
    _container_types.add(cls)
    return cls


class Uncacheable(TypeError):
    pass


@contextmanager
def prefer_conditioning_cache():
    # retain reusable negative conditioning ahead of changing positive prompts
    token = _prefer_cache.set(True)
    try:
        yield
    finally:
        _prefer_cache.reset(token)


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


@dataclass
class _CacheEntry:
    output: object
    size: int
    ready: tuple[torch.cuda.Event, ...]
    owner: int
    preferred: bool

    def wait(self):
        for event in self.ready:
            event.synchronize()


@dataclass
class _GroupEntry:
    output: object
    owner: int
    copied_bytes: int
    streams: tuple[torch.cuda.Stream, ...]

    def wait(self):
        consumers = {}
        for producer in self.streams:
            consumer = torch.cuda.current_stream(producer.device)
            if consumer != producer:
                consumer.wait_stream(producer)
                consumers[producer.device] = consumer

        if not consumers:
            return

        def record(t):
            if t.device in consumers:
                t.record_stream(consumers[t.device])
            return t

        _map_output(self.output, record)


def _copy_output(value, *, share_tensors=False):
    tensors = {}

    def copy_tensor(t):
        if id(t) not in tensors:
            tensors[id(t)] = t if share_tensors else t.clone()
        return tensors[id(t)]

    return _map_output(value, copy_tensor)


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
    """One cache per executor/rank with private host entries and scoped device reuse."""

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
        self._group_entries = ContextVar("conditioning_group_entries", default=None)
        self.group_hits = 0

    def clear(self):
        group_entries = self._group_entries.get()
        if group_entries is not None:
            group_entries.clear()
        for entry in self._entries.values():
            entry.wait()
        self._entries.clear()
        self.bytes = 0

    def invalidate(self, parameters):
        if parameters is None:
            self.clear()
            return
        # shared parameters invalidate both owners, including nested encoders
        owners = {
            identity
            for model, identity in self._models.items()
            if isinstance(model, torch.nn.Module)
            and any(id(p) in parameters for p in model.parameters())
        }
        for key, entry in list(self._entries.items()):
            if entry.owner in owners:
                entry.wait()
                self.bytes -= entry.size
                del self._entries[key]
        group_entries = self._group_entries.get()
        if group_entries is not None:
            for key, entry in list(group_entries.items()):
                if entry.owner in owners:
                    del group_entries[key]

    def stats(self):
        return dict(
            hits=self.hits,
            misses=self.misses,
            evictions=self.evictions,
            bypasses=self.bypasses,
            entries=len(self._entries),
            bytes=self.bytes,
            group_hits=self.group_hits,
        )

    def _identity(self, owner):
        if owner not in self._models:
            self._models[owner] = self._next_model
            self._next_model += 1
        return self._models[owner]

    @contextmanager
    def group_scope(self, enabled=True):
        """Keep device results only while a stage executes a group of requests."""
        token = self._group_entries.set({} if enabled else None)
        try:
            yield
        finally:
            self._group_entries.reset(token)

    def _remember_group(self, key, output, owner, share_in_group):
        entries = self._group_entries.get()
        if entries is None:
            return
        tensors = {}

        def visit(t):
            tensors[id(t)] = t
            return t

        _map_output(output, visit)
        size = (
            0
            if share_in_group
            else sum(t.numel() * t.element_size() for t in tensors.values())
        )
        # private snapshots must not retain unbounded intermediate encoder states
        if not share_in_group and (
            len(entries) >= 128
            or size + sum(entry.copied_bytes for entry in entries.values())
            > 512 * 1024**2
        ):
            return
        stored = _copy_output(output, share_tensors=share_in_group)
        streams = tuple(
            torch.cuda.current_stream(device)
            for device in {
                t.device for t in tensors.values() if t.device.type == "cuda"
            }
        )
        entries[key] = _GroupEntry(stored, owner, size, streams)

    @contextmanager
    def scope(self, enabled=True, *, refresh=False, cross_request=True):
        # Zero-capacity ranks still participate in encoder hit consensus.
        if refresh and self._group_entries.get() is not None:
            self._group_entries.get().clear()
        token = _active_cache.set(self if enabled else None)
        refresh_token = _refresh_cache.set(refresh or _refresh_cache.get())
        cross_request_token = _cross_request_cache.set(
            cross_request and _cross_request_cache.get()
        )
        try:
            yield
        finally:
            _cross_request_cache.reset(cross_request_token)
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
        namespace=None,
        share_in_group=False,
        cross_request=True,
    ):
        group_entries = self._group_entries.get()
        cross_request = cross_request and _cross_request_cache.get()
        if not cross_request and group_entries is None:
            return compute()
        try:
            if (not cross_request or not self.max_bytes) and group_entries is None:
                raise Uncacheable("cache disabled")
            if kwargs.get("use_cache") or kwargs.get("past_key_values") is not None:
                raise Uncacheable("stateful autoregressive encoding")
            parameter = next(model.parameters(), None)
            precision = str(parameter.dtype) if parameter is not None else None
            key = (
                self._identity(model),
                self._identity(namespace) if namespace is not None else None,
                method,
                share_in_group,
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
        entry = self._entries.get(key) if cross_request else None
        group_entry = group_entries.get(key) if group_entries is not None else None
        hit = group_entry is not None or (
            entry is not None and not _refresh_cache.get()
        )
        if group is not None and group.world_size > 1:
            # Encoders may issue TP/folding collectives. A rank-local eviction
            # must never leave another rank returning early from the encoder.
            flag = torch.tensor(int(hit), dtype=torch.int32)
            dist.all_reduce(flag, op=dist.ReduceOp.MIN, group=group.cpu_group)
            hit = bool(flag.item())
        if hit:
            self.hits += 1
            if entry is not None:
                entry.preferred |= _prefer_cache.get()
                self._entries.move_to_end(key)
            if group_entry is not None:
                self.group_hits += 1
                group_entry.wait()
                return _copy_output(group_entry.output, share_tensors=share_in_group)
            entry.wait()
            logger.debug("Conditioning cache hit: %s.%s", type(model).__name__, method)
            restored = {}

            def restore(t):
                if id(t) not in restored:
                    restored[id(t)] = t.data.to(t.device, copy=True, non_blocking=True)
                return restored[id(t)]

            output = _map_output(entry.output, restore, restore=True)
            self._remember_group(key, output, entry.owner, share_in_group)
            return output
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
            self._remember_group(key, output, self._identity(model), share_in_group)
            if not cross_request or not size or size > self.max_bytes:
                self.bypasses += 1
                return output
        except Uncacheable:
            self.bypasses += 1
            return output
        old = self._entries.pop(key, None)
        if old is not None:
            old.wait()
            self.bytes -= old.size
        preferred = _prefer_cache.get()
        evictable = [
            key
            for key, entry in self._entries.items()
            if preferred or not entry.preferred
        ]
        available = (
            self.max_bytes
            - self.bytes
            + sum(self._entries[key].size for key in evictable)
        )
        if size > available or (len(self._entries) >= 128 and not evictable):
            self.bypasses += 1
            return output
        for evicted in evictable:
            if self.bytes + size <= self.max_bytes and len(self._entries) < 128:
                break
            removed = self._entries.pop(evicted)
            removed.wait()
            self.bytes -= removed.size
            self.evictions += 1
        tensors = {}
        copy_streams = {}

        def snapshot(t):
            if id(t) not in tensors:
                if t.device.type == "cuda":
                    copy_streams[t.device] = torch.cuda.current_stream(t.device)
                    host = torch.empty_like(t, device="cpu", pin_memory=True)
                    host.copy_(t.detach(), non_blocking=True)
                else:
                    host = t.detach().to("cpu", copy=True)
                tensors[id(t)] = _HostTensor(host, t.device)
            return tensors[id(t)]

        stored = _map_output(output, snapshot)
        # Copies precede downstream mutations on each producing stream. Wait
        # only before reading or freeing host storage, not on the cold path.
        ready = []
        for stream in copy_streams.values():
            event = torch.cuda.Event()
            event.record(stream)
            ready.append(event)
        self._entries[key] = _CacheEntry(
            stored, size, tuple(ready), self._identity(model), preferred
        )
        self.bytes += size
        logger.debug(
            "Conditioning cache store: %s.%s, %d bytes",
            type(model).__name__,
            method,
            size,
        )
        return output


def _inference_cache(model, *, share_in_group=False):
    if torch.compiler.is_compiling():
        return None
    cache = _active_cache.get()
    if cache is None or model.training or torch.is_grad_enabled():
        return None
    # FSDP reuses only consumed stage outputs, never sharded intermediate states
    if not _cross_request_cache.get() and not share_in_group:
        return None
    # graph capture cannot hash or copy CUDA values through host memory
    if torch.cuda.is_available() and torch.cuda.is_current_stream_capturing():
        return None
    return cache


def cached_encoder_call(
    model,
    args,
    kwargs,
    compute,
    group=None,
    *,
    namespace=None,
    nested=True,
    share_in_group=False,
    cross_request=True,
):
    cache = _inference_cache(model, share_in_group=share_in_group)
    if (
        cache is None
        or model is _stage_encoder.get()
        or kwargs.get("use_cache")
        or kwargs.get("past_key_values") is not None
    ):
        return compute()

    if not _cross_request_cache.get() and model_parallel_is_initialized():
        # the encoder's TP group may not cover its FSDP shard group
        group = get_replica_group()

    def compute_conditioning():
        # a stage owns the consumed output; preserve nested vision-method caches
        token = _stage_encoder.set(model)
        try:
            return compute()
        finally:
            _stage_encoder.reset(token)

    return cache.run(
        model,
        "forward",
        args,
        kwargs,
        compute_conditioning if namespace is not None and cross_request else compute,
        group,
        nested=nested,
        namespace=namespace,
        share_in_group=share_in_group,
        cross_request=cross_request,
    )


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
