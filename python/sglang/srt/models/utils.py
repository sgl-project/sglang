# Copyright 2023-2025 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
from __future__ import annotations

import itertools
import logging
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from functools import lru_cache
from typing import TYPE_CHECKING, Any, Optional, Tuple

import numpy as np
import torch

from sglang.jit_kernel.norm import can_use_fused_inplace_qknorm, fused_inplace_qknorm
from sglang.srt.environ import envs
from sglang.srt.layers.radix_attention import RadixAttention
from sglang.srt.layers.utils import PPMissingLayer
from sglang.srt.model_executor.cuda_graph_runner import get_is_capture_mode
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.model_loader.weight_utils import default_weight_loader
from sglang.srt.utils import is_cuda
from sglang.srt.utils.custom_op import register_custom_op

logger = logging.getLogger(__name__)
_is_cuda = is_cuda()

if TYPE_CHECKING:
    from sglang.srt.layers.layernorm import RMSNorm

_is_cuda = is_cuda()

WeightsMapping = Mapping[str, Optional[str]]
"""If a key maps to a value of `None`, the corresponding weight is ignored."""


@dataclass
class WeightsMapper:
    """Maps the name of each weight if they match the following patterns."""

    orig_to_new_substr: WeightsMapping = field(default_factory=dict)
    orig_to_new_prefix: WeightsMapping = field(default_factory=dict)
    orig_to_new_suffix: WeightsMapping = field(default_factory=dict)

    def _map_name(self, key: str) -> Optional[str]:
        for substr, new_key in sorted(
            self.orig_to_new_substr.items(), key=lambda i: len(i[0]), reverse=True
        ):
            if substr in key:
                if new_key is None:
                    return None

                key = key.replace(substr, new_key, 1)
                break

        for prefix, new_key in sorted(
            self.orig_to_new_prefix.items(), key=lambda i: len(i[0]), reverse=True
        ):
            if key.startswith(prefix):
                if new_key is None:
                    return None

                key = key.replace(prefix, new_key, 1)
                break

        for suffix, new_key in sorted(
            self.orig_to_new_suffix.items(), key=lambda i: len(i[0]), reverse=True
        ):
            if key.endswith(suffix):
                if new_key is None:
                    return None

                key = new_key.join(key.rsplit(suffix, 1))
                break

        return key

    def apply(
        self, weights: Iterable[tuple[str, torch.Tensor]]
    ) -> Iterable[tuple[str, torch.Tensor]]:
        return (
            (out_name, data)
            for name, data in weights
            if (out_name := self._map_name(name)) is not None
        )

    def apply_list(self, values: list[str]) -> list[str]:
        return [
            out_name
            for name in values
            if (out_name := self._map_name(name)) is not None
        ]

    def apply_dict(self, values: dict[str, Any]) -> dict[str, Any]:
        return {
            out_name: value
            for name, value in values.items()
            if (out_name := self._map_name(name)) is not None
        }


class AutoWeightsLoader:
    """Single-pass weight loader with module/param overrides and strict mismatch checks."""

    ROTARY_EMBEDS_UNUSED_WEIGHTS = [
        "rotary_emb.inv_freq",
        "rotary_emb.cos_cached",
        "rotary_emb.sin_cached",
    ]

    def __init__(
        self,
        module: torch.nn.Module,
        *,
        skip_prefixes: Optional[list[str]] = None,
        skip_substrs: Optional[list[str]] = None,
        ignore_unexpected_prefixes: Optional[list[str]] = None,
        ignore_unexpected_suffixes: Optional[list[str]] = None,
    ) -> None:
        self.module = module
        # Copy to avoid mutating caller lists
        self.skip_prefixes = list(skip_prefixes) if skip_prefixes else []
        self.skip_substrs = list(skip_substrs) if skip_substrs else []
        self.ignore_unexpected_prefixes = (
            list(ignore_unexpected_prefixes) if ignore_unexpected_prefixes else []
        )
        self.ignore_unexpected_suffixes = (
            list(ignore_unexpected_suffixes) if ignore_unexpected_suffixes else []
        )
        self.skip_substrs.extend(self.ROTARY_EMBEDS_UNUSED_WEIGHTS)

    def groupby_prefix(
        self, weights: Iterable[tuple[str, torch.Tensor]]
    ) -> Iterable[tuple[str, Iterable[tuple[str, torch.Tensor]]]]:
        weights_by_parts = (
            (weight_name.split(".", 1), weight_data)
            for weight_name, weight_data in weights
        )
        for prefix, group in itertools.groupby(weights_by_parts, key=lambda x: x[0][0]):
            yield (
                prefix,
                (
                    ("" if len(parts) == 1 else parts[1], weight_data)
                    for parts, weight_data in group
                ),
            )

    def qualname(self, prefix: str, rest: str) -> str:
        if not prefix:
            return rest
        if not rest:
            return prefix
        return f"{prefix}.{rest}"

    def can_skip(self, qualname: str) -> bool:
        return any(qualname.startswith(p) for p in self.skip_prefixes) or any(
            substr in qualname for substr in self.skip_substrs
        )

    def can_ignore_unexpected(self, qualname: str) -> bool:
        return any(
            qualname.startswith(p) for p in self.ignore_unexpected_prefixes
        ) or any(qualname.endswith(s) for s in self.ignore_unexpected_suffixes)

    def load_param(
        self,
        prefix: str,
        param: torch.Tensor,
        weights: Iterable[tuple[str, torch.Tensor]],
    ) -> Iterable[str]:
        for weight_name, weight_tensor in weights:
            qualname = self.qualname(prefix, weight_name)

            if self.can_skip(qualname):
                logger.debug("Skipping weight %s", qualname)
                continue

            if weight_name:
                if self.can_ignore_unexpected(qualname):
                    logger.debug("Ignoring unexpected nested weight %s", qualname)
                    continue
                raise ValueError(
                    f"Attempted to load nested weight '{qualname}' into parameter "
                    f"'{prefix}'"
                )

            weight_loader = getattr(param, "weight_loader", default_weight_loader)
            weight_loader(param, weight_tensor)
            logger.debug("Loaded weight %s with shape %s", qualname, tuple(param.shape))
            yield qualname

    def add_loadable_non_param_tensors(
        self, module: torch.nn.Module, child_params: dict[str, torch.Tensor]
    ) -> None:
        """Add non-parameter tensors (e.g., batchnorm stats) that may appear in weights."""
        if isinstance(
            module,
            (
                torch.nn.BatchNorm1d,
                torch.nn.BatchNorm2d,
                torch.nn.BatchNorm3d,
                torch.nn.LazyBatchNorm1d,
                torch.nn.LazyBatchNorm2d,
                torch.nn.LazyBatchNorm3d,
                torch.nn.SyncBatchNorm,
            ),
        ):
            module_state_dict = module.state_dict()
            for stat_name in ("running_mean", "running_var", "num_batches_tracked"):
                if stat_name in module_state_dict:
                    child_params[stat_name] = module_state_dict[stat_name]

    def load_module(
        self,
        prefix: str,
        module: torch.nn.Module,
        weights: Iterable[tuple[str, torch.Tensor]],
    ) -> Iterable[str]:
        if isinstance(module, PPMissingLayer):
            return

        if module is not self.module:
            custom_loader = getattr(module, "load_weights", None)
            if callable(custom_loader):
                loaded = custom_loader(weights)
                if loaded is None:
                    logger.warning(
                        "Unable to collect loaded parameters for module %s", module
                    )
                else:
                    for name in loaded:
                        yield self.qualname(prefix, name)

        child_modules = dict(module.named_children())
        direct_params = dict(module.named_parameters(recurse=False))
        self.add_loadable_non_param_tensors(module, direct_params)

        for child_prefix, child_weights in self.groupby_prefix(weights):
            qualname = self.qualname(prefix, child_prefix)

            if child_prefix in child_modules:
                if self.can_skip(f"{qualname}."):
                    logger.debug("Skipping module %s", qualname)
                    continue

                yield from self.load_module(
                    qualname, child_modules[child_prefix], child_weights
                )
                continue

            if child_prefix in direct_params:
                if self.can_skip(qualname):
                    logger.debug("Skipping param %s", qualname)
                    continue
                yield from self.load_param(
                    qualname, direct_params[child_prefix], child_weights
                )
                continue

            if self.can_skip(f"{qualname}.") or self.can_skip(qualname):
                logger.debug("Skipping missing %s", qualname)
                continue

            if self.can_ignore_unexpected(f"{qualname}.") or self.can_ignore_unexpected(
                qualname
            ):
                logger.debug("Ignoring unexpected %s", qualname)
                continue

            raise ValueError(
                f"There is no module or parameter named '{qualname}' "
                f"in {type(self.module).__name__}"
            )

    def load_weights(
        self,
        weights: Iterable[tuple[str, torch.Tensor]],
        *,
        mapper: Optional[WeightsMapper] = None,
    ) -> set[str]:
        if mapper is not None:
            weights = mapper.apply(weights)

        filtered = (
            (name, tensor) for name, tensor in weights if not self.can_skip(name)
        )
        return set(self.load_module("", self.module, filtered))


def enable_fused_set_kv_buffer(forward_batch: ForwardBatch):
    """Enable fused set_kv_buffer only on CUDA with bfloat16 KV cache."""
    return (
        _is_cuda
        and hasattr(forward_batch.token_to_kv_pool, "dtype")
        and forward_batch.token_to_kv_pool.dtype == torch.bfloat16
    )


def create_fused_set_kv_buffer_arg(
    value: torch.Tensor,
    layer: RadixAttention,
    forward_batch: ForwardBatch,
):
    from sgl_kernel import FusedSetKVBufferArg

    layer_id = layer.layer_id
    token_to_kv_pool = forward_batch.token_to_kv_pool

    k_buffer = token_to_kv_pool.get_key_buffer(layer_id)
    v_buffer = token_to_kv_pool.get_value_buffer(layer_id)

    return FusedSetKVBufferArg(
        value=value,
        k_buffer=k_buffer.view(k_buffer.shape[0], -1),
        v_buffer=v_buffer.view(v_buffer.shape[0], -1),
        k_scale=layer.k_scale,
        v_scale=layer.v_scale,
        cache_loc=forward_batch.out_cache_loc,
    )


def permute_inv(perm: torch.Tensor) -> torch.Tensor:
    inv_perm = torch.empty_like(perm)
    inv_perm[perm] = torch.arange(perm.numel(), device=perm.device, dtype=perm.dtype)
    return inv_perm


def compute_cu_seqlens_from_grid_numpy(grid_thw: torch.Tensor) -> torch.Tensor:
    """
    Compute cu_seqlens from grid_thw using NumPy.

    grid_thw: [T, 3] int tensor on CPU.
              columns: [repeat_count, H, W]
    Returns:
        cu_seqlens: 1D int32 tensor on CPU, shape [N + 1]
    """
    assert (
        grid_thw.device.type == "cpu"
    ), "compute_cu_seqlens_from_grid_numpy expects a CPU tensor"
    arr = grid_thw.numpy()

    cu_seqlens = np.repeat(arr[:, 1] * arr[:, 2], arr[:, 0]).cumsum(
        axis=0, dtype=np.int32
    )
    cu_seqlens = np.concatenate([np.zeros(1, dtype=np.int32), cu_seqlens])
    cu_seqlens = torch.from_numpy(cu_seqlens)
    return cu_seqlens


class RotaryPosMixin:

    @staticmethod
    @lru_cache(maxsize=1024)
    def rot_pos_ids(h: int, w: int, spatial_merge_size: int) -> torch.Tensor:
        if isinstance(h, torch.Tensor):
            h = int(h.item())
        if isinstance(w, torch.Tensor):
            w = int(w.item())
        if isinstance(spatial_merge_size, torch.Tensor):
            spatial_merge_size = int(spatial_merge_size.item())
        hpos_ids = np.broadcast_to(np.arange(h).reshape(h, 1), (h, w))
        h_div = h // spatial_merge_size
        w_div = w // spatial_merge_size
        hpos_ids = hpos_ids.reshape(
            h_div,
            spatial_merge_size,
            w_div,
            spatial_merge_size,
        )
        hpos_ids = hpos_ids.transpose(0, 2, 1, 3)
        hpos_ids = hpos_ids.flatten()

        wpos_ids = np.broadcast_to(np.arange(w).reshape(1, w), (h, w))
        wpos_ids = wpos_ids.reshape(
            h_div,
            spatial_merge_size,
            w_div,
            spatial_merge_size,
        )
        wpos_ids = wpos_ids.transpose(0, 2, 1, 3)
        wpos_ids = wpos_ids.flatten()

        return torch.from_numpy(np.stack([hpos_ids, wpos_ids], axis=-1))


def apply_qk_norm(
    q: torch.Tensor,
    k: torch.Tensor,
    q_norm: RMSNorm,
    k_norm: RMSNorm,
    head_dim: int,
    alt_stream: Optional[torch.cuda.Stream] = None,
    allow_inplace: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Apply QK normalization for query and key tensors.
    If eligible, we will use JIT fused inplace QK normalization for better performance.

    Args:
        q: Query tensor of shape [batch_size, ...]
        k: Key tensor of shape [batch_size, ...]
        q_norm: RMSNorm layer for query normalization
        k_norm: RMSNorm layer for key normalization
        head_dim: Dimension of each attention head
        alt_stream: Optional alternative CUDA stream for overlapping computation
        allow_inplace: Whether to allow inplace normalization. (True for better performance)

    Returns:
        Tuple of normalized query and key tensors
    """

    batch_size = q.size(0)
    q_eps = q_norm.variance_epsilon
    k_eps = k_norm.variance_epsilon
    if (
        _is_cuda  # TODO(dark): have not tested on ROCm or other backends
        and allow_inplace  # TODO(dark): this can be relaxed if needed
        and (q_eps == k_eps)  # TODO(dark): this can also be relaxed
        and not envs.SGLANG_ENABLE_DETERMINISTIC_INFERENCE.get()
        and can_use_fused_inplace_qknorm(head_dim)
    ):
        fused_inplace_qknorm(
            q=q.view(batch_size, -1, head_dim),
            k=k.view(batch_size, -1, head_dim),
            q_weight=q_norm.weight,
            k_weight=k_norm.weight,
            head_dim=head_dim,
            eps=q_eps,
        )
        return q, k

    if alt_stream is not None and get_is_capture_mode():
        current_stream = torch.cuda.current_stream()
        alt_stream.wait_stream(current_stream)
        q_by_head = q.reshape(-1, head_dim)
        q_by_head = q_norm(q_by_head)
        with torch.cuda.stream(alt_stream):
            k_by_head = k.reshape(-1, head_dim)
            k_by_head = k_norm(k_by_head)
        current_stream.wait_stream(alt_stream)
    else:
        q_by_head = q.reshape(-1, head_dim)
        q_by_head = q_norm(q_by_head)
        k_by_head = k.reshape(-1, head_dim)
        k_by_head = k_norm(k_by_head)
    q = q_by_head.view(q.shape)
    k = k_by_head.view(k.shape)
    return q, k


# Register the inplace op
fused_inplace_qknorm = register_custom_op(fused_inplace_qknorm, mutates_args=["q", "k"])
