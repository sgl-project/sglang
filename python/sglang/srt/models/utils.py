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

import itertools
import logging
from typing import Iterable, Optional, Tuple

import torch
import torch.nn as nn

from sglang.srt.layers.radix_attention import RadixAttention
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.model_loader.weight_utils import default_weight_loader
from sglang.srt.utils import is_cuda

logger = logging.getLogger(__name__)

_is_cuda = is_cuda()


class AutoWeightsLoader:
    """
    Helper class to load weights into a torch.nn.Module. It automatically
    detects child modules and parameters while iterating weights only once.

    This simplifies model code by abstracting the common weight loading logic,
    reducing code duplication across different model implementations.

    Adapted from vLLM's AutoWeightsLoader implementation.

    Args:
        module: The root module to load weights into
        skip_prefixes: List of weight name prefixes to skip
        skip_substrs: List of substrings to skip in weight names
        ignore_unexpected_prefixes: List of prefixes for unexpected weights to ignore
        ignore_unexpected_suffixes: List of suffixes for unexpected weights to ignore

    Example:
        >>> def load_weights(self, weights):
        ...     loader = AutoWeightsLoader(
        ...         self,
        ...         skip_prefixes=["lm_head"] if self.config.tie_word_embeddings else [],
        ...         skip_substrs=["rotary_emb.inv_freq"],
        ...     )
        ...     return loader.load_weights(weights)
    """

    # Common weights that should be skipped (e.g., ColossalAI rotary embeddings)
    ROTARY_EMBEDS_UNUSED_WEIGHTS = [
        "rotary_emb.inv_freq",
        "rotary_emb.cos_cached",
        "rotary_emb.sin_cached",
    ]

    def __init__(
        self,
        module: nn.Module,
        *,
        skip_prefixes: Optional[list] = None,
        skip_substrs: Optional[list] = None,
        ignore_unexpected_prefixes: Optional[list] = None,
        ignore_unexpected_suffixes: Optional[list] = None,
    ) -> None:
        self.module = module
        # Create copies to avoid mutating caller's lists
        self.skip_prefixes = list(skip_prefixes) if skip_prefixes else []
        self.skip_substrs = list(skip_substrs) if skip_substrs else []
        self.ignore_unexpected_prefixes = (
            list(ignore_unexpected_prefixes) if ignore_unexpected_prefixes else []
        )
        self.ignore_unexpected_suffixes = (
            list(ignore_unexpected_suffixes) if ignore_unexpected_suffixes else []
        )
        # Always skip common rotary embedding weights
        self.skip_substrs.extend(self.ROTARY_EMBEDS_UNUSED_WEIGHTS)

    def _groupby_prefix(
        self,
        weights: Iterable[Tuple[str, torch.Tensor]],
    ) -> Iterable[Tuple[str, Iterable[Tuple[str, torch.Tensor]]]]:
        """Group weights by their first prefix component (before the first dot)."""
        weights_by_parts = (
            (weight_name.split(".", 1), weight_data)
            for weight_name, weight_data in weights
        )
        for prefix, group in itertools.groupby(weights_by_parts, key=lambda x: x[0][0]):
            yield (
                prefix,
                (
                    ("" if len(parts) == 1 else parts[1], weights_data)
                    for parts, weights_data in group
                ),
            )

    def _get_qualname(self, prefix: str, rest: str) -> str:
        """Construct fully qualified name from prefix and rest."""
        if prefix == "":
            return rest
        if rest == "":
            return prefix
        return ".".join((prefix, rest))

    def _can_skip(self, qualname: str) -> bool:
        """Check if parameter should be skipped based on skip rules."""
        return any(qualname.startswith(p) for p in self.skip_prefixes) or any(
            substr in qualname for substr in self.skip_substrs
        )

    def _can_ignore_unexpected(self, qualname: str) -> bool:
        """Check if unexpected weight can be ignored based on ignore rules."""
        starts_with_ignored = any(
            qualname.startswith(p) for p in self.ignore_unexpected_prefixes
        )
        ends_with_ignored = any(
            qualname.endswith(s) for s in self.ignore_unexpected_suffixes
        )
        return starts_with_ignored or ends_with_ignored

    def _load_param(
        self,
        base_prefix: str,
        param: torch.Tensor,
        weights: Iterable[Tuple[str, torch.Tensor]],
    ) -> Iterable[str]:
        """Load weights into a single parameter."""
        for weight_name, weight_data in weights:
            weight_qualname = self._get_qualname(base_prefix, weight_name)

            if self._can_skip(weight_qualname):
                logger.debug("Skipping weight %s", weight_qualname)
                continue

            if weight_name != "":
                if self._can_ignore_unexpected(weight_qualname):
                    logger.debug("Ignoring unexpected weight %s", weight_qualname)
                    continue
                raise ValueError(
                    f"Attempted to load nested weight '{weight_qualname}' "
                    f"into a single parameter '{base_prefix}'"
                )

            weight_loader = getattr(param, "weight_loader", default_weight_loader)
            weight_loader(param, weight_data)
            logger.debug("Loaded weight %s with shape %s", weight_qualname, param.shape)
            yield weight_qualname

    def _add_loadable_non_param_tensors(self, module: nn.Module, child_params: dict):
        """Add tensor names not in model params (e.g., batchnorm statistics)."""
        if isinstance(
            module,
            (
                nn.BatchNorm1d,
                nn.BatchNorm2d,
                nn.BatchNorm3d,
                nn.LazyBatchNorm1d,
                nn.LazyBatchNorm2d,
                nn.LazyBatchNorm3d,
                nn.SyncBatchNorm,
            ),
        ):
            module_state_dict = module.state_dict()
            for stat_name in ("running_mean", "running_var", "num_batches_tracked"):
                if stat_name in module_state_dict:
                    child_params[stat_name] = module_state_dict[stat_name]

    def _load_module(
        self,
        base_prefix: str,
        module: nn.Module,
        weights: Iterable[Tuple[str, torch.Tensor]],
    ) -> Iterable[str]:
        """Recursively load weights into a module and its children."""
        # If module has a custom load_weights method, use it
        if module != self.module:
            module_load_weights = getattr(module, "load_weights", None)
            if callable(module_load_weights):
                loaded_params = module_load_weights(weights)
                if loaded_params is None:
                    logger.warning(
                        "Unable to collect loaded parameters for module %s. "
                        "Module.load_weights() should return an iterable of "
                        "loaded parameter names.",
                        module,
                    )
                else:
                    yield from (
                        self._get_qualname(base_prefix, x) for x in loaded_params
                    )
                return

        # Get child modules and parameters
        child_modules = dict(module.named_children())
        child_params = dict(module.named_parameters(recurse=False))
        self._add_loadable_non_param_tensors(module, child_params)

        # Process weights grouped by prefix
        for child_prefix, child_weights in self._groupby_prefix(weights):
            prefix = self._get_qualname(base_prefix, child_prefix)

            if child_prefix in child_modules:
                if self._can_skip(prefix + "."):
                    logger.debug("Skipping module %s", prefix)
                    continue
                yield from self._load_module(
                    prefix, child_modules[child_prefix], child_weights
                )
            elif child_prefix in child_params:
                if self._can_skip(prefix):
                    logger.debug("Skipping param %s", prefix)
                    continue
                yield from self._load_param(
                    prefix, child_params[child_prefix], child_weights
                )
            else:
                # Check if we should skip or ignore this missing parameter
                can_skip_module = self._can_skip(prefix + ".")
                can_skip_param = self._can_skip(prefix)
                if can_skip_module or can_skip_param:
                    logger.debug("Skipping missing %s", prefix)
                    continue

                can_ignore_module = self._can_ignore_unexpected(prefix + ".")
                can_ignore_param = self._can_ignore_unexpected(prefix)
                if can_ignore_module or can_ignore_param:
                    logger.debug("Ignoring missing %s", prefix)
                    continue

                msg = (
                    f"There is no module or parameter named '{prefix}' "
                    f"in {type(self.module).__name__}"
                )
                raise ValueError(msg)

    def load_weights(
        self,
        weights: Iterable[Tuple[str, torch.Tensor]],
    ) -> set:
        """
        Load weights into the module.

        Args:
            weights: Iterable of (name, tensor) tuples to load

        Returns:
            Set of weight names that were successfully loaded
        """
        # Filter out skippable weights early
        weights = (
            (name, weight) for name, weight in weights if not self._can_skip(name)
        )
        autoloaded_weights = set(self._load_module("", self.module, weights))
        return autoloaded_weights


if _is_cuda:
    from sgl_kernel import FusedSetKVBufferArg


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
