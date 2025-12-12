# Copied and adapted from: https://github.com/hao-ai-lab/FastVideo

# SPDX-License-Identifier: Apache-2.0
import os
from collections import defaultdict
from collections.abc import Hashable
from typing import Any

import torch
import torch.distributed as dist
from safetensors.torch import load_file

from sglang.multimodal_gen.runtime.distributed import get_local_torch_device
from sglang.multimodal_gen.runtime.layers.lora.linear import (
    BaseLayerWithLoRA,
    replace_submodule,
    wrap_with_lora_layer,
)
from sglang.multimodal_gen.runtime.loader.utils import get_param_names_mapping
from sglang.multimodal_gen.runtime.pipelines_core.composed_pipeline_base import (
    ComposedPipelineBase,
)
from sglang.multimodal_gen.runtime.server_args import ServerArgs
from sglang.multimodal_gen.runtime.utils.hf_diffusers_utils import maybe_download_lora
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger

# to avoid deadlocks when forking
os.environ["TOKENIZERS_PARALLELISM"] = "false"

logger = init_logger(__name__)


class LoRAPipeline(ComposedPipelineBase):
    """
    Pipeline that supports injecting LoRA adapters into the diffusion transformer.
    """

    # Type annotations for instance attributes (initialized in __init__)
    # [lora_nickname][target_LoRA_weight_name_in_SGLang_dit] = weight
    # e.g., [jinx][transformer_blocks.0.attn.to_v.lora_A]
    lora_adapters: dict[str, dict[str, torch.Tensor]]
    loaded_adapter_paths: dict[str, str]  # nickname -> lora_path
    # Track current adapter per module: {"transformer": "high_lora", "transformer_2": "low_lora"}
    cur_adapter_name: dict[str, str]
    cur_adapter_path: dict[str, str]
    # [dit_layer_name] = wrapped_lora_layer
    lora_layers: dict[str, BaseLayerWithLoRA]
    lora_layers_critic: dict[str, BaseLayerWithLoRA]
    lora_layers_transformer_2: dict[str, BaseLayerWithLoRA]
    server_args: ServerArgs
    exclude_lora_layers: list[str]
    device: torch.device
    lora_target_modules: list[str] | None
    lora_path: str | None
    lora_nickname: str
    lora_rank: int | None
    lora_alpha: int | None
    lora_initialized: bool
    # Track merge status per module: {"transformer": True, "transformer_2": False}
    is_lora_merged: dict[str, bool]
    # Valid target values for set_lora (class constant, immutable)
    VALID_TARGETS: list[str] = ["all", "transformer", "transformer_2", "critic"]

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        # Initialize all mutable instance attributes to avoid sharing across instances
        self.lora_adapters = defaultdict(dict)
        self.loaded_adapter_paths = {}
        self.cur_adapter_name = {}
        self.cur_adapter_path = {}
        self.lora_layers = {}
        self.lora_layers_critic = {}
        self.lora_layers_transformer_2 = {}
        self.is_lora_merged = {}
        self.lora_initialized = False
        self.lora_rank = None
        self.lora_alpha = None
        self.lora_path = None
        self.lora_nickname = "default"

        # Initialize from server_args
        self.device = get_local_torch_device()
        self.exclude_lora_layers = (
            self.server_args.pipeline_config.dit_config.arch_config.exclude_lora_layers
        )
        self.lora_target_modules = self.server_args.lora_target_modules
        self.lora_path = self.server_args.lora_path
        self.lora_nickname = self.server_args.lora_nickname
        if self.lora_path is not None:
            self.convert_to_lora_layers()
            self.set_lora(
                self.lora_nickname, self.lora_path  # type: ignore
            )  # type: ignore

    def is_target_layer(self, module_name: str) -> bool:
        if self.lora_target_modules is None:
            return True
        return any(
            target_name in module_name for target_name in self.lora_target_modules
        )

    def _get_target_lora_layers(
        self, target: str
    ) -> tuple[list[tuple[str, dict[str, BaseLayerWithLoRA]]], str | None]:
        """
        Return a list of (module_name, lora_layers_dict) based on the target.

        Args:
            target: One of "all", "transformer", "transformer_2", "critic".

        Returns:
            A tuple of (result, error_message):
            - result: List of tuples (module_name, lora_layers_dict) to operate on.
            - error_message: Error description if target is invalid or module doesn't exist, None otherwise.
        """
        if target == "all":
            result: list[tuple[str, dict[str, BaseLayerWithLoRA]]] = [
                ("transformer", self.lora_layers)
            ]
            if self.lora_layers_transformer_2:
                result.append(("transformer_2", self.lora_layers_transformer_2))
            if self.lora_layers_critic:
                result.append(("critic", self.lora_layers_critic))
            return result, None
        elif target == "transformer":
            return [("transformer", self.lora_layers)], None
        elif target == "transformer_2":
            if not self.lora_layers_transformer_2:
                return [], "transformer_2 does not exist in this pipeline"
            return [("transformer_2", self.lora_layers_transformer_2)], None
        elif target == "critic":
            if not self.lora_layers_critic:
                return (
                    [],
                    "critic (fake_score_transformer) does not exist in this pipeline",
                )
            return [("critic", self.lora_layers_critic)], None
        else:
            return [], f"Invalid target: {target}. Valid targets: {self.VALID_TARGETS}"

    def convert_module_lora_layers(
        self,
        module: torch.nn.Module,
        module_name: str,
        target_lora_layers: dict[str, BaseLayerWithLoRA],
        check_exclude: bool = True,
    ) -> int:
        """
        Convert layers in a module to LoRA layers.

        Args:
            module: The module to convert.
            module_name: The name of the module (for replace_submodule).
            target_lora_layers: The dictionary to store the converted LoRA layers.
            check_exclude: Whether to check the exclude_lora_layers list.

        Returns:
            The number of layers converted.
        """
        converted_count = 0
        for name, layer in module.named_modules():
            if not self.is_target_layer(name):
                continue

            if check_exclude:
                excluded = any(
                    exclude_layer in name for exclude_layer in self.exclude_lora_layers
                )
                if excluded:
                    continue

            lora_layer = wrap_with_lora_layer(
                layer,
                lora_rank=self.lora_rank,
                lora_alpha=self.lora_alpha,
            )
            if lora_layer is not None:
                target_lora_layers[name] = lora_layer
                replace_submodule(self.modules[module_name], name, lora_layer)
                converted_count += 1
        return converted_count

    def convert_to_lora_layers(self) -> None:
        """
        Unified method to convert the transformer to a LoRA transformer.
        """
        if self.lora_initialized:
            return
        self.lora_initialized = True

        # Convert transformer
        converted_count = self.convert_module_lora_layers(
            self.modules["transformer"],
            "transformer",
            self.lora_layers,
            check_exclude=True,
        )
        logger.info("Converted %d layers to LoRA layers", converted_count)

        # Convert transformer_2 if exists (e.g., Wan2.2 A14B dual-transformer)
        if (
            "transformer_2" in self.modules
            and self.modules["transformer_2"] is not None
        ):
            converted_count_2 = self.convert_module_lora_layers(
                self.modules["transformer_2"],
                "transformer_2",
                self.lora_layers_transformer_2,
                check_exclude=True,
            )
            logger.info(
                "Converted %d layers to LoRA layers in transformer_2", converted_count_2
            )

        # Convert fake_score_transformer if exists
        if "fake_score_transformer" in self.modules:
            converted_count_critic = self.convert_module_lora_layers(
                self.modules["fake_score_transformer"],
                "fake_score_transformer",
                self.lora_layers_critic,
                check_exclude=False,
            )
            logger.info(
                "Converted %d layers to LoRA layers in the critic model",
                converted_count_critic,
            )

    def _apply_lora_to_layers(
        self,
        lora_layers: dict[str, BaseLayerWithLoRA],
        lora_nickname: str,
        lora_path: str | None,
        rank: int,
    ) -> int:
        """
        Apply LoRA weights to the given lora_layers.

        Args:
            lora_layers: The dictionary of LoRA layers to apply weights to.
            lora_nickname: The nickname of the LoRA adapter.
            lora_path: The path to the LoRA adapter.
            rank: The distributed rank (for logging).

        Returns:
            The number of layers that had LoRA weights applied.
        """
        adapted_count = 0
        for name, layer in lora_layers.items():
            lora_A_name = name + ".lora_A"
            lora_B_name = name + ".lora_B"
            if (
                lora_A_name in self.lora_adapters[lora_nickname]
                and lora_B_name in self.lora_adapters[lora_nickname]
            ):
                layer.set_lora_weights(
                    self.lora_adapters[lora_nickname][lora_A_name],
                    self.lora_adapters[lora_nickname][lora_B_name],
                    lora_path=lora_path,
                )
                adapted_count += 1
            else:
                if rank == 0:
                    logger.warning(
                        "LoRA adapter %s does not contain the weights for layer '%s'. LoRA will not be applied to it.",
                        lora_path,
                        name,
                    )
                layer.disable_lora = True
        return adapted_count

    def is_lora_effective(self, target: str = "all") -> bool:
        """
        Check if LoRA is currently effective (merged) for the specified target.

        Args:
            target: Which transformer to check. "all" returns True if any is merged.
        """
        if target == "all":
            return any(self.is_lora_merged.values())
        return self.is_lora_merged.get(target, False)

    def is_lora_set(self, target: str = "all") -> bool:
        """
        Check if LoRA has been set for the specified target.

        Args:
            target: Which transformer to check. "all" returns True if any is set.
        """
        if not self.lora_initialized:
            return False
        if target == "all":
            return bool(self.cur_adapter_name)
        return target in self.cur_adapter_name

    def load_lora_adapter(self, lora_path: str, lora_nickname: str, rank: int):
        """
        Load the LoRA, and setup the lora_adapters for later weight replacement
        """
        assert lora_path is not None
        lora_local_path = maybe_download_lora(lora_path)
        lora_state_dict = load_file(lora_local_path)

        if lora_nickname in self.lora_adapters:
            self.lora_adapters[lora_nickname].clear()

        config = self.server_args.pipeline_config.dit_config.arch_config

        param_names_mapping_fn = get_param_names_mapping(
            config.param_names_mapping
            or self.modules["transformer"].param_names_mapping
        )
        lora_param_names_mapping_fn = get_param_names_mapping(
            config.lora_param_names_mapping
            or self.modules["transformer"].lora_param_names_mapping
        )

        to_merge_params: defaultdict[Hashable, dict[Any, Any]] = defaultdict(dict)
        for name, weight in lora_state_dict.items():
            name = name.replace("diffusion_model.", "")
            name = name.replace(".weight", "")
            # misc-format -> HF-format
            name, _, _ = lora_param_names_mapping_fn(name)
            # HF-format (LoRA) -> SGLang-dit-format
            target_name, merge_index, num_params_to_merge = param_names_mapping_fn(name)
            # for (in_dim, r) @ (r, out_dim), we only merge (r, out_dim * n) where n is the number of linear layers to fuse
            # see param mapping in HunyuanVideoArchConfig
            if merge_index is not None and "lora_B" in name:
                to_merge_params[target_name][merge_index] = weight
                if len(to_merge_params[target_name]) == num_params_to_merge:
                    # cat at output dim according to the merge_index order
                    sorted_tensors = [
                        to_merge_params[target_name][i]
                        for i in range(num_params_to_merge)
                    ]
                    weight = torch.cat(sorted_tensors, dim=1)
                    del to_merge_params[target_name]
                else:
                    continue

            if target_name in self.lora_adapters[lora_nickname]:
                raise ValueError(
                    f"Dit target weight name {target_name} already exists in lora_adapters[{lora_nickname}]"
                )
            self.lora_adapters[lora_nickname][target_name] = weight.to(self.device)
        self.loaded_adapter_paths[lora_nickname] = lora_path
        logger.info("Rank %d: loaded LoRA adapter %s", rank, lora_path)

    def set_lora(
        self, lora_nickname: str, lora_path: str | None = None, target: str = "all"
    ):  # type: ignore
        """
        Load a LoRA adapter into the pipeline and apply it to the specified transformer(s).

        Args:
            lora_nickname: The "nick name" of the adapter when referenced in the pipeline.
            lora_path: The path to the adapter, either a local path or a Hugging Face repo id.
            target: Which transformer(s) to apply the LoRA to. One of:
                - "all": Apply to all transformers (default, backward compatible)
                - "transformer": Apply only to the primary transformer (high noise for Wan2.2)
                - "transformer_2": Apply only to transformer_2 (low noise for Wan2.2)
                - "critic": Apply only to the critic model (fake_score_transformer)
        """
        if target not in self.VALID_TARGETS:
            raise ValueError(
                f"Invalid target: {target}. Valid targets: {self.VALID_TARGETS}"
            )

        # Check if any target module has a different LoRA merged
        target_modules, error = self._get_target_lora_layers(target)
        if error:
            logger.warning("set_lora: %s", error)
        for module_name, _ in target_modules:
            if (
                self.is_lora_merged.get(module_name, False)
                and self.cur_adapter_name.get(module_name) != lora_nickname
            ):
                raise ValueError(
                    f"LoRA '{self.cur_adapter_name.get(module_name)}' is currently merged in {module_name}. "
                    "Please call 'unmerge_lora_weights' before setting a new LoRA."
                )

        if lora_nickname not in self.lora_adapters and lora_path is None:
            raise ValueError(
                f"Adapter {lora_nickname} not found in the pipeline. Please provide lora_path to load it."
            )
        if not self.lora_initialized:
            self.convert_to_lora_layers()

        # Re-fetch target_modules after convert_to_lora_layers() to get populated dicts
        target_modules, error = self._get_target_lora_layers(target)
        if error:
            logger.warning("set_lora: %s", error)

        adapter_updated = False
        rank = dist.get_rank()

        should_load = False
        if lora_path is not None:
            if lora_nickname not in self.loaded_adapter_paths:
                should_load = True
            elif self.loaded_adapter_paths[lora_nickname] != lora_path:
                should_load = True

        if should_load:
            adapter_updated = True
            self.load_lora_adapter(lora_path, lora_nickname, rank)

        # Check if we can skip (same adapter already applied to all target modules)
        all_already_applied = all(
            not adapter_updated
            and self.cur_adapter_name.get(module_name) == lora_nickname
            and self.is_lora_merged.get(module_name, False)
            for module_name, _ in target_modules
        )
        if all_already_applied:
            return

        # Apply LoRA to target modules
        adapted_count = 0
        for module_name, lora_layers_dict in target_modules:
            count = self._apply_lora_to_layers(
                lora_layers_dict, lora_nickname, lora_path, rank
            )
            adapted_count += count
            self.cur_adapter_name[module_name] = lora_nickname
            self.cur_adapter_path[module_name] = (
                lora_path or self.loaded_adapter_paths.get(lora_nickname, "")
            )
            self.is_lora_merged[module_name] = True

        logger.info(
            "Rank %d: LoRA adapter %s applied to %d layers (target: %s)",
            rank,
            lora_path,
            adapted_count,
            target,
        )

    def merge_lora_weights(self, target: str = "all") -> None:
        """
        Merge LoRA weights into the base model for the specified target.

        This operation is idempotent - calling it when LoRA is already merged is safe.

        Args:
            target: Which transformer(s) to merge. One of "all", "transformer",
                    "transformer_2", "critic".
        """
        target_modules, error = self._get_target_lora_layers(target)
        if error:
            logger.warning("merge_lora_weights: %s", error)
        if not target_modules:
            return

        for module_name, lora_layers_dict in target_modules:
            if self.is_lora_merged.get(module_name, False):
                logger.warning("LoRA weights are already merged for %s", module_name)
                continue
            for name, layer in lora_layers_dict.items():
                # Only re-enable LoRA for layers that actually have LoRA weights
                has_lora_weights = hasattr(layer, "lora_A") and layer.lora_A is not None
                if not has_lora_weights:
                    continue
                if hasattr(layer, "disable_lora"):
                    layer.disable_lora = False
                try:
                    layer.merge_lora_weights()
                except Exception as e:
                    logger.warning("Could not merge layer %s: %s", name, e)
                    continue
            self.is_lora_merged[module_name] = True
            logger.info("LoRA weights merged for %s", module_name)

    def unmerge_lora_weights(self, target: str = "all") -> None:
        """
        Unmerge LoRA weights from the base model for the specified target.
        This also disables LoRA so it won't be computed on-the-fly.

        This operation is idempotent - calling it when LoRA is not merged is safe.

        Args:
            target: Which transformer(s) to unmerge. One of "all", "transformer",
                    "transformer_2", "critic".
        """
        target_modules, error = self._get_target_lora_layers(target)
        if error:
            logger.warning("unmerge_lora_weights: %s", error)
        if not target_modules:
            return

        for module_name, lora_layers_dict in target_modules:
            if not self.is_lora_merged.get(module_name, False):
                logger.warning(
                    "LoRA weights are not merged for %s, skipping", module_name
                )
                continue
            for name, layer in lora_layers_dict.items():
                # Check layer-level state to avoid raising exception
                if hasattr(layer, "merged") and not layer.merged:
                    logger.warning("Layer %s is not merged, skipping", name)
                    # Still disable LoRA to prevent on-the-fly computation
                    if hasattr(layer, "disable_lora"):
                        layer.disable_lora = True
                    continue
                try:
                    layer.unmerge_lora_weights()
                    # Disable LoRA after unmerge to prevent on-the-fly computation
                    if hasattr(layer, "disable_lora"):
                        layer.disable_lora = True
                except ValueError as e:
                    logger.warning("Could not unmerge layer %s: %s", name, e)
                    # Still disable LoRA even if unmerge failed
                    if hasattr(layer, "disable_lora"):
                        layer.disable_lora = True
                    continue
            self.is_lora_merged[module_name] = False
            logger.info("LoRA weights unmerged for %s", module_name)
