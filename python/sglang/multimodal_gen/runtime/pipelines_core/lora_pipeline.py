# Copied and adapted from: https://github.com/hao-ai-lab/FastVideo

# SPDX-License-Identifier: Apache-2.0
import os
from collections import defaultdict
from collections.abc import Hashable
from contextlib import contextmanager
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
from sglang.multimodal_gen.runtime.pipelines_core.lora_format_adapter import (
    normalize_lora_state_dict,
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
    cur_adapter_strength: dict[str, float]  # Track current strength per module
    cur_adapter_config: dict[str, tuple[list[str], list[float]]]
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
        self.cur_adapter_strength = {}
        # Track full LoRA config: {module_name: (nickname_list, strength_list)}
        self.cur_adapter_config = {}
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

    @contextmanager
    def _temporarily_disable_offload(
        self,
        target_modules: list[tuple[str, dict[str, BaseLayerWithLoRA]]] | None = None,
        target: str | None = None,
        use_module_names_only: bool = False,
    ):
        """
        Context manager to temporarily disable layerwise offload for the given modules.

        Args:
            target_modules: List of (module_name, lora_layers_dict) tuples. If None, will be determined from target.
            target: Target string ("all", "transformer", etc.). Used if target_modules is None.
            use_module_names_only: If True, determine module names directly from target without requiring
                                   LoRA initialization. Used for convert_to_lora_layers scenario.

        Yields:
            List of modules that had offload disabled.
        """
        from sglang.multimodal_gen.runtime.utils.layerwise_offload import (
            OffloadableDiTMixin,
        )

        module_names = []
        if target_modules is not None:
            # Extract module names from target_modules
            module_names = [module_name for module_name, _ in target_modules]
        elif target is not None:
            if use_module_names_only:
                if target == "all":
                    module_names = ["transformer", "transformer_2"]
                elif target in ["transformer", "transformer_2", "critic"]:
                    module_names = [target]
            else:
                target_modules, _ = self._get_target_lora_layers(target)
                if target_modules:
                    module_names = [module_name for module_name, _ in target_modules]
        else:
            yield []
            return

        if not module_names:
            yield []
            return

        # clear CUDA cache to free up unused memory
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()

        offload_disabled_modules = []
        for module_name in module_names:
            module = self.modules.get(module_name)
            if module is not None and isinstance(module, OffloadableDiTMixin):
                if module.layerwise_offload_managers is not None:
                    module.disable_offload()
                    offload_disabled_modules.append(module)

        try:
            yield offload_disabled_modules
        finally:
            # Re-enable layerwise offload: sync weights to CPU and restore hooks
            for module in offload_disabled_modules:
                module.enable_offload()

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

    def _normalize_lora_params(
        self,
        lora_nickname: str | list[str],
        lora_path: str | None | list[str | None],
        strength: float | list[float],
        target: str | list[str],
    ) -> tuple[list[str], list[str | None], list[float], list[str]]:
        """
        Normalize LoRA parameters to lists for multi-LoRA support.

        Requirements:
        - each nickname must have a corresponding lora_path (no implicit repeat)
        - strength / target if scalar broadcast, else length must match nickname
        """
        # nickname
        if isinstance(lora_nickname, str):
            lora_nicknames = [lora_nickname]
        else:
            lora_nicknames = lora_nickname

        # lora_path: require 1:1 mapping with nickname (no implicit repeat)
        if isinstance(lora_path, list):
            lora_paths = lora_path
        else:
            lora_paths = [lora_path]
        if len(lora_paths) != len(lora_nicknames):
            raise ValueError(
                f"Length mismatch: lora_nickname has {len(lora_nicknames)} items, "
                f"but lora_path has {len(lora_paths)} items. "
                "Provide one path per nickname."
            )

        # strength and target: allow scalar broadcast, else length must match
        if isinstance(strength, (int, float)):
            strengths = [float(strength)] * len(lora_nicknames)
        else:
            strengths = [float(s) for s in strength]
        if len(strengths) != len(lora_nicknames):
            raise ValueError(
                f"Length mismatch: lora_nickname has {len(lora_nicknames)} items, "
                f"but strength has {len(strengths)} items"
            )

        if isinstance(target, str):
            targets = [target] * len(lora_nicknames)
        else:
            targets = target
        if len(targets) != len(lora_nicknames):
            raise ValueError(
                f"Length mismatch: lora_nickname has {len(lora_nicknames)} items, "
                f"but target has {len(targets)} items"
            )
        return lora_nicknames, lora_paths, strengths, targets

    def _check_lora_config_matches(
        self,
        module_name: str,
        target_nicknames: list[str],
        target_strengths: list[float],
        adapter_updated: bool,
    ) -> bool:
        """
        Check if current LoRA configuration matches the target configuration.

        Args:
            module_name: The name of the module to check.
            target_nicknames: List of LoRA nicknames to apply.
            target_strengths: List of LoRA strengths to apply.
            adapter_updated: Whether any adapter was updated/loaded.

        Returns:
            True if the configuration matches exactly (including order and strength), False otherwise.
        """
        if not self.is_lora_merged.get(module_name, False):
            return False
        if adapter_updated:
            return False  # Adapter was updated, need to reapply

        stored_config = self.cur_adapter_config.get(module_name)
        if stored_config is None:
            return False

        stored_nicknames, stored_strengths = stored_config
        # Compare: nickname list and strength list must match exactly (including order)
        return (
            stored_nicknames == target_nicknames
            and stored_strengths == target_strengths
        )

    def _apply_lora_to_layers(
        self,
        lora_layers: dict[str, BaseLayerWithLoRA],
        lora_nicknames: list[str],
        lora_paths: list[str | None],
        rank: int,
        strengths: list[float],
        clear_existing: bool = False,
    ) -> int:
        """
        Apply LoRA weights to the given lora_layers. Supports multiple LoRA adapters.

        Args:
            lora_layers: The dictionary of LoRA layers to apply weights to.
            lora_nicknames: The list of nicknames of the LoRA adapters.
            lora_paths: The list of paths to the LoRA adapters. Must match length of lora_nicknames.
            rank: The distributed rank (for logging).
            strengths: The list of LoRA strengths for merge. Must match length of lora_nicknames.
            clear_existing: If True, clear existing LoRA weights before adding new ones.

        Returns:
            The number of layers that had LoRA weights applied.
        """
        if len(lora_paths) != len(lora_nicknames):
            raise ValueError(
                f"Length mismatch: lora_nicknames has {len(lora_nicknames)} items, "
                f"but lora_paths has {len(lora_paths)} items"
            )
        if len(strengths) != len(lora_nicknames):
            raise ValueError(
                f"Length mismatch: lora_nicknames has {len(lora_nicknames)} items, "
                f"but strengths has {len(strengths)} items"
            )

        adapted_count = 0
        for name, layer in lora_layers.items():
            # Apply all LoRA adapters in order
            for idx, (nickname, path, lora_strength) in enumerate(
                zip(lora_nicknames, lora_paths, strengths)
            ):
                lora_A_name = name + ".lora_A"
                lora_B_name = name + ".lora_B"
                if (
                    lora_A_name in self.lora_adapters[nickname]
                    and lora_B_name in self.lora_adapters[nickname]
                ):
                    # Some LoRA checkpoints (e.g. Lightning distill) store per-layer alpha as "<layer>.alpha".
                    # If present, we must apply the standard LoRA scaling: scale = alpha / rank.
                    try:
                        inferred_rank = int(
                            self.lora_adapters[nickname][lora_A_name].shape[0]
                        )
                    except Exception:
                        inferred_rank = None
                    # Default to None for some checkpoints without "<layer>.alpha"
                    inferred_alpha: int | None = None
                    alpha_key = name + ".alpha"
                    if alpha_key in self.lora_adapters[nickname]:
                        try:
                            inferred_alpha = int(
                                self.lora_adapters[nickname][alpha_key].item()
                            )
                        except Exception:
                            inferred_alpha = None

                    if inferred_rank is not None:
                        layer.lora_rank = inferred_rank
                        layer.lora_alpha = (
                            inferred_alpha
                            if inferred_alpha is not None
                            else inferred_rank
                        )

                    layer.set_lora_weights(
                        self.lora_adapters[nickname][lora_A_name],
                        self.lora_adapters[nickname][lora_B_name],
                        lora_path=path,
                        strength=lora_strength,
                        clear_existing=(
                            clear_existing and idx == 0
                        ),  # Only clear on first LoRA
                    )
                    adapted_count += 1
                else:
                    if rank == 0 and idx == 0:  # Only warn for first missing LoRA
                        logger.warning(
                            "LoRA adapter %s does not contain the weights for layer '%s'. LoRA will not be applied to it.",
                            path,
                            name,
                        )
                    # Only disable if no LoRA was applied at all
                    if idx == len(lora_nicknames) - 1:
                        has_any_lora = any(
                            name + ".lora_A" in self.lora_adapters[n]
                            and name + ".lora_B" in self.lora_adapters[n]
                            for n in lora_nicknames
                        )
                        if not has_any_lora:
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

        # Only rank 0 downloads to avoid race conditions where other ranks
        # try to load incomplete downloads
        if rank == 0:
            lora_local_path = maybe_download_lora(lora_path)
        else:
            lora_local_path = None

        # Synchronize all ranks after download completes
        if dist.is_initialized():
            dist.barrier()

        # Non-rank-0 workers now download (will hit cache since rank 0 completed)
        if rank != 0:
            lora_local_path = maybe_download_lora(lora_path)

        raw_state_dict = load_file(lora_local_path)
        lora_state_dict = normalize_lora_state_dict(raw_state_dict, logger=logger)

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
            # for fuse B(out_dim, r) @ A(r, in_dim) -> (N, out_dim, r) @ (N, r, in_dim)
            # see param mapping in HunyuanVideoArchConfig
            if merge_index is not None:
                to_merge_params[target_name][merge_index] = weight
                if len(to_merge_params[target_name]) == num_params_to_merge:
                    sorted_tensors = [
                        to_merge_params[target_name][i]
                        for i in range(num_params_to_merge)
                    ]
                    # Use stack instead of cat because it needs to be compatible with TP.
                    weight = torch.stack(sorted_tensors, dim=0)
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
        self,
        lora_nickname: str | list[str],
        lora_path: str | None | list[str | None] = None,
        target: str | list[str] = "all",
        strength: float | list[float] = 1.0,
    ):  # type: ignore
        """
        Load LoRA adapter(s) into the pipeline and apply them to the specified transformer(s).
        Supports both single LoRA (backward compatible) and multiple LoRA adapters.
        """
        # Normalize inputs to lists for multi-LoRA support
        lora_nicknames, lora_paths, strengths, targets = self._normalize_lora_params(
            lora_nickname, lora_path, strength, target
        )

        # Validate targets
        invalid_targets = [t for t in targets if t not in self.VALID_TARGETS]
        if invalid_targets:
            raise ValueError(
                f"Invalid target(s): {invalid_targets}. Valid targets: {self.VALID_TARGETS}"
            )

        # Disable layerwise offload before convert_to_lora_layers to ensure weights are accessible
        # This is critical because convert_to_lora_layers needs to save cpu_weight from actual weights,
        # not from offloaded placeholder tensors
        if not self.lora_initialized:
            with self._temporarily_disable_offload(
                target="all", use_module_names_only=True
            ):
                self.convert_to_lora_layers()

        # Check adapter presence and load missing adapters
        adapter_updated = False
        rank = dist.get_rank()

        # load required adapters
        for nickname, path in zip(lora_nicknames, lora_paths):
            if nickname not in self.lora_adapters and path is None:
                raise ValueError(
                    f"Adapter {nickname} not found in the pipeline. Please provide lora_path to load it."
                )
            # Check if adapter needs to be loaded
            should_load = False
            if path is not None:
                if nickname not in self.loaded_adapter_paths:
                    should_load = True
                elif self.loaded_adapter_paths[nickname] != path:
                    should_load = True
            if should_load:
                adapter_updated = True
                self.load_lora_adapter(path, nickname, rank)

        # Group by target to apply separately
        target_to_indices = {}
        for idx, tgt in enumerate(targets):
            if tgt not in target_to_indices:
                target_to_indices[tgt] = []
            target_to_indices[tgt].append(idx)

        adapted_count = 0
        for tgt, idx_list in target_to_indices.items():
            target_modules, error = self._get_target_lora_layers(tgt)
            if error:
                logger.warning("set_lora: %s", error)
            if not target_modules:
                continue

            # Disable layerwise offload if enabled: load all layers to GPU
            # the LoRA weights merging process requires weights being on device
            with self._temporarily_disable_offload(target_modules=target_modules):
                tgt_nicknames = [lora_nicknames[i] for i in idx_list]
                tgt_paths = [lora_paths[i] for i in idx_list]
                tgt_strengths = [strengths[i] for i in idx_list]

                merged_name = (
                    ",".join(tgt_nicknames)
                    if len(tgt_nicknames) > 1
                    else tgt_nicknames[0]
                )

                # Skip if LoRA configuration matches exactly (including order and strength)
                # Since all modules for the same target apply the same config, checking one is sufficient
                first_module_name, _ = target_modules[0]
                if self._check_lora_config_matches(
                    first_module_name, tgt_nicknames, tgt_strengths, adapter_updated
                ):
                    logger.info("LoRA configuration matches exactly, skipping")
                    continue

                # Apply LoRA to modules for this target
                for module_name, lora_layers_dict in target_modules:
                    count = self._apply_lora_to_layers(
                        lora_layers_dict,
                        tgt_nicknames,
                        tgt_paths,
                        rank,
                        tgt_strengths,
                        clear_existing=True,
                    )
                    adapted_count += count
                    self.cur_adapter_name[module_name] = merged_name
                    self.cur_adapter_path[module_name] = ",".join(
                        str(p or self.loaded_adapter_paths.get(n, ""))
                        for n, p in zip(tgt_nicknames, tgt_paths)
                    )
                    self.is_lora_merged[module_name] = True
                    self.cur_adapter_strength[module_name] = tgt_strengths[0]
                    # Store full configuration for multi-LoRA support (preserves order and all strengths)
                    self.cur_adapter_config[module_name] = (
                        tgt_nicknames.copy(),
                        tgt_strengths.copy(),
                    )

        logger.info(
            "Rank %d: LoRA adapter(s) %s applied to %d layers (targets: %s, strengths: %s)",
            rank,
            ", ".join(map(str, lora_paths)) if lora_paths else None,
            adapted_count,
            ", ".join(targets) if len(set(targets)) > 1 else targets[0],
            (
                ", ".join(f"{s:.2f}" for s in strengths)
                if len(strengths) > 1
                else f"{strengths[0]:.2f}"
            ),
        )

    def merge_lora_weights(self, target: str = "all", strength: float = 1.0) -> None:
        """
        Merge LoRA weights into the base model for the specified target.

        This operation is idempotent - calling it when LoRA is already merged is safe.

        Args:
            target: Which transformer(s) to merge. One of "all", "transformer",
                    "transformer_2", "critic".
            strength: LoRA strength for merge, default 1.0.
        """
        target_modules, error = self._get_target_lora_layers(target)
        if error:
            logger.warning("merge_lora_weights: %s", error)
        if not target_modules:
            return

        # Disable layerwise offload if enabled: load all layers to GPU
        with self._temporarily_disable_offload(target_modules=target_modules):
            for module_name, lora_layers_dict in target_modules:
                if self.is_lora_merged.get(module_name, False):
                    # Check if strength is the same - if so, skip (idempotent)
                    if self.cur_adapter_strength.get(module_name) == strength:
                        logger.warning(
                            "LoRA weights are already merged for %s with same strength",
                            module_name,
                        )
                        continue
                    # Different strength requested - allow re-merge (layer handles unmerge internally)
                    logger.info(
                        "Re-merging LoRA weights for %s with new strength %s",
                        module_name,
                        strength,
                    )
                for name, layer in lora_layers_dict.items():
                    # Only re-enable LoRA for layers that actually have LoRA weights
                    has_lora_weights = (
                        hasattr(layer, "lora_A") and layer.lora_A is not None
                    )
                    if not has_lora_weights:
                        continue
                    if hasattr(layer, "disable_lora"):
                        layer.disable_lora = False
                    try:
                        layer.merge_lora_weights(strength=strength)
                    except Exception as e:
                        logger.warning("Could not merge layer %s: %s", name, e)
                        continue
                self.is_lora_merged[module_name] = True
                self.cur_adapter_strength[module_name] = strength
                logger.info(
                    "LoRA weights merged for %s (strength: %s)", module_name, strength
                )

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

        # Disable layerwise offload if enabled: load all layers to GPU

        for module_name, lora_layers_dict in target_modules:
            if not self.is_lora_merged.get(module_name, False):
                logger.warning(
                    "LoRA weights are not merged for %s, skipping", module_name
                )
                continue
            with self._temporarily_disable_offload(target_modules=target_modules):
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
                self.cur_adapter_strength.pop(module_name, None)
                self.cur_adapter_config.pop(module_name, None)
            logger.info("LoRA weights unmerged for %s", module_name)

    def get_lora_status(self) -> dict[str, Any]:
        """
        Summarize loaded LoRA adapters and current application status per module.

        Returns a plain Python dict with no tensor values to allow safe JSON serialization.
        """
        # Loaded adapters: list of {nickname, path}
        loaded_adapters = [
            {"nickname": nickname, "path": path}
            for nickname, path in self.loaded_adapter_paths.items()
        ]

        def _module_status(module_name: str) -> list[dict] | None:
            # return list of dict to support multi-lora in the future
            if not self.is_lora_merged.get(module_name, False):
                return None
            else:
                return [
                    {
                        "nickname": self.cur_adapter_name.get(module_name, None),
                        "path": self.cur_adapter_path.get(module_name, None),
                        "merged": self.is_lora_merged.get(module_name, False),
                        "strength": self.cur_adapter_strength.get(module_name, None),
                    }
                ]

        # Build active usage per module only for modules that exist in this pipeline
        active: dict[str, Any] = {}
        if (
            "transformer" in self.modules
            and self.modules["transformer"] is not None
            and (status := _module_status("transformer")) is not None
        ):
            active["transformer"] = status
        if (
            "transformer_2" in self.modules
            and self.modules["transformer_2"] is not None
            and (status := _module_status("transformer_2")) is not None
        ):
            active["transformer_2"] = status
        if (
            "fake_score_transformer" in self.modules
            and self.modules["fake_score_transformer"] is not None
            and (status := _module_status("critic")) is not None
        ):
            active["critic"] = status

        return {
            "loaded_adapters": loaded_adapters,
            "active": active,
        }
