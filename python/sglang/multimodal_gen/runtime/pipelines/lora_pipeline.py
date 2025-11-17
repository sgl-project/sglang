# Copied and adapted from: https://github.com/hao-ai-lab/FastVideo

# SPDX-License-Identifier: Apache-2.0
from collections import defaultdict
from collections.abc import Hashable
from typing import Any

import torch
import torch.distributed as dist
from safetensors.torch import load_file

from sglang.multimodal_gen.runtime.distributed import get_local_torch_device
from sglang.multimodal_gen.runtime.layers.lora.linear import (
    BaseLayerWithLoRA,
    get_lora_layer,
    replace_submodule,
)
from sglang.multimodal_gen.runtime.loader.utils import get_param_names_mapping
from sglang.multimodal_gen.runtime.pipelines.composed_pipeline_base import (
    ComposedPipelineBase,
)
from sglang.multimodal_gen.runtime.server_args import ServerArgs
from sglang.multimodal_gen.runtime.utils.hf_diffusers_utils import maybe_download_lora
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger

logger = init_logger(__name__)


class LoRAPipeline(ComposedPipelineBase):
    """
    Pipeline that supports injecting LoRA adapters into the diffusion transformer.
    TODO: support training.
    """

    lora_adapters: dict[str, dict[str, torch.Tensor]] = defaultdict(
        dict
    )  # state dicts of loaded lora adapters
    cur_adapter_name: str = ""
    cur_adapter_path: str = ""
    lora_layers: dict[str, BaseLayerWithLoRA] = {}
    lora_layers_critic: dict[str, BaseLayerWithLoRA] = {}
    server_args: ServerArgs
    exclude_lora_layers: list[str] = []
    device: torch.device = get_local_torch_device()
    lora_target_modules: list[str] | None = None
    lora_path: str | None = None
    lora_nickname: str = "default"
    lora_rank: int | None = None
    lora_alpha: int | None = None
    lora_initialized: bool = False

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.device = get_local_torch_device()
        self.exclude_lora_layers = self.modules[
            "transformer"
        ].config.arch_config.exclude_lora_layers
        self.lora_target_modules = self.server_args.lora_target_modules
        self.lora_path = self.server_args.lora_path
        self.lora_nickname = self.server_args.lora_nickname
        if self.lora_path is not None:
            self.convert_to_lora_layers()
            self.set_lora_adapter(
                self.lora_nickname, self.lora_path  # type: ignore
            )  # type: ignore

    def is_target_layer(self, module_name: str) -> bool:
        if self.lora_target_modules is None:
            return True
        return any(
            target_name in module_name for target_name in self.lora_target_modules
        )

    def convert_to_lora_layers(self) -> None:
        """
        Unified method to convert the transformer to a LoRA transformer.
        """
        if self.lora_initialized:
            return
        self.lora_initialized = True
        converted_count = 0
        for name, layer in self.modules["transformer"].named_modules():
            if not self.is_target_layer(name):
                continue

            excluded = False
            for exclude_layer in self.exclude_lora_layers:
                if exclude_layer in name:
                    excluded = True
                    break
            if excluded:
                continue

            layer = get_lora_layer(
                layer,
                lora_rank=self.lora_rank,
                lora_alpha=self.lora_alpha,
            )
            if layer is not None:
                self.lora_layers[name] = layer
                replace_submodule(self.modules["transformer"], name, layer)
                converted_count += 1
        logger.info("Converted %d layers to LoRA layers", converted_count)

        if "fake_score_transformer" in self.modules:
            for name, layer in self.modules["fake_score_transformer"].named_modules():
                if not self.is_target_layer(name):
                    continue
                layer = get_lora_layer(
                    layer,
                    lora_rank=self.lora_rank,
                    lora_alpha=self.lora_alpha,
                )
                if layer is not None:
                    self.lora_layers_critic[name] = layer
                    replace_submodule(
                        self.modules["fake_score_transformer"], name, layer
                    )
                    converted_count += 1
            logger.info(
                "Converted %d layers to LoRA layers in the critic model",
                converted_count,
            )

    def set_lora_adapter(
        self, lora_nickname: str, lora_path: str | None = None
    ):  # type: ignore
        """
        Load a LoRA adapter into the pipeline and merge it into the transformer.
        Args:
            lora_nickname: The "nick name" of the adapter when referenced in the pipeline.
            lora_path: The path to the adapter, either a local path or a Hugging Face repo id.
        """

        if lora_nickname not in self.lora_adapters and lora_path is None:
            raise ValueError(
                f"Adapter {lora_nickname} not found in the pipeline. Please provide lora_path to load it."
            )
        if not self.lora_initialized:
            self.convert_to_lora_layers()
        adapter_updated = False
        rank = dist.get_rank()
        if lora_path is not None and lora_path != self.cur_adapter_path:
            lora_local_path = maybe_download_lora(lora_path)
            lora_state_dict = load_file(lora_local_path)

            # Map the hf layer names to our custom layer names
            param_names_mapping_fn = get_param_names_mapping(
                self.modules["transformer"].param_names_mapping
            )
            lora_param_names_mapping_fn = get_param_names_mapping(
                self.modules["transformer"].lora_param_names_mapping
            )

            to_merge_params: defaultdict[Hashable, dict[Any, Any]] = defaultdict(dict)
            for name, weight in lora_state_dict.items():
                name = name.replace("diffusion_model.", "")
                name = name.replace(".weight", "")
                name, _, _ = lora_param_names_mapping_fn(name)
                target_name, merge_index, num_params_to_merge = param_names_mapping_fn(
                    name
                )
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
                        f"Target name {target_name} already exists in lora_adapters[{lora_nickname}]"
                    )
                self.lora_adapters[lora_nickname][target_name] = weight.to(self.device)
            adapter_updated = True
            self.cur_adapter_path = lora_path
            logger.info("Rank %d: loaded LoRA adapter %s", rank, lora_path)

        if not adapter_updated and self.cur_adapter_name == lora_nickname:
            return
        self.cur_adapter_name = lora_nickname

        # Merge the new adapter
        adapted_count = 0
        for name, layer in self.lora_layers.items():
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
                        "LoRA adapter %s does not contain the weights for layer %s. LoRA will not be applied to it.",
                        lora_path,
                        name,
                    )
                layer.disable_lora = True
        logger.info(
            "Rank %d: LoRA adapter %s applied to %d layers",
            rank,
            lora_path,
            adapted_count,
        )

    def merge_lora_weights(self) -> None:
        for name, layer in self.lora_layers.items():
            layer.merge_lora_weights()

    def unmerge_lora_weights(self) -> None:
        for name, layer in self.lora_layers.items():
            layer.unmerge_lora_weights()
