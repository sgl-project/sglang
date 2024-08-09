"""
Copyright 2023-2024 SGLang Team
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

# temporarily adapted from https://github.com/vllm-project/vllm/blob/10383887e03412196a2689b9398290719c4797bf/vllm/model_executor/model_loader/loader.py
# FIXME: in progress of refactoring the model loader

import glob
import os
import re
from typing import Any, Dict, Generator, List, Optional, Tuple, Type

import torch
from torch import nn
from tqdm import tqdm
from vllm.config import (
    CacheConfig,
    DeviceConfig,
    LoadConfig,
    LoadFormat,
    LoRAConfig,
    ModelConfig,
    MultiModalConfig,
    ParallelConfig,
    SchedulerConfig,
)
from vllm.model_executor.layers.quantization.base_config import QuantizationConfig
from vllm.model_executor.model_loader.utils import (
    get_model_architecture,
    set_default_torch_dtype,
)
from vllm.platforms import current_platform

from sglang.srt.model_loader.utils import (
    download_safetensors_index_file_from_hf,
    download_weights_from_hf,
    filter_duplicate_safetensors_files,
    get_quant_config,
    safetensors_weights_iterator,
)


def _get_quantization_config(
    model_config: ModelConfig, load_config: LoadConfig
) -> Optional[QuantizationConfig]:
    """Get the quantization config."""
    if model_config.quantization is not None:
        quant_config = get_quant_config(model_config, load_config)
        capability = current_platform.get_device_capability()
        capability = capability[0] * 10 + capability[1]
        if capability < quant_config.get_min_capability():
            raise ValueError(
                f"The quantization method {model_config.quantization} is not "
                "supported for the current GPU. "
                f"Minimum capability: {quant_config.get_min_capability()}. "
                f"Current capability: {capability}."
            )
        supported_dtypes = quant_config.get_supported_act_dtypes()
        if model_config.dtype not in supported_dtypes:
            raise ValueError(
                f"{model_config.dtype} is not supported for quantization "
                f"method {model_config.quantization}. Supported dtypes: "
                f"{supported_dtypes}"
            )
        return quant_config
    return None


def _get_model_initialization_kwargs(
    model_class: Type[nn.Module],
    lora_config: Optional[LoRAConfig],
    multimodal_config: Optional[MultiModalConfig],
) -> Dict[str, Any]:
    """Get extra kwargs for model initialization."""
    extra_kwargs: Dict[str, Any] = {}

    assert lora_config is None
    assert multimodal_config is None

    return extra_kwargs


def _initialize_model(
    model_config: ModelConfig,
    load_config: LoadConfig,
    lora_config: Optional[LoRAConfig],
    multimodal_config: Optional[MultiModalConfig],
    cache_config: CacheConfig,
) -> nn.Module:
    """Initialize a model with the given configurations."""
    model_class = get_model_architecture(model_config)[0]
    quant_config = _get_quantization_config(model_config, load_config)

    return model_class(
        config=model_config.hf_config,
        cache_config=cache_config,
        quant_config=quant_config,
        efficient_weight_load=True,
        **_get_model_initialization_kwargs(model_class, lora_config, multimodal_config),
    )


class ModelLoader:
    """Model loader that can load different file types from disk."""

    def __init__(self, load_config: LoadConfig):
        self.load_config = load_config

    def _prepare_weights(
        self, model_name_or_path: str, revision: Optional[str], fall_back_to_pt: bool
    ) -> Tuple[str, List[str], bool]:
        """Prepare weights for the model.

        If the model is not local, it will be downloaded."""

        is_local = os.path.isdir(model_name_or_path)
        load_format = self.load_config.load_format
        use_safetensors = False
        # Some quantized models use .pt files for storing the weights.
        if load_format == LoadFormat.AUTO:
            allow_patterns = ["*.safetensors", "*.bin"]
        elif load_format == LoadFormat.SAFETENSORS:
            use_safetensors = True
            allow_patterns = ["*.safetensors"]
        elif load_format == LoadFormat.PT:
            allow_patterns = ["*.pt"]
        elif load_format == LoadFormat.NPCACHE:
            allow_patterns = ["*.bin"]
        else:
            raise ValueError(f"Unknown load_format: {load_format}")

        if fall_back_to_pt:
            allow_patterns += ["*.pt"]

        if not is_local:
            hf_folder = download_weights_from_hf(
                model_name_or_path,
                self.load_config.download_dir,
                allow_patterns,
                revision,
            )
        else:
            hf_folder = model_name_or_path

        hf_weights_files: List[str] = []
        for pattern in allow_patterns:
            hf_weights_files += glob.glob(os.path.join(hf_folder, pattern))
            if len(hf_weights_files) > 0:
                if pattern == "*.safetensors":
                    use_safetensors = True
                break

        if use_safetensors:
            # For models like Mistral-7B-Instruct-v0.3
            # there are both sharded safetensors files and a consolidated
            # safetensors file. Using both breaks.
            # Here, we download the `model.safetensors.index.json` and filter
            # any files not found in the index.
            if not is_local:
                download_safetensors_index_file_from_hf(
                    model_name_or_path, self.load_config.download_dir, revision
                )
            hf_weights_files = filter_duplicate_safetensors_files(
                hf_weights_files, hf_folder
            )
        else:
            hf_weights_files = filter_files_not_needed_for_inference(hf_weights_files)

        if len(hf_weights_files) == 0:
            raise RuntimeError(
                f"Cannot find any model weights with `{model_name_or_path}`"
            )

        return hf_folder, hf_weights_files, use_safetensors

    def _get_weights_iterator(
        self, model_name_or_path: str, revision: Optional[str], fall_back_to_pt: bool
    ) -> Generator[Tuple[str, torch.Tensor], None, None]:
        """Get an iterator for the model weights based on the load format."""
        hf_folder, hf_weights_files, use_safetensors = self._prepare_weights(
            model_name_or_path, revision, fall_back_to_pt
        )
        if self.load_config.load_format == LoadFormat.NPCACHE:
            # Currently np_cache only support *.bin checkpoints
            assert use_safetensors is False
            weights_iterator = np_cache_weights_iterator(
                model_name_or_path,
                self.load_config.download_dir,
                hf_folder,
                hf_weights_files,
            )
        elif use_safetensors:
            weights_iterator = safetensors_weights_iterator(hf_weights_files)
        else:
            weights_iterator = pt_weights_iterator(hf_weights_files)

        return weights_iterator

    def load_model(
        self,
        *,
        model_config: ModelConfig,
        device_config: DeviceConfig,
        lora_config: Optional[LoRAConfig],
        multimodal_config: Optional[MultiModalConfig],
        parallel_config: ParallelConfig,
        scheduler_config: SchedulerConfig,
        cache_config: CacheConfig,
    ) -> nn.Module:
        with set_default_torch_dtype(model_config.dtype):
            with torch.device(device_config.device):
                model = _initialize_model(
                    model_config,
                    self.load_config,
                    lora_config,
                    multimodal_config,
                    cache_config,
                )
            weights = self._get_weights_iterator(
                model_config.model,
                model_config.revision,
                fall_back_to_pt=getattr(model, "fall_back_to_pt_during_load", True),
            )

            modules = {}
            for name, module in model.named_modules():
                modules[name] = module

            def apply_quant_method(module):
                quant_method = getattr(module, "quant_method", None)
                if quant_method is not None:
                    # print("before apply quant", module.weight, module.weight.dtype)
                    quant_method.process_weights_after_loading(module)
                    # print("after apply quant", module.weight, module.weight.dtype)
                # FIXME: Remove this after Mixtral is updated
                # to use quant_method.
                if hasattr(module, "process_weights_after_loading"):
                    module.process_weights_after_loading()

            if torch.cuda.current_device() == 0:
                weights = tqdm(
                    weights, total=model.get_num_params() * 1.5, desc="load model"
                )

            num_shard = {}
            num_loaded = {}
            for name, loaded_weight in weights:
                model.load_weights(None, name, loaded_weight)
                module_name, shard_num = model.get_module_name(name)
                num_shard[module_name] = shard_num
                if module_name not in num_loaded:
                    num_loaded[module_name] = 1
                else:
                    num_loaded[module_name] += 1
                if num_loaded[module_name] == num_shard[module_name]:
                    apply_quant_method(modules[module_name])

        return model.eval()


def get_model(
    *,
    model_config: ModelConfig,
    load_config: LoadConfig,
    device_config: DeviceConfig,
    parallel_config: ParallelConfig,
    scheduler_config: SchedulerConfig,
    lora_config: Optional[LoRAConfig],
    multimodal_config: Optional[MultiModalConfig],
    cache_config: CacheConfig,
) -> nn.Module:
    loader = ModelLoader(load_config)
    return loader.load_model(
        model_config=model_config,
        device_config=device_config,
        lora_config=lora_config,
        multimodal_config=multimodal_config,
        parallel_config=parallel_config,
        scheduler_config=scheduler_config,
        cache_config=cache_config,
    )
