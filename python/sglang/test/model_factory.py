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

from contextlib import contextmanager
from typing import Callable, Optional

import torch
from transformers import PretrainedConfig

from sglang.srt.configs.load_config import LoadConfig, LoadFormat
from sglang.srt.configs.model_config import ModelConfig
from sglang.srt.distributed import (
    destroy_distributed_environment,
    destroy_model_parallel,
    init_distributed_environment,
    initialize_model_parallel,
)
from sglang.srt.layers.attention.dummy_backend import DummyAttentionBackend
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.model_loader.loader import _get_quantization_config
from sglang.srt.model_loader.utils import (
    get_model_architecture,
    post_load_weights,
    set_default_torch_dtype,
)
from sglang.srt.model_loader.weight_utils import initialize_dummy_weights
from sglang.srt.server_args import ServerArgs


@contextmanager
def model_layer(
    server_args: ServerArgs,
    layer_initializer: Callable[
        [PretrainedConfig, Optional[QuantizationConfig]], torch.nn.Module
    ],
):
    # distributed setup required for parallel layers
    init_distributed_environment(
        backend="nccl",
        world_size=server_args.tp_size * server_args.pp_size,
        rank=0,
        local_rank=server_args.base_gpu_id,
        distributed_init_method=f"tcp://127.0.0.1:{server_args.nccl_port}",
        timeout=server_args.dist_timeout,
    )
    initialize_model_parallel(
        tensor_model_parallel_size=server_args.tp_size,
        pipeline_model_parallel_size=server_args.pp_size,
        expert_model_parallel_size=server_args.ep_size,
        duplicate_tp_group=server_args.enable_pdmux,
    )

    # Pre processing required for model loader
    model_config = ModelConfig.from_server_args(server_args)
    load_config = LoadConfig(LoadFormat.DUMMY)
    model_class, _ = get_model_architecture(model_config)
    packed_modules_mapping = getattr(model_class, "packed_modules_mapping", {})
    quant_config = _get_quantization_config(
        model_config, load_config, packed_modules_mapping
    )

    with set_default_torch_dtype(model_config.dtype):
        # layer initialization
        layer = layer_initializer(model_config.hf_config, quant_config)
        layer.to(device="cuda")

        # Post processing step of dummy model loader
        for _, module in layer.named_modules():
            quant_method = getattr(module, "quant_method", None)
            if quant_method is not None:
                quant_method.process_weights_after_loading(module)

        initialize_dummy_weights(layer)
        post_load_weights(layer, model_config)

    attn_backend = DummyAttentionBackend()

    try:
        yield layer.eval(), model_config, attn_backend
    finally:
        destroy_model_parallel()
        destroy_distributed_environment()
