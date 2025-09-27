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

import copy
import os
from typing import Optional

import pytest
import torch
from torch._inductor.utils import run_and_get_code
from transformers import LlamaConfig

from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.model_executor.cuda_graph_runner import torch_compile
from sglang.srt.models.llama import LlamaMLP
from sglang.srt.server_args import ServerArgs
from sglang.test.model_factory import model_layer


def init_llama_mlp(
    config: LlamaConfig, quant_config: Optional[QuantizationConfig]
) -> torch.nn.Module:
    return LlamaMLP(
        config.hidden_size,
        config.intermediate_size,
        config.hidden_act,
        quant_config,
        "llama_mlp",
    )


test_data = [
    {
        "models": [
            "meta-llama/Llama-3.2-1B",
            "RedHatAI/Llama-2-7b-chat-hf-FP8",
        ],
        "layer_initializer": init_llama_mlp,
    }
]


test_cases = []
for data in test_data:
    for model in data["models"]:
        test_cases.append((model, data["layer_initializer"]))


@pytest.mark.parametrize("model, layer_initializer", test_cases)
def test_fused_activation_pass(model, layer_initializer):
    server_args = ServerArgs(
        model_path=model,
        enable_torch_compile=True,
        enable_torch_compile_fusion=True,
        enable_fused_activation_pass=True,
        nccl_port=12345
        + int(os.environ.get("PYTEST_XDIST_WORKER", "gw0").split("gw")[1]),
    )

    # NOTE: Uncomment these lines for graph debugging
    # server_args.log_level = "debug"
    # server_args.enable_torch_compile_graph_trace_logs = True
    # from sglang.srt.utils import configure_logger
    # configure_logger(server_args, " TEST_LOGGER")

    with model_layer(server_args, layer_initializer) as (layer, model_config, _):
        ref_layer = copy.deepcopy(layer)

        # prepare input
        num_tokens = 2
        hidden_states = torch.rand(
            (num_tokens, model_config.hf_config.hidden_size),
            device="cuda",
            dtype=model_config.dtype,
        )

        ref_res = ref_layer(hidden_states)

        # same torch compile setup as model runner
        torch_compile(layer, server_args, model_config)

        res, source_codes = run_and_get_code(layer.forward, hidden_states)
        code = "\n".join(source_codes)

        torch.testing.assert_close(ref_res, res)
        assert "sglang.fused_swiglu" in code
        assert "sgl_kernel.silu_and_mul" not in code


if __name__ == "__main__":
    # pytest.main([__file__])
    test_fused_activation_pass("meta-llama/Llama-3.2-1B", init_llama_mlp)
