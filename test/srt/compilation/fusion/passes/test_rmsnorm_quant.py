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
from types import SimpleNamespace
from typing import Optional

import pytest
import torch
from torch._inductor.utils import run_and_get_code
from transformers import LlamaConfig

from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.model_executor.cuda_graph_runner import torch_compile
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.models.llama import LlamaDecoderLayer
from sglang.srt.server_args import ServerArgs
from sglang.test.model_factory import model_layer


def init_llama_decoder(
    config: LlamaConfig, quant_config: Optional[QuantizationConfig]
) -> torch.nn.Module:
    return LlamaDecoderLayer(
        config,
        0,
        quant_config,
        "llama",
    )


test_data = [
    {
        "models": [
            "RedHatAI/Llama-2-7b-chat-hf-FP8",
            "RedHatAI/Meta-Llama-3.1-8B-Instruct-FP8",
        ],
        "layer_initializer": init_llama_decoder,
    }
]


test_cases = []
for data in test_data:
    for model in data["models"]:
        test_cases.append((model, data["layer_initializer"]))


@pytest.mark.parametrize("model, layer_initializer", test_cases)
def test_fused_add_rmsnorm_quant_pass(model, layer_initializer):
    server_args = ServerArgs(
        model_path=model,
        enable_torch_compile=True,
        enable_torch_compile_fusion=True,
        enable_rmsnorm_quant_pass=True,
        nccl_port=12345
        + int(os.environ.get("PYTEST_XDIST_WORKER", "gw0").split("gw")[1]),
    )

    # NOTE: Uncomment these lines for graph debugging
    # server_args.log_level = "debug"
    # server_args.enable_torch_compile_graph_trace_logs = True
    # from sglang.srt.utils import configure_logger
    # configure_logger(server_args, " TEST_LOGGER")

    with model_layer(server_args, layer_initializer) as (
        layer,
        model_config,
        dummy_atten_backend,
    ):
        ref_layer = copy.deepcopy(layer)

        # prepare input
        num_tokens = 2
        dummy_atten_backend.set_out(
            torch.rand(
                (
                    num_tokens,
                    model_config.hf_config.hidden_size
                    * model_config.hf_config.num_attention_heads,
                ),
                device="cuda",
                dtype=model_config.dtype,
            )
        )
        position = torch.randint(100, (num_tokens,), dtype=torch.int64, device="cuda")
        hidden_states = torch.rand(
            (num_tokens, model_config.hf_config.hidden_size),
            device="cuda",
            dtype=model_config.dtype,
        )
        forward_batch = SimpleNamespace(
            forward_mode=ForwardMode.EXTEND,
            attn_backend=dummy_atten_backend,
        )

        ref_res = ref_layer(position, hidden_states, forward_batch, None)

        # same torch compile setup as model runner
        torch_compile(layer, server_args, model_config)

        res, source_codes = run_and_get_code(
            layer.forward, position, hidden_states, forward_batch, None
        )
        code = "\n".join(source_codes)

        torch.testing.assert_close(ref_res, res)
        assert "_C.fused_add_rms_norm_static_fp8_quant" in code
        assert "sgl_kernel.fused_add_rmsnorm" not in code


if __name__ == "__main__":
    pytest.main([__file__])
