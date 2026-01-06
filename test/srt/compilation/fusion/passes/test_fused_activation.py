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
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.server_args import ServerArgs
from sglang.test.model_bench import LlamaBench, ModelBenchArgs


def init_llama_mlp(
    bench: LlamaBench, config: LlamaConfig, quant_config: Optional[QuantizationConfig]
) -> torch.nn.Module:
    return bench.init_mlp()


test_data = [
    {
        "models": [
            "meta-llama/Llama-3.2-1B",
            "RedHatAI/Llama-2-7b-chat-hf-FP8",
            "RedHatAI/Meta-Llama-3.1-8B-Instruct-FP8",
        ],
        "model_initializer": init_llama_mlp,
    }
]


test_cases = []
for data in test_data:
    for model in data["models"]:
        test_cases.append((model, data["model_initializer"]))


@pytest.mark.parametrize("model, model_initializer", test_cases)
def test_fused_activation_pass(model, model_initializer):
    server_args = ServerArgs(
        model_path=model,
        attention_backend="none",
        enable_torch_compile=True,
        enable_torch_compile_fusion=True,
        disable_rmsnorm_quant_pass=True,
        nccl_port=12345
        + int(os.environ.get("PYTEST_XDIST_WORKER", "gw0").split("gw")[1]),
    )

    # NOTE: Uncomment these lines for graph debugging
    # server_args.log_level = "debug"
    # server_args.enable_torch_compile_graph_trace_logs = True

    bench_args = ModelBenchArgs(
        num_tokens=1,
        forward_mode=ForwardMode.DECODE,
    )

    with LlamaBench(server_args, bench_args, model_initializer) as bench:
        # prepare input
        hidden_states = bench.get_rand_input_hidden_states()

        # reference should be done before torch compile
        ref_model = copy.deepcopy(bench.model)
        ref_res = ref_model(hidden_states)

        # torch compile run
        bench.torch_compile()
        res, source_codes = run_and_get_code(bench.model, hidden_states)
        code = "\n".join(source_codes)

        torch.testing.assert_close(ref_res, res)

        assert "sglang.fused_swiglu" in code
        assert "sgl_kernel.silu_and_mul" not in code


if __name__ == "__main__":
    pytest.main([__file__])
