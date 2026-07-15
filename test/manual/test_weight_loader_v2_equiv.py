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
# Manual verification for weight loader v2 (Qwen2 native path).
#
# Run:
#   CUDA_VISIBLE_DEVICES=0 python test/manual/test_weight_loader_v2_equiv.py
#
# Engine-level e2e (Qwen2 + transformers backend) lives in:
#   test/registered/model_loading/test_weight_loader_v2_e2e.py

import unittest

import torch

from sglang.srt.environ import envs

MODEL = "Qwen/Qwen2-0.5B"
QWEN3_MODEL = "Qwen/Qwen3-0.6B"


def _init_model_parallel() -> None:
    from sglang.srt.distributed import (
        init_distributed_environment,
        initialize_model_parallel,
    )
    from sglang.srt.distributed.parallel_state import monkey_patch_vllm_parallel_state

    try:
        init_distributed_environment(
            backend="nccl",
            world_size=1,
            rank=0,
            local_rank=0,
            distributed_init_method="tcp://127.0.0.1:29634",
        )
        initialize_model_parallel(tensor_model_parallel_size=1)
        monkey_patch_vllm_parallel_state()
    except AssertionError:
        pass


def _load_qwen2_native(v2: bool) -> torch.nn.Module:
    return _load_native_model(MODEL, v2)


def _load_native_model(model_path: str, v2: bool) -> torch.nn.Module:
    from sglang.srt.configs.device_config import DeviceConfig
    from sglang.srt.configs.load_config import LoadConfig
    from sglang.srt.configs.model_config import ModelConfig
    from sglang.srt.model_loader import get_model
    from sglang.srt.server_args import ServerArgs, set_global_server_args_for_scheduler
    from sglang.srt.utils import get_device

    server_args = ServerArgs(
        model_path=model_path,
        dtype=torch.float16,
        trust_remote_code=True,
    )
    set_global_server_args_for_scheduler(server_args)
    model_config = ModelConfig.from_server_args(server_args)

    with envs.SGLANG_ENABLE_WEIGHT_LOADER_V2.override(v2):
        return get_model(
            model_config=model_config,
            load_config=LoadConfig(),
            device_config=DeviceConfig(get_device()),
        )


def _state_dict_cpu(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {
        name: param.detach().cpu().clone() for name, param in model.state_dict().items()
    }


class TestWeightLoaderV2Equiv(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        _init_model_parallel()

    @unittest.skipIf(not torch.cuda.is_available(), "needs GPU")
    def test_qwen2_v1_v2_state_dict_identical(self):
        model_v1 = _load_qwen2_native(v2=False)
        state_v1 = _state_dict_cpu(model_v1)
        del model_v1
        torch.cuda.empty_cache()

        model_v2 = _load_qwen2_native(v2=True)
        state_v2 = _state_dict_cpu(model_v2)
        del model_v2
        torch.cuda.empty_cache()

        self.assertEqual(set(state_v1.keys()), set(state_v2.keys()))
        for name in sorted(state_v1.keys()):
            torch.testing.assert_close(
                state_v1[name],
                state_v2[name],
                rtol=0,
                atol=0,
                msg=name,
            )

    @unittest.skipIf(not torch.cuda.is_available(), "needs GPU")
    def test_qwen3_v1_v2_state_dict_identical(self):
        model_v1 = _load_native_model(QWEN3_MODEL, v2=False)
        state_v1 = _state_dict_cpu(model_v1)
        del model_v1
        torch.cuda.empty_cache()

        model_v2 = _load_native_model(QWEN3_MODEL, v2=True)
        state_v2 = _state_dict_cpu(model_v2)
        del model_v2
        torch.cuda.empty_cache()

        self.assertEqual(set(state_v1.keys()), set(state_v2.keys()))
        for name in sorted(state_v1.keys()):
            torch.testing.assert_close(
                state_v1[name],
                state_v2[name],
                rtol=0,
                atol=0,
                msg=name,
            )


if __name__ == "__main__":
    unittest.main()
