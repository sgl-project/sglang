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
# Manual state-dict verification for representative weight-loader-v2 paths.
#
# Run:
#   CUDA_VISIBLE_DEVICES=0 pytest -v test/manual/test_weight_loader_v2_equiv.py
#
# Only checkpoints already present in the Hugging Face cache are used. Override
# any model with SGLANG_WEIGHT_LOADER_V2_<CATEGORY>_MODEL=/path/to/checkpoint.
# Engine-level e2e coverage lives in:
#   test/registered/model_loading/test_weight_loader_v2_e2e.py

import os
from dataclasses import dataclass
from pathlib import Path

import pytest
import torch

from sglang.srt.environ import envs


@dataclass(frozen=True)
class ModelCase:
    category: str
    default_model: str
    architecture: str
    model_type: str = "generation"

    @property
    def model(self) -> str:
        env_name = f"SGLANG_WEIGHT_LOADER_V2_{self.category.upper()}_MODEL"
        return os.environ.get(env_name, self.default_model)


# These public fixtures are intentionally small. Their architecture strings map
# directly to SGLang native loaders (rather than the transformers fallback).
MODEL_CASES = [
    ModelCase(
        "standard_dense",
        "hf-internal-testing/tiny-random-Gemma2ForCausalLM",
        "Gemma2ForCausalLM",
    ),
    ModelCase(
        "packed_dense",
        "hf-internal-testing/tiny-random-GPTBigCodeForCausalLM",
        "GPTBigCodeForCausalLM",
    ),
    ModelCase(
        "standard_moe",
        "peft-internal-testing/tiny-random-qwen-1.5-MoE",
        "Qwen2MoeForCausalLM",
    ),
    ModelCase(
        "shared_special_moe",
        "hf-internal-testing/tiny-random-MixtralForCausalLM",
        "MixtralForCausalLM",
    ),
    ModelCase(
        "wrapper",
        "trl-internal-testing/tiny-Qwen2ForSequenceClassification-2.5",
        "Qwen2ForSequenceClassification",
        model_type="embedding",
    ),
]


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


def _load_native_model(
    model_path: str, v2: bool, model_type: str
) -> torch.nn.Module:
    from sglang.srt.configs.device_config import DeviceConfig
    from sglang.srt.configs.load_config import LoadConfig
    from sglang.srt.configs.model_config import ModelConfig
    from sglang.srt.model_loader import get_model
    from sglang.srt.server_args import ServerArgs, set_global_server_args_for_scheduler
    from sglang.srt.utils import get_device

    server_args = ServerArgs(
        model_path=model_path,
        dtype=torch.float16,
        model_type=model_type,
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


def _require_cached_checkpoint(case: ModelCase) -> str:
    model = case.model
    path = Path(model).expanduser()
    if path.is_dir():
        cached_path = path
    else:
        try:
            from huggingface_hub import snapshot_download

            cached_path = Path(snapshot_download(model, local_files_only=True))
        except Exception as exc:
            pytest.skip(
                f"{case.category}: {model!r} is not cached locally ({exc})"
            )

    weight_files = [
        *cached_path.glob("*.safetensors"),
        *cached_path.glob("*.bin"),
    ]
    if not weight_files:
        pytest.skip(
            f"{case.category}: cached checkpoint {str(cached_path)!r} has no weights"
        )

    try:
        from transformers import AutoConfig

        config = AutoConfig.from_pretrained(
            cached_path, local_files_only=True, trust_remote_code=True
        )
    except Exception as exc:
        pytest.skip(
            f"{case.category}: cached config for {model!r} is unavailable ({exc})"
        )

    architectures = getattr(config, "architectures", None) or []
    if case.architecture not in architectures:
        pytest.skip(
            f"{case.category}: expected architecture {case.architecture!r}, "
            f"but {model!r} declares {architectures!r}"
        )
    return str(cached_path)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs GPU")
@pytest.mark.parametrize("case", MODEL_CASES, ids=lambda case: case.category)
def test_v1_v2_state_dict_identical(case: ModelCase):
    model_path = _require_cached_checkpoint(case)
    _init_model_parallel()

    model_v1 = _load_native_model(model_path, v2=False, model_type=case.model_type)
    state_v1 = _state_dict_cpu(model_v1)
    del model_v1
    torch.cuda.empty_cache()

    model_v2 = _load_native_model(model_path, v2=True, model_type=case.model_type)
    state_v2 = _state_dict_cpu(model_v2)
    del model_v2
    torch.cuda.empty_cache()

    assert set(state_v1) == set(state_v2)
    for name in sorted(state_v1):
        torch.testing.assert_close(
            state_v1[name],
            state_v2[name],
            rtol=0,
            atol=0,
            msg=name,
        )


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
