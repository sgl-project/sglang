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

import multiprocessing as mp
import os
from dataclasses import dataclass
from functools import lru_cache

import torch

from sglang.srt.environ import envs
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.runners import SRTRunner, check_close_model_outputs
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=600, stage="base-b", runner_config="1-gpu-small")

SHORT_PROMPT = "Hello"


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


# Public tiny-random checkpoints, each tied to the exact native SGLang loader
# named by architecture. Individual cases skip if the fixture cannot be
# resolved, so a transient Hub outage does not fail unrelated base CI coverage.
STANDARD_DENSE = ModelCase(
    "standard_dense",
    "hf-internal-testing/tiny-random-Gemma2ForCausalLM",
    "Gemma2ForCausalLM",
)
PACKED_DENSE = ModelCase(
    "packed_dense",
    "hf-internal-testing/tiny-random-GPTBigCodeForCausalLM",
    "GPTBigCodeForCausalLM",
)
STANDARD_MOE = ModelCase(
    "standard_moe",
    "peft-internal-testing/tiny-random-qwen-1.5-MoE",
    "Qwen2MoeForCausalLM",
)
SHARED_SPECIAL_MOE = ModelCase(
    "shared_special_moe",
    "hf-internal-testing/tiny-random-MixtralForCausalLM",
    "MixtralForCausalLM",
)
WRAPPER = ModelCase(
    "wrapper",
    "trl-internal-testing/tiny-Qwen2ForSequenceClassification-2.5",
    "Qwen2ForSequenceClassification",
    model_type="embedding",
)


@lru_cache(maxsize=None)
def _resolve_model(case: ModelCase) -> tuple[str | None, str | None]:
    try:
        from huggingface_hub import snapshot_download
        from transformers import AutoConfig

        if os.path.isdir(case.model):
            model_path = case.model
        else:
            model_path = snapshot_download(case.model)
        config = AutoConfig.from_pretrained(
            model_path, local_files_only=True, trust_remote_code=True
        )
    except Exception as exc:
        reason = (
            f"{case.category}: cannot resolve checkpoint/config for {case.model!r}; "
            f"override SGLANG_WEIGHT_LOADER_V2_{case.category.upper()}_MODEL "
            f"with an available checkpoint ({exc})"
        )
        return None, reason

    architectures = getattr(config, "architectures", None) or []
    if case.architecture not in architectures:
        reason = (
            f"{case.category}: {case.model!r} declares {architectures!r}, "
            f"not the required native SGLang architecture {case.architecture!r}"
        )
        return None, reason
    return model_path, None


class TestWeightLoaderV2E2E(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        mp.set_start_method("spawn", force=True)

    def _require_model(self, case: ModelCase) -> str:
        model_path, reason = _resolve_model(case)
        if reason:
            self.skipTest(reason)
        assert model_path is not None
        return model_path

    def _runner_kwargs(self, case: ModelCase):
        return dict(
            torch_dtype=torch.float16,
            model_type=case.model_type,
            disable_cuda_graph=True,
            disable_radix_cache=True,
            trust_remote_code=True,
            max_total_tokens=512,
        )

    def _assert_generation_equivalent(self, case: ModelCase):
        model_path = self._require_model(case)
        prompts = [SHORT_PROMPT]
        max_new_tokens = 8
        kwargs = self._runner_kwargs(case)

        with envs.SGLANG_ENABLE_WEIGHT_LOADER_V2.override(False):
            with SRTRunner(model_path, **kwargs) as runner_v1:
                out_v1 = runner_v1.forward(prompts, max_new_tokens=max_new_tokens)

        with envs.SGLANG_ENABLE_WEIGHT_LOADER_V2.override(True):
            with SRTRunner(model_path, **kwargs) as runner_v2:
                out_v2 = runner_v2.forward(prompts, max_new_tokens=max_new_tokens)

        check_close_model_outputs(
            hf_outputs=out_v1,
            srt_outputs=out_v2,
            prefill_tolerance=1e-6,
            decode_tolerance=1e-6,
            rouge_l_tolerance=1.0,
            debug_text=f"{case.category} native v1 vs v2 weight loader",
        )

    def _assert_embedding_wrapper_equivalent(self, case: ModelCase):
        model_path = self._require_model(case)
        prompts = [SHORT_PROMPT]
        kwargs = self._runner_kwargs(case)

        with envs.SGLANG_ENABLE_WEIGHT_LOADER_V2.override(False):
            with SRTRunner(model_path, **kwargs) as runner_v1:
                out_v1 = runner_v1.forward(prompts)

        with envs.SGLANG_ENABLE_WEIGHT_LOADER_V2.override(True):
            with SRTRunner(model_path, **kwargs) as runner_v2:
                out_v2 = runner_v2.forward(prompts)

        self.assertEqual(len(out_v1.embed_logits), len(out_v2.embed_logits))
        torch.testing.assert_close(
            torch.tensor(out_v1.embed_logits),
            torch.tensor(out_v2.embed_logits),
            rtol=0,
            atol=0,
        )

    def test_standard_dense_gemma2_v1_v2_generation_match(self):
        self._assert_generation_equivalent(STANDARD_DENSE)

    def test_packed_dense_gpt_bigcode_v1_v2_generation_match(self):
        self._assert_generation_equivalent(PACKED_DENSE)

    def test_standard_moe_qwen2_v1_v2_generation_match(self):
        self._assert_generation_equivalent(STANDARD_MOE)

    def test_shared_special_moe_mixtral_v1_v2_generation_match(self):
        self._assert_generation_equivalent(SHARED_SPECIAL_MOE)

    def test_wrapper_qwen2_sequence_classification_v1_v2_embedding_match(self):
        self._assert_embedding_wrapper_equivalent(WRAPPER)


if __name__ == "__main__":
    import unittest

    unittest.main()
