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
"""Differential validation: v1 vs v2 weight loaders produce identical state."""

import unittest

import torch

from sglang.srt.environ import envs


class TestWeightLoaderV2Equivalence(unittest.TestCase):
    MODEL_PATH = "Qwen/Qwen2-0.5B"

    @unittest.skipIf(not torch.cuda.is_available(), "needs GPU")
    def test_qwen2_v1_v2_equivalence(self):
        from transformers import AutoConfig

        from sglang.srt.model_loader.weight_utils import get_model_weights_iter
        from sglang.srt.models.qwen2 import Qwen2ForCausalLM

        config = AutoConfig.from_pretrained(self.MODEL_PATH, trust_remote_code=True)

        with torch.device("cuda"):
            model_v1 = Qwen2ForCausalLM(config)
            model_v2 = Qwen2ForCausalLM(config)

        torch.manual_seed(0)
        for param in (*model_v1.parameters(), *model_v2.parameters()):
            param.data.random_()

        weights = list(get_model_weights_iter(self.MODEL_PATH))

        with envs.SGLANG_ENABLE_WEIGHT_LOADER_V2.override(False):
            model_v1.load_weights(iter(weights))

        with envs.SGLANG_ENABLE_WEIGHT_LOADER_V2.override(True):
            loaded_names = model_v2.load_weights(iter(weights))

        self.assertGreater(len(loaded_names), 0)

        v2_params = dict(model_v2.named_parameters())
        mismatches = []
        for name, p1 in model_v1.named_parameters():
            p2 = v2_params[name]
            if not torch.equal(p1.data, p2.data):
                max_diff = (p1.data - p2.data).abs().max().item()
                mismatches.append(f"  {name}: max_diff={max_diff:.6e}")

        if mismatches:
            self.fail(
                f"Parameter mismatches ({len(mismatches)}):\n"
                + "\n".join(mismatches[:20])
            )

        self.assertTrue(loaded_names.issubset(set(v2_params.keys())))

        v1_buffers = dict(model_v1.named_buffers())
        v2_buffers = dict(model_v2.named_buffers())
        for name, b1 in v1_buffers.items():
            if name not in v2_buffers:
                continue
            b2 = v2_buffers[name]
            if not torch.equal(b1, b2):
                max_diff = (b1 - b2).abs().max().item()
                mismatches.append(f"  buffer {name}: max_diff={max_diff:.6e}")

        if mismatches:
            self.fail(
                f"Buffer mismatches ({len(mismatches)}):\n" + "\n".join(mismatches[:20])
            )


if __name__ == "__main__":
    unittest.main()
