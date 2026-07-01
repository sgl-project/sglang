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
"""Differential validation: v1 vs v2 weight loaders produce identical state.

This test loads the same weights through both the legacy and AutoWeightsLoader
paths, then asserts bit-identical parameter state.

Run with:
    SGLANG_IS_IN_CI=1 python -m pytest test/manual/test_weight_loader_v2_equiv.py -v

Requires: GPU, small model downloadable from HF.
"""

import unittest

import torch

from sglang.srt.environ import envs


class TestWeightLoaderV2Equivalence(unittest.TestCase):
    """Prove v1 and v2 weight loaders produce bit-identical model state."""

    MODEL_PATH = "Qwen/Qwen2-0.5B"

    @unittest.skipIf(not torch.cuda.is_available(), "needs GPU")
    def test_qwen2_v1_v2_equivalence(self):
        from transformers import AutoConfig

        from sglang.srt.model_loader.weight_utils import get_model_weights_iter

        # Load weights once into memory
        config = AutoConfig.from_pretrained(self.MODEL_PATH, trust_remote_code=True)

        # We need to instantiate two Qwen2 models and load them separately.
        # For this test we use a simplified approach: load weights list once,
        # then replay through both paths.
        from sglang.srt.models.qwen2 import Qwen2ForCausalLM

        # Minimal construction args (TP=1, no quant, no PP)
        with torch.device("cuda"):
            model_v1 = Qwen2ForCausalLM(config)
            model_v2 = Qwen2ForCausalLM(config)

        # Collect all weights
        weights = list(get_model_weights_iter(self.MODEL_PATH))

        # Load with v1 (legacy)
        with envs.SGLANG_ENABLE_WEIGHT_LOADER_V2.override(False):
            model_v1.load_weights(iter(weights))

        # Load with v2 (AutoWeightsLoader)
        with envs.SGLANG_ENABLE_WEIGHT_LOADER_V2.override(True):
            loaded_names = model_v2.load_weights(iter(weights))

        # loaded_names should be non-empty
        self.assertIsNotNone(loaded_names)
        self.assertGreater(len(loaded_names), 0)

        # Compare all parameters
        v1_params = dict(model_v1.named_parameters())
        v2_params = dict(model_v2.named_parameters())

        self.assertEqual(
            set(v1_params.keys()),
            set(v2_params.keys()),
            "Parameter name sets differ between v1 and v2",
        )

        mismatches = []
        for name in sorted(v1_params.keys()):
            p1 = v1_params[name]
            p2 = v2_params[name]
            if not torch.equal(p1.data, p2.data):
                max_diff = (p1.data - p2.data).abs().max().item()
                mismatches.append(f"  {name}: max_diff={max_diff:.6e}")

        if mismatches:
            self.fail(
                f"Parameter mismatches ({len(mismatches)}):\n"
                + "\n".join(mismatches[:20])
            )

        # Compare all buffers
        v1_buffers = dict(model_v1.named_buffers())
        v2_buffers = dict(model_v2.named_buffers())

        for name in v1_buffers:
            if name not in v2_buffers:
                continue
            b1 = v1_buffers[name]
            b2 = v2_buffers[name]
            if not torch.equal(b1, b2):
                max_diff = (b1 - b2).abs().max().item()
                mismatches.append(f"  buffer {name}: max_diff={max_diff:.6e}")

        if mismatches:
            self.fail(
                f"Buffer mismatches ({len(mismatches)}):\n" + "\n".join(mismatches[:20])
            )

        print(f"✓ V1 and V2 produce identical state for {self.MODEL_PATH}")
        print(f"✓ V2 loaded {len(loaded_names)} parameter names")


if __name__ == "__main__":
    unittest.main()
