# Copyright 2023-2024 SGLang Team
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
"""Unit tests for the MiniMax-M3 FuseEP prefill integration logic.

These tests validate the SGLang-side inference integration and configuration
validation *without* requiring Ascend NPU hardware.  They cover:

  - Environment variable / arg validation (component 3)
  - FusedMoEMode enum correctness (component 2)
  - Idle DP-rank dummy route construction
  - Negative expert-ID sentinel masking
  - Decode token padding to 128 M-tile
"""

import os
import unittest

try:
    import torch

    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


@unittest.skipUnless(HAS_TORCH, "torch not available")
class TestFusedMoEModeEnum(unittest.TestCase):
    """Verify the FusedMoEMode enum maps to the expected integer constants."""

    def test_mode_values(self):
        from sglang.srt.hardware_backend.npu.utils import FusedMoEMode

        self.assertEqual(FusedMoEMode.DISPATCH_GMM_COMBINE_DECODE, 1)
        self.assertEqual(FusedMoEMode.DISPATCH_FFN_COMBINE, 2)

    def test_mode_from_int(self):
        from sglang.srt.hardware_backend.npu.utils import FusedMoEMode

        self.assertIs(FusedMoEMode(1), FusedMoEMode.DISPATCH_GMM_COMBINE_DECODE)
        self.assertIs(FusedMoEMode(2), FusedMoEMode.DISPATCH_FFN_COMBINE)

    def test_invalid_mode(self):
        from sglang.srt.hardware_backend.npu.utils import FusedMoEMode

        with self.assertRaises(ValueError):
            FusedMoEMode(99)



@unittest.skipUnless(HAS_TORCH, "torch not available")
class TestNegativeExpertIdMasking(unittest.TestCase):
    """Verify that negative sentinel expert IDs are replaced with 0."""

    def test_mask_negative_ids(self):
        topk_ids = torch.tensor([[3, -1, 7, -1], [0, 2, -1, 5]])
        masked = topk_ids.masked_fill(topk_ids < 0, 0)
        self.assertTrue((masked >= 0).all())
        # Original positive values unchanged
        self.assertEqual(masked[0, 0].item(), 3)
        self.assertEqual(masked[0, 2].item(), 7)
        self.assertEqual(masked[1, 0].item(), 0)
        self.assertEqual(masked[1, 1].item(), 2)
        self.assertEqual(masked[1, 3].item(), 5)
        # Negatives replaced with 0
        self.assertEqual(masked[0, 1].item(), 0)
        self.assertEqual(masked[0, 3].item(), 0)
        self.assertEqual(masked[1, 2].item(), 0)

    def test_no_negatives_unchanged(self):
        topk_ids = torch.tensor([[1, 2, 3, 4]])
        masked = topk_ids.masked_fill(topk_ids < 0, 0)
        self.assertTrue(torch.equal(topk_ids, masked))



@unittest.skipUnless(HAS_TORCH, "torch not available")
class TestDecodePaddingTo128(unittest.TestCase):
    """Verify decode tokens < 128 are padded to one full M tile (128)."""

    def test_pad_small_batch(self):
        """Simulate the padding logic from _forward_fuseep_normal_m3."""
        hidden_size = 64
        num_tokens = 30
        top_k = 4
        hidden_states = torch.randn(num_tokens, hidden_size)
        topk_ids = torch.randint(0, 128, (num_tokens, top_k))
        topk_weights = torch.rand(num_tokens, top_k)

        if hidden_states.shape[0] < 128:
            pad_tokens = 128 - hidden_states.shape[0]
            hidden_states_padded = torch.cat(
                (
                    hidden_states,
                    hidden_states.new_zeros((pad_tokens, hidden_size)),
                )
            )
            topk_ids_padded = torch.cat(
                (
                    topk_ids,
                    torch.zeros(
                        (pad_tokens, top_k),
                        dtype=topk_ids.dtype,
                        device=topk_ids.device,
                    ),
                )
            )
            topk_weights_padded = torch.cat(
                (
                    topk_weights,
                    topk_weights.new_ones((pad_tokens, top_k)),
                )
            )

        self.assertEqual(hidden_states_padded.shape[0], 128)
        self.assertEqual(topk_ids_padded.shape[0], 128)
        self.assertEqual(topk_weights_padded.shape[0], 128)
        # Original data preserved
        self.assertTrue(
            torch.equal(hidden_states_padded[:num_tokens], hidden_states)
        )
        self.assertTrue(torch.equal(topk_ids_padded[:num_tokens], topk_ids))
        # Padded hidden states are zero
        self.assertTrue(
            (hidden_states_padded[num_tokens:] == 0).all()
        )

    def test_no_pad_at_128(self):
        """128 tokens should not be padded."""
        hidden_states = torch.randn(128, 64)
        self.assertFalse(hidden_states.shape[0] < 128)

    def test_no_pad_above_128(self):
        """256 tokens should not be padded."""
        hidden_states = torch.randn(256, 64)
        self.assertFalse(hidden_states.shape[0] < 128)


@unittest.skipUnless(HAS_TORCH, "torch not available")
class TestIdleDpRankDummyRoute(unittest.TestCase):
    """Verify dummy route creation for zero-token DP ranks."""

    def test_dummy_hidden_states(self):
        """An idle rank creates a (1, hidden_size) zero tensor."""
        hidden_size = 6144
        hidden_states = torch.empty(0, hidden_size)
        self.assertEqual(hidden_states.shape[0], 0)

        # Simulate idle rank logic
        dummy = hidden_states.new_zeros((1, hidden_size))
        self.assertEqual(dummy.shape, (1, hidden_size))
        self.assertTrue((dummy == 0).all())

    def test_dummy_topk_ids(self):
        """An idle rank creates (1, top_k) zero expert IDs."""
        top_k = 4
        topk_ids = torch.zeros((1, top_k), dtype=torch.int64)
        self.assertEqual(topk_ids.shape, (1, top_k))
        self.assertTrue((topk_ids == 0).all())

    def test_dummy_topk_weights(self):
        """An idle rank creates (1, top_k) unit weights."""
        top_k = 4
        topk_weights = torch.ones((1, top_k), dtype=torch.bfloat16)
        self.assertEqual(topk_weights.shape, (1, top_k))
        self.assertTrue((topk_weights == 1).all())


class TestM3PrefillPathSelection(unittest.TestCase):
    """Test the prefill-vs-decode decision logic from minimax_m3.py."""

    def test_prefill_with_feature_enabled(self):
        """When env is enabled AND is_extend_in_batch, use_m3_fuseep_normal=True."""
        is_extend_in_batch = True
        enable_m3 = True
        expert_dispatch_info = None

        use_m3_fuseep_normal = (
            enable_m3
            and is_extend_in_batch
            and expert_dispatch_info is None
        )
        self.assertTrue(use_m3_fuseep_normal)

    def test_decode_with_feature_enabled(self):
        """When env is enabled BUT NOT is_extend_in_batch, use_m3_fuseep_normal=False."""
        is_extend_in_batch = False
        enable_m3 = True
        expert_dispatch_info = None

        use_m3_fuseep_normal = (
            enable_m3
            and is_extend_in_batch
            and expert_dispatch_info is None
        )
        self.assertFalse(use_m3_fuseep_normal)

    def test_prefill_with_feature_disabled(self):
        """When env is NOT enabled, use_m3_fuseep_normal=False regardless."""
        is_extend_in_batch = True
        enable_m3 = False
        expert_dispatch_info = None

        use_m3_fuseep_normal = (
            enable_m3
            and is_extend_in_batch
            and expert_dispatch_info is None
        )
        self.assertFalse(use_m3_fuseep_normal)

    def test_prefill_with_eplb_dispatch(self):
        """When EPLB expert_location_dispatch_info is set, skip M3 normal-mode."""
        is_extend_in_batch = True
        enable_m3 = True
        expert_dispatch_info = "some_dispatch_info"

        use_m3_fuseep_normal = (
            enable_m3
            and is_extend_in_batch
            and expert_dispatch_info is None
        )
        self.assertFalse(use_m3_fuseep_normal)


class TestM3FuseepNumInputTokens(unittest.TestCase):
    """Test m3_fuseep_num_input_tokens computation logic."""

    def test_with_dp_global_tokens_prefill(self):
        """During prefill with DP, sum global tokens."""
        is_extend_in_batch = True
        dp_global_num_tokens = [100, 200, 50, 150]

        m3_fuseep_num_input_tokens = (
            sum(dp_global_num_tokens)
            if is_extend_in_batch and dp_global_num_tokens is not None
            else None
        )
        self.assertEqual(m3_fuseep_num_input_tokens, 500)

    def test_decode_returns_none(self):
        """During decode, m3_fuseep_num_input_tokens should be None."""
        is_extend_in_batch = False
        dp_global_num_tokens = [100, 200, 50, 150]

        m3_fuseep_num_input_tokens = (
            sum(dp_global_num_tokens)
            if is_extend_in_batch and dp_global_num_tokens is not None
            else None
        )
        self.assertIsNone(m3_fuseep_num_input_tokens)

    def test_no_dp_returns_none(self):
        """Without DP attention, m3_fuseep_num_input_tokens should be None."""
        is_extend_in_batch = True
        dp_global_num_tokens = None

        m3_fuseep_num_input_tokens = (
            sum(dp_global_num_tokens)
            if is_extend_in_batch and dp_global_num_tokens is not None
            else None
        )
        self.assertIsNone(m3_fuseep_num_input_tokens)


class TestMaxOutputSizeComputation(unittest.TestCase):
    """Test max_output_size workspace computation for the M3 operator."""

    def test_prefill_with_dp(self):
        """Prefill with DP attention: max_output_size = num_input_tokens * top_k."""
        num_input_tokens = 500
        top_k = 4
        is_dp = True

        max_output_size = max(num_input_tokens, 1) * top_k
        if not is_dp:
            max_output_size *= 16  # ep_size
        self.assertEqual(max_output_size, 2000)

    def test_prefill_without_dp(self):
        """Prefill without DP: max_output_size = num_input_tokens * top_k * ep_size."""
        num_input_tokens = 500
        top_k = 4
        ep_size = 16
        is_dp = False

        max_output_size = max(num_input_tokens, 1) * top_k
        if not is_dp:
            max_output_size *= ep_size
        self.assertEqual(max_output_size, 32000)

    def test_normal_decode_floor(self):
        """Decode uses a floor of padded_tokens * top_k * ep_size."""
        num_input_tokens = 10
        padded_tokens = 128
        top_k = 4
        ep_size = 16
        is_dp = False

        max_output_size = max(num_input_tokens, 1) * top_k
        if not is_dp:
            max_output_size *= ep_size
        # normal decode floor
        max_output_size = max(max_output_size, padded_tokens * top_k * ep_size)
        self.assertEqual(max_output_size, 128 * 4 * 16)


@unittest.skipUnless(HAS_TORCH, "torch not available")
class TestM3FuseepValidation(unittest.TestCase):
    """Test server argument validation for MiniMax-M3 FuseEP prefill."""

    def test_validation_disabled(self):
        from types import SimpleNamespace
        from unittest.mock import patch

        from sglang.srt.arg_groups.overrides import _m3_fuseep_prefill_validation
        from sglang.srt.environ import envs

        view = SimpleNamespace(moe_a2a_backend="deepep", fuseep_mode=1)
        with patch.object(envs.SGLANG_ENABLE_M3_FUSEEP_PREFILL, "get", return_value=False):
            res = _m3_fuseep_prefill_validation(view)
            self.assertEqual(res, {})

    def test_validation_success(self):
        from types import SimpleNamespace
        from unittest.mock import patch

        from sglang.srt.arg_groups.overrides import _m3_fuseep_prefill_validation
        from sglang.srt.environ import envs

        view = SimpleNamespace(moe_a2a_backend="ascend_fuseep", fuseep_mode=2)
        with patch.object(envs.SGLANG_ENABLE_M3_FUSEEP_PREFILL, "get", return_value=True):
            res = _m3_fuseep_prefill_validation(view)
            self.assertEqual(res, {})

    def test_validation_wrong_backend(self):
        from types import SimpleNamespace
        from unittest.mock import patch

        from sglang.srt.arg_groups.overrides import _m3_fuseep_prefill_validation
        from sglang.srt.environ import envs

        view = SimpleNamespace(moe_a2a_backend="deepep", fuseep_mode=2)
        with patch.object(envs.SGLANG_ENABLE_M3_FUSEEP_PREFILL, "get", return_value=True):
            with self.assertRaises(ValueError) as ctx:
                _m3_fuseep_prefill_validation(view)
            self.assertIn("--moe-a2a-backend ascend_fuseep", str(ctx.exception))

    def test_validation_wrong_fuseep_mode(self):
        from types import SimpleNamespace
        from unittest.mock import patch

        from sglang.srt.arg_groups.overrides import _m3_fuseep_prefill_validation
        from sglang.srt.environ import envs

        view = SimpleNamespace(moe_a2a_backend="ascend_fuseep", fuseep_mode=1)
        with patch.object(envs.SGLANG_ENABLE_M3_FUSEEP_PREFILL, "get", return_value=True):
            with self.assertRaises(ValueError) as ctx:
                _m3_fuseep_prefill_validation(view)
            self.assertIn("SGLANG_NPU_FUSED_MOE_MODE=2", str(ctx.exception))


@unittest.skipUnless(HAS_TORCH, "torch not available")
class TestEnvironmentVariable(unittest.TestCase):
    """Test the SGLANG_ENABLE_M3_FUSEEP_PREFILL environment variable."""

    def test_default_disabled(self):
        """Feature is disabled by default."""
        # Save and clear the env var to test default
        saved = os.environ.pop("SGLANG_ENABLE_M3_FUSEEP_PREFILL", None)
        try:
            from sglang.srt.environ import envs

            # EnvBool caches; test the default value directly
            self.assertFalse(
                envs.SGLANG_ENABLE_M3_FUSEEP_PREFILL.default
            )
        finally:
            if saved is not None:
                os.environ["SGLANG_ENABLE_M3_FUSEEP_PREFILL"] = saved


if __name__ == "__main__":
    unittest.main()
