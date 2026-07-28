"""Configuration validation tests for GLM-5.2 EAGLE3 TP4×PP2.

Tests that the strict fail-fast validator correctly:
1. Accepts the one supported configuration.
2. Rejects all unsupported configurations.
3. Validates the topk=1 chain invariant.
4. Validates PP split layer boundaries.
5. Tests uneven PP split ownership changes.
"""

import os
from unittest.mock import MagicMock, patch

import pytest

from sglang.srt.speculative.glm52_eagle3_pp import (
    build_slot_ownership_map,
    get_pp_split_layer,
    validate_glm52_eagle3_tp4_pp2_configuration,
)
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm


def make_server_args(**kwargs):
    """Create a mock server_args with the supported config defaults."""
    defaults = dict(
        pp_size=2,
        tp_size=4,
        speculative_eagle_topk=1,
        speculative_num_steps=4,
        speculative_num_draft_tokens=5,
        disable_overlap_schedule=True,
        speculative_adaptive=False,
        speculative_draft_model_path="/fake/eagle3/draft/path",
        speculative_use_rejection_sampling=False,
        enable_disaggregation=False,
        enable_dp_attention=False,
        enable_ep_moe=False,
        pp_async_batch_depth=0,
        speculative_token_map=None,
        enable_context_parallel=False,
    )
    defaults.update(kwargs)
    return MagicMock(**defaults)


class TestConfigValidation:
    """Test the strict configuration validator."""

    def test_valid_config_accepted(self):
        """The one supported configuration must pass."""
        sa = make_server_args()
        validate_glm52_eagle3_tp4_pp2_configuration(
            server_args=sa,
            spec_algorithm=SpeculativeAlgorithm.EAGLE3,
            is_draft_worker=False,
            pp_rank=0,
            tp_rank=0,
        )

    def test_pp_size_1_skipped(self):
        """PP=1 should skip validation (no PP)."""
        sa = make_server_args(pp_size=1)
        validate_glm52_eagle3_tp4_pp2_configuration(
            server_args=sa,
            spec_algorithm=SpeculativeAlgorithm.EAGLE3,
            is_draft_worker=False,
            pp_rank=0,
            tp_rank=0,
        )

    @patch("sglang.srt.environ.envs")
    def test_pp_spec_disabled_skipped(self, mock_envs):
        """SGLANG_ENABLE_PP_SPEC=False should skip validation."""
        mock_envs.SGLANG_ENABLE_PP_SPEC.get.return_value = False
        sa = make_server_args()
        validate_glm52_eagle3_tp4_pp2_configuration(
            server_args=sa,
            spec_algorithm=SpeculativeAlgorithm.EAGLE3,
            is_draft_worker=False,
            pp_rank=0,
            tp_rank=0,
        )

    @patch("sglang.srt.environ.envs")
    def test_reject_pp_size_3(self, mock_envs):
        """PP=3 is not supported."""
        mock_envs.SGLANG_ENABLE_PP_SPEC.get.return_value = True
        sa = make_server_args(pp_size=3)
        with pytest.raises(ValueError, match="pp_size=3"):
            validate_glm52_eagle3_tp4_pp2_configuration(
                server_args=sa,
                spec_algorithm=SpeculativeAlgorithm.EAGLE3,
                is_draft_worker=False,
                pp_rank=0,
                tp_rank=0,
            )

    @patch("sglang.srt.environ.envs")
    def test_reject_tp_size_8(self, mock_envs):
        """TP=8 is not supported."""
        mock_envs.SGLANG_ENABLE_PP_SPEC.get.return_value = True
        sa = make_server_args(tp_size=8)
        with pytest.raises(ValueError, match="tp_size=8"):
            validate_glm52_eagle3_tp4_pp2_configuration(
                server_args=sa,
                spec_algorithm=SpeculativeAlgorithm.EAGLE3,
                is_draft_worker=False,
                pp_rank=0,
                tp_rank=0,
            )

    @patch("sglang.srt.environ.envs")
    def test_reject_eagle_not_eagle3(self, mock_envs):
        """EAGLE (not EAGLE3) is not supported."""
        mock_envs.SGLANG_ENABLE_PP_SPEC.get.return_value = True
        sa = make_server_args()
        with pytest.raises(ValueError, match="EAGLE3"):
            validate_glm52_eagle3_tp4_pp2_configuration(
                server_args=sa,
                spec_algorithm=SpeculativeAlgorithm.EAGLE,
                is_draft_worker=False,
                pp_rank=0,
                tp_rank=0,
            )

    @patch("sglang.srt.environ.envs")
    def test_reject_topk_2(self, mock_envs):
        """topk=2 is not supported."""
        mock_envs.SGLANG_ENABLE_PP_SPEC.get.return_value = True
        sa = make_server_args(speculative_eagle_topk=2)
        with pytest.raises(ValueError, match="topk=2"):
            validate_glm52_eagle3_tp4_pp2_configuration(
                server_args=sa,
                spec_algorithm=SpeculativeAlgorithm.EAGLE3,
                is_draft_worker=False,
                pp_rank=0,
                tp_rank=0,
            )

    @patch("sglang.srt.environ.envs")
    def test_reject_overlap_schedule(self, mock_envs):
        """Overlap schedule is not supported."""
        mock_envs.SGLANG_ENABLE_PP_SPEC.get.return_value = True
        sa = make_server_args(disable_overlap_schedule=False)
        with pytest.raises(ValueError, match="overlap"):
            validate_glm52_eagle3_tp4_pp2_configuration(
                server_args=sa,
                spec_algorithm=SpeculativeAlgorithm.EAGLE3,
                is_draft_worker=False,
                pp_rank=0,
                tp_rank=0,
            )

    @patch("sglang.srt.environ.envs")
    def test_reject_adaptive(self, mock_envs):
        """Adaptive speculation is not supported."""
        mock_envs.SGLANG_ENABLE_PP_SPEC.get.return_value = True
        sa = make_server_args(speculative_adaptive=True)
        with pytest.raises(ValueError, match="adaptive"):
            validate_glm52_eagle3_tp4_pp2_configuration(
                server_args=sa,
                spec_algorithm=SpeculativeAlgorithm.EAGLE3,
                is_draft_worker=False,
                pp_rank=0,
                tp_rank=0,
            )

    @patch("sglang.srt.environ.envs")
    def test_reject_no_draft_model_path(self, mock_envs):
        """Missing draft model path (MTP/NextN fallback) is not supported."""
        mock_envs.SGLANG_ENABLE_PP_SPEC.get.return_value = True
        sa = make_server_args(speculative_draft_model_path=None)
        with pytest.raises(ValueError, match="draft_model_path"):
            validate_glm52_eagle3_tp4_pp2_configuration(
                server_args=sa,
                spec_algorithm=SpeculativeAlgorithm.EAGLE3,
                is_draft_worker=False,
                pp_rank=0,
                tp_rank=0,
            )

    @patch("sglang.srt.environ.envs")
    def test_reject_wrong_draft_tokens_invariant(self, mock_envs):
        """num_draft_tokens != num_steps + 1 must fail for topk=1."""
        mock_envs.SGLANG_ENABLE_PP_SPEC.get.return_value = True
        sa = make_server_args(
            speculative_num_steps=4,
            speculative_num_draft_tokens=6,  # should be 5
        )
        with pytest.raises(ValueError, match="num_draft_tokens"):
            validate_glm52_eagle3_tp4_pp2_configuration(
                server_args=sa,
                spec_algorithm=SpeculativeAlgorithm.EAGLE3,
                is_draft_worker=False,
                pp_rank=0,
                tp_rank=0,
            )

    @patch("sglang.srt.environ.envs")
    def test_reject_mtp_algorithm(self, mock_envs):
        """FROZEN_KV_MTP (MTP) is not supported."""
        mock_envs.SGLANG_ENABLE_PP_SPEC.get.return_value = True
        sa = make_server_args()
        with pytest.raises(ValueError, match="EAGLE3"):
            validate_glm52_eagle3_tp4_pp2_configuration(
                server_args=sa,
                spec_algorithm=SpeculativeAlgorithm.FROZEN_KV_MTP,
                is_draft_worker=False,
                pp_rank=0,
                tp_rank=0,
            )

    @patch("sglang.srt.environ.envs")
    def test_reject_standalone_algorithm(self, mock_envs):
        """STANDALONE is not supported."""
        mock_envs.SGLANG_ENABLE_PP_SPEC.get.return_value = True
        sa = make_server_args()
        with pytest.raises(ValueError, match="EAGLE3"):
            validate_glm52_eagle3_tp4_pp2_configuration(
                server_args=sa,
                spec_algorithm=SpeculativeAlgorithm.STANDALONE,
                is_draft_worker=False,
                pp_rank=0,
                tp_rank=0,
            )


class TestPPSplitLayer:
    """Test the PP split layer via SGLANG_PP_LAYER_PARTITION."""

    def test_default_even_split(self):
        """Default should be even split."""
        os.environ.pop("SGLANG_PP_LAYER_PARTITION", None)
        assert get_pp_split_layer(78, 2) == 39

    def test_explicit_split_44(self):
        """SGLANG_PP_LAYER_PARTITION=44,34 gives 44/34 split."""
        os.environ["SGLANG_PP_LAYER_PARTITION"] = "44,34"
        try:
            assert get_pp_split_layer(78, 2) == 44
        finally:
            del os.environ["SGLANG_PP_LAYER_PARTITION"]

    def test_explicit_split_42(self):
        """SGLANG_PP_LAYER_PARTITION=42,36 gives 42/36 split."""
        os.environ["SGLANG_PP_LAYER_PARTITION"] = "42,36"
        try:
            assert get_pp_split_layer(78, 2) == 42
        finally:
            del os.environ["SGLANG_PP_LAYER_PARTITION"]

    def test_no_partition_default(self):
        """No partition env means default (even)."""
        os.environ.pop("SGLANG_PP_LAYER_PARTITION", None)
        assert get_pp_split_layer(78, 2) == 39


class TestUnevenSplitOwnership:
    """Test that capture layer ownership changes correctly with uneven splits."""

    def test_even_split_39_39(self):
        """Default 39/39 split: layer 2 on PP0, layers 39 and 75 on PP1."""
        os.environ.pop("SGLANG_PP_LAYER_PARTITION", None)
        layers = [2, 39, 75]
        ownership = build_slot_ownership_map(layers, pp_size=2, num_hidden_layers=78)
        assert ownership == {2: 0, 39: 1, 75: 1}

    def test_split_44_34(self):
        """44/34 split: layer 2 on PP0, layers 39 and 75 on PP1 (39 < 44)."""
        os.environ["SGLANG_PP_LAYER_PARTITION"] = "44,34"
        try:
            layers = [2, 39, 75]
            ownership = build_slot_ownership_map(layers, pp_size=2, num_hidden_layers=78)
            assert ownership == {2: 0, 39: 0, 75: 1}
        finally:
            del os.environ["SGLANG_PP_LAYER_PARTITION"]

    def test_split_46_32(self):
        """46/32 split: layers 2 and 39 on PP0, layer 75 on PP1."""
        os.environ["SGLANG_PP_LAYER_PARTITION"] = "46,32"
        try:
            layers = [2, 39, 75]
            ownership = build_slot_ownership_map(layers, pp_size=2, num_hidden_layers=78)
            assert ownership == {2: 0, 39: 0, 75: 1}
        finally:
            del os.environ["SGLANG_PP_LAYER_PARTITION"]

    def test_split_moves_layer_from_pp1_to_pp0(self):
        """A split that moves a capture layer from PP1 to PP0 must update ownership."""
        # With default 39/39: layer 39 is on PP1
        os.environ.pop("SGLANG_PP_LAYER_PARTITION", None)
        layers = [2, 39, 75]
        ownership_even = build_slot_ownership_map(layers, pp_size=2, num_hidden_layers=78)
        assert ownership_even[39] == 1

        # With 40/38 split: layer 39 moves to PP0
        os.environ["SGLANG_PP_LAYER_PARTITION"] = "40,38"
        try:
            ownership_uneven = build_slot_ownership_map(layers, pp_size=2, num_hidden_layers=78)
            assert ownership_uneven[39] == 0
        finally:
            del os.environ["SGLANG_PP_LAYER_PARTITION"]

    def test_explicit_partition_overrides_default(self):
        """SGLANG_PP_LAYER_PARTITION overrides the default even split."""
        os.environ["SGLANG_PP_LAYER_PARTITION"] = "40,38"
        try:
            layers = [2, 39, 75]
            ownership = build_slot_ownership_map(layers, pp_size=2, num_hidden_layers=78)
            # With partition 40,38: layer 39 is on PP0 (0..39)
            assert ownership[39] == 0
        finally:
            del os.environ["SGLANG_PP_LAYER_PARTITION"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
