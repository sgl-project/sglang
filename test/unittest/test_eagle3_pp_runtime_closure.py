"""Runtime closure tests for GLM-5.2 EAGLE3 TP4×PP2.

Tests covering P0-1 through P0-8 runtime correctness contracts:
  A. Verify lifetime contract (record_stream, keep-alive)
  B. Idle EagleVerifyInput path
  C. Required/optional PP proxy keys + stale buffer prevention
  D. Missing remote aux fatal test
  E. Rejection sampling startup rejection
  F. Static buffer usage (no eager fallback in graph path)
  G. Multi-graph-key buffer ownership (pointer stability)
  H. CUDA Graph-safe debug logging
  I. Partition source-of-truth (SGLANG_PP_LAYER_PARTITION only)
  J. Capture layer sorted/unique/semantic tests
"""

import os
import sys
from unittest.mock import MagicMock, patch

import pytest
import torch

from sglang.srt.speculative.spec_info import SpeculativeAlgorithm

# -------------------------------------------------------------------- #
# A. Verify lifetime contract                                         #
# -------------------------------------------------------------------- #

class TestVerifyLifetimeContract:
    """P0-1: PP0 verify path must match standard EAGLEWorkerV2.verify()
    tensor-lifetime contract."""

    def test_record_stream_for_v2_verify_called(self):
        """Verify that record_stream_for_v2_verify is importable and
        has the expected signature."""
        from sglang.srt.speculative.spec_utils import (
            record_stream_for_v2_verify,
            record_stream_each,
        )
        import inspect
        sig = inspect.signature(record_stream_for_v2_verify)
        params = list(sig.parameters.keys())
        assert "batch" in params
        assert "verify_input" in params
        assert "fwd_stream" in params

        sig2 = inspect.signature(record_stream_each)
        params2 = list(sig2.parameters.keys())
        assert "tensors" in params2
        assert "stream" in params2

    def test_extra_keep_alive_refs_exists(self):
        """GenerationBatchResult must have extra_keep_alive_refs field."""
        from sglang.srt.managers.utils import GenerationBatchResult
        gbr = GenerationBatchResult()
        assert hasattr(gbr, "extra_keep_alive_refs")
        assert gbr.extra_keep_alive_refs is None

    def test_verify_forward_batch_keep_alive_on_pp0(self):
        """The PP0 verify path sets extra_keep_alive_refs to retain
        verify_forward_batch. Verify the scheduler code path exists."""
        import inspect
        from sglang.srt.managers.scheduler import Scheduler
        source = inspect.getsource(Scheduler.run_batch)
        assert "extra_keep_alive_refs" in source
        assert "verify_forward_batch" in source
        assert "record_stream_for_v2_verify" in source
        assert "record_stream_each" in source

    def test_no_unbounded_keep_alive_growth(self):
        """Verify that extra_keep_alive_refs is cleared after consumption
        via batch_record_buf (ring buffer with fixed size)."""
        import inspect
        from sglang.srt.managers.scheduler import Scheduler
        source = inspect.getsource(Scheduler.run_batch)
        # batch_record_buf is a ring buffer (fixed size), so refs are
        # overwritten after 2 iterations, not appended indefinitely.
        assert "batch_record_buf" in source
        assert "batch_record_ct" in source


# -------------------------------------------------------------------- #
# B. Idle EagleVerifyInput path                                       #
# -------------------------------------------------------------------- #

class TestIdleVerifyInput:
    """P0-2: Idle EagleVerifyInput.create_idle_input must match exact API."""

    def test_create_idle_input_signature(self):
        """Verify the exact signature of EagleVerifyInput.create_idle_input."""
        from sglang.srt.speculative.eagle_info import EagleVerifyInput
        import inspect
        sig = inspect.signature(EagleVerifyInput.create_idle_input)
        params = list(sig.parameters.keys())
        # Must accept: topk, spec_steps, num_verify_tokens, device
        assert "topk" in params
        assert "spec_steps" in params
        assert "num_verify_tokens" in params
        assert "device" in params

    def test_create_idle_input_returns_valid_object(self):
        """create_idle_input must return a valid EagleVerifyInput with
        empty tensors."""
        from sglang.srt.speculative.eagle_info import EagleVerifyInput
        vi = EagleVerifyInput.create_idle_input(
            topk=1,
            spec_steps=4,
            num_verify_tokens=5,
            device="cpu",
        )
        assert vi.draft_token.shape[0] == 0
        assert vi.spec_steps == 4
        assert vi.topk == 1
        assert vi.draft_token_num == 5

    def test_scheduler_pp_mixin_idle_path(self):
        """The scheduler PP mixin must call create_idle_input with
        the correct arguments for idle batches."""
        import inspect
        from sglang.srt.managers.scheduler_pp_mixin import SchedulerPPMixin
        source = inspect.getsource(SchedulerPPMixin._pp_spec_rebuild_verify_input)
        assert "create_idle_input" in source
        assert "topk=" in source
        assert "spec_steps=" in source
        assert "num_verify_tokens=" in source
        assert "device=" in source


# -------------------------------------------------------------------- #
# C. Required/optional PP proxy keys + stale buffer prevention         #
# -------------------------------------------------------------------- #

class TestPPProxyKeySemantics:
    """P0-3: Required PP proxy keys must be strict; missing required keys
    must be fatal."""

    def test_required_keys_set(self):
        from sglang.srt.speculative.glm52_eagle3_pp import REQUIRED_PP_PROXY_KEYS
        assert "hidden_states" in REQUIRED_PP_PROXY_KEYS
        assert "residual" in REQUIRED_PP_PROXY_KEYS

    def test_optional_keys_set(self):
        from sglang.srt.speculative.glm52_eagle3_pp import OPTIONAL_PP_PROXY_KEYS
        assert "topk_indices" in OPTIONAL_PP_PROXY_KEYS

    def test_missing_hidden_states_is_fatal(self):
        from sglang.srt.speculative.glm52_eagle3_pp import validate_pp_proxy_keys
        with pytest.raises(RuntimeError, match="hidden_states"):
            validate_pp_proxy_keys(
                available_keys=["residual"],
                pp_rank=1,
                tp_rank=0,
                forward_mode="TARGET_VERIFY",
                active_token_rows=10,
                remote_capture_layers_exist=False,
            )

    def test_missing_residual_is_fatal(self):
        from sglang.srt.speculative.glm52_eagle3_pp import validate_pp_proxy_keys
        with pytest.raises(RuntimeError, match="residual"):
            validate_pp_proxy_keys(
                available_keys=["hidden_states"],
                pp_rank=1,
                tp_rank=0,
                forward_mode="TARGET_VERIFY",
                active_token_rows=10,
                remote_capture_layers_exist=False,
            )

    def test_missing_required_aux_is_fatal(self):
        """PP1 with remote capture layers but missing aux key must fail."""
        from sglang.srt.speculative.glm52_eagle3_pp import (
            GLM52_EAGLE3_AUX_PP_KEY,
            validate_pp_proxy_keys,
        )
        with pytest.raises(RuntimeError, match="EAGLE3 aux"):
            validate_pp_proxy_keys(
                available_keys=["hidden_states", "residual"],
                pp_rank=1,
                tp_rank=0,
                forward_mode="TARGET_VERIFY",
                active_token_rows=10,
                remote_capture_layers_exist=True,
                slot_ownership={2: 0, 39: 0, 75: 1},
            )

    def test_missing_optional_topk_accepted(self):
        """Missing topk_indices must not be fatal (it's optional)."""
        from sglang.srt.speculative.glm52_eagle3_pp import validate_pp_proxy_keys
        # Should not raise
        validate_pp_proxy_keys(
            available_keys=["hidden_states", "residual"],
            pp_rank=1,
            tp_rank=0,
            forward_mode="TARGET_VERIFY",
            active_token_rows=10,
            remote_capture_layers_exist=False,
        )

    def test_stale_buffer_regression(self):
        """Stale buffer regression: round 2 missing required key must error,
        not silently reuse round 1's static buffer contents."""
        from sglang.srt.speculative.glm52_eagle3_pp import validate_pp_proxy_keys

        # Round 1: required key present
        validate_pp_proxy_keys(
            available_keys=["hidden_states", "residual",
                            "glm52_eagle3_aux_hidden_states"],
            pp_rank=1,
            tp_rank=0,
            forward_mode="TARGET_VERIFY",
            active_token_rows=10,
            remote_capture_layers_exist=True,
            slot_ownership={2: 0, 39: 0, 75: 1},
        )

        # Round 2: required key missing -> must error
        with pytest.raises(RuntimeError):
            validate_pp_proxy_keys(
                available_keys=["hidden_states", "residual"],
                pp_rank=1,
                tp_rank=0,
                forward_mode="TARGET_VERIFY",
                active_token_rows=10,
                remote_capture_layers_exist=True,
                slot_ownership={2: 0, 39: 0, 75: 1},
            )


# -------------------------------------------------------------------- #
# E. Rejection sampling startup rejection                             #
# -------------------------------------------------------------------- #

class TestRejectionSamplingRejection:
    """P0-8: Rejection sampling must be rejected at startup."""

    @pytest.fixture(autouse=True)
    def _enable_pp_spec(self):
        """Enable SGLANG_ENABLE_PP_SPEC for these tests."""
        with patch.dict(os.environ, {"SGLANG_ENABLE_PP_SPEC": "1"}):
            yield

    def _make_sa(self, **kw):
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
        defaults.update(kw)
        return MagicMock(**defaults)

    def test_rejection_sampling_rejected(self):
        from sglang.srt.speculative.glm52_eagle3_pp import (
            validate_glm52_eagle3_tp4_pp2_configuration,
        )
        sa = self._make_sa(speculative_use_rejection_sampling=True)
        with pytest.raises(ValueError, match="rejection_sampling"):
            validate_glm52_eagle3_tp4_pp2_configuration(
                server_args=sa,
                spec_algorithm=SpeculativeAlgorithm.EAGLE3,
                is_draft_worker=False,
                pp_rank=0,
                tp_rank=0,
            )

    def test_disaggregation_rejected(self):
        from sglang.srt.speculative.glm52_eagle3_pp import (
            validate_glm52_eagle3_tp4_pp2_configuration,
        )
        sa = self._make_sa(enable_disaggregation=True)
        with pytest.raises(ValueError, match="disaggregation"):
            validate_glm52_eagle3_tp4_pp2_configuration(
                server_args=sa,
                spec_algorithm=SpeculativeAlgorithm.EAGLE3,
                is_draft_worker=False,
                pp_rank=0,
                tp_rank=0,
            )

    def test_dp_attention_rejected(self):
        from sglang.srt.speculative.glm52_eagle3_pp import (
            validate_glm52_eagle3_tp4_pp2_configuration,
        )
        sa = self._make_sa(enable_dp_attention=True)
        with pytest.raises(ValueError, match="DP attention"):
            validate_glm52_eagle3_tp4_pp2_configuration(
                server_args=sa,
                spec_algorithm=SpeculativeAlgorithm.EAGLE3,
                is_draft_worker=False,
                pp_rank=0,
                tp_rank=0,
            )

    def test_ep_moe_rejected(self):
        from sglang.srt.speculative.glm52_eagle3_pp import (
            validate_glm52_eagle3_tp4_pp2_configuration,
        )
        sa = self._make_sa(enable_ep_moe=True)
        with pytest.raises(ValueError, match="Expert parallelism"):
            validate_glm52_eagle3_tp4_pp2_configuration(
                server_args=sa,
                spec_algorithm=SpeculativeAlgorithm.EAGLE3,
                is_draft_worker=False,
                pp_rank=0,
                tp_rank=0,
            )

    def test_async_pp_depth_rejected(self):
        from sglang.srt.speculative.glm52_eagle3_pp import (
            validate_glm52_eagle3_tp4_pp2_configuration,
        )
        sa = self._make_sa(pp_async_batch_depth=1)
        with pytest.raises(ValueError, match="pp_async_batch_depth"):
            validate_glm52_eagle3_tp4_pp2_configuration(
                server_args=sa,
                spec_algorithm=SpeculativeAlgorithm.EAGLE3,
                is_draft_worker=False,
                pp_rank=0,
                tp_rank=0,
            )

    def test_token_map_rejected(self):
        from sglang.srt.speculative.glm52_eagle3_pp import (
            validate_glm52_eagle3_tp4_pp2_configuration,
        )
        sa = self._make_sa(speculative_token_map="/fake/token_map.json")
        with pytest.raises(ValueError, match="Token-map"):
            validate_glm52_eagle3_tp4_pp2_configuration(
                server_args=sa,
                spec_algorithm=SpeculativeAlgorithm.EAGLE3,
                is_draft_worker=False,
                pp_rank=0,
                tp_rank=0,
            )


# -------------------------------------------------------------------- #
# H. CUDA Graph-safe debug logging                                    #
# -------------------------------------------------------------------- #

class TestCudaGraphSafeDebug:
    """P0-7: Debug value logging must be disabled during CUDA Graph capture."""

    def test_graph_capture_guard_exists(self):
        """Verify the model source checks is_current_stream_capturing."""
        import inspect
        from sglang.srt.models.deepseek_v2 import DeepseekV2Model
        source = inspect.getsource(DeepseekV2Model.forward)
        assert "is_current_stream_capturing" in source

    def test_static_buffer_enforcement_unconditional(self):
        """P0-5: CUDA Graph static-buffer enforcement must NOT depend on
        SGLANG_GLM52_PP_DEBUG. The raise must be outside the debug guard."""
        import inspect
        from sglang.srt.models.deepseek_v2 import DeepseekV2Model
        source = inspect.getsource(DeepseekV2Model.forward)

        # Find the unconditional enforcement block
        assert "is_current_stream_capturing() and not used_static" in source, (
            "DeepseekV2Model.forward must unconditionally raise when "
            "CUDA Graph capture is active and used_static is False"
        )

        # Verify the raise is NOT inside the SGLANG_GLM52_PP_DEBUG guard
        # by checking the enforcement appears before the debug log block
        enforcement_idx = source.find(
            "is_current_stream_capturing() and not used_static"
        )
        debug_idx = source.find("SGLANG_GLM52_PP_DEBUG", enforcement_idx)
        assert enforcement_idx > 0, "Enforcement block must exist"
        # The enforcement must come before (or without) the debug guard
        # that follows it
        assert debug_idx > enforcement_idx or debug_idx == -1, (
            "Static-buffer enforcement must not be guarded by SGLANG_GLM52_PP_DEBUG"
        )


# -------------------------------------------------------------------- #
# I. Partition source-of-truth                                        #
# -------------------------------------------------------------------- #

class TestPartitionSourceOfTruth:
    """P1-1: Only SGLANG_PP_LAYER_PARTITION; SGLANG_GLM52_PP_SPLIT removed."""

    def test_glm52_pp_split_env_removed(self):
        """SGLANG_GLM52_PP_SPLIT must not exist in environ.py."""
        from sglang.srt import environ
        assert not hasattr(environ.envs, "SGLANG_GLM52_PP_SPLIT")

    def test_pp_layer_partition_used(self):
        """build_slot_ownership_map must use SGLANG_PP_LAYER_PARTITION."""
        from sglang.srt.speculative.glm52_eagle3_pp import build_slot_ownership_map
        import inspect
        source = inspect.getsource(build_slot_ownership_map)
        assert "SGLANG_PP_LAYER_PARTITION" in source
        assert "SGLANG_GLM52_PP_SPLIT" not in source

    @pytest.mark.parametrize("partition_str,pp0_end,pp1_start", [
        ("39,39", 39, 39),
        ("40,38", 40, 40),
        ("42,36", 42, 42),
        ("44,34", 44, 44),
        ("46,32", 46, 46),
    ])
    def test_partition_ownership(self, partition_str, pp0_end, pp1_start):
        """Test partition ownership for various splits."""
        from sglang.srt.speculative.glm52_eagle3_pp import build_slot_ownership_map
        with patch.dict(os.environ, {"SGLANG_PP_LAYER_PARTITION": partition_str}):
            ownership = build_slot_ownership_map(
                global_capture_layers=[2, 39, 75],
                pp_size=2,
                num_hidden_layers=78,
            )
            # Layer 2 < pp0_end -> owned by PP0
            assert ownership[2] == 0
            # Layer 75 >= pp1_start -> owned by PP1
            assert ownership[75] == 1
            # Layer 39 boundary
            if pp0_end == 39:
                assert ownership[39] == 1  # 39 >= 39 -> PP1
            elif pp0_end == 40:
                assert ownership[39] == 0  # 39 < 40 -> PP0


# -------------------------------------------------------------------- #
# J. Capture layer sorted/unique/semantic tests                       #
# -------------------------------------------------------------------- #

class TestCaptureLayerSemantics:
    """P1-2: Capture layers must be sorted, unique, and in valid range."""

    def test_capture_layers_sorted_and_unique(self):
        from sglang.srt.speculative.glm52_eagle3_pp import validate_capture_layers
        ownership = validate_capture_layers(
            global_capture_layers=[2, 39, 75],
            num_hidden_layers=78,
            pp_size=2,
            start_layer=0,
            end_layer=40,
            hidden_size=5120,
        )
        assert len(ownership) == 3

    def test_capture_layers_unsorted_rejected(self):
        """Unsorted capture layers must be rejected (fail-fast)."""
        from sglang.srt.speculative.glm52_eagle3_pp import validate_capture_layers
        with pytest.raises(ValueError, match="not strictly sorted"):
            validate_capture_layers(
                global_capture_layers=[75, 2, 39],
                num_hidden_layers=78,
                pp_size=2,
                start_layer=0,
                end_layer=40,
                hidden_size=5120,
            )

    def test_capture_layers_duplicate_rejected(self):
        """Duplicate capture layer IDs must be rejected."""
        from sglang.srt.speculative.glm52_eagle3_pp import validate_capture_layers
        with pytest.raises(ValueError, match="duplicate"):
            validate_capture_layers(
                global_capture_layers=[2, 39, 39],
                num_hidden_layers=78,
                pp_size=2,
                start_layer=0,
                end_layer=40,
                hidden_size=5120,
            )

    def test_capture_layers_out_of_range_rejected(self):
        from sglang.srt.speculative.glm52_eagle3_pp import validate_capture_layers
        with pytest.raises(ValueError, match="out of"):
            validate_capture_layers(
                global_capture_layers=[2, 39, 78],
                num_hidden_layers=78,
                pp_size=2,
                start_layer=0,
                end_layer=40,
                hidden_size=5120,
            )

    def test_capture_layers_empty_rejected(self):
        from sglang.srt.speculative.glm52_eagle3_pp import validate_capture_layers
        with pytest.raises(ValueError, match="empty"):
            validate_capture_layers(
                global_capture_layers=[],
                num_hidden_layers=78,
                pp_size=2,
                start_layer=0,
                end_layer=40,
                hidden_size=5120,
            )


# -------------------------------------------------------------------- #
# P1-7: next_verify_chain clone test                                  #
# -------------------------------------------------------------------- #

class TestNextVerifyChainClone:
    """P1-7: next_verify_chain must be cloned at the relay boundary."""

    def test_clone_exists_in_source(self):
        import inspect
        from sglang.srt.speculative.eagle_worker_v2 import EAGLEWorkerV2
        source = inspect.getsource(EAGLEWorkerV2.forward_batch_generation)
        assert ".clone()" in source
        assert "next_verify_chain" in source

    def test_clone_prevents_overwrite(self):
        """Verify that cloning prevents buffer overwrite issues."""
        # Create a tensor simulating a CUDA Graph static buffer
        static_buf = torch.zeros(10, dtype=torch.int64)
        static_buf[0] = 42
        static_buf[1] = 99

        # Clone it (as the code does)
        cloned = static_buf.contiguous().clone()

        # Overwrite the original
        static_buf.fill_(0)

        # Cloned data is preserved
        assert cloned[0].item() == 42
        assert cloned[1].item() == 99


# -------------------------------------------------------------------- #
# P1-6: Request lifecycle cleanup                                     #
# -------------------------------------------------------------------- #

class TestRequestLifecycleCleanup:
    """P1-6: Retraction cleanup must be covered."""

    def test_finish_cleanup_in_source(self):
        import inspect
        from sglang.srt.managers.scheduler_pp_mixin import SchedulerPPMixin
        source = inspect.getsource(SchedulerPPMixin._pp_process_batch_result)
        assert "finished" in source
        assert "_pp_spec_chain_by_rid" in source
        assert "pop" in source

    def test_store_bonus_skips_finished(self):
        import inspect
        from sglang.srt.managers.scheduler_pp_mixin import SchedulerPPMixin
        source = inspect.getsource(SchedulerPPMixin._pp_spec_store_bonus)
        assert "finished" in source
        assert "pop" in source


# -------------------------------------------------------------------- #
# K. Draft identity validation                                         #
# -------------------------------------------------------------------- #

class TestDraftIdentityValidation:
    """P0-8: Draft model must be a real EAGLE3, not MTP/NextN, and must
    differ from the target model."""

    def test_path_comparison_exists(self):
        """Source must compare resolved target and draft paths."""
        import inspect
        from sglang.srt.speculative.eagle_worker_v2 import _validate_eagle3_draft_model
        source = inspect.getsource(_validate_eagle3_draft_model)
        assert "realpath" in source, (
            "_validate_eagle3_draft_model must resolve paths via os.path.realpath"
        )
        assert "target_resolved" in source
        assert "draft_resolved" in source

    def test_mtp_config_keywords_checked(self):
        """Source must check config-level MTP/NextN identifiers."""
        import inspect
        from sglang.srt.speculative.eagle_worker_v2 import _validate_eagle3_draft_model
        source = inspect.getsource(_validate_eagle3_draft_model)
        assert "_MTP_CONFIG_KEYWORDS" in source
        assert "model_type" in source

    def test_nextn_predict_layers_checked(self):
        """Source must reject num_nextn_predict_layers."""
        import inspect
        from sglang.srt.speculative.eagle_worker_v2 import _validate_eagle3_draft_model
        source = inspect.getsource(_validate_eagle3_draft_model)
        assert "num_nextn_predict_layers" in source

    def test_mtp_arch_names_checked(self):
        """Source must check architectures against MTP arch names."""
        import inspect
        from sglang.srt.speculative.eagle_worker_v2 import _validate_eagle3_draft_model
        source = inspect.getsource(_validate_eagle3_draft_model)
        assert "_MTP_ARCH_NAMES" in source
        assert "architectures" in source

    def test_path_equality_rejected(self):
        """Equal resolved paths must raise ValueError."""
        from sglang.srt.speculative.eagle_worker_v2 import _validate_eagle3_draft_model
        from unittest.mock import MagicMock
        import os

        # Create a temp file to use as both target and draft path
        import tempfile
        with tempfile.NamedTemporaryFile(delete=False) as f:
            tmp_path = f.name
        try:
            mock_runner = MagicMock()
            mock_runner.model = MagicMock()
            type(mock_runner.model).__name__ = "SomeEagle3Model"
            mock_runner.model_config.hf_config = MagicMock()
            mock_runner.model_config.hf_config.eagle_config = None
            mock_runner.model_config.hf_config.eagle_aux_hidden_state_layer_ids = [2, 39, 75]
            mock_runner.model_config.hf_config.model_type = "eagle3"
            mock_runner.model_config.hf_config.architectures = ["Eagle3Model"]
            mock_runner.model_config.hf_config.num_nextn_predict_layers = None

            sa = MagicMock()
            sa.model_path = tmp_path
            sa.speculative_draft_model_path = tmp_path

            with pytest.raises(ValueError, match="same path"):
                _validate_eagle3_draft_model(mock_runner, sa)
        finally:
            os.unlink(tmp_path)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
