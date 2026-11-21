"""Bole pool allocation and commit routing across the optional snapshot layout."""

import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from sglang.srt.configs.mamba_utils import (
    Mamba2CacheParams,
    Mamba2StateDType,
    Mamba2StateShape,
)
from sglang.srt.layers.attention.hybrid_linear_attn_backend import (
    HybridLinearAttnBackend,
)
from sglang.srt.layers.attention.mamba.mamba2_metadata import ForwardStateSlots
from sglang.srt.mem_cache.memory_pool import HybridReqToTokenPool
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

_BACKEND = "sglang.srt.layers.attention.hybrid_linear_attn_backend"


def _pool(*, bole, allocate):
    shape = Mamba2StateShape.create(
        tp_world_size=1,
        intermediate_size=64,
        n_groups=2,
        num_heads=4,
        head_dim=16,
        state_size=16,
        conv_kernel=4,
    )
    return HybridReqToTokenPool(
        size=2,
        mamba_size=4,
        mamba_spec_state_size=2,
        max_context_len=16,
        device="cpu",
        enable_memory_saver=False,
        cache_params=Mamba2CacheParams(
            shape=shape,
            dtype=Mamba2StateDType(conv=torch.bfloat16, temporal=torch.float32),
            layers=[0, 1],
        ),
        mamba_layer_ids=[0, 1],
        enable_mamba_extra_buffer=False,
        speculative_num_draft_tokens=4,
        enable_linear_bole_spec=bole,
        allocate_intermediate_ssm=allocate,
    )


class TestBoleSpecIntegration(CustomTestCase):
    def test_pool_preserves_optional_snapshot_and_conv_layouts(self):
        for bole in (False, True):
            for allocate in (False, True):
                with self.subTest(bole=bole, allocate=allocate):
                    pool = _pool(bole=bole, allocate=allocate).mamba_pool
                    cache = pool.get_speculative_mamba2_params_all_layers()
                    if allocate:
                        self.assertIsNotNone(cache.intermediate_ssm)
                        self.assertEqual(
                            cache.intermediate_ssm.shape[2], 0 if bole else 4
                        )
                    else:
                        self.assertIsNone(cache.intermediate_ssm)
                    self.assertEqual(cache.intermediate_conv_window[0].shape[2], 4)
                    self.assertEqual(pool.bole_verifier is not None, bole)
                    if bole:
                        self.assertIs(pool.bole_verifier.temporal, cache.temporal)

    def test_bole_commits_without_snapshots_and_uses_resolved_track_slots(self):
        pool = _pool(bole=True, allocate=False).mamba_pool
        pool.bole_verifier = Mock()
        working = torch.tensor([1, 2], dtype=torch.int32)
        tracked = torch.tensor([3, -1], dtype=torch.int32)
        steps = torch.tensor([2, 0], dtype=torch.int32)
        track_steps = torch.tensor([1, -1], dtype=torch.int32)
        backend = object.__new__(HybridLinearAttnBackend)
        backend.linear_attn_backend = SimpleNamespace(
            state_pool=pool,
            accept_lens_pool=None,
            forward_metadata=SimpleNamespace(query_start_loc=torch.arange(5)),
            forward_state_slots=lambda *_: ForwardStateSlots(working, tracked),
        )
        backend._update_ple_state_after_mtp_verify = Mock()
        with (
            patch(f"{_BACKEND}.pp_spec_stable_rows_enabled", return_value=False),
            patch(f"{_BACKEND}.fused_conv_window_scatter_with_mask") as conv,
            patch(f"{_BACKEND}.scatter_mamba_states_after_mtp_verify") as snapshots,
        ):
            backend.update_mamba_state_after_mtp_verify(
                last_correct_step_indices=steps,
                mamba_track_indices=torch.tensor([10, 11]),
                mamba_steps_to_track=track_steps,
                model=None,
            )
        snapshots.assert_not_called()
        pool.bole_verifier.commit.assert_called_once()
        args = pool.bole_verifier.commit.call_args.args
        self.assertEqual(args[0], 4)
        torch.testing.assert_close(args[1], working)
        torch.testing.assert_close(args[2], steps)
        torch.testing.assert_close(args[3], tracked)
        torch.testing.assert_close(args[4], track_steps)
        self.assertEqual(conv.call_count, 2)
        torch.testing.assert_close(conv.call_args_list[0].args[2], working)
        torch.testing.assert_close(conv.call_args_list[1].args[2], tracked)


if __name__ == "__main__":
    unittest.main()
