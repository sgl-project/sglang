"""V4.1 prefill CP token ownership and MegaMoE buffer-budget regressions."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.arg_groups import deepseek_v4_hook
from sglang.srt.layers.cp.utils import cp_shard_model_inputs
from sglang.srt.layers.moe.utils import MoeA2ABackend
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.dsv41_cp_test_utils import cp_context
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestDSV41CPMegaMoE(CustomTestCase):
    def test_local_token_count_masks_padding_and_restores_global_count(self):
        for size in (2, 4):
            for length in (5, 9, 17, 129):
                for rank in range(size):
                    with (
                        self.subTest(size=size, length=length, rank=rank),
                        cp_context(size, rank, (1, length - 1), (0, 16384)) as (
                            _,
                            batch,
                        ),
                        patch(
                            "sglang.srt.layers.cp.utils.get_moe_a2a_backend",
                            return_value=MoeA2ABackend.MEGAMOE,
                        ),
                    ):
                        full = torch.arange(length * 3, dtype=torch.float32).reshape(
                            length, 3
                        )
                        original = torch.tensor(length, dtype=torch.int32)
                        batch.num_token_non_padded = original
                        with cp_shard_model_inputs(
                            full, batch.positions, batch, batch.input_ids
                        ) as (local, positions, ids):
                            expected_ids = batch.input_ids[rank::size]
                            count = batch.num_token_non_padded.item()
                            self.assertEqual(count, len(expected_ids))
                            torch.testing.assert_close(ids[:count], expected_ids)
                            torch.testing.assert_close(batch.input_ids_global, ids)
                            torch.testing.assert_close(local[:count], full[rank::size])
                            torch.testing.assert_close(
                                positions[:count], batch.positions[rank::size]
                            )
                            self.assertGreaterEqual(len(local), count)
                            self.assertEqual(
                                torch.count_nonzero(local[count:]).item(), 0
                            )
                        self.assertIs(batch.num_token_non_padded, original)
                        self.assertEqual(original.item(), length)

    def test_local_token_count_restored_after_failed_forward(self):
        with (
            cp_context(4, 3) as (_, batch),
            patch(
                "sglang.srt.layers.cp.utils.get_moe_a2a_backend",
                return_value=MoeA2ABackend.MEGAMOE,
            ),
        ):
            original = torch.tensor(9, dtype=torch.int32)
            batch.num_token_non_padded = original
            with self.assertRaisesRegex(RuntimeError, "injected MoE error"):
                with cp_shard_model_inputs(
                    torch.zeros(9, 3), batch.positions, batch, batch.input_ids
                ):
                    self.assertEqual(batch.num_token_non_padded.item(), 2)
                    raise RuntimeError("injected MoE error")
            self.assertIs(batch.num_token_non_padded, original)

    def test_no_a2a_preserves_existing_count(self):
        with cp_context(4, 1) as (_, batch):
            original = torch.tensor(9, dtype=torch.int32)
            batch.num_token_non_padded = original
            with cp_shard_model_inputs(
                torch.zeros(9, 3), batch.positions, batch, batch.input_ids
            ):
                self.assertIs(batch.num_token_non_padded, original)
            self.assertIs(batch.num_token_non_padded, original)

    def validate_budget(self, chunk, size, cap):
        cfg = SimpleNamespace(
            moe_a2a_backend="megamoe",
            disaggregation_mode="prefill",
            pp_size=1,
            enable_dynamic_chunking=False,
            chunked_prefill_size=chunk,
            enable_prefill_cp=True,
            attn_cp_size=size,
        )
        with (
            patch.object(deepseek_v4_hook, "resolving_view", return_value=cfg),
            patch.object(
                deepseek_v4_hook.envs.SGLANG_OPT_DEEPGEMM_MEGA_MOE_NUM_MAX_TOKENS_PER_RANK,
                "get",
                return_value=cap,
            ),
        ):
            deepseek_v4_hook.validate_deepseek_v4_mega_moe_token_budget(object())

    def test_buffer_budget_covers_physical_cp_padding(self):
        for size in (2, 4):
            for chunk in (5, 17, 16385):
                with (
                    self.subTest(size=size, chunk=chunk),
                    cp_context(size, 0, (1, chunk - 1), (0, 0)) as (_, batch),
                ):
                    physical = max(batch.attn_cp_metadata.per_rank_actual_token)
                    with self.assertRaisesRegex(ValueError, "required_per_rank"):
                        self.validate_budget(chunk, size, physical - 1)
                    self.validate_budget(chunk, size, physical)

    def test_aligned_buffer_budget_still_supported(self):
        self.validate_budget(16384, 4, 4096)
        self.validate_budget(16384, 2, 8192)


if __name__ == "__main__":
    unittest.main()
