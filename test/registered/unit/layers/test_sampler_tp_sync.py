import types
import unittest
from unittest.mock import patch

import torch

from sglang.srt.layers import sampler as sampler_module
from sglang.srt.layers.sampler import Sampler
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestSamplerTpSync(CustomTestCase):
    def _sync(self, world_size: int, force_sync: bool, has_grammar: bool):
        group = object()
        sampler = types.SimpleNamespace(
            tp_sync_group=group,
            tp_sync_group_world_size=world_size,
        )
        token_ids = torch.tensor([7], dtype=torch.int64)
        sampling_info = types.SimpleNamespace(
            grammars=[object()] if has_grammar else None
        )

        with (
            patch.object(sampler_module, "SYNC_TOKEN_IDS_ACROSS_TP", force_sync),
            patch.object(torch.distributed, "all_reduce") as all_reduce,
        ):
            Sampler._sync_token_ids_across_tp(sampler, token_ids, sampling_info)

        return all_reduce, token_ids, group

    def test_skips_single_rank_group(self):
        for force_sync in (False, True):
            for has_grammar in (False, True):
                with self.subTest(force_sync=force_sync, has_grammar=has_grammar):
                    all_reduce, token_ids, _ = self._sync(1, force_sync, has_grammar)
                    all_reduce.assert_not_called()
                    torch.testing.assert_close(token_ids, torch.tensor([7]))

    def test_syncs_multi_rank_group(self):
        for force_sync in (False, True):
            for has_grammar in (False, True):
                with self.subTest(force_sync=force_sync, has_grammar=has_grammar):
                    all_reduce, token_ids, group = self._sync(
                        2, force_sync, has_grammar
                    )
                    if force_sync or has_grammar:
                        all_reduce.assert_called_once_with(
                            token_ids,
                            op=torch.distributed.ReduceOp.MIN,
                            group=group,
                        )
                    else:
                        all_reduce.assert_not_called()

    def _make_sampler(
        self,
        tp_size: int,
        dp_attention: bool,
        attn_tp_size: int = 1,
        attn_cp_size: int = 1,
    ):
        parallel = types.SimpleNamespace(
            tp_group=types.SimpleNamespace(device_group=object(), world_size=tp_size),
            attn_tp_group=types.SimpleNamespace(
                device_group=object(), world_size=attn_tp_size
            ),
            attn_cp_size=attn_cp_size,
        )
        if attn_cp_size > 1:
            parallel.attn_cp_group = types.SimpleNamespace(device_group=object())
        execution = types.SimpleNamespace(
            deterministic=types.SimpleNamespace(
                rl_on_policy_target=None, enable_deterministic_inference=False
            ),
            kernel=types.SimpleNamespace(sampling_backend="pytorch"),
            features=types.SimpleNamespace(sampling_mask_max_tokens=8),
        )
        with (
            patch.object(sampler_module, "get_parallel", return_value=parallel),
            patch.object(sampler_module, "get_exec", return_value=execution),
            patch.object(
                sampler_module, "is_dp_attention_enabled", return_value=dp_attention
            ),
            patch.object(sampler_module, "OutputLogprobProcessor"),
        ):
            sampler = Sampler()
        return sampler, parallel

    def _assert_grammar_sync(self, sampler, group):
        token_ids = torch.tensor([7], dtype=torch.int64)
        with (
            patch.object(sampler_module, "SYNC_TOKEN_IDS_ACROSS_TP", False),
            patch.object(torch.distributed, "all_reduce") as all_reduce,
        ):
            sampler._sync_token_ids_across_tp(
                token_ids, types.SimpleNamespace(grammars=[object()])
            )
        if group.world_size == 1:
            all_reduce.assert_not_called()
        else:
            all_reduce.assert_called_once_with(
                token_ids, op=torch.distributed.ReduceOp.MIN, group=group.device_group
            )
        torch.testing.assert_close(token_ids, torch.tensor([7]))

    def test_uses_tp_group_without_dp_attention(self):
        for tp_size in (1, 2):
            with self.subTest(tp_size=tp_size):
                sampler, parallel = self._make_sampler(tp_size, dp_attention=False)
                self._assert_grammar_sync(sampler, parallel.tp_group)
                self.assertIsNone(sampler.cp_sync_group)

    def test_uses_attention_tp_group_with_dp_attention(self):
        for attn_tp_size in (1, 2):
            with self.subTest(attn_tp_size=attn_tp_size):
                sampler, parallel = self._make_sampler(
                    8, dp_attention=True, attn_tp_size=attn_tp_size
                )
                self._assert_grammar_sync(sampler, parallel.attn_tp_group)
                self.assertIsNone(sampler.cp_sync_group)

    def test_preserves_context_parallel_group(self):
        for attn_tp_size in (1, 2):
            with self.subTest(attn_tp_size=attn_tp_size):
                sampler, parallel = self._make_sampler(
                    8, dp_attention=True, attn_tp_size=attn_tp_size, attn_cp_size=2
                )
                self._assert_grammar_sync(sampler, parallel.attn_tp_group)
                self.assertIs(
                    sampler.cp_sync_group, parallel.attn_cp_group.device_group
                )


if __name__ == "__main__":
    unittest.main()
