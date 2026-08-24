import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.attn_parallel import (
    AttnParallelMode,
    KvResidency,
    kv_storage_dcp_size,
    resolve_kv_residency,
    select_attn_parallel_mode,
)
from sglang.srt.layers.cp.base import init_cp_strategy
from sglang.srt.layers.cp.utils import is_cp_v2_active, prepare_cp_forward
from sglang.srt.managers.scheduler_components.dp_attn import MLPSyncBatchInfo
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.model_executor.runner.shape_key import ShapeKey
from sglang.srt.runtime_context import get_parallel
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=3, suite="base-a-test-cpu")


class _Mode:
    def __init__(
        self,
        name,
        *,
        cp_extend=False,
        target_verify=False,
        draft_extend_v2=False,
    ):
        self.name = name
        self._cp_extend = cp_extend
        self._target_verify = target_verify
        self._draft_extend_v2 = draft_extend_v2

    def is_context_parallel_extend(self):
        return self._cp_extend

    def is_target_verify(self):
        return self._target_verify

    def is_draft_extend_v2(self):
        return self._draft_extend_v2


class TestAttnParallelModeSelector(unittest.TestCase):
    def test_zigzag_selects_cp_at_floor(self):
        decision = select_attn_parallel_mode(
            forward_mode=_Mode("EXTEND", cp_extend=True),
            extend_seq_lens=[16, 32],
            num_tokens=48,
            strategy="zigzag",
            cp_size=8,
        )
        self.assertEqual(decision.mode, AttnParallelMode.CP)
        self.assertIsNone(decision.veto_reason)

    def test_mixed_is_vetoed_for_interleave(self):
        decision = select_attn_parallel_mode(
            forward_mode=_Mode("MIXED", cp_extend=True),
            extend_seq_lens=[128, 1],
            num_tokens=129,
            strategy="interleave",
            cp_size=8,
        )
        self.assertEqual(decision.mode, AttnParallelMode.TP)
        self.assertEqual(decision.veto_reason, "mixed")

    def test_prefix_hit_short_request_vetoes_zigzag(self):
        decision = select_attn_parallel_mode(
            forward_mode=_Mode("EXTEND", cp_extend=True),
            extend_seq_lens=[256, 8],
            num_tokens=264,
            strategy="zigzag",
            cp_size=8,
        )
        self.assertEqual(decision.mode, AttnParallelMode.TP)
        self.assertEqual(decision.veto_reason, "short_request")

    def test_policy_threshold_is_separate_from_layout_floor(self):
        decision = select_attn_parallel_mode(
            forward_mode=_Mode("EXTEND", cp_extend=True),
            extend_seq_lens=[32],
            num_tokens=32,
            strategy="zigzag",
            cp_size=8,
            min_prefill_tokens=64,
        )
        self.assertEqual(decision.mode, AttnParallelMode.TP)
        self.assertEqual(decision.veto_reason, "below_threshold")

    def test_decode_selects_dcp_only_above_context_threshold(self):
        short = select_attn_parallel_mode(
            forward_mode=_Mode("DECODE"),
            extend_seq_lens=None,
            num_tokens=0,
            strategy="zigzag",
            cp_size=8,
            enable_decode_dcp=True,
            dcp_size=8,
            decode_seq_lens=[4096, 8191],
            min_decode_context=8192,
        )
        long = select_attn_parallel_mode(
            forward_mode=_Mode("DECODE"),
            extend_seq_lens=None,
            num_tokens=0,
            strategy="zigzag",
            cp_size=8,
            enable_decode_dcp=True,
            dcp_size=8,
            decode_seq_lens=[4096, 8192],
            min_decode_context=8192,
        )
        self.assertEqual(short.mode, AttnParallelMode.TP)
        self.assertEqual(short.veto_reason, "short_context")
        self.assertEqual(long.mode, AttnParallelMode.DCP)

    def test_striped_prefill_uses_full_prefix_assembly_path(self):
        decision = select_attn_parallel_mode(
            forward_mode=_Mode("EXTEND", cp_extend=True),
            extend_seq_lens=[32768],
            num_tokens=32768,
            strategy="zigzag",
            cp_size=8,
            enable_decode_dcp=True,
            dcp_size=8,
            kv_residency=KvResidency.STRIPED,
        )
        self.assertEqual(decision.mode, AttnParallelMode.TP)
        self.assertEqual(decision.veto_reason, "striped_prefill")

    def test_stamped_mode_is_authoritative(self):
        self.assertTrue(
            is_cp_v2_active(SimpleNamespace(attn_parallel_mode=AttnParallelMode.CP))
        )
        self.assertFalse(
            is_cp_v2_active(SimpleNamespace(attn_parallel_mode=AttnParallelMode.TP))
        )

    def test_replicated_dynamic_dcp_uses_unscaled_loc_space(self):
        parallel = SimpleNamespace(
            dcp_enabled=True,
            attn_dcp_size=8,
            dynamic_attn_parallel_enable_dcp=True,
        )
        self.assertEqual(resolve_kv_residency(parallel), KvResidency.REPLICATED)
        self.assertEqual(kv_storage_dcp_size(parallel), 1)

        parallel.dynamic_attn_parallel_enable_dcp = False
        self.assertEqual(resolve_kv_residency(parallel), KvResidency.STRIPED)
        self.assertEqual(kv_storage_dcp_size(parallel), 8)

    def test_decode_graph_key_separates_tp_and_dcp(self):
        tp_key = ShapeKey(size=8, attn_parallel_mode=AttnParallelMode.TP.value)
        dcp_key = ShapeKey(size=8, attn_parallel_mode=AttnParallelMode.DCP.value)
        self.assertNotEqual(tp_key, dcp_key)

    def test_mlp_sync_uses_permissive_idle_vote(self):
        info = MLPSyncBatchInfo(
            dp_size=2,
            tp_size=1,
            cp_size=1,
            num_tokens=32,
            num_tokens_for_logprob=1,
            can_run_decode_cuda_graph=False,
            can_run_prefill_cuda_graph=False,
            is_extend_in_batch=True,
            local_can_run_tbo=False,
            local_forward_mode=1,
            local_attn_parallel_mode=AttnParallelMode.TP.value,
        )
        local = info._get_local_tensor("cpu")
        fallback = info._get_fallback_tensor("cpu")
        self.assertEqual(local.numel(), 8)
        self.assertEqual(fallback.numel(), 8)
        self.assertEqual(int(fallback[-1]), AttnParallelMode.DCP.value)
        self.assertEqual(local.dtype, torch.int64)

    def test_cp_metadata_and_cache_locations_survive_mode_flip(self):
        init_cp_strategy(
            SimpleNamespace(
                enable_prefill_cp=True,
                cp_strategy="zigzag",
                attn_cp_size=2,
            )
        )
        forward_batch = SimpleNamespace(
            attn_parallel_mode=AttnParallelMode.CP,
            input_ids=torch.arange(32),
            forward_mode=ForwardMode.EXTEND,
            extend_seq_lens_cpu=[32],
            seq_lens_cpu=torch.tensor([32], dtype=torch.int32),
            attn_cp_metadata=None,
            global_num_tokens_cpu=None,
            out_cache_loc=torch.arange(40),
        )

        try:
            with (
                get_parallel().override(attn_cp_rank=0, attn_cp_size=2),
                patch(
                    "sglang.srt.layers.cp.padding.get_cp_padding_align_size",
                    return_value=4,
                ),
            ):
                prepare_cp_forward(forward_batch)
                metadata = forward_batch.attn_cp_metadata
                cache_locations = forward_batch.out_cache_loc
                prepare_cp_forward(forward_batch)

                self.assertIs(forward_batch.attn_cp_metadata, metadata)
                self.assertIs(forward_batch.out_cache_loc, cache_locations)
                self.assertEqual(cache_locations.tolist(), list(range(32)))

                forward_batch.attn_parallel_mode = AttnParallelMode.TP
                self.assertFalse(is_cp_v2_active(forward_batch))
                self.assertIs(forward_batch.attn_cp_metadata, metadata)
                self.assertIs(forward_batch.out_cache_loc, cache_locations)

                forward_batch.attn_parallel_mode = AttnParallelMode.CP
                self.assertTrue(is_cp_v2_active(forward_batch))
        finally:
            init_cp_strategy(SimpleNamespace(enable_prefill_cp=False))


if __name__ == "__main__":
    unittest.main()
