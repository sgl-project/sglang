import unittest
from unittest import mock

import torch

from sglang.srt.layers.attention.dsa import dsa_indexer as indexer_module
from sglang.srt.layers.attention.dsa.dsa_indexer import Indexer
from sglang.srt.model_executor.runner_utils.capture_owner import (
    collect_full_cuda_graph_owners,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class TestRetainFullGraphCaptureOwner(unittest.TestCase):
    def test_noop_outside_capture_scope(self):
        indexer = object.__new__(Indexer)
        tensor = torch.empty(4)
        result = indexer._retain_full_graph_capture_owner(tensor)
        self.assertIs(result, tensor)

    def test_records_owner_inside_capture_scope(self):
        indexer = object.__new__(Indexer)
        tensor = torch.empty(4)
        with collect_full_cuda_graph_owners() as owners:
            result = indexer._retain_full_graph_capture_owner(tensor)
        self.assertIs(result, tensor)
        self.assertEqual(owners, [tensor])

    def test_retention_survives_enclosing_torch_compile(self):
        # Dynamo traces away plain Python side effects; the helper's
        # torch.compiler.disable boundary must force it to execute.
        indexer = object.__new__(Indexer)

        def outer(value):
            produced = value.float().square()
            produced = indexer._retain_full_graph_capture_owner(produced)
            return produced.to(torch.int32)

        compiled_outer = torch.compile(outer, dynamic=False)
        values = torch.randn(8)
        compiled_outer(values)

        with collect_full_cuda_graph_owners() as owners:
            output = compiled_outer(values)

        self.assertEqual(output.dtype, torch.int32)
        self.assertEqual(len(owners), 1)
        self.assertEqual(owners[0].dtype, torch.float32)
        torch.testing.assert_close(owners[0], values.float().square())

    def test_paged_path_retains_weights_and_logits_before_consumers(self):
        indexer = object.__new__(Indexer)
        indexer.sm_count = 1
        indexer.index_topk = 2048
        indexer.paged_mqa_logits_backend = mock.Mock()
        indexer.paged_mqa_logits_backend.is_cutedsl.return_value = False
        indexer.paged_mqa_logits_backend.is_aiter.return_value = False
        indexer._get_index_k_read_buffer = mock.Mock(
            return_value=torch.empty(1, 64 * 132)
        )
        indexer._mask_init_and_local_tokens = mock.Mock(
            side_effect=lambda logits, seqlens: events.append(("mask", logits))
        )

        q_fp8 = torch.empty(24, 32, 128)
        weights = torch.empty(24, 32, 1)
        seqlens = torch.full((24,), 2240, dtype=torch.int32)
        block_tables = torch.zeros(24, 35, dtype=torch.int32)
        logits = torch.empty(24, 2240)
        topk = torch.empty(24, 2051, dtype=torch.int32)
        events = []

        forward_batch = mock.Mock()
        forward_batch.forward_mode.is_target_verify.return_value = True
        forward_batch.forward_mode.is_draft_extend_v2.return_value = False
        metadata = mock.Mock()
        metadata.get_page_table_64.return_value = block_tables
        metadata.get_seqlens_expanded.return_value = seqlens
        metadata.get_seqlens_int32.return_value = seqlens
        metadata.get_dsa_extend_len_cpu.return_value = [1] * 24
        metadata.paged_mqa_schedule_metadata = torch.empty(1, dtype=torch.int32)

        def topk_transform(candidate, topk_count):
            events.append(("topk", candidate))
            return topk

        metadata.topk_transform.side_effect = topk_transform

        def retain(candidate):
            events.append(("retain", candidate))
            return candidate

        def produce_logits(*args, **kwargs):
            events.append(("produce", args[3]))
            return logits

        indexer._retain_full_graph_capture_owner = retain
        pool = mock.Mock(page_size=64)
        with (
            mock.patch.object(
                indexer_module, "get_token_to_kv_pool", return_value=pool
            ),
            mock.patch.object(indexer_module, "_is_hip", False),
            mock.patch.object(indexer_module, "_is_cuda", True),
            mock.patch.object(indexer_module, "deep_gemm", mock.Mock(), create=True),
            mock.patch.object(
                indexer_module,
                "deepgemm_paged_mqa_logits_split",
                side_effect=produce_logits,
            ),
        ):
            result = indexer._get_topk_paged(forward_batch, 0, q_fp8, weights, metadata)

        self.assertIs(result, topk)
        kinds = [kind for kind, _ in events]
        self.assertEqual(kinds, ["retain", "produce", "retain", "mask", "topk"])
        self.assertIs(events[0][1], weights)
        # The producer receives the squeezed view of the retained owner.
        self.assertEqual(events[1][1].shape, (24, 32))
        self.assertEqual(
            events[1][1].untyped_storage().data_ptr(),
            weights.untyped_storage().data_ptr(),
        )
        self.assertIs(events[2][1], logits)
        self.assertIs(events[3][1], logits)
        self.assertIs(events[4][1], logits)


if __name__ == "__main__":
    unittest.main()
