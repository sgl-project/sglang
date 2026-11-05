import gc
import unittest
import weakref
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.layers.attention.hybrid_linear_attn_backend import MambaAttnBackendBase
from sglang.srt.layers.attention.linear import gdn_backend
from sglang.srt.layers.attention.linear.gdn_backend import build_gdn_mis_metadata
from sglang.srt.model_executor.forward_mode import ForwardMode
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=11, suite="base-a-test-cpu")


class TestGDNMISMetadata(CustomTestCase):
    @staticmethod
    def _batch():
        return SimpleNamespace(
            is_prefill_only=True,
            extend_prefix_lens_cpu=[0, 0],
            extend_seq_lens_cpu=[12, 8],
            multi_item_delimiter_indices=[
                torch.tensor([3, 5, 11]),
                torch.tensor([0, 1, 7]),
            ],
            input_ids=torch.zeros(23, dtype=torch.int64),
            forward_mode=ForwardMode.EXTEND,
            tbo_parent_token_range=None,
        )

    def test_batch_tensors_follow_slot_assignment(self):
        for dtype in (torch.int32, torch.int64):
            with self.subTest(dtype=dtype):
                slots = torch.tensor([1, 3], dtype=dtype)
                first = build_gdn_mis_metadata(
                    self._batch(), cache_indices=slots
                ).batch_tensors
                slots[:] = torch.tensor([3, 1], dtype=dtype)
                second = build_gdn_mis_metadata(
                    self._batch(), cache_indices=slots
                ).batch_tensors
                for tensors, query, items in (
                    (first, [1], [1, 1, 1, 3, 3, 3]),
                    (second, [3], [3, 3, 3, 1, 1, 1]),
                ):
                    self.assertEqual(tensors.query_cache_indices.tolist(), query)
                    self.assertEqual(tensors.item_ssm_indices.tolist(), items)
                    self.assertEqual(tensors.item_conv_indices.tolist(), list(range(6)))
                    self.assertEqual(tensors.query_has_initial_states.tolist(), [False])
                    self.assertEqual(
                        tensors.item_has_initial_states.tolist(), [True] * 6
                    )
                    self.assertEqual(tensors.item_ssm_indices.dtype, dtype)
                self.assertNotEqual(
                    first.item_ssm_indices.data_ptr(),
                    second.item_ssm_indices.data_ptr(),
                )

    def test_invalid_slot_metadata_rejected(self):
        for slots in (
            torch.tensor([1.0, 3.0]),
            torch.tensor([1]),
            torch.tensor([[1, 3]]),
            torch.empty(2, dtype=torch.int64, device="meta"),
        ):
            with self.subTest(
                shape=slots.shape, dtype=slots.dtype, device=slots.device
            ):
                with self.assertRaisesRegex(ValueError, "cache_indices"):
                    build_gdn_mis_metadata(self._batch(), cache_indices=slots)

    def test_batch_tensor_lifecycle_and_layer_reuse(self):
        backend = object.__new__(gdn_backend.GDNAttnBackend)
        backend.enable_mis = True
        slots = torch.tensor([1, 3])
        backend.forward_metadata = SimpleNamespace(
            mamba_cache_indices=slots,
            has_mamba_track_mask=False,
            query_start_loc=torch.tensor([0, 12, 20]),
        )
        batch = self._batch()
        with patch.object(MambaAttnBackendBase, "init_forward_metadata"):
            backend.prefill_graph_metadata = object()
            backend.init_forward_metadata(batch)
            self.assertIsNone(backend.prefill_graph_metadata)
            self.assertFalse(backend._use_strided_target_verify_qkv)
            self.assertIsNone(backend.forward_metadata.conv_block_table)
            metadata = backend.mis_metadata
            observations = []

            def segments(**kwargs):
                observations.append(kwargs)
                kwargs["out"][:, kwargs["output_indices"]] = 7
                # Private item convolution writes must not affect the query's pool.
                if not kwargs["inplace_update"]:
                    kwargs["conv_states"].fill_(11)

            conv = torch.full((5, 12, 3), 5.0)
            ssm = torch.full((5, 2, 4, 4), 5.0)
            with (
                patch.object(backend, "_forward_mis_segments", side_effect=segments),
                patch.object(
                    torch, "arange", side_effect=AssertionError("rebuilt map")
                ),
                patch.object(
                    torch, "zeros", side_effect=AssertionError("rebuilt mask")
                ),
                patch.object(torch, "ones", side_effect=AssertionError("rebuilt mask")),
            ):
                for _ in range(2):
                    output = backend._forward_extend_mis(
                        layer=SimpleNamespace(num_v_heads=2, head_v_dim=4),
                        mixed_qkv=torch.empty(23, 12),
                        a=torch.empty(23, 2),
                        b=torch.empty(23, 2),
                        conv_states=conv,
                        ssm_states=ssm,
                        cache_indices=slots,
                    )
                    self.assertTrue(bool((output[:, :20] == 7).all()))
                    self.assertTrue(bool((output[:, 20:] == 0).all()))
                    self.assertTrue(bool((conv[slots] == 0).all()))
                    self.assertTrue(bool((ssm[slots] == 0).all()))
                    self.assertTrue(bool((conv[[0, 2, 4]] == 5).all()))
                self.assertEqual(len(observations), 4)
                for query, item in zip(observations[::2], observations[1::2]):
                    tensors = metadata.batch_tensors
                    for actual, expected in (
                        (query["conv_cache_indices"], tensors.query_cache_indices),
                        (query["ssm_cache_indices"], tensors.query_cache_indices),
                        (query["has_initial_states"], tensors.query_has_initial_states),
                        (item["conv_cache_indices"], tensors.item_conv_indices),
                        (item["ssm_cache_indices"], tensors.item_ssm_indices),
                        (item["has_initial_states"], tensors.item_has_initial_states),
                        (query["out"], item["out"]),
                        (query["output_indices"], metadata.query_token_indices),
                        (item["output_indices"], metadata.item_token_indices),
                    ):
                        self.assertIs(actual, expected)
            batch.multi_item_delimiter_indices = None
            backend.init_forward_metadata(batch)
            self.assertIsNone(backend.mis_metadata)
            previous_maps = weakref.ref(metadata.batch_tensors.item_ssm_indices)
            observations.clear()
            del metadata, query, item, tensors
            gc.collect()
            self.assertIsNone(previous_maps())
            batch = self._batch()
            batch.multi_item_delimiter_indices = [
                torch.tensor([0, 11]),
                torch.tensor([0, 7]),
            ]
            backend.init_forward_metadata(batch)
            self.assertEqual(
                backend.mis_metadata.batch_tensors.query_cache_indices.numel(), 0
            )

    def test_execution_requires_prepared_batch_tensors(self):
        backend = object.__new__(gdn_backend.GDNAttnBackend)
        backend.mis_metadata = build_gdn_mis_metadata(self._batch())
        with self.assertRaisesRegex(ValueError, "must be prepared"):
            backend._forward_extend_mis(
                layer=None,
                mixed_qkv=None,
                a=None,
                b=None,
                conv_states=None,
                ssm_states=None,
                cache_indices=None,
            )

    def test_allows_trailing_attention_padding(self):
        forward_batch = SimpleNamespace(
            input_ids=torch.empty(8, dtype=torch.int64),
            extend_seq_lens_cpu=[5],
            extend_prefix_lens_cpu=[0],
            multi_item_delimiter_indices=[torch.tensor([1, 4], dtype=torch.int64)],
            is_prefill_only=True,
        )

        metadata = build_gdn_mis_metadata(forward_batch)

        torch.testing.assert_close(
            torch.cat([metadata.query_token_indices, metadata.item_token_indices])
            .sort()
            .values,
            torch.arange(5, dtype=torch.int64),
        )

    def test_mixed_batch_with_empty_query(self):
        forward_batch = SimpleNamespace(
            input_ids=torch.empty(16, dtype=torch.int64),
            extend_seq_lens_cpu=[9, 7],
            extend_prefix_lens_cpu=[0, 0],
            multi_item_delimiter_indices=[
                torch.tensor([0, 3, 8], dtype=torch.int64),
                torch.tensor([4, 6], dtype=torch.int64),
            ],
            is_prefill_only=True,
        )

        metadata = build_gdn_mis_metadata(forward_batch)

        self.assertEqual(metadata.query_seq_lens_cpu, [4])
        self.assertEqual(metadata.item_seq_lens_cpu, [3, 5, 1, 2, 1])
        torch.testing.assert_close(
            metadata.query_token_indices,
            torch.tensor([9, 10, 11, 12], dtype=torch.int64),
        )
        torch.testing.assert_close(
            metadata.query_cu_seqlens,
            torch.tensor([0, 4], dtype=torch.int32),
        )
        torch.testing.assert_close(
            metadata.query_request_indices,
            torch.tensor([1], dtype=torch.int64),
        )
        torch.testing.assert_close(
            metadata.item_token_indices,
            torch.tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 13, 14, 15], dtype=torch.int64),
        )
        torch.testing.assert_close(
            metadata.item_cu_seqlens,
            torch.tensor([0, 3, 8, 9, 11, 12], dtype=torch.int32),
        )
        torch.testing.assert_close(
            metadata.item_request_indices,
            torch.tensor([0, 0, 0, 1, 1], dtype=torch.int64),
        )

    def test_rejects_sequence_lengths_beyond_input(self):
        forward_batch = SimpleNamespace(
            input_ids=torch.empty(4, dtype=torch.int64),
            extend_seq_lens_cpu=[5],
            extend_prefix_lens_cpu=[0],
            multi_item_delimiter_indices=[torch.tensor([1, 4], dtype=torch.int64)],
            is_prefill_only=True,
        )

        with self.assertRaisesRegex(ValueError, "exceed the input tokens"):
            build_gdn_mis_metadata(forward_batch)

    def test_rejects_missing_final_delimiter(self):
        forward_batch = SimpleNamespace(
            input_ids=torch.empty(5, dtype=torch.int64),
            extend_seq_lens_cpu=[5],
            extend_prefix_lens_cpu=[0],
            multi_item_delimiter_indices=[torch.tensor([2, 3], dtype=torch.int64)],
            is_prefill_only=True,
        )

        with self.assertRaisesRegex(ValueError, "final delimiter"):
            build_gdn_mis_metadata(forward_batch)


if __name__ == "__main__":
    unittest.main()
