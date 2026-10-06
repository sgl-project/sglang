"""Cached attention and host metadata ownership must survive replanning."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch
import torch.nn.functional as F
from flashinfer import BatchPrefillWithPagedKVCacheWrapper

from sglang.srt.layers.attention.flashinfer_backend import (
    FlashInferIndicesUpdaterPrefill,
)
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=15, stage="base-b", runner_config="1-gpu-small")


class TestFlashInferPrefillCpuMetadata(CustomTestCase):
    def setUp(self):
        torch.manual_seed(42)
        self.updater = FlashInferIndicesUpdaterPrefill.__new__(
            FlashInferIndicesUpdaterPrefill
        )
        self.updater.num_qo_heads = 4
        self.updater.num_kv_heads = 2
        self.updater.head_dim = 64
        self.updater.data_type = torch.bfloat16
        self.updater.q_data_type = torch.bfloat16
        self.updater.kv_indptr = [torch.zeros(5, dtype=torch.int32, device="cuda")]
        self.updater.qo_indptr = [torch.zeros(5, dtype=torch.int32, device="cuda")]
        self.updater.kv_last_page_len = torch.ones(4, dtype=torch.int32, device="cuda")
        self.updater.prefill_wrapper_ragged = None
        self.updater._swa_kv_pool = None
        self.updater.attn_backend = SimpleNamespace(dq_paged_kernel_lens=None)
        self.wrapper = BatchPrefillWithPagedKVCacheWrapper(
            torch.empty(32 * 1024 * 1024, dtype=torch.uint8, device="cuda"),
            kv_layout="NHD",
            backend="fa2",
        )

    def _inputs(self, lengths, prefixes):
        total = sum(lengths)
        query_total = sum(length - prefix for length, prefix in zip(lengths, prefixes))
        q = torch.randn(query_total, 4, 64, device="cuda", dtype=torch.bfloat16)
        k = torch.randn(total, 1, 2, 64, device="cuda", dtype=torch.bfloat16)
        v = torch.randn_like(k)
        # Non-contiguous physical placement prevents a wrong identity mapping
        # from producing the same attention result as the reference.
        indices = torch.randperm(total, device="cuda").to(torch.int32)

        def fill_indices(*, out, total_tokens, **kwargs):
            out[:total_tokens].copy_(indices)

        self.updater.attn_backend.kv_index_translator = SimpleNamespace(
            reads_are_translated=False, fill_packed_read_stream=fill_indices
        )
        return q, k, v, indices

    def _reference(self, q, k, v, indices, lengths, prefixes):
        outputs = []
        q_start = kv_start = 0
        for length, prefix in zip(lengths, prefixes):
            q_len = length - prefix
            selected = indices[kv_start : kv_start + length].long()
            q_req = q[q_start : q_start + q_len].float().transpose(0, 1)
            k_req = k[selected, 0].float().repeat_interleave(2, dim=1).transpose(0, 1)
            v_req = v[selected, 0].float().repeat_interleave(2, dim=1).transpose(0, 1)
            # Cached-prefix causal masking is bottom-right aligned, including
            # the mixed batch's decode row where q_len is exactly one.
            mask = torch.arange(length, device="cuda")[None, :] <= (
                torch.arange(q_len, device="cuda")[:, None] + prefix
            )
            outputs.append(
                F.scaled_dot_product_attention(
                    q_req, k_req, v_req, attn_mask=mask
                ).transpose(0, 1)
            )
            q_start += q_len
            kv_start += length
        return torch.cat(outputs)

    def _plan(self, lengths, prefixes, use_cpu_metadata, indices, fallback=None):
        self.updater.use_cpu_metadata = use_cpu_metadata
        # Already-pinned int32 inputs catch accidental borrowing: neither a
        # dtype conversion nor pin_memory() would copy this caller-owned data.
        seq_cpu = torch.tensor(lengths, dtype=torch.int32, pin_memory=True)
        prefix_cpu = list(prefixes)
        seq_gpu = seq_cpu.to(device="cuda", dtype=torch.int32)
        prefix_gpu = torch.tensor(prefixes, dtype=torch.int32, device="cuda")
        captured = []
        original_plan = self.wrapper.begin_forward

        def record_plan(*args, **kwargs):
            captured.append((args, kwargs))
            return original_plan(*args, **kwargs)

        host_lengths = seq_cpu
        if fallback == "missing_lengths":
            host_lengths = None
        elif fallback == "device_lengths":
            host_lengths = seq_gpu
        with patch.object(self.wrapper, "begin_forward", record_plan):
            self.updater.update_single_wrapper(
                req_pool_indices=torch.arange(
                    len(lengths), device="cuda", dtype=torch.int32
                ),
                seq_lens=seq_gpu,
                seq_lens_cpu=host_lengths,
                seq_lens_sum=sum(lengths),
                prefix_lens=prefix_gpu,
                prefill_wrappers=[self.wrapper],
                use_ragged=False,
                encoder_lens=None,
                spec_info=None,
                extend_prefix_lens_cpu=(
                    None if fallback == "missing_prefixes" else prefix_cpu
                ),
                custom_kv_indices=indices if fallback == "custom_indices" else None,
            )

        self.assertEqual(len(captured), 1)
        # Mutating the caller's host buffers must not alter a submitted plan,
        # including asynchronous H2D copies still reading pinned snapshots.
        seq_cpu.fill_(1)
        prefix_cpu[:] = [0] * len(prefix_cpu)
        return captured[0]

    def test_mixed_cached_attention_and_resized_batch_snapshots(self):
        """ON/OFF and SDPA agree across mixed, uncached, and resized batches."""
        snapshots = []
        for lengths, prefixes in (
            ([585, 586], [304, 585]),
            ([17], [0]),
            ([33, 9, 24], [16, 8, 0]),
        ):
            with self.subTest(lengths=lengths, prefixes=prefixes):
                q, k, v, indices = self._inputs(lengths, prefixes)
                expected = self._reference(q, k, v, indices, lengths, prefixes)
                outputs = []
                for enabled in (False, True):
                    args, kwargs = self._plan(lengths, prefixes, enabled, indices)
                    metadata = (args[0], args[1], args[3])
                    if enabled:
                        self.assertIn("seq_lens", kwargs)
                        metadata += (kwargs["seq_lens"],)
                        self.assertEqual(kwargs["seq_lens"].tolist(), lengths)
                    self.assertEqual(args[7], 1)  # FlashInfer uses token-sized pages.
                    self.assertEqual(args[2].device.type, "cuda")
                    for tensor in metadata:
                        self.assertEqual(
                            tensor.device.type, "cpu" if enabled else "cuda"
                        )
                        self.assertEqual(tensor.dtype, torch.int32)
                        if enabled:
                            self.assertTrue(tensor.is_pinned())
                    if enabled:
                        snapshots.extend(
                            (tensor, tensor.clone()) for tensor in metadata
                        )
                    output = self.wrapper.forward(q, (k, v), causal=True)
                    torch.testing.assert_close(
                        output.float(), expected, atol=0.02, rtol=0.02
                    )
                    outputs.append(output)
                torch.testing.assert_close(outputs[0], outputs[1], atol=0, rtol=0)
                for original, saved in snapshots:
                    torch.testing.assert_close(original, saved, atol=0, rtol=0)

    def test_fallback_keeps_device_metadata_and_attention(self):
        """Missing host data and custom KV layouts must retain the legacy path."""
        lengths, prefixes = [33, 9], [16, 8]
        q, k, v, indices = self._inputs(lengths, prefixes)
        expected = self._reference(q, k, v, indices, lengths, prefixes)
        for fallback in (
            "missing_lengths",
            "device_lengths",
            "missing_prefixes",
            "custom_indices",
        ):
            with self.subTest(fallback=fallback):
                args, _ = self._plan(
                    lengths, prefixes, True, indices, fallback=fallback
                )
                for tensor in (args[0], args[1], args[3]):
                    self.assertEqual(tensor.device.type, "cuda")
                output = self.wrapper.forward(q, (k, v), causal=True)
                torch.testing.assert_close(
                    output.float(), expected, atol=0.02, rtol=0.02
                )


if __name__ == "__main__":
    unittest.main()
