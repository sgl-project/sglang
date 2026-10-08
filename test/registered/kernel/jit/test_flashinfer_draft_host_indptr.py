"""Draft planning matches device metadata across live lengths and cache slots."""

import unittest
from types import SimpleNamespace

import torch

from sglang.kernels.ops.speculative.cache_locs import generate_draft_decode_kv_indices
from sglang.srt.environ import envs
from sglang.srt.layers.attention import flashinfer_backend as backend
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=30, stage="base-b-kernel-unit", runner_config="1-gpu-large")


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
class TestFlashInferDraftHostIndptr(CustomTestCase):
    def _build(self, topk, steps, page_size, lengths):
        n = len(lengths)
        pool_len = 16640
        pool = torch.randperm((n + 1) * pool_len, device="cuda", dtype=torch.int32)
        pool = pool.reshape(n + 1, pool_len)
        obj = backend.FlashInferMultiStepDraftBackend.__new__(
            backend.FlashInferMultiStepDraftBackend
        )
        obj.topk = topk
        obj.speculative_num_steps = steps
        obj.generate_draft_decode_kv_indices = generate_draft_decode_kv_indices
        obj.page_size = page_size
        obj.pool_len = pool_len
        obj.req_to_token_pool = SimpleNamespace(req_to_token=pool)
        obj.kv_indptr = torch.zeros(
            (steps, n * topk + 1), device="cuda", dtype=torch.int32
        )
        indices = torch.full(
            (steps, n * topk * pool_len), -17, device="cuda", dtype=torch.int32
        )
        host = torch.tensor(lengths, dtype=torch.int64)
        seq = host.cuda()
        fb = SimpleNamespace(
            batch_size=n,
            seq_lens=seq,
            seq_lens_cpu=host,
            seq_lens_sum=sum(lengths),
            req_pool_indices=torch.arange(n, device="cuda", dtype=torch.int32),
            positions=seq.repeat_interleave(topk),
            spec_info=SimpleNamespace(is_draft_input=lambda: True),
        )
        return obj, indices, fb

    def test_production_metadata_and_live_graph_after_length_rollback(self):
        for topk, steps, page_size, initial in [
            (1, 3, 1, [8192]),
            (1, 3, 64, [16300, 1, 32]),
            (2, 5, 64, [17, 4097, 1]),
            (4, 3, 1, [0, 8192]),
            (1, 1, 1, [1]),
        ]:
            with self.subTest(topk=topk, steps=steps, page_size=page_size):
                obj, indices, fb = self._build(topk, steps, page_size, initial)
                observed = []

                def plan(i, batch):
                    observed.append(backend.global_override_indptr_cpu.clone())

                with envs.SGLANG_FLASHINFER_DRAFT_HOST_INDPTR.override(True):
                    obj.common_template(fb, indices, plan)
                graph = torch.cuda.CUDAGraph()
                consumer = torch.empty_like(indices)
                consumer_indptr = torch.empty_like(obj.kv_indptr)
                with torch.cuda.graph(graph):
                    consumer.copy_(indices)
                    consumer_indptr.copy_(obj.kv_indptr)

                for case in range(4):
                    # Shrink, grow, then remap cache slots; padded length-one
                    # rows and zero prefixes are mixed with long contexts.
                    live = [max(0, x - 7 * case) for x in initial]
                    if case == 2:
                        live = [min(16300, x + 13) for x in live]
                    fb.seq_lens_cpu.copy_(torch.tensor(live))
                    fb.seq_lens.copy_(fb.seq_lens_cpu)
                    fb.seq_lens_sum = sum(live)
                    fb.positions.copy_(fb.seq_lens.repeat_interleave(topk))
                    fb.req_pool_indices.add_(1).remainder_(len(initial) + 1)
                    indices.fill_(-17)
                    with envs.SGLANG_FLASHINFER_DRAFT_HOST_INDPTR.override(False):
                        obj.common_template(fb, indices, lambda i, batch: None)
                    original = indices.clone()
                    original_indptr = obj.kv_indptr.clone()
                    indices.fill_(-17)
                    observed.clear()
                    with envs.SGLANG_FLASHINFER_DRAFT_HOST_INDPTR.override(True):
                        obj.common_template(fb, indices, plan)
                    torch.testing.assert_close(indices, original, rtol=0, atol=0)
                    torch.testing.assert_close(
                        obj.kv_indptr, original_indptr, rtol=0, atol=0
                    )
                    for i, host in enumerate(observed):
                        torch.testing.assert_close(
                            host, original_indptr[i].cpu(), rtol=0, atol=0
                        )
                    graph.replay()
                    torch.testing.assert_close(consumer, original, rtol=0, atol=0)
                    torch.testing.assert_close(
                        consumer_indptr, original_indptr, rtol=0, atol=0
                    )

    def test_missing_host_mirror_and_failed_plan_restore_override(self):
        obj, indices, fb = self._build(1, 3, 1, [8192])
        with envs.SGLANG_FLASHINFER_DRAFT_HOST_INDPTR.override(False):
            obj.common_template(fb, indices, lambda i, batch: None)
        expected = indices.clone()
        fb.seq_lens_cpu = None
        indices.fill_(-17)
        sentinel = torch.tensor([123], dtype=torch.int32)
        previous = backend.global_override_indptr_cpu
        backend.global_override_indptr_cpu = sentinel
        try:
            with envs.SGLANG_FLASHINFER_DRAFT_HOST_INDPTR.override(True):
                obj.common_template(fb, indices, lambda i, batch: None)
                self.assertIs(backend.global_override_indptr_cpu, sentinel)
                torch.testing.assert_close(indices, expected, rtol=0, atol=0)

                def fail(i, batch):
                    raise RuntimeError("test plan failure")

                with self.assertRaisesRegex(RuntimeError, "test plan failure"):
                    obj.common_template(fb, indices, fail)
                self.assertIs(backend.global_override_indptr_cpu, sentinel)
        finally:
            backend.global_override_indptr_cpu = previous


if __name__ == "__main__":
    unittest.main()
