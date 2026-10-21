"""Host planning preserves live draft indices, verify masks and graph attention."""

import unittest
from functools import partial
from types import SimpleNamespace

import flashinfer
import torch
from flashinfer.quantization import segment_packbits

from sglang.kernels.ops.speculative.cache_locs import generate_draft_decode_kv_indices
from sglang.kernels.ops.speculative.pack_verify_mask import pack_verify_mask_into
from sglang.srt.environ import envs
from sglang.srt.layers.attention import flashinfer_backend as backend
from sglang.srt.layers.attention.flashinfer_backend import (
    FlashInferIndicesUpdaterPrefill,
    fast_prefill_plan,
)
from sglang.srt.speculative.eagle_info import EagleVerifyInput
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=70, stage="base-b-kernel-unit", runner_config="1-gpu-large")


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
        obj.draft_window_size = obj.draft_sink_size = 0
        obj.kv_index_translator = SimpleNamespace(
            read_source=lambda plan, **kw: SimpleNamespace(
                row_ids=kw["req_pool_indices"],
                ids=pool,
                row_stride=pool_len,
                entry_page_size=1,
                v2p=None,
            )
        )
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
            kv_loc_plan=None,
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

    def test_sliding_window_uses_bounded_device_metadata(self):
        obj, indices, fb = self._build(1, 3, 1, [8192, 1024])
        obj.draft_window_size = 256
        obj.draft_sink_size = 16
        with envs.SGLANG_FLASHINFER_DRAFT_HOST_INDPTR.override(False):
            obj.common_template(fb, indices, lambda i, batch: None)
        expected = indices.clone()
        expected_indptr = obj.kv_indptr.clone()
        observed = []
        indices.fill_(-17)
        with envs.SGLANG_FLASHINFER_DRAFT_HOST_INDPTR.override(True):
            obj.common_template(
                fb,
                indices,
                lambda i, batch: observed.append(
                    backend.global_override_indptr_cpu.clone()
                ),
            )
        torch.testing.assert_close(indices, expected, rtol=0, atol=0)
        torch.testing.assert_close(obj.kv_indptr, expected_indptr, rtol=0, atol=0)
        for i, host in enumerate(observed):
            torch.testing.assert_close(host, expected_indptr[i].cpu(), rtol=0, atol=0)

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


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
class TestFlashInferVerifyHostPlan(CustomTestCase):
    def test_packed_bits_and_padding(self):
        for width, lengths in [(4, [8196]), (3, [1, 7, 8191]), (5, [13, 17, 16379])]:
            with self.subTest(width=width, lengths=lengths):
                indptr = (
                    torch.tensor([0] + lengths, dtype=torch.int32)
                    .cumsum(0)
                    .int()
                    .cuda()
                )
                mask = torch.randint(
                    0, 2, (sum(lengths) * width,), device="cuda"
                ).bool()
                expected, packed_indptr = segment_packbits(
                    mask, indptr * width, bitorder="little"
                )
                result = torch.full(
                    (expected.numel() + 256,), 173, dtype=torch.uint8, device="cuda"
                )
                pack_verify_mask_into(
                    mask,
                    indptr,
                    packed_indptr,
                    result,
                    batch_size=len(lengths),
                    query_width=width,
                    max_kv_len=max(lengths),
                )
                torch.testing.assert_close(
                    result[: expected.numel()], expected, rtol=0, atol=0
                )
                self.assertTrue(torch.all(result[expected.numel() :] == 173).item())

    def test_real_wrapper_live_lengths_and_masks(self):
        torch.manual_seed(20261008)
        for dtype, batch in [(torch.bfloat16, 1), (torch.float8_e4m3fn, 3)]:
            with self.subTest(dtype=dtype, batch=batch):
                width, capacity, heads, kv_heads, dim = 4, 9216, 32, 2, 128
                q = torch.randn(
                    batch * width, heads, dim, device="cuda", dtype=torch.bfloat16
                )
                kv = torch.randn(
                    capacity, 2, 1, kv_heads, dim, device="cuda", dtype=torch.bfloat16
                ).to(dtype)
                wrappers = []
                for _ in range(2):
                    wrappers.append(
                        flashinfer.BatchPrefillWithPagedKVCacheWrapper(
                            torch.empty(
                                512 * 1024 * 1024, dtype=torch.uint8, device="cuda"
                            ),
                            use_cuda_graph=True,
                            qo_indptr_buf=torch.empty(
                                batch + 1, dtype=torch.int32, device="cuda"
                            ),
                            paged_kv_indptr_buf=torch.empty(
                                batch + 1, dtype=torch.int32, device="cuda"
                            ),
                            paged_kv_indices_buf=torch.empty(
                                capacity * batch, dtype=torch.int32, device="cuda"
                            ),
                            paged_kv_last_page_len_buf=torch.ones(
                                batch, dtype=torch.int32, device="cuda"
                            ),
                            custom_mask_buf=torch.full(
                                (capacity * batch * width,),
                                173,
                                dtype=torch.uint8,
                                device="cuda",
                            ),
                            mask_indptr_buf=torch.empty(
                                batch + 1, dtype=torch.int32, device="cuda"
                            ),
                            backend="fa2",
                        )
                    )
                graphs, outputs = [], []
                for step, first in enumerate([8196, 8189, 8213, 13, 9213]):
                    lengths = [max(width, first - i * 13) for i in range(batch)]
                    host_lens = torch.tensor(lengths, dtype=torch.int32)
                    host_kv = torch.cat(
                        [torch.zeros(1, dtype=torch.int32), host_lens.cumsum(0).int()]
                    )
                    host_q = torch.arange(batch + 1, dtype=torch.int32) * width
                    gpu_kv, gpu_q = host_kv.cuda(), host_q.cuda()
                    indices = torch.cat(
                        [
                            torch.randperm(capacity, device="cuda", dtype=torch.int32)[
                                :n
                            ]
                            for n in lengths
                        ]
                    )
                    masks = []
                    for n in lengths:
                        # Actual live masks may differ after rejection. Random
                        # branch visibility exercises more than a causal mask.
                        row = torch.rand(width, n, device="cuda") > 0.15
                        row[:, 0] = True
                        row[:, -width:] = torch.ones(
                            width, width, device="cuda", dtype=torch.bool
                        ).tril()
                        masks.append(row.flatten())
                    mask = torch.cat(masks)
                    last = torch.ones(batch, dtype=torch.int32, device="cuda")
                    args = (gpu_q, gpu_kv, indices, last, heads, kv_heads, dim, 1)
                    kwargs = {
                        "q_data_type": q.dtype,
                        "kv_data_type": dtype,
                        "custom_mask": mask,
                    }
                    wrappers[0].plan(*args, **kwargs)
                    if step == 0:
                        wrappers[1].plan(*args, **kwargs)
                        wrappers[1].begin_forward = partial(
                            fast_prefill_plan, wrappers[1]
                        )
                    else:
                        # Exercise the SGLang entry point, including the
                        # committed-prefix + candidate-window host layout.
                        pool = torch.empty(
                            batch, capacity, dtype=torch.int32, device="cuda"
                        )
                        for i, n in enumerate(lengths):
                            pool[i, :n] = indices[int(host_kv[i]) : int(host_kv[i + 1])]
                        spec = EagleVerifyInput.__new__(EagleVerifyInput)
                        from sglang.srt.speculative.spec_info import SpecInputType

                        spec.spec_input_type = SpecInputType.EAGLE_VERIFY
                        spec.custom_mask = mask
                        spec.draft_token_num = spec.num_tokens_per_req = width
                        updater = FlashInferIndicesUpdaterPrefill.__new__(
                            FlashInferIndicesUpdaterPrefill
                        )

                        def pack_read_stream(plan, **kw):
                            lengths_cpu = kw["seq_lens"].cpu().tolist()
                            rows_cpu = kw["req_pool_indices"].cpu().tolist()
                            kw["out"].copy_(
                                torch.cat(
                                    [
                                        pool[row, :n]
                                        for row, n in zip(rows_cpu, lengths_cpu)
                                    ]
                                )
                            )

                        updater.attn_backend = SimpleNamespace(
                            kv_index_translator=SimpleNamespace(
                                reads_are_translated=False,
                                pack_read_stream=pack_read_stream,
                            )
                        )
                        updater.req_to_token = pool
                        updater.kv_last_page_len = last
                        updater.num_qo_heads, updater.num_kv_heads = heads, kv_heads
                        updater.head_dim, updater.q_data_type, updater.data_type = (
                            dim,
                            q.dtype,
                            dtype,
                        )
                        committed = host_lens - width
                        updater.call_begin_forward(
                            None,
                            wrappers[1],
                            torch.arange(batch, dtype=torch.int32, device="cuda"),
                            committed.cuda(),
                            int(committed.sum()),
                            committed.cuda(),
                            None,
                            None,
                            gpu_kv,
                            gpu_q,
                            False,
                            spec,
                            seq_lens_cpu=committed,
                            plan=None,
                        )
                    expected_bits, expected_ptr = segment_packbits(
                        mask, gpu_kv * width, bitorder="little"
                    )
                    self.assertEqual(wrappers[0]._plan_info, wrappers[1]._plan_info)
                    for wrapper in wrappers:
                        torch.testing.assert_close(
                            wrapper._mask_indptr_buf, expected_ptr, rtol=0, atol=0
                        )
                        torch.testing.assert_close(
                            wrapper._custom_mask_buf[: expected_bits.numel()],
                            expected_bits,
                            rtol=0,
                            atol=0,
                        )
                    torch.testing.assert_close(
                        wrappers[0].run(q, kv), wrappers[1].run(q, kv), rtol=0, atol=0
                    )
                    if step == 0:
                        for wrapper in wrappers:
                            graph = torch.cuda.CUDAGraph()
                            with torch.cuda.graph(graph):
                                out = wrapper.run(q, kv)
                            graphs.append(graph)
                            outputs.append(out)
                    q.normal_()
                    for graph in graphs:
                        graph.replay()
                    torch.testing.assert_close(outputs[0], outputs[1], rtol=0, atol=0)
                    torch.testing.assert_close(
                        outputs[1], wrappers[0].run(q, kv), rtol=0, atol=0
                    )


if __name__ == "__main__":
    unittest.main()
