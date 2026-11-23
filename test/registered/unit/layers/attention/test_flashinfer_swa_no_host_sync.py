"""Regression tests for sync-free FlashInfer SWA index translation and masking.

Static SWA pools (`SWAKVPool`) map full-pool token ids to SWA slots through
`full_to_swa_index_mapping`. Fusing that lookup into `RunnerIds.pack_read_stream`
eliminates the post-gather `kv_indices[:kv_indptr[-1]]` slice that forced two
blocking D2H syncs on every decode step (including CUDA-graph replay) and every
prefill/verify step. Ragged SWA prefill also derives the paged length sum and
custom-mask necessity on the host when `seq_lens_cpu` / `extend_prefix_lens_cpu`
are provided.
"""

import unittest
from types import SimpleNamespace

import torch

from sglang.srt.layers.attention import flashinfer_backend as fb
from sglang.srt.mem_cache.kv.alloc.id_space import window_read
from sglang.srt.mem_cache.kv.alloc.iteration_ids import IterationIds
from sglang.srt.mem_cache.kv.alloc.runner_ids import RunnerIds
from sglang.srt.mem_cache.swa_memory_pool import SWAKVPool
from sglang.srt.speculative.eagle_info import EagleVerifyInput
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=6, stage="base-b", runner_config="1-gpu-small")


def _make_static_swa_fixtures(bs: int = 3, max_seq: int = 32):
    device = torch.device("cuda")
    # Distinct full-pool token ids per (req, pos) plus a -1 sentinel entry.
    req_to_token = (
        torch.arange(1, bs * max_seq + 1, dtype=torch.int32, device=device)
        .view(bs, max_seq)
        .contiguous()
    )
    # Non-identity full -> swa mapping with trailing -1 sentinel matching SWAKVPool.
    full_size = bs * max_seq + 2
    mapping = (torch.arange(full_size, dtype=torch.int64, device=device) * 3 + 7) % 1000
    mapping[-1] = -1

    swa_pool = SWAKVPool.__new__(SWAKVPool)
    swa_pool.full_to_swa_index_mapping = mapping
    swa_pool.request_window = None

    allocator = SimpleNamespace(hands_out_virtual_ids=False)
    runner_ids = RunnerIds(
        req_to_token=req_to_token,
        token_to_kv_pool_allocator=allocator,
        token_to_kv_pool=swa_pool,
        page_size=1,
        device="cuda",
    )
    plan = IterationIds(
        source=runner_ids,
        req_pool_indices=torch.arange(bs, dtype=torch.int32, device=device),
        seq_lens=torch.full((bs,), max_seq, dtype=torch.int32, device=device),
        seq_lens_cpu=None,
        write_virtual=None,
    )

    def _forbid_post_translate(_ids):
        raise AssertionError("post-gather _swa_space.ids() should not be called")

    swa_space = SimpleNamespace(ids=_forbid_post_translate)
    return req_to_token, mapping, swa_pool, runner_ids, plan, swa_space


class _StubWrapper:
    def __init__(self, *, cuda_graph: bool = False, buf_size: int = 256):
        self.is_cuda_graph_enabled = cuda_graph
        self._paged_kv_indices_buf = (
            torch.full((buf_size,), -99, dtype=torch.int32, device="cuda")
            if cuda_graph
            else None
        )
        self._custom_mask_buf = None
        self.calls = []

    def begin_forward(self, *args, **kwargs):
        self.calls.append((args, kwargs))


@unittest.skipUnless(torch.cuda.is_available(), "CUDA is required for this test.")
class TestFlashInferSWANoHostSync(CustomTestCase):
    def test_decode_static_swa_fuses_translation_without_host_sync(self):
        """Decode (eager and CUDA-graph replay) fuses full->swa translation into
        pack_read_stream with zero CUDA syncs and exact index parity."""
        req_to_token, mapping, swa_pool, runner_ids, plan, swa_space = (
            _make_static_swa_fixtures()
        )
        updater = fb.FlashInferIndicesUpdaterDecode.__new__(
            fb.FlashInferIndicesUpdaterDecode
        )
        updater.num_qo_heads = 8
        updater.num_kv_heads = 2
        updater.head_dim = 128
        updater.data_type = torch.bfloat16
        updater.q_data_type = torch.bfloat16
        updater.sliding_window_size = 8
        updater.attn_backend = SimpleNamespace(runner_ids=runner_ids)
        updater.kv_last_page_len = torch.ones(4, dtype=torch.int32, device="cuda")
        updater._swa_kv_pool = swa_pool
        updater._swa_space = swa_space

        req_pool_indices = torch.tensor([0, 1, 2], dtype=torch.int32, device="cuda")
        seq_lens_cpu = torch.tensor([5, 9, 9], dtype=torch.int32)
        paged_kernel_lens = seq_lens_cpu.to("cuda")
        kv_start_idx = torch.tensor([0, 4, 11], dtype=torch.int32, device="cuda")
        expected = torch.cat(
            [
                mapping[req_to_token[r, int(s) : int(s) + int(l)].long()]
                for r, s, l in zip([0, 1, 2], [0, 4, 11], [5, 9, 9])
            ]
        ).to(torch.int32)

        for cuda_graph in (False, True):
            with self.subTest(cuda_graph=cuda_graph):
                wrapper = _StubWrapper(cuda_graph=cuda_graph)
                kv_indptr = torch.zeros(4, dtype=torch.int32, device="cuda")
                torch.cuda.synchronize()
                torch.cuda.set_sync_debug_mode("error")
                try:
                    updater.call_begin_forward(
                        wrapper,
                        paged_kernel_lens,
                        int(seq_lens_cpu.sum()),
                        kv_indptr,
                        kv_start_idx,
                        spec_info=None,
                        seq_lens_cpu=seq_lens_cpu,
                        use_sliding_window_kv_pool=True,
                        plan=plan,
                        req_pool_indices=req_pool_indices,
                    )
                finally:
                    torch.cuda.set_sync_debug_mode("default")
                torch.cuda.synchronize()
                out_kv_indices = wrapper.calls[0][0][1]
                torch.testing.assert_close(out_kv_indices[: expected.numel()], expected)

    def test_prefill_static_swa_fuses_translation_without_host_sync(self):
        """Normal extend fuses full->swa translation into pack_read_stream with
        zero CUDA syncs and exact index parity."""
        req_to_token, mapping, swa_pool, runner_ids, plan, swa_space = (
            _make_static_swa_fixtures()
        )
        updater = fb.FlashInferIndicesUpdaterPrefill.__new__(
            fb.FlashInferIndicesUpdaterPrefill
        )
        updater.num_qo_heads = 8
        updater.num_kv_heads = 2
        updater.head_dim = 128
        updater.data_type = torch.bfloat16
        updater.q_data_type = torch.bfloat16
        updater.sliding_window_size = 8
        updater.attn_backend = SimpleNamespace(
            runner_ids=runner_ids, dq_paged_kernel_lens=None
        )
        updater.kv_last_page_len = torch.ones(4, dtype=torch.int32, device="cuda")
        updater._swa_kv_pool = swa_pool
        updater._swa_space = swa_space

        req_pool_indices = torch.tensor([0, 1, 2], dtype=torch.int32, device="cuda")
        prefix_lens = torch.tensor([4, 12, 0], dtype=torch.int32, device="cuda")
        seq_lens = torch.tensor([7, 15, 5], dtype=torch.int32, device="cuda")
        read = window_read(
            seq_lens=seq_lens, prefix_lens=prefix_lens, window=8, prefix_only=True
        )
        expected = torch.cat(
            [
                mapping[req_to_token[0, 0:4].long()],
                mapping[req_to_token[1, 4:12].long()],
            ]
        ).to(torch.int32)

        wrapper_ragged = _StubWrapper()
        wrapper_paged = _StubWrapper()
        kv_indptr = torch.zeros(4, dtype=torch.int32, device="cuda")
        qo_indptr = torch.zeros(4, dtype=torch.int32, device="cuda")
        torch.cuda.synchronize()
        torch.cuda.set_sync_debug_mode("error")
        try:
            updater.call_begin_forward(
                wrapper_ragged,
                wrapper_paged,
                req_pool_indices,
                read.lens,
                12,
                seq_lens,
                prefix_lens,
                read.start,
                kv_indptr,
                qo_indptr,
                use_ragged=True,
                spec_info=None,
                use_sliding_window_kv_pool=True,
                window=read,
                plan=plan,
            )
        finally:
            torch.cuda.set_sync_debug_mode("default")
        torch.cuda.synchronize()
        out_kv_indices = wrapper_paged.calls[0][0][2]
        torch.testing.assert_close(out_kv_indices[: expected.numel()], expected)

    def test_prefill_spec_verify_static_swa_fuses_translation_without_host_sync(self):
        """Speculative verify (EagleVerifyInput) threads token_mapping into
        pack_read_stream with zero CUDA syncs and exact index parity."""
        req_to_token, mapping, swa_pool, runner_ids, plan, swa_space = (
            _make_static_swa_fixtures()
        )
        updater = fb.FlashInferIndicesUpdaterPrefill.__new__(
            fb.FlashInferIndicesUpdaterPrefill
        )
        updater.num_qo_heads = 8
        updater.num_kv_heads = 2
        updater.head_dim = 128
        updater.data_type = torch.bfloat16
        updater.q_data_type = torch.bfloat16
        updater.sliding_window_size = 8
        updater.attn_backend = SimpleNamespace(
            runner_ids=runner_ids, dq_paged_kernel_lens=None
        )
        updater.kv_last_page_len = torch.ones(4, dtype=torch.int32, device="cuda")
        updater._swa_kv_pool = swa_pool
        updater._swa_space = swa_space

        draft_token_num = 3
        req_pool_indices = torch.tensor([0, 1], dtype=torch.int32, device="cuda")
        seq_lens = torch.tensor([5, 14], dtype=torch.int32, device="cuda")
        read = window_read(
            seq_lens=seq_lens, prefix_lens=seq_lens, window=8, prefix_only=False
        )
        # req 0: start=0, len=5+3=8; req 1: start=6, len=8+3=11.
        expected = torch.cat(
            [
                mapping[req_to_token[0, 0:8].long()],
                mapping[req_to_token[1, 6:17].long()],
            ]
        ).to(torch.int32)

        spec_info = EagleVerifyInput.__new__(EagleVerifyInput)
        spec_info.spec_input_type = fb.SpecInputType.EAGLE_VERIFY
        spec_info.draft_token_num = draft_token_num
        spec_info.num_tokens_per_req = draft_token_num
        spec_info.custom_mask = torch.ones(512, dtype=torch.bool, device="cuda")

        wrapper_ragged = _StubWrapper()
        wrapper_paged = _StubWrapper()
        kv_indptr = torch.zeros(3, dtype=torch.int32, device="cuda")
        qo_indptr = torch.zeros(3, dtype=torch.int32, device="cuda")
        torch.cuda.synchronize()
        torch.cuda.set_sync_debug_mode("error")
        try:
            updater.call_begin_forward(
                wrapper_ragged,
                wrapper_paged,
                req_pool_indices,
                read.lens,
                13,
                seq_lens,
                seq_lens,
                read.start,
                kv_indptr,
                qo_indptr,
                use_ragged=False,
                spec_info=spec_info,
                use_sliding_window_kv_pool=True,
                window=read,
                plan=plan,
            )
        finally:
            torch.cuda.set_sync_debug_mode("default")
        torch.cuda.synchronize()
        out_kv_indices = wrapper_paged.calls[0][0][2]
        torch.testing.assert_close(out_kv_indices[: expected.numel()], expected)

    def test_ragged_swa_prefix_custom_mask_no_host_sync_and_exact_parity(self):
        """Ragged SWA prefix mask uses host lengths and an analytical boundary
        check with zero CUDA syncs, matching the dense reference bit for bit."""
        updater = fb.FlashInferIndicesUpdaterPrefill.__new__(
            fb.FlashInferIndicesUpdaterPrefill
        )
        window = 4
        updater.sliding_window_size = window

        cases = (
            # All queries within window (exact boundary: kv_start == p + e - 1 - w).
            ("all_in_window_boundary", [0, 3, 10], [5, 2, 1], False),
            # One token past window on the second request.
            ("one_past_window", [3, 10], [2, 2], True),
            # Mixed zero-prefix, short prefix, and long prefix straddling window.
            ("mixed_batch", [0, 2, 10], [6, 5, 4], True),
        )
        for name, prefix_list, extend_list, expect_mask in cases:
            with self.subTest(case=name):
                seq_list = [p + e for p, e in zip(prefix_list, extend_list)]
                prefix_t = torch.tensor(prefix_list, dtype=torch.int32, device="cuda")
                seq_t = torch.tensor(seq_list, dtype=torch.int32, device="cuda")
                seq_cpu = torch.tensor(seq_list, dtype=torch.int32)
                kv_start_t = torch.clamp(prefix_t - window, min=0)

                torch.cuda.synchronize()
                torch.cuda.set_sync_debug_mode("error")
                try:
                    mask = updater._build_swa_prefix_custom_mask(
                        prefix_t,
                        seq_t,
                        kv_start_t,
                        seq_lens_cpu=seq_cpu,
                        extend_prefix_lens_cpu=prefix_list,
                    )
                finally:
                    torch.cuda.set_sync_debug_mode("default")
                torch.cuda.synchronize()

                if not expect_mask:
                    self.assertIsNone(mask)
                else:
                    self.assertIsNotNone(mask)
                    ref_parts = []
                    for p, e in zip(prefix_list, extend_list):
                        s = max(0, p - window)
                        paged_len = p - s
                        if paged_len == 0 or e == 0:
                            continue
                        q_abs = torch.arange(e, device="cuda").view(-1, 1) + p
                        k_abs = torch.arange(paged_len, device="cuda").view(1, -1) + s
                        ref_parts.append(
                            (k_abs >= (q_abs - window)).to(torch.uint8).view(-1)
                        )
                    torch.testing.assert_close(mask, torch.cat(ref_parts))


if __name__ == "__main__":
    unittest.main()
