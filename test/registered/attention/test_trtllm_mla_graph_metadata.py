"""Correctness tests for the fused TRTLLM-MLA cuda-graph metadata kernel."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

import sglang.srt.layers.attention.trtllm_mla_backend as trtllm_mla_backend
from sglang.kernels.ops.kvcache.kv_indices import (
    create_flashmla_kv_indices_triton,
    get_num_kv_index_blocks_flashmla,
)
from sglang.kernels.ops.kvcache.trtllm_mla_graph_metadata import (
    update_trtllm_mla_graph_metadata,
)
from sglang.srt.layers.attention.trtllm_mla_backend import TRTLLMMLABackend
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=30, stage="base-b", runner_config="1-gpu-large")

DEVICE = "cuda"


def _make_backend_for_hook_test(page_size=64, num_draft_tokens=4, device="cpu"):
    backend = TRTLLMMLABackend.__new__(TRTLLMMLABackend)
    backend.device = torch.device(device)
    backend.max_context_len = 256
    backend.page_size = page_size
    backend.num_draft_tokens = num_draft_tokens
    backend.req_to_token = torch.zeros(4, 256, dtype=torch.int32, device=device)
    backend.decode_cuda_graph_metadata = {}
    max_blocks = backend._calc_padded_blocks(backend.max_context_len)
    backend.decode_cuda_graph_kv_indices = torch.full(
        (4, max_blocks), -1, dtype=torch.int32, device=device
    )
    backend.forward_decode_metadata = None
    return backend


def _make_fb(bs, forward_mode, device="cpu"):
    return SimpleNamespace(
        batch_size=bs,
        req_pool_indices=torch.arange(bs, dtype=torch.int64, device=device),
        seq_lens=torch.ones(bs, dtype=torch.int64, device=device),
        forward_mode=forward_mode,
        spec_info=None,
        positions=torch.arange(bs, dtype=torch.int64, device=device),
    )


class TestTRTLLMMLAGraphMetadataHooks(CustomTestCase):
    def _patched_calls(self):
        calls = []
        patcher = patch.object(
            trtllm_mla_backend,
            "update_trtllm_mla_graph_metadata",
            lambda **kwargs: calls.append(kwargs),
        )
        self.addCleanup(patcher.stop)
        patcher.start()
        return calls

    def test_decode_rebuild_runs_in_graph_hook_only(self):
        calls = self._patched_calls()
        backend = _make_backend_for_hook_test()
        fb = _make_fb(bs=2, forward_mode=ForwardMode.DECODE)

        backend.init_forward_metadata_out_graph(fb, in_capture=True)
        self.assertEqual(calls, [])
        self.assertIs(
            backend.forward_decode_metadata, backend.decode_cuda_graph_metadata[2]
        )

        backend.init_forward_metadata_in_graph(fb)
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0]["seqlen_offset"], 0)
        self.assertIsNone(calls[0]["seq_lens_k"])

        calls.clear()
        backend.forward_decode_metadata = None
        backend.init_forward_metadata_out_graph(fb)
        self.assertEqual(calls, [])
        self.assertIs(
            backend.forward_decode_metadata, backend.decode_cuda_graph_metadata[2]
        )

    def test_target_verify_applies_draft_token_offset(self):
        calls = self._patched_calls()
        backend = _make_backend_for_hook_test(num_draft_tokens=4)
        fb = _make_fb(bs=2, forward_mode=ForwardMode.TARGET_VERIFY)

        backend.init_forward_metadata_out_graph(fb, in_capture=True)
        backend.init_forward_metadata_in_graph(fb)

        metadata = backend.decode_cuda_graph_metadata[2]
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0]["seqlen_offset"], 4)
        self.assertIs(calls[0]["seq_lens_k"], metadata.seq_lens_k)

    def test_draft_extend_v2_uses_static_q_metadata(self):
        calls = self._patched_calls()
        backend = _make_backend_for_hook_test(num_draft_tokens=4)
        fb = _make_fb(bs=2, forward_mode=ForwardMode.DRAFT_EXTEND_V2)

        backend.init_forward_metadata_out_graph(fb, in_capture=True)
        backend.init_forward_metadata_in_graph(fb)

        metadata = backend.decode_cuda_graph_metadata[2]
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0]["seqlen_offset"], 0)
        self.assertIs(calls[0]["seq_lens_k"], metadata.seq_lens_k)
        self.assertEqual(metadata.max_seq_len_q, 4)
        self.assertEqual(metadata.sum_seq_lens_q, 8)
        torch.testing.assert_close(
            metadata.cu_seqlens_q,
            torch.tensor([0, 4, 8], dtype=torch.int32),
            rtol=0,
            atol=0,
        )


@unittest.skipIf(not torch.cuda.is_available(), "CUDA required")
class TestTRTLLMMLAGraphMetadataKernel(CustomTestCase):
    def _run_parity_case(
        self,
        bs,
        page_size,
        seqlen_offset,
        with_seq_lens_k,
        seed,
        pool_size=8,
        max_context_len=2048,
        seq_lens=None,
    ):
        g = torch.Generator(device="cpu").manual_seed(seed)
        req_to_token = torch.randint(
            0,
            pool_size * max_context_len,
            (pool_size, max_context_len),
            generator=g,
            dtype=torch.int32,
        ).to(DEVICE)
        req_pool_indices = torch.randperm(pool_size, generator=g)[:bs].to(
            DEVICE, dtype=torch.int64
        )
        if seq_lens is None:
            seq_lens = torch.randint(
                1,
                max_context_len - seqlen_offset,
                (bs,),
                generator=g,
                dtype=torch.int64,
            )
        seq_lens = seq_lens.to(DEVICE)

        max_blocks = max_context_len // page_size
        block_kv_indices = torch.full(
            (bs, max_blocks), -1, dtype=torch.int32, device=DEVICE
        )
        block_kv_indices_ref = torch.full(
            (bs, max_blocks), -1, dtype=torch.int32, device=DEVICE
        )

        seq_lens_k = (
            torch.zeros(bs, dtype=torch.int32, device=DEVICE)
            if with_seq_lens_k
            else None
        )

        update_trtllm_mla_graph_metadata(
            req_pool_indices=req_pool_indices,
            seq_lens=seq_lens,
            req_to_token=req_to_token,
            block_kv_indices=block_kv_indices,
            bs=bs,
            seqlen_offset=seqlen_offset,
            page_size=page_size,
            seq_lens_k=seq_lens_k,
        )

        seq_lens_k_ref = (seq_lens + seqlen_offset).to(torch.int32)
        create_flashmla_kv_indices_triton[
            (bs, get_num_kv_index_blocks_flashmla(max_blocks, page_size))
        ](
            req_to_token,
            req_pool_indices,
            seq_lens_k_ref,
            None,
            block_kv_indices_ref,
            req_to_token.stride(0),
            max_blocks,
            PAGED_SIZE=page_size,
        )
        torch.cuda.synchronize()

        torch.testing.assert_close(
            block_kv_indices, block_kv_indices_ref, rtol=0, atol=0
        )
        if with_seq_lens_k:
            torch.testing.assert_close(seq_lens_k, seq_lens_k_ref, rtol=0, atol=0)

    def test_parity_with_reference_pipeline(self):
        for bs in (1, 3, 8):
            for page_size in (32, 64):
                for seqlen_offset in (0, 4):
                    for with_seq_lens_k in (False, True):
                        with self.subTest(
                            bs=bs,
                            page_size=page_size,
                            seqlen_offset=seqlen_offset,
                            with_seq_lens_k=with_seq_lens_k,
                        ):
                            self._run_parity_case(
                                bs=bs,
                                page_size=page_size,
                                seqlen_offset=seqlen_offset,
                                with_seq_lens_k=with_seq_lens_k,
                                seed=1234
                                + bs * 31
                                + page_size
                                + seqlen_offset * 7
                                + int(with_seq_lens_k),
                            )

    def test_large_seqlen_coverage(self):
        max_context_len = 1_000_064
        self._run_parity_case(
            bs=2,
            page_size=64,
            seqlen_offset=4,
            with_seq_lens_k=True,
            seed=7,
            pool_size=2,
            max_context_len=max_context_len,
            seq_lens=torch.tensor([1_000_000, 999_983], dtype=torch.int64),
        )

    def test_large_batch_coverage(self):
        bs = 16 * 1024
        seq_lens = torch.arange(bs, dtype=torch.int64) % 257 + 1
        self._run_parity_case(
            bs=bs,
            page_size=32,
            seqlen_offset=2,
            with_seq_lens_k=True,
            seed=8,
            pool_size=bs,
            max_context_len=512,
            seq_lens=seq_lens,
        )

    def test_metadata_update_records_inside_cuda_graph(self):
        backend = _make_backend_for_hook_test(
            page_size=64, num_draft_tokens=2, device=DEVICE
        )
        backend.req_to_token = torch.arange(
            4 * 256, dtype=torch.int32, device=DEVICE
        ).reshape(4, 256)
        fb = _make_fb(bs=2, forward_mode=ForwardMode.TARGET_VERIFY, device=DEVICE)
        fb.seq_lens = torch.tensor([3, 4], dtype=torch.int64, device=DEVICE)

        backend.init_forward_metadata_out_graph(fb, in_capture=True)
        # Warmup launch (triton JIT) before capture.
        backend.init_forward_metadata_in_graph(fb)
        torch.cuda.synchronize()

        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            backend.init_forward_metadata_in_graph(fb)

        fb.seq_lens.copy_(torch.tensor([65, 130], dtype=torch.int64, device=DEVICE))
        graph.replay()
        torch.cuda.synchronize()

        metadata = backend.decode_cuda_graph_metadata[2]
        torch.testing.assert_close(
            metadata.seq_lens_k,
            torch.tensor([67, 132], dtype=torch.int32, device=DEVICE),
            rtol=0,
            atol=0,
        )
        # Row 1 covers ceil(132 / 64) = 3 pages after the replayed rebuild.
        torch.testing.assert_close(
            metadata.block_kv_indices[1, :3],
            backend.req_to_token[1, ::64][:3] // 64,
            rtol=0,
            atol=0,
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
