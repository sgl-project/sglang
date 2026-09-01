"""DCP cuda-graph metadata for the trtllm_mla backend family.

The rank-local KV-length and page-table plumbing lives on
:class:`TRTLLMMLABackend`, so the same expectations are asserted for the base
backend and for both subclasses that inherit it.
"""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.layers.attention import trtllm_mla_backend as backend_module
from sglang.srt.layers.attention.cutedsl_mla_backend import CuteDslMLABackend
from sglang.srt.layers.attention.tokenspeed_mla_backend import TokenspeedMLABackend
from sglang.srt.layers.attention.trtllm_mla_backend import (
    TRTLLMMLABackend,
    TRTLLMMLADecodeMetadata,
)
from sglang.srt.layers.dcp.layout import get_dcp_lens
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=60, stage="base-b", runner_config="4-gpu-b200")

NUM_DRAFT_TOKENS = 8
DCP_SIZE = 4
DCP_RANK = 2


def _make_backend(backend_cls, bs: int):
    backend = object.__new__(backend_cls)
    backend.num_draft_tokens = NUM_DRAFT_TOKENS
    metadata = TRTLLMMLADecodeMetadata(
        block_kv_indices=torch.full((bs, 4), -1, dtype=torch.int32, device="cuda"),
        seq_lens_k=torch.zeros(bs, dtype=torch.int32, device="cuda"),
        global_seq_lens_k=torch.zeros(bs, dtype=torch.int32, device="cuda"),
    )
    backend.decode_cuda_graph_metadata = {bs: metadata}
    return backend, metadata


def _apply(backend, *, bs: int, seq_lens: torch.Tensor, forward_mode):
    parallel = SimpleNamespace(dcp_enabled=True, dcp_size=DCP_SIZE, dcp_rank=DCP_RANK)
    with (
        patch.object(backend_module, "get_parallel", return_value=parallel),
        patch.object(backend, "_fill_dcp_block_kv_indices") as fill,
    ):
        backend._apply_cuda_graph_metadata(
            bs=bs,
            req_pool_indices=torch.arange(bs, dtype=torch.int32, device="cuda"),
            seq_lens=seq_lens,
            forward_mode=forward_mode,
        )
    return fill


@unittest.skipUnless(torch.cuda.is_available(), "DCP metadata buffers live on CUDA")
class _DCPMetadataTests:
    backend_cls = None

    def test_target_verify_splits_global_and_local_lengths(self):
        bs = 3
        backend, metadata = _make_backend(self.backend_cls, bs)
        prefix_lens = torch.tensor([10, 20, 30], dtype=torch.int32, device="cuda")

        fill = _apply(
            backend,
            bs=bs,
            seq_lens=prefix_lens,
            forward_mode=ForwardMode.TARGET_VERIFY,
        )

        expected_global = prefix_lens + NUM_DRAFT_TOKENS
        expected_local = get_dcp_lens(expected_global, DCP_SIZE, DCP_RANK).to(
            torch.int32
        )
        torch.testing.assert_close(metadata.global_seq_lens_k, expected_global)
        torch.testing.assert_close(metadata.seq_lens_k, expected_local)
        fill.assert_called_once()
        torch.testing.assert_close(fill.call_args.args[2], expected_local)

    def test_decode_does_not_add_draft_tokens(self):
        bs = 3
        backend, metadata = _make_backend(self.backend_cls, bs)
        seq_lens = torch.tensor([10, 20, 30], dtype=torch.int32, device="cuda")

        fill = _apply(
            backend, bs=bs, seq_lens=seq_lens, forward_mode=ForwardMode.DECODE
        )

        expected_local = get_dcp_lens(seq_lens, DCP_SIZE, DCP_RANK).to(torch.int32)
        fill.assert_called_once()
        torch.testing.assert_close(fill.call_args.args[2], expected_local)
        # Plain decode keeps both views in the capture-stable buffers.
        torch.testing.assert_close(metadata.global_seq_lens_k, seq_lens)
        torch.testing.assert_close(metadata.seq_lens_k, expected_local)


class TestTRTLLMMLADCPMetadata(_DCPMetadataTests, CustomTestCase):
    backend_cls = TRTLLMMLABackend


class TestTokenspeedMLADCPMetadata(_DCPMetadataTests, CustomTestCase):
    backend_cls = TokenspeedMLABackend


class TestCuteDslMLADCPMetadata(_DCPMetadataTests, CustomTestCase):
    backend_cls = CuteDslMLABackend


@unittest.skipUnless(torch.cuda.is_available(), "needs the flashinfer decode hook")
class TestTRTLLMMLARejectsDcpMultiTokenQuery(CustomTestCase):
    """``_run_decode_kernel`` must refuse every spec path under DCP.

    trtllm-gen takes no global causal bound, which a ``q_len > 1`` batch needs.
    Single-token spec batches still read as ``q_len == 1``, so the refusal also
    keys on ``causal_seqs`` and ``return_lse``.
    """

    def _make_backend(self):
        backend = object.__new__(TRTLLMMLABackend)
        backend.backend = "trtllm-gen"
        backend.qk_nope_head_dim = 128
        backend.kv_lora_rank = 512
        backend.qk_rope_head_dim = 64
        backend.workspace_buffer = None
        backend._multi_ctas_kv_counter_buffer = None
        return backend

    def _call(
        self,
        backend,
        q_len,
        *,
        dcp_enabled,
        kernel=None,
        causal_seqs=None,
        return_lse=False,
    ):
        parallel = SimpleNamespace(
            dcp_enabled=dcp_enabled,
            dcp_size=DCP_SIZE if dcp_enabled else 1,
            dcp_rank=DCP_RANK if dcp_enabled else 0,
        )
        flashinfer_stub = SimpleNamespace(
            decode=SimpleNamespace(
                trtllm_batch_decode_with_kv_cache_mla=kernel or (lambda **kw: None)
            )
        )
        with (
            patch.object(backend_module, "get_parallel", return_value=parallel),
            patch.object(backend_module, "flashinfer", flashinfer_stub),
            patch.object(backend, "_compute_decode_bmm1_scale", return_value=1.0),
        ):
            return backend._run_decode_kernel(
                query=torch.zeros(
                    (2, q_len, 16, 576), dtype=torch.bfloat16, device="cuda"
                ),
                kv_cache=torch.zeros((4, 1, 64, 576), dtype=torch.bfloat16),
                block_tables=torch.zeros((2, 4), dtype=torch.int32, device="cuda"),
                seq_lens=torch.ones(2, dtype=torch.int32, device="cuda"),
                max_seq_len=64,
                layer=SimpleNamespace(scaling=1.0, k_scale_float=None),
                causal_seqs=causal_seqs,
                return_lse=return_lse,
            )

    def test_multi_token_query_under_dcp_raises(self):
        with self.assertRaises(NotImplementedError):
            self._call(self._make_backend(), q_len=NUM_DRAFT_TOKENS, dcp_enabled=True)

    def test_explicit_causal_bound_under_dcp_raises_at_q_len_one(self):
        # Nothing forces speculative_num_draft_tokens > 1, so a q_len == 1
        # target-verify is expressible and must not slip past the q_len proxy.
        with self.assertRaises(NotImplementedError):
            self._call(
                self._make_backend(),
                q_len=1,
                dcp_enabled=True,
                causal_seqs=torch.ones(2, dtype=torch.int32, device="cuda"),
            )

    def test_single_token_query_under_dcp_is_allowed(self):
        # The premise of trtllm_mla DCP decode: q_len == 1 needs no global
        # bound, so the guard must not swallow the path it exists to protect.
        calls = []
        self._call(
            self._make_backend(),
            q_len=1,
            dcp_enabled=True,
            kernel=lambda **kw: calls.append(kw),
            return_lse=True,
        )
        self.assertEqual(len(calls), 1)

    def test_single_token_draft_extend_under_dcp_raises(self):
        # A single-token draft-extend reads as q_len == 1 with no causal_seqs;
        # skipping the cross-rank merge (no LSE requested) is the only signal.
        with self.assertRaises(NotImplementedError):
            self._call(self._make_backend(), q_len=1, dcp_enabled=True)

    def test_multi_token_query_without_dcp_is_allowed(self):
        calls = []
        self._call(
            self._make_backend(),
            q_len=NUM_DRAFT_TOKENS,
            dcp_enabled=False,
            kernel=lambda **kw: calls.append(kw),
        )
        self.assertEqual(len(calls), 1)


class TestDcpDecodeLayout(CustomTestCase):
    """Rank-local length math the decode page table above is built from."""

    SIZES = [1, 2, 3, 4, 8]
    LENS = list(range(0, 41))

    def test_ranks_partition_the_global_length(self):
        lens = torch.tensor(self.LENS, dtype=torch.int32)
        for n in self.SIZES:
            total = sum(
                get_dcp_lens(lens, n, rank).to(torch.int64) for rank in range(n)
            )
            self.assertTrue(
                torch.equal(total, lens.to(torch.int64)),
                f"per-rank lengths do not sum to the global length at n={n}",
            )

    def test_newest_token_is_owned_by_exactly_one_rank(self):
        # A decode step appends one token; the cross-rank merge double counts
        # or drops it unless exactly one rank sees its length grow.
        for n in self.SIZES:
            for global_len in self.LENS[1:]:
                prev = torch.tensor([global_len - 1], dtype=torch.int32)
                cur = torch.tensor([global_len], dtype=torch.int32)
                grew = [
                    int(get_dcp_lens(cur, n, rank).item())
                    - int(get_dcp_lens(prev, n, rank).item())
                    for rank in range(n)
                ]
                self.assertEqual(sum(grew), 1, f"n={n}, global_len={global_len}")
                self.assertEqual(grew[(global_len - 1) % n], 1)


if __name__ == "__main__":
    unittest.main()
