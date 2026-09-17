"""Verify-window semantics of trtllm-gen for ENCODER_ONLY layers.

Two pins against a paged SDPA reference: the spec-decode call
(``q_len_per_req = L``) is causal inside the window (wrong for ENCODER_ONLY
layers, which need bidirectional attention), and the expanded formulation
(bs*L single-token rows, kv length = prefix + L) matches the full-window
reference -- what TRTLLMHAAttnBackend runs for ENCODER_ONLY layers on the
draft worker. Causal-window cases also check the fixed-query decode helper used
by DSpark against the same paged reference.
"""

import math
import unittest
from types import SimpleNamespace

import torch

from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

# trtllm_mha kernels are sm100-only; run this kernel-unit test on Blackwell.
register_cuda_ci(est_time=10, stage="base-b", runner_config="4-gpu-b200")

DEVICE = "cuda"
PAGE_SIZE = 32
BS = 2
PREFIX = 40
L = 7
NUM_Q_HEADS = 8
NUM_KV_HEADS = 2
HEAD_DIM = 64


def _build_inputs(seed=3, *, prefix=PREFIX):
    torch.manual_seed(seed)
    dtype = torch.bfloat16
    seq_len = prefix + L
    pages_per_req = math.ceil(seq_len / PAGE_SIZE)
    num_pages = BS * pages_per_req + 1

    k_cache = torch.randn(
        num_pages, NUM_KV_HEADS, PAGE_SIZE, HEAD_DIM, dtype=dtype, device=DEVICE
    )
    v_cache = torch.randn(
        num_pages, NUM_KV_HEADS, PAGE_SIZE, HEAD_DIM, dtype=dtype, device=DEVICE
    )
    # Distinct page rows per request; page 0 left unused.
    block_tables = torch.arange(
        1, 1 + BS * pages_per_req, dtype=torch.int32, device=DEVICE
    ).view(BS, pages_per_req)
    q = torch.randn(BS * L, NUM_Q_HEADS, HEAD_DIM, dtype=dtype, device=DEVICE)
    workspace = torch.zeros(256 * 1024 * 1024, dtype=torch.uint8, device=DEVICE)
    return q, (k_cache, v_cache), block_tables, workspace


def _gather_kv(kv_cache, block_tables, req, *, prefix=PREFIX):
    k_cache, v_cache = kv_cache
    seq_len = prefix + L
    pages = block_tables[req].long()
    # [pages, kv_heads, page, dim] -> [kv_heads, pages*page, dim]
    k = k_cache[pages].permute(1, 0, 2, 3).reshape(NUM_KV_HEADS, -1, HEAD_DIM)
    v = v_cache[pages].permute(1, 0, 2, 3).reshape(NUM_KV_HEADS, -1, HEAD_DIM)
    return k[:, :seq_len], v[:, :seq_len]


def _sdpa_reference(
    q, kv_cache, block_tables, *, bidirectional, prefix=PREFIX, window_left=-1
):
    """Per-request SDPA over the paged KV; the L query tokens sit at the last
    L positions. bidirectional=True lets every query see all prefix+L keys;
    False applies the verify-style causal mask (query i sees prefix+i+1)."""
    seq_len = prefix + L
    group = NUM_Q_HEADS // NUM_KV_HEADS
    outs = []
    for req in range(BS):
        k, v = _gather_kv(kv_cache, block_tables, req, prefix=prefix)
        k = k.repeat_interleave(group, dim=0).float()
        v = v.repeat_interleave(group, dim=0).float()
        qi = q.view(BS, L, NUM_Q_HEADS, HEAD_DIM)[req].permute(1, 0, 2).float()
        scores = torch.einsum("hqd,hkd->hqk", qi, k) / math.sqrt(HEAD_DIM)
        if not bidirectional:
            kv_pos = torch.arange(seq_len, device=DEVICE).view(1, 1, -1)
            q_pos = (prefix + torch.arange(L, device=DEVICE)).view(1, -1, 1)
            mask = kv_pos > q_pos
            if window_left >= 0:
                mask = mask | (kv_pos < q_pos - window_left)
            scores = scores.masked_fill(mask, float("-inf"))
        out = torch.einsum("hqk,hkd->hqd", torch.softmax(scores, dim=-1), v)
        outs.append(out.permute(1, 0, 2))
    return torch.cat(outs, dim=0).to(q.dtype)


class TestTrtllmMhaEncoderOnlyVerify(CustomTestCase):
    def test_spec_decode_call_is_causal_in_window(self):
        import flashinfer

        q, kv_cache, block_tables, workspace = _build_inputs()
        seq_lens = torch.full((BS,), PREFIX + L, dtype=torch.int32, device=DEVICE)
        o = flashinfer.decode.trtllm_batch_decode_with_kv_cache(
            query=q,
            kv_cache=kv_cache,
            workspace_buffer=workspace,
            block_tables=block_tables,
            seq_lens=seq_lens,
            max_seq_len=PREFIX + L,
            bmm1_scale=1.0 / math.sqrt(HEAD_DIM),
            bmm2_scale=1.0,
            out_dtype=torch.bfloat16,
            q_len_per_req=L,
        )
        causal_ref = _sdpa_reference(q, kv_cache, block_tables, bidirectional=False)
        full_ref = _sdpa_reference(q, kv_cache, block_tables, bidirectional=True)
        torch.testing.assert_close(
            o.view(-1, NUM_Q_HEADS, HEAD_DIM).float(),
            causal_ref.float(),
            atol=2e-2,
            rtol=2e-2,
        )
        # And it is NOT full-window bidirectional attention (the two
        # references would only coincide if they degenerate).
        self.assertFalse(
            torch.allclose(causal_ref.float(), full_ref.float(), atol=2e-2, rtol=2e-2)
        )

    def test_expanded_rows_match_bidirectional_reference(self):
        import flashinfer

        q, kv_cache, block_tables, workspace = _build_inputs()
        row_map = torch.arange(BS * L, device=DEVICE) // L
        expanded_seq_lens = torch.full(
            (BS * L,), PREFIX + L, dtype=torch.int32, device=DEVICE
        )
        expanded_block_tables = block_tables[row_map].contiguous()
        o = flashinfer.decode.trtllm_batch_decode_with_kv_cache(
            query=q,
            kv_cache=kv_cache,
            workspace_buffer=workspace,
            block_tables=expanded_block_tables,
            seq_lens=expanded_seq_lens,
            max_seq_len=PREFIX + L,
            bmm1_scale=1.0 / math.sqrt(HEAD_DIM),
            bmm2_scale=1.0,
            out_dtype=torch.bfloat16,
            q_len_per_req=1,
        )
        full_ref = _sdpa_reference(q, kv_cache, block_tables, bidirectional=True)
        torch.testing.assert_close(
            o.view(-1, NUM_Q_HEADS, HEAD_DIM).float(),
            full_ref.float(),
            atol=2e-2,
            rtol=2e-2,
        )


class TestTrtllmMhaCausalWindow(CustomTestCase):
    def test_dspark_fixed_query_window_boundaries(self):
        from sglang.srt.layers.attention.trtllm_mha_backend import TRTLLMHAAttnBackend

        # The query block crosses the window and page edges; long prefixes must
        # keep bottom-right causal alignment, not rebase to window-local positions.
        cases = [(1, 0), (1, 40), (2, 0), (2, 40)]
        cases += [(w, w - L + delta) for w in (32, 2048) for delta in (-1, 0, 1)]
        cases += [(2048, 4097)]
        for window, prefix in cases:
            with self.subTest(window=window, prefix=prefix):
                q, kv_cache, block_tables, workspace = _build_inputs(prefix=prefix)
                seq_lens = torch.full(
                    (BS,), prefix + L, dtype=torch.int32, device=DEVICE
                )
                backend = SimpleNamespace(
                    workspace_buffer=workspace,
                    max_context_len=prefix + L,
                    q_data_type=q.dtype,
                    decode_seq_len_splits=1,
                    _multi_ctas_kv_counter_buffer=None,
                )
                out = TRTLLMHAAttnBackend._run_fixed_q_len_decode(
                    backend,
                    q,
                    kv_cache,
                    block_tables,
                    seq_lens,
                    bmm1_scale=1.0 / math.sqrt(HEAD_DIM),
                    bmm2_scale=1.0,
                    window_left=window - 1,
                    sinks=None,
                    q_len_per_req=L,
                )
                ref = _sdpa_reference(
                    q,
                    kv_cache,
                    block_tables,
                    bidirectional=False,
                    prefix=prefix,
                    window_left=window - 1,
                )
                torch.testing.assert_close(
                    out.view_as(ref).float(), ref.float(), atol=2e-2, rtol=2e-2
                )


if __name__ == "__main__":
    unittest.main()
