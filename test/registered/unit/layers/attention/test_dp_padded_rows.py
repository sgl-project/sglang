# SPDX-License-Identifier: Apache-2.0
"""Attention over a batch that DP-attention padding grew after it was planned.

prepare_mlp_sync_batch pads each DP rank's token rows to a multiple of the
attention TP size. In the eager paths the attention metadata can be planned on
the real rows first, so the backend sees more rows than its plan covers; the
padding rows belong to no request and their output is dropped.
"""

import contextlib
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.layers.attention import trtllm_mha_backend
from sglang.srt.layers.attention.linear.gdn_backend import GDNAttnBackend
from sglang.srt.layers.attention.mamba.mamba2_metadata import ForwardMetadata
from sglang.srt.layers.attention.trtllm_mha_backend import TRTLLMHAAttnBackend
from sglang.srt.mem_cache.memory_pool import MambaPool
from sglang.srt.model_executor.forward_mode import ForwardMode
from sglang.srt.utils.common import ceil_align
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=4, suite="base-a-test-cpu")

HEADS, DIM, PAGES = 4, 8, 32


class _TrtllmDecodeKernel:
    """Indexes like trtllm_batch_decode_with_kv_cache: its batch is
    rows // q_len_per_req, and it reads seq_lens / block_tables for each."""

    def __init__(self):
        self.calls = []

    def __call__(self, **kw):
        q, q_len = kw["query"], kw.get("q_len_per_req", 1)
        batch = q.shape[0] // q_len
        self.calls.append((q.shape[0], q_len))
        if q.shape[0] % q_len:
            raise RuntimeError(f"{q.shape[0]} query rows split into {q_len}-row runs")
        if batch > kw["seq_lens"].shape[0] or batch > kw["block_tables"].shape[0]:
            raise IndexError(
                f"decode batch {batch} reads past a "
                f"{kw['seq_lens'].shape[0]}-request plan"
            )
        if kw["out"] is None:
            return (q.float() * 2 + 1).to(kw["out_dtype"] or q.dtype)
        return kw["out"].copy_(q.float() * 2 + 1)


def _trtllm_backend():
    backend = object.__new__(TRTLLMHAAttnBackend)
    backend.decode_seq_len_splits = 1
    backend.decode_split_hoist = False
    backend.workspace_buffer = None
    backend.max_context_len = 4096
    backend.q_data_type = torch.bfloat16
    backend._multi_ctas_kv_counter_buffer = None
    backend._full_attn_kv_split_scope = lambda **kw: contextlib.nullcontext()
    return backend


def _trtllm_decode(query, requests, q_len, out=None):
    """Decode over a plan of ``requests``; returns the output and kernel calls."""
    kernel = _TrtllmDecodeKernel()
    flashinfer = SimpleNamespace(
        decode=SimpleNamespace(trtllm_batch_decode_with_kv_cache=kernel)
    )
    with patch.object(trtllm_mha_backend, "flashinfer", flashinfer, create=True):
        output = _trtllm_backend()._run_fixed_q_len_decode(
            query,
            None,
            torch.zeros((requests, PAGES), dtype=torch.int32),
            torch.full((requests,), 100, dtype=torch.int32),
            bmm1_scale=1.0,
            bmm2_scale=1.0,
            window_left=-1,
            sinks=None,
            q_len_per_req=q_len,
            out=out,
        )
    return output, kernel.calls


class TestTrtllmMhaDecodeOverPadding(CustomTestCase):
    def test_padding_rows_past_the_plan_are_not_read(self):
        # The eager EAGLE draft decode plans on its real rows, then DP padding
        # adds rows; the kernel must not index requests past the plan.
        for requests, q_len, attn_tp, preallocated in (
            (9, 1, 8, False),
            (9, 7, 8, False),
            (0, 1, 8, False),
            (9, 1, 8, True),
        ):
            with self.subTest(requests=requests, q_len=q_len, out=preallocated):
                planned = requests * q_len
                rows = ceil_align(max(planned, 1), attn_tp)
                query = torch.randn(rows, HEADS, DIM).to(torch.bfloat16)
                # Native NVFP4 decode passes a reused buffer covering every row;
                # its stale padding must be zeroed.
                out = None
                if preallocated:
                    out = torch.full((rows, HEADS, DIM), 7.0, dtype=torch.bfloat16)
                output, calls = _trtllm_decode(
                    query, requests=requests, q_len=q_len, out=out
                )
                self.assertEqual(calls, [(planned, q_len)] if planned else [])
                self.assertEqual(tuple(output.shape), (rows, HEADS, DIM))
                if preallocated:
                    self.assertEqual(output.data_ptr(), out.data_ptr())
                expected = (query[:planned].float() * 2 + 1).to(torch.bfloat16)
                self.assertTrue(torch.equal(output[:planned], expected))
                self.assertFalse(output[planned:].any())


DRAFT, VERIFY_REQUESTS, HEAD = 7, 9, 2


def _gdn_backend(verify_calls):
    """A GDN backend whose conv and verify kernels index like the real ones:
    one request per conv-state index, one token run per query_start_loc span."""

    def verify_conv_update(x, *args, conv_state_indices, **kwargs):
        assert x.shape[0] == conv_state_indices.shape[0]
        return x + 1

    def target_verify(*, q, k, v, query_start_loc, **kwargs):
        assert q.shape[1] == int(query_start_loc[-1])
        verify_calls.append(q.shape[1])
        return v * 2

    backend = object.__new__(GDNAttnBackend)
    backend.forward_metadata = ForwardMetadata(
        query_start_loc=torch.arange(0, (VERIFY_REQUESTS + 1) * DRAFT, DRAFT),
        mamba_cache_indices=torch.arange(VERIFY_REQUESTS),
    )
    backend.mis_metadata = None
    backend.requires_contiguous_prefill_state = False
    state = MambaPool.SpeculativeState(
        conv=[torch.zeros(VERIFY_REQUESTS, 3 * HEAD, 3)],
        temporal=torch.zeros(VERIFY_REQUESTS, 1, HEAD, HEAD),
        intermediate_ssm=torch.zeros(VERIFY_REQUESTS + 1, DRAFT, 1, HEAD, HEAD),
        intermediate_conv_window=[torch.zeros(VERIFY_REQUESTS + 1, DRAFT, 3 * HEAD)],
    )
    backend.state_layer_cache = lambda layer_id: state
    backend.verify_intermediate_state_indices = torch.arange(VERIFY_REQUESTS + 1)
    backend.req_to_token_pool = SimpleNamespace(size=VERIFY_REQUESTS)
    backend.state_pool = SimpleNamespace(
        replayssm_spec_fold=False, replayssm_cache_base=None, replayssm_is_kda=False
    )
    backend.kernel_dispatcher = SimpleNamespace(
        verify_conv_update=verify_conv_update, target_verify=target_verify
    )
    return backend


def _gdn_verify(backend, mixed_qkv, a, b):
    layer = SimpleNamespace(
        layer_id=0,
        q_dim=HEAD,
        k_dim=HEAD,
        v_dim=HEAD,
        num_q_heads=1,
        num_k_heads=1,
        num_v_heads=1,
        head_q_dim=HEAD,
        head_k_dim=HEAD,
        head_v_dim=HEAD,
        conv_weights=None,
        bias=None,
        activation="silu",
        A_log=torch.zeros(1),
        dt_bias=torch.zeros(1),
    )
    batch = SimpleNamespace(
        forward_mode=ForwardMode.TARGET_VERIFY,
        spec_info=SimpleNamespace(draft_token_num=DRAFT),
        req_pool_indices=torch.arange(VERIFY_REQUESTS),
    )
    return backend.forward_extend(layer, batch, mixed_qkv, a, b)


class TestGdnVerifyOverPadding(CustomTestCase):
    def test_verify_runs_the_real_rows_and_zeroes_the_padding(self):
        # 9 requests x 7 draft tokens = 63 rows reach the backend as 64 at
        # attention TP 8; the trailing row belongs to no request.
        real = VERIFY_REQUESTS * DRAFT
        padded = ceil_align(real, 8)
        inputs = [torch.randn(padded, n) for n in (3 * HEAD, 1, 1)]
        for x in inputs:
            # Padding rows hold whatever the buffer had; make them loud.
            x[real:] = 1e3
        verify_calls = []
        backend = _gdn_backend(verify_calls)
        reference = _gdn_verify(backend, *(x[:real] for x in inputs))
        output = _gdn_verify(backend, *inputs)
        self.assertEqual(verify_calls, [real, real])
        self.assertEqual(output.shape[-3], padded)
        self.assertTrue(torch.equal(output.narrow(-3, 0, real), reference))
        self.assertFalse(output.narrow(-3, real, padded - real).any())


if __name__ == "__main__":
    unittest.main()
