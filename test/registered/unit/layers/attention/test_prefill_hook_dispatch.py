"""Contracts the chunked-prefix prefill call site depends on.

The call site passes the row-check arguments by keyword, so an override left on
a stale signature raises TypeError at dispatch, before any kernel runs -- on
every chunked-prefix batch, and only for the backend that owns the override.
Host length mirrors have to sum to the rows actually handed to the kernel, which
the padding that grows the query buffer otherwise breaks.
"""

import ast
import inspect
import sys
import types
import unittest
from pathlib import Path
from unittest import mock

import torch

from sglang.srt.layers.attention import trtllm_mla_backend
from sglang.srt.layers.attention.dsa_backend import (
    DeepseekSparseAttnBackend,
    DSAMetadata,
)
from sglang.srt.layers.attention.tokenspeed_mla_backend import TokenspeedMLABackend
from sglang.srt.layers.attention.trtllm_mla_backend import TRTLLMMLABackend
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=6, suite="base-a-test-cpu")

_ATTENTION_DIR = Path(inspect.getfile(TRTLLMMLABackend)).parent

_PREFILL_HOOK_OWNERS = {"trtllm_mla_backend.py", "tokenspeed_mla_backend.py"}

# What the chunked-prefix call site passes when a chunk may carry an empty row.
_HOOK_CALL = dict(
    q=None,
    k=None,
    v=None,
    layer=None,
    batch_size=3,
    cum_seq_lens_q=None,
    max_q_len=257,
    seq_lens_kv=None,
    cum_seq_lens_kv=None,
    max_kv_len=512,
    is_causal=False,
    return_lse=True,
    out_buffer=None,
    q_seq_lens_cpu=None,
    kv_seq_lens_cpu=None,
    all_rows_active=False,
    o_sf_scale=-1.0,
)


class TestPrefillHookDispatch(CustomTestCase):
    def test_every_override_is_accounted_for(self):
        owners = set()
        for path in _ATTENTION_DIR.glob("*.py"):
            for node in ast.walk(ast.parse(path.read_text())):
                if (
                    isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
                    and node.name == "_run_prefill_kernel"
                ):
                    owners.add(path.name)
        self.assertEqual(
            owners,
            _PREFILL_HOOK_OWNERS,
            "a _run_prefill_kernel override was added or removed; it must take "
            "q_seq_lens_cpu / kv_seq_lens_cpu / all_rows_active, then update "
            "this pin",
        )

    def test_overrides_bind_the_chunked_prefix_arguments(self):
        for cls in (TRTLLMMLABackend, TokenspeedMLABackend):
            with self.subTest(cls.__name__):
                inspect.signature(cls._run_prefill_kernel).bind(
                    object.__new__(cls), **_HOOK_CALL
                )


class _RecordingKernel:
    """Stands in for the wrapper, enforcing the contracts it checks."""

    # Written into every covered row so callers can prove the buffer they got
    # back is the storage the kernel filled, not same-shaped scratch.
    SENTINEL = 7.0

    def __init__(self):
        self.query_rows = None

    def __call__(self, **kwargs):
        for name, packed in (("q_seq_lens_cpu", "query"), ("kv_seq_lens_cpu", "key")):
            if kwargs.get(name) is not None:
                total = int(kwargs[name].sum())
                if total != kwargs[packed].shape[0]:
                    raise ValueError(
                        f"{name} sums to {total}, but expected "
                        f"{kwargs[packed].shape[0]} tokens"
                    )
        query = kwargs["query"]
        self.query_rows = query.shape[0]
        out = kwargs.get("out")
        if out is None:
            out = torch.empty(
                query.shape[0], *kwargs["value"].shape[1:], dtype=query.dtype
            )
        elif out.shape[0] != query.shape[0]:
            raise ValueError(
                f"out must have shape ({query.shape[0]}, ...), got {tuple(out.shape)}"
            )
        out.fill_(self.SENTINEL)
        if not kwargs["return_lse"]:
            return out
        lse = kwargs.get("lse")
        if lse is None:
            lse = torch.zeros(out.shape[0], out.shape[1], dtype=torch.float32)
        lse.fill_(self.SENTINEL)
        return out, lse


class TestPaddedQueryBuffer(CustomTestCase):
    """A padded query buffer must still reach the kernel with mirrors attached.

    Passing the real extend lengths against a padded buffer makes the wrapper
    reject the call outright instead of attending the rows that do exist.
    """

    def _run(self, *, padded_rows, q_lens, kv_lens):
        backend = object.__new__(TRTLLMMLABackend)
        backend.data_type = torch.bfloat16
        backend.workspace_buffer = None
        kernel = _RecordingKernel()
        fake = types.SimpleNamespace(
            prefill=types.SimpleNamespace(trtllm_ragged_attention_deepseek=kernel)
        )
        num_kv = int(kv_lens.sum())
        with mock.patch.object(trtllm_mla_backend, "flashinfer", fake, create=True):
            out, lse = backend._run_prefill_kernel(
                q=torch.zeros(padded_rows, 2, 4),
                k=torch.zeros(num_kv, 2, 4),
                v=torch.zeros(num_kv, 2, 4),
                layer=types.SimpleNamespace(scaling=1.0),
                batch_size=q_lens.numel(),
                cum_seq_lens_q=None,
                max_q_len=int(q_lens.max()),
                seq_lens_kv=None,
                cum_seq_lens_kv=None,
                max_kv_len=max(int(kv_lens.max()), 1),
                is_causal=False,
                return_lse=True,
                out_buffer=torch.zeros(padded_rows, 2, 4),
                q_seq_lens_cpu=q_lens,
                kv_seq_lens_cpu=kv_lens,
                all_rows_active=False,
                o_sf_scale=-1.0,
            )
        return kernel, out, lse

    def test_padded_buffer_is_trimmed_to_the_mirrored_rows(self):
        q_lens = torch.tensor([7, 8], dtype=torch.int32)
        kernel, out, lse = self._run(
            padded_rows=16,
            q_lens=q_lens,
            kv_lens=torch.tensor([4, 0], dtype=torch.int32),
        )
        self.assertEqual(kernel.query_rows, 15)
        # Callers size their buffers by the padded query and index them that way.
        self.assertEqual(out.shape[0], 16)
        self.assertEqual(lse.shape[0], 16)
        self.assertTrue(bool((out[:15] == _RecordingKernel.SENTINEL).all()))
        self.assertTrue(bool((lse[:15] == _RecordingKernel.SENTINEL).all()))

    def test_unpadded_buffer_is_passed_through(self):
        q_lens = torch.tensor([7, 8], dtype=torch.int32)
        kernel, out, _ = self._run(
            padded_rows=15,
            q_lens=q_lens,
            kv_lens=torch.tensor([4, 3], dtype=torch.int32),
        )
        self.assertEqual(kernel.query_rows, 15)
        self.assertEqual(out.shape[0], 15)


class TestDsaPaddedQueryBuffer(CustomTestCase):
    """DSA one-shot MHA must trim a padded query before mirroring its lengths.

    MLP-sync (DP) padding appends zero-length extend rows and grows the token
    count, so the unsliced lengths then fail the wrapper's sum check.
    """

    def _run(self, *, padded_rows, q_lens, kv_lens):
        backend = object.__new__(DeepseekSparseAttnBackend)
        backend.device_sm_major = 10
        backend.workspace_buffer = None
        backend.use_mha = True
        kernel = _RecordingKernel()
        stub = types.ModuleType("flashinfer")
        stub.prefill = types.SimpleNamespace(trtllm_ragged_attention_deepseek=kernel)
        heads, head_dim, v_head_dim = 2, 4, 4
        layer = types.SimpleNamespace(
            tp_q_head_num=heads,
            tp_k_head_num=heads,
            tp_v_head_num=heads,
            head_dim=head_dim,
            v_head_dim=v_head_dim,
            scaling=1.0,
        )
        num_kv = int(kv_lens.sum())
        metadata = DSAMetadata(
            page_size=64,
            cache_seqlens_int32=None,
            max_seq_len_q=int(q_lens.max()),
            max_seq_len_k=int(kv_lens.max()),
            cu_seqlens_q=torch.zeros(q_lens.numel() + 1, dtype=torch.int32),
            cu_seqlens_k=torch.zeros(kv_lens.numel() + 1, dtype=torch.int32),
            page_table_1=None,
            real_page_table=None,
            dsa_cache_seqlens_int32=None,
            dsa_cu_seqlens_q=None,
            dsa_cu_seqlens_k=None,
            dsa_extend_seq_lens_list=q_lens.tolist(),
            dsa_seqlens_expanded=None,
            mha_q_seq_lens_cpu=q_lens,
            mha_kv_seq_lens_cpu=kv_lens,
            mha_all_rows_active=False,
        )
        with mock.patch.dict(sys.modules, {"flashinfer": stub}):
            out = backend._forward_standard_mha(
                torch.zeros(padded_rows * heads * head_dim),
                torch.zeros(num_kv * heads * head_dim),
                torch.zeros(num_kv * heads * v_head_dim),
                layer,
                types.SimpleNamespace(batch_size=q_lens.numel()),
                metadata,
            )
        return kernel, out

    def test_padded_buffer_is_trimmed_and_returned_padded(self):
        kernel, out = self._run(
            padded_rows=8,
            q_lens=torch.tensor([7, 0], dtype=torch.int32),
            kv_lens=torch.tensor([11, 4], dtype=torch.int32),
        )
        self.assertEqual(kernel.query_rows, 7)
        self.assertEqual(out.shape[0], 8)
        self.assertTrue(bool((out[:7] == _RecordingKernel.SENTINEL).all()))

    def test_unpadded_buffer_is_passed_through(self):
        kernel, out = self._run(
            padded_rows=7,
            q_lens=torch.tensor([7, 0], dtype=torch.int32),
            kv_lens=torch.tensor([11, 4], dtype=torch.int32),
        )
        self.assertEqual(kernel.query_rows, 7)
        self.assertEqual(out.shape[0], 7)
        self.assertTrue(bool((out == _RecordingKernel.SENTINEL).all()))


if __name__ == "__main__":
    unittest.main()
