"""trtllm_mla passes host q/kv lengths to flashinfer's ragged prefill. CPU-only."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.layers.attention import trtllm_mla_backend
from sglang.srt.layers.attention.trtllm_mla_backend import (
    HostSeqLens,
    TRTLLMMLABackend,
    _host_seq_lens,
    _prefix_chunk_host_seq_lens,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def _lens(*values):
    return torch.tensor(values, dtype=torch.int32)


def _host(*values):
    return HostSeqLens(lens=_lens(*values), total=sum(values))


class TestHostSeqLens(CustomTestCase):
    def test_passes_lens_that_cover_q_and_k(self):
        q_lens, kv_lens = _host(3, 5), _host(0, 7)
        kwargs = _host_seq_lens(torch.empty(8, 2), torch.empty(7, 2), q_lens, kv_lens)
        self.assertIs(kwargs["q_seq_lens_cpu"], q_lens.lens)
        self.assertIs(kwargs["kv_seq_lens_cpu"], kv_lens.lens)

    def test_stock_call_on_padding_or_missing_lens(self):
        q, k = torch.empty(8, 2), torch.empty(7, 2)
        for args in (
            (torch.empty(16, 2), k, _host(3, 5), _host(4, 3)),  # padded q
            (q, torch.empty(16, 2), _host(3, 5), _host(4, 3)),  # padded k
            (q, k, None, _host(4, 3)),
            (q, k, _host(3, 5), None),
        ):
            self.assertEqual(_host_seq_lens(*args), {})

    def test_prefix_chunk_lens(self):
        lens = torch.tensor([[4, 3], [0, 2]], dtype=torch.int32)
        fb = SimpleNamespace(
            prefix_chunk_seq_lens_cpu=lens, prefix_chunk_num_tokens=[7, 2]
        )
        chunk = _prefix_chunk_host_seq_lens(fb, 1)
        self.assertTrue(torch.equal(chunk.lens, lens[1]))
        self.assertEqual(chunk.total, 2)
        unset = SimpleNamespace(
            prefix_chunk_seq_lens_cpu=None, prefix_chunk_num_tokens=None
        )
        self.assertIsNone(_prefix_chunk_host_seq_lens(unset, 0))

    def test_run_prefill_kernel_forwards_lens(self):
        received = {}

        def fake_ragged(**kwargs):
            received.update(kwargs)
            return kwargs["out"]

        backend = TRTLLMMLABackend.__new__(TRTLLMMLABackend)
        backend.data_type = torch.bfloat16
        backend.workspace_buffer = torch.empty(16, dtype=torch.uint8)
        q = torch.zeros(8, 2, 4, dtype=torch.bfloat16)
        lens = _lens(3, 5)
        # flashinfer is imported only on CUDA builds.
        fake_flashinfer = SimpleNamespace(
            prefill=SimpleNamespace(trtllm_ragged_attention_deepseek=fake_ragged)
        )
        with patch.object(
            trtllm_mla_backend, "flashinfer", fake_flashinfer, create=True
        ):
            backend._run_prefill_kernel(
                q=q,
                k=q,
                v=q,
                layer=SimpleNamespace(scaling=1.0),
                batch_size=2,
                cum_seq_lens_q=_lens(0, 3, 8),
                max_q_len=5,
                seq_lens_kv=lens,
                cum_seq_lens_kv=_lens(0, 3, 8),
                max_kv_len=5,
                is_causal=True,
                return_lse=False,
                out_buffer=torch.empty_like(q),
                q_seq_lens_cpu=lens,
                kv_seq_lens_cpu=lens,
            )
        self.assertIs(received["q_seq_lens_cpu"], lens)
        self.assertIs(received["kv_seq_lens_cpu"], lens)


if __name__ == "__main__":
    unittest.main()
