import sys
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch

# Loaded before the patch.dict stub below, which unloads every module first
# imported inside it while their torch op registrations stay.
import sglang.srt.model_executor.forward_batch_info  # noqa: F401
from sglang.test.ci.ci_register import register_cpu_ci

with patch.dict(
    sys.modules,
    {
        module: MagicMock()
        for module in (
            "sgl_kernel",
            "sgl_kernel.flash_attn",
            "sgl_kernel.quantization",
            "sgl_kernel.scalar_type",
        )
    },
):
    from sglang.srt.layers.attention.xpu_backend import XPUAttentionBackend

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestXpuBackendHostSeqLens(unittest.TestCase):
    """With the XPU monokernel the attention backend opts out of the scheduler's host
    copy of seq_lens (decide_needs_cpu_seq_lens), which spares every speculative step the wait for the
    previous verify's lengths; an eager step without the copy reads them back."""

    def test_the_opt_out_is_per_instance_and_off_by_default(self):
        """Only a backend built with --enable-xpu-monokernel opts out (in __init__);
        the class keeps the default, so every other XPU server is unchanged."""
        self.assertTrue(XPUAttentionBackend.needs_cpu_seq_lens)

    def test_the_longest_sequence_comes_from_the_host_copy_when_there_is_one(self):
        batch = SimpleNamespace(
            seq_lens_cpu=torch.tensor([5, 9, 7]),
            # a device read would give 100: it must not happen
            seq_lens=torch.tensor([100, 1, 1]),
        )
        self.assertEqual(XPUAttentionBackend._max_seq_len(batch), 9)

    def test_an_eager_step_without_the_host_copy_reads_the_lengths_back(self):
        batch = SimpleNamespace(seq_lens_cpu=None, seq_lens=torch.tensor([5, 9, 7]))
        self.assertEqual(XPUAttentionBackend._max_seq_len(batch), 9)

    def test_graph_replay_metadata_needs_no_device_read(self):
        """Without the host copy, a replay bounds the key length by what the static
        page table holds (its width is in pages) instead of reading seq_lens back: the
        page-table refresh rewrites cdiv(bound, page_size) pages, so a bound in pages
        would leave the pages of a long context stale."""
        bs, width = 2, 64
        backend = XPUAttentionBackend.__new__(XPUAttentionBackend)
        backend.page_size = 64
        backend.is_encoder_decoder = False
        backend.use_sliding_window_kv_pool = False
        metadata = SimpleNamespace(
            page_table=torch.zeros(bs, width, dtype=torch.int32),
            cu_seqlens_q=torch.zeros(bs + 1, dtype=torch.int32),
            max_seq_len_k=None,
            max_seq_len_q=None,
        )
        backend.decode_cuda_graph_metadata = {bs: metadata}
        backend._set_decode_page_metadata = MagicMock()
        backend.req_to_token = torch.zeros(4, width, dtype=torch.int32)
        seq_lens = torch.tensor([10, 20])
        batch = SimpleNamespace(
            batch_size=bs,
            req_pool_indices=torch.arange(bs),
            seq_lens=seq_lens,
            seq_lens_cpu=None,
            forward_mode=SimpleNamespace(
                is_target_verify=lambda: True,
                is_decode_or_idle=lambda: False,
                is_draft_extend_v2=lambda: False,
            ),
            spec_info=SimpleNamespace(num_tokens_per_req=2),
            encoder_lens=None,
            out_cache_loc=None,
        )
        # stop after the part under test: the call after it needs the real backend
        backend._set_decode_page_metadata.side_effect = RuntimeError("stop")
        with (
            patch.object(
                torch.Tensor, "item", side_effect=AssertionError("device read")
            ),
            self.assertRaisesRegex(RuntimeError, "stop"),
        ):
            backend._init_full_cg_decode_metadata(batch, in_capture=False)
        self.assertEqual(metadata.max_seq_len_k, width * backend.page_size)
        # the refresh rewrites every page the table holds
        pages = -(-metadata.max_seq_len_k // backend.page_size)
        self.assertEqual(pages, width)


if __name__ == "__main__":
    unittest.main()
