"""Prefill metadata for prefix-cache state tracking issues no host sync. It is
built while the previous forward still runs; a sync there parks the scheduler
until that forward ends, so the new forward is launched with no lead."""

import unittest
from types import SimpleNamespace

import torch

from sglang.srt.layers.attention.hybrid_linear_attn_backend import (
    MambaAttnBackendBase,
)
from sglang.srt.layers.attention.linear.kernels.gdn_flashinfer import (
    maybe_build_flashinfer_checkpoint_plan,
)
from sglang.srt.runtime_context import get_context
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=10, stage="base-b", runner_config="1-gpu-small")


def _sync_error(fn):
    """The RuntimeError torch raises if `fn` synchronizes, or None."""
    torch.cuda.synchronize()
    torch.cuda.set_sync_debug_mode("error")
    try:
        fn()
    except RuntimeError as exc:
        return exc
    finally:
        torch.cuda.set_sync_debug_mode("default")
        torch.cuda.synchronize()
    return None


def _cuda(values, dtype=torch.int64):
    return torch.tensor(values, dtype=dtype).cuda()


@unittest.skipUnless(torch.cuda.is_available(), "needs CUDA")
class TestPrefillTrackMetadataNoHostSync(CustomTestCase):
    def test_conv_track_indices_clamp_on_device(self):
        # Two requests of 70 and 130 tokens; a 4-wide conv window ending at each
        # tracked boundary. Row 0's window starts before the batch and row 1's
        # runs past its end, so both clamp bounds are exercised.
        backend = SimpleNamespace(conv_states_shape=(8, 4), device="cuda")
        forward_batch = SimpleNamespace(
            mamba_track_aligned_lens=lambda: _cuda([2, 132]),
            mamba_track_mask=_cuda([True, True], dtype=torch.bool),
        )
        query_start_loc = _cuda([0, 70, 200], dtype=torch.int32)
        result = {}

        def build():
            result["indices"] = MambaAttnBackendBase._init_track_conv_indices(
                backend,
                query_start_loc,
                forward_batch,
                track_mask_indices=_cuda([0, 1]),
            )

        self.assertIsNone(_sync_error(build))
        self.assertEqual(
            result["indices"].tolist(), [[0, 0, 0, 1], [198, 199, 199, 199]]
        )

    def test_flashinfer_checkpoint_plan_reads_host_copies(self):
        override = get_context().override_server_args(_mamba_cache_chunk_size=64)
        override.install()
        self.addCleanup(override.restore)
        extend_seq_lens = [63, 64, 65, 127, 128, 129]
        track_mask = [False, True, True, True, True, True]
        track_seqlens = [63, 64, 65, 127, 65, 129]
        prefix_lens = [0] * 6
        forward_batch = SimpleNamespace(
            extend_seq_lens=_cuda(extend_seq_lens),
            mamba_track_mask=_cuda(track_mask, dtype=torch.bool),
            mamba_track_seqlens=_cuda(track_seqlens),
            extend_prefix_lens=_cuda(prefix_lens),
            extend_seq_lens_cpu=extend_seq_lens,
            mamba_prefill_track_mask_cpu=track_mask,
            mamba_track_seqlens_cpu=track_seqlens,
            extend_prefix_lens_cpu=prefix_lens,
        )
        metadata = SimpleNamespace(
            track_ssm_h_src=torch.empty(4, device="cuda"),
            track_ssm_h_dst=torch.empty(4, device="cuda"),
        )

        error = _sync_error(
            lambda: maybe_build_flashinfer_checkpoint_plan(
                forward_batch, metadata, "cuda", use_host_metadata=True
            )
        )

        self.assertIsNone(error)
        self.assertEqual(metadata.track_ssm_h_src.tolist(), [1, 2, 3, 6])
        self.assertEqual(metadata.num_state_checkpoints, 7)


if __name__ == "__main__":
    unittest.main()
