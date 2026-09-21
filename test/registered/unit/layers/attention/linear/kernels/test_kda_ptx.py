"""Unit tests for the PTX KDA prefill routing wrapper."""

import unittest
from unittest.mock import Mock, patch

import torch

from sglang.srt.layers.attention.linear.kernels.kda_ptx import PtxKDAKernel
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class _RejectTriton:
    def extend(self, *args, **kwargs):
        raise AssertionError("native-eligible batch unexpectedly fell back to Triton")


class TestPtxKDATrackRouting(CustomTestCase):
    """Regression: a batch carrying the fp32 track snapshot buffer must not
    take the native PTX path — the kernel cannot write the buffer, and the
    backend copies it into the prefix-cache track slots unconditionally, so an
    unwritten buffer silently corrupts later cache restores.
    """

    def _make_kernel(self):
        kernel = PtxKDAKernel()
        kernel._ensure_loaded = lambda: None
        kernel._fwd = Mock(side_effect=AssertionError("native path must not run"))
        return kernel

    @staticmethod
    def _inputs(seq_lens=(64, 100)):
        total = sum(seq_lens)
        H, D = 2, 128

        def vals(offset):
            return torch.full((1, total, H, D), offset, dtype=torch.bfloat16)

        return {
            "q": vals(0.1),
            "k": vals(0.2),
            "v": vals(0.3),
            "g": vals(0.4),
            "beta": torch.zeros(1, total, H, dtype=torch.bfloat16),
            "ssm_states": torch.zeros(8, H, D, D, dtype=torch.float32),
            "cache_indices": torch.tensor([1, 3], dtype=torch.int32),
            "query_start_loc": torch.tensor(
                [0] + list(torch.tensor(seq_lens).cumsum(0).tolist()),
                dtype=torch.int32,
            ),
            "A_log": torch.zeros(H, dtype=torch.float32),
            "dt_bias": torch.zeros(H * D, dtype=torch.float32),
            "extend_seq_lens_cpu": list(seq_lens),
        }

    def test_batch_with_track_state_routes_to_triton(self):
        kernel = self._make_kernel()
        kernel._triton.extend = Mock(return_value="triton-out")
        x = self._inputs()
        track_state = torch.zeros(2, 2, 128, 128, dtype=torch.float32)
        track_chunk_idx = torch.tensor([1, -1], dtype=torch.int32)

        with patch(
            "sglang.srt.layers.attention.linear.kernels.kda_ptx.mamba_cache_chunk_size",
            return_value=64,
        ):
            out = kernel.extend(
                x["q"],
                x["k"],
                x["v"],
                x["g"],
                x["beta"],
                ssm_states=x["ssm_states"],
                cache_indices=x["cache_indices"],
                query_start_loc=x["query_start_loc"],
                A_log=x["A_log"],
                dt_bias=x["dt_bias"],
                return_intermediate_states=True,
                track_ssm_h_src=torch.tensor([1], dtype=torch.long),
                track_state=track_state,
                track_chunk_idx=track_chunk_idx,
                extend_seq_lens_cpu=x["extend_seq_lens_cpu"],
            )

        self.assertEqual(out, "triton-out")
        kernel._fwd.assert_not_called()
        kernel._triton.extend.assert_called_once()
        forwarded = kernel._triton.extend.call_args.kwargs
        self.assertIs(forwarded["track_state"], track_state)
        self.assertIs(forwarded["track_chunk_idx"], track_chunk_idx)

    def test_batch_without_track_state_stays_native(self):
        kernel = self._make_kernel()
        kernel._triton = _RejectTriton()
        h = torch.zeros(3, 2, 128, 128, dtype=torch.float32)

        def fake_fwd(*args, **kwargs):
            return [
                args[2].clone(),  # out == v
                kwargs["initial_state"].clone(),  # final_state
                *([None] * 8),
                h,  # result[10]
            ]

        kernel._fwd = fake_fwd
        x = self._inputs()

        out, h_out = kernel.extend(
            x["q"],
            x["k"],
            x["v"],
            x["g"],
            x["beta"],
            ssm_states=x["ssm_states"],
            cache_indices=x["cache_indices"],
            query_start_loc=x["query_start_loc"],
            A_log=x["A_log"],
            dt_bias=x["dt_bias"],
            return_intermediate_states=True,
            track_ssm_h_src=torch.empty(0, dtype=torch.long),
            extend_seq_lens_cpu=x["extend_seq_lens_cpu"],
        )

        self.assertEqual(tuple(out.shape), (1, 164, 2, 128))
        self.assertIs(h_out, h)


if __name__ == "__main__":
    unittest.main()
