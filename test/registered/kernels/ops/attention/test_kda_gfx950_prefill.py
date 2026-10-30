# SPDX-License-Identifier: Apache-2.0

import itertools
import unittest
from unittest.mock import patch

import torch

from sglang.kernels.ops.attention.fla.kda import chunk_kda
from sglang.kernels.ops.attention.linear import kda_gfx950_prefill
from sglang.srt.layers.attention.linear.kernels import kda_triton
from sglang.test.ci.ci_register import register_amd_ci
from sglang.test.test_utils import CustomTestCase

register_amd_ci(est_time=300, suite="stage-b-test-1-gpu-small-amd-mi35x")

SLOTS = 40


def _is_gfx950() -> bool:
    return bool(
        torch.version.hip
        and torch.cuda.is_available()
        and torch.cuda.get_device_properties().gcnArchName.split(":", 1)[0] == "gfx950"
    )


def _inputs(lengths, heads=4, seed=42) -> dict[str, torch.Tensor]:
    generator = torch.Generator(device="cuda").manual_seed(seed)
    shape = (1, sum(lengths), heads, 128)

    def randn(shape, dtype, scale):
        return (torch.randn(shape, generator=generator, device="cuda") * scale).to(
            dtype
        )

    return {
        "q": randn(shape, torch.bfloat16, 0.5),
        "k": randn(shape, torch.bfloat16, 0.5),
        "v": randn(shape, torch.bfloat16, 0.5),
        "g": randn(shape, torch.bfloat16, 0.5),
        "beta": randn(shape[:3], torch.bfloat16, 0.5),
        "A_log": torch.randn(heads, generator=generator, device="cuda"),
        "dt_bias": torch.randn(heads * 128, generator=generator, device="cuda"),
        "ssm_states": randn((SLOTS, heads, 128, 128), torch.float32, 0.1),
        "cache_indices": torch.randperm(SLOTS, generator=generator, device="cuda")[
            : len(lengths)
        ].to(torch.int32),
        "query_start_loc": torch.tensor(
            (0, *itertools.accumulate(lengths)), device="cuda", dtype=torch.int32
        ),
    }


def _clone(inputs):
    return {name: value.clone() for name, value in inputs.items()}


def _args(inputs):
    return (inputs[name] for name in ("q", "k", "v", "g", "beta"))


def _reference(inputs, **track):
    return chunk_kda(
        *_args(inputs),
        initial_state=inputs["ssm_states"],
        initial_state_indices=inputs["cache_indices"],
        use_qk_l2norm_in_kernel=True,
        cu_seqlens=inputs["query_start_loc"],
        A_log=inputs["A_log"],
        dt_bias=inputs["dt_bias"],
        lower_bound=-5.0,
        beta_is_raw=True,
        output_intermediate_states=bool(track),
        **track,
    )


def _hip(inputs, lower_bound=-5.0, **kwargs):
    return kda_gfx950_prefill.extend(
        *_args(inputs),
        ssm_states=inputs["ssm_states"],
        cache_indices=inputs["cache_indices"],
        query_start_loc=inputs["query_start_loc"],
        A_log=inputs["A_log"],
        dt_bias=inputs["dt_bias"],
        lower_bound=lower_bound,
        beta_is_raw=True,
        return_intermediate_states="track_state" in kwargs
        or "track_ssm_h_src" in kwargs,
        extend_seq_lens_cpu=inputs["query_start_loc"].diff().tolist(),
        **kwargs,
    )


@unittest.skipUnless(_is_gfx950(), "gfx950 KDA prefill requires MI35x")
class TestGfx950KDAPrefill(CustomTestCase):
    def _assert_close(self, actual, expected, inputs, actual_inputs, expected_inputs):
        torch.testing.assert_close(
            actual.float(), expected.float(), rtol=2e-2, atol=3e-2
        )
        torch.testing.assert_close(
            actual_inputs["ssm_states"],
            expected_inputs["ssm_states"],
            rtol=2e-2,
            atol=3e-2,
        )
        selected = torch.zeros(SLOTS, dtype=torch.bool, device="cuda")
        selected[inputs["cache_indices"].long()] = True
        self.assertTrue(
            torch.equal(
                actual_inputs["ssm_states"][~selected], inputs["ssm_states"][~selected]
            )
        )

    @torch.inference_mode()
    def test_triton_extend_runs_the_hip_kernel(self):
        inputs = _inputs((193, 137))
        with patch.object(
            kda_triton,
            "chunk_kda",
            side_effect=AssertionError("unexpected fallback to Triton chunk_kda"),
        ):
            result = kda_triton.TritonKDAKernel().extend(
                *_args(inputs),
                ssm_states=inputs["ssm_states"],
                cache_indices=inputs["cache_indices"],
                query_start_loc=inputs["query_start_loc"],
                A_log=inputs["A_log"],
                dt_bias=inputs["dt_bias"],
                lower_bound=-5.0,
                beta_is_raw=True,
            )
        self.assertIsInstance(result, torch.Tensor)

    @torch.inference_mode()
    def test_uncovered_inputs_fall_back(self):
        inputs = _inputs((193, 137))
        self.assertIsNone(_hip(inputs, lower_bound=None))
        self.assertIsNone(_hip(inputs, is_spec_decode=True))
        self.assertIsNone(_hip(inputs, chunk_indices=torch.zeros(1, 2, device="cuda")))
        # unaligned tracking without a snapshot buffer needs the per-chunk h
        self.assertIsNone(
            _hip(inputs, track_ssm_h_src=torch.ones(1, device="cuda").long())
        )

    @torch.inference_mode()
    def test_layouts_match_triton(self):
        # every recurrence path of kda_recur_path on a 256-CU MI35x
        for lengths, heads in (
            ((40, 63, 1), 12),  # fused: sequences of one chunk
            ((1024,) * 16, 12),  # fused: balanced, 3/4 pair per CU
            ((8192,), 12),  # grp: one long sequence
            ((300, 300), 4),  # ws4: few pairs
            ((4096,) * 4, 12),  # ws4
            ((17, 4096, 333, 8000), 12),  # ws4: a dominant long sequence
            ((2048,) * 8, 12),  # ws2
            ((1024,) * 12, 12),  # vs1
        ):
            with self.subTest(lengths=lengths[:4], sequences=len(lengths), heads=heads):
                inputs = _inputs(lengths, heads)
                actual_inputs, expected_inputs = _clone(inputs), _clone(inputs)
                expected = _reference(expected_inputs)
                actual = _hip(actual_inputs)
                torch.cuda.synchronize()
                self._assert_close(
                    actual, expected, inputs, actual_inputs, expected_inputs
                )

    @torch.inference_mode()
    def test_aligned_track_extend_skips_snapshot(self):
        inputs = _inputs((193, 137))
        actual_inputs, expected_inputs = _clone(inputs), _clone(inputs)
        expected = _reference(expected_inputs, track_state=None, track_chunk_idx=None)
        actual, h = _hip(
            actual_inputs, track_ssm_h_src=torch.empty(0, device="cuda").long()
        )
        torch.cuda.synchronize()
        self.assertIsNone(h)
        self._assert_close(actual, expected[0], inputs, actual_inputs, expected_inputs)

    @torch.inference_mode()
    def test_track_snapshots_match_triton(self):
        # chunk 3 lies in the first sequence group of the 4000-token walk, chunk
        # 40 in a later one
        for lengths, chunks, heads in (
            ((193, 137), (2, 1), 4),  # vs2
            ((4000,), (3,), 4),  # grp
            ((4000,), (40,), 4),  # grp
            ((1024,) * 8, (5,) * 8, 12),  # vs2
            ((256,) * 24, (1,) * 24, 12),  # vs1
        ):
            with self.subTest(lengths=lengths[:4], chunks=chunks[:4], heads=heads):
                inputs = _inputs(lengths, heads)
                actual_inputs, expected_inputs = _clone(inputs), _clone(inputs)
                track_chunk_idx = torch.tensor(chunks, device="cuda", dtype=torch.int64)
                expected_track = torch.zeros(
                    len(lengths), heads, 128, 128, device="cuda"
                )
                actual_track = torch.zeros_like(expected_track)
                expected = _reference(
                    expected_inputs,
                    track_state=expected_track,
                    track_chunk_idx=track_chunk_idx,
                )
                actual, h = _hip(
                    actual_inputs,
                    track_state=actual_track,
                    track_chunk_idx=track_chunk_idx,
                )
                torch.cuda.synchronize()
                self.assertIsNone(h)
                self._assert_close(
                    actual, expected[0], inputs, actual_inputs, expected_inputs
                )
                torch.testing.assert_close(
                    actual_track, expected_track, rtol=2e-2, atol=3e-2
                )


if __name__ == "__main__":
    unittest.main()
