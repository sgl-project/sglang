# SPDX-License-Identifier: Apache-2.0

import itertools
import unittest
from unittest.mock import patch

import torch

from sglang.kernels.ops.attention.fla.kda import chunk_kda
from sglang.kernels.ops.attention.linear import kda_aiter_prefill
from sglang.srt.environ import envs
from sglang.srt.layers.attention.linear.kernels import kda_triton
from sglang.test.ci.ci_register import register_amd_ci
from sglang.test.test_utils import CustomTestCase

register_amd_ci(est_time=60, suite="stage-b-test-1-gpu-small-amd-mi35x")

SLOTS = 40
LOWER_BOUND = -5.0


def _inputs(lengths, state_dtype, heads=12, seed=0) -> dict[str, torch.Tensor]:
    generator = torch.Generator(device="cuda").manual_seed(seed)
    shape = (1, sum(lengths), heads, 128)

    def randn(shape, dtype, scale=1.0):
        return (torch.randn(shape, generator=generator, device="cuda") * scale).to(
            dtype
        )

    return {
        "q": randn(shape, torch.bfloat16),
        "k": randn(shape, torch.bfloat16),
        "v": randn(shape, torch.bfloat16),
        "g": randn(shape, torch.bfloat16),
        "beta": randn(shape[:3], torch.bfloat16),
        "A_log": torch.log(
            torch.empty(heads, device="cuda").uniform_(1, 16, generator=generator)
        ),
        "dt_bias": torch.randn(heads * 128, generator=generator, device="cuda"),
        "ssm_states": randn((SLOTS, heads, 128, 128), state_dtype, 0.1),
        # slot 0 is the pool's reserved padding slot
        "cache_indices": (
            torch.randperm(SLOTS - 1, generator=generator, device="cuda")[
                : len(lengths)
            ]
            + 1
        ).to(torch.int32),
        "query_start_loc": torch.tensor(
            (0, *itertools.accumulate(lengths)), device="cuda", dtype=torch.int32
        ),
    }


def _clone(inputs):
    return {name: value.clone() for name, value in inputs.items()}


def _args(inputs):
    return [inputs[name] for name in ("q", "k", "v", "g", "beta")]


def _kwargs(inputs, **overrides):
    kwargs = {
        "ssm_states": inputs["ssm_states"],
        "cache_indices": inputs["cache_indices"],
        "query_start_loc": inputs["query_start_loc"],
        "A_log": inputs["A_log"],
        "dt_bias": inputs["dt_bias"],
        "lower_bound": LOWER_BOUND,
        "beta_is_raw": True,
    }
    kwargs.update(overrides)
    return kwargs


def _reference(inputs):
    return chunk_kda(
        *_args(inputs),
        initial_state=inputs["ssm_states"],
        initial_state_indices=inputs["cache_indices"],
        use_qk_l2norm_in_kernel=True,
        cu_seqlens=inputs["query_start_loc"],
        A_log=inputs["A_log"],
        dt_bias=inputs["dt_bias"],
        lower_bound=LOWER_BOUND,
        beta_is_raw=True,
    )


def _rms_ratio(actual: torch.Tensor, expected: torch.Tensor) -> float:
    actual, expected = actual.double(), expected.double()
    return ((actual - expected).norm() / expected.norm()).item()


class TestAiterKDAPrefill(CustomTestCase):
    def setUp(self):
        if kda_aiter_prefill._aiter_chunk_kda_module() is None:
            self.skipTest("needs gfx950 and AITER Gluon chunk_kda")

    @torch.inference_mode()
    def test_bf16_pool_extend_runs_aiter(self):
        inputs = _inputs((193, 137), torch.bfloat16)
        with patch.object(
            kda_triton,
            "chunk_kda",
            side_effect=AssertionError("unexpected fallback to Triton chunk_kda"),
        ):
            result = kda_triton.TritonKDAKernel().extend(
                *_args(inputs), **_kwargs(inputs)
            )
        self.assertIsInstance(result, torch.Tensor)

    @torch.inference_mode()
    def test_matches_triton(self):
        for state_dtype, lengths in itertools.product(
            (torch.bfloat16, torch.float32),
            ((64, 96), (40, 63, 1), (8192,), (17, 4096, 333)),
        ):
            with self.subTest(state_dtype=state_dtype, lengths=lengths):
                inputs = _inputs(lengths, state_dtype)
                actual_inputs, expected_inputs = _clone(inputs), _clone(inputs)
                expected = _reference(expected_inputs)
                actual = kda_aiter_prefill.extend(
                    *_args(actual_inputs), **_kwargs(actual_inputs)
                )
                rows = inputs["cache_indices"].long()
                self.assertLess(_rms_ratio(actual, expected), 1e-2)
                self.assertLess(
                    _rms_ratio(
                        actual_inputs["ssm_states"][rows],
                        expected_inputs["ssm_states"][rows],
                    ),
                    1e-2,
                )
                untouched = torch.ones(SLOTS, dtype=torch.bool, device="cuda")
                untouched[rows] = False
                self.assertTrue(
                    torch.equal(
                        actual_inputs["ssm_states"][untouched],
                        inputs["ssm_states"][untouched],
                    )
                )

    @torch.inference_mode()
    def test_uncovered_inputs_fall_back(self):
        inputs = _inputs((193, 137), torch.bfloat16)
        args = _args(inputs)
        for overrides in (
            {"return_intermediate_states": True},
            {"track_state": torch.zeros(2, 12, 128, 128, device="cuda")},
            {"chunk_indices": torch.zeros(1, 2, device="cuda", dtype=torch.int32)},
            {"lower_bound": None},
            {"lower_bound": -6.0},
            {"beta_is_raw": False},
        ):
            with self.subTest(overrides=list(overrides)):
                self.assertIsNone(
                    kda_aiter_prefill.extend(*args, **_kwargs(inputs, **overrides))
                )
        kda_aiter_prefill._aiter_chunk_kda_module.cache_clear()
        try:
            with envs.SGLANG_OPT_HIP_KDA_AITER_PREFILL.override(False):
                self.assertIsNone(kda_aiter_prefill.extend(*args, **_kwargs(inputs)))
        finally:
            kda_aiter_prefill._aiter_chunk_kda_module.cache_clear()


if __name__ == "__main__":
    unittest.main()
