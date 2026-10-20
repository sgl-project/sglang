# SPDX-License-Identifier: Apache-2.0
"""Cache-DiT step count = DiT calls per ComfyUI sampler run.

The table was checked on CPU against ComfyUI's k-diffusion samplers with a
denoiser that counts its calls, for N = 1, 2, 5, 10 and both terminal sigmas.
"""

import pytest

from sglang.multimodal_gen.apps.ComfyUI_SGLDiffusion.executors.cache_options import (
    expected_model_calls,
)

ONE_CALL = [
    "sample_euler",
    "sample_euler_ancestral",
    "sample_lms",
    "sample_dpmpp_2m",
    "sample_unipc",
]
TWO_CALL = [
    "sample_heun",
    "sample_dpm_2",
    "sample_dpm_2_ancestral",
    "sample_dpmpp_2s_ancestral",
    "sample_dpmpp_sde",
]
STEPS = [1, 2, 5, 10]


def _sigmas(n: int, zero_end: bool) -> list[float]:
    head = [10.0 - i for i in range(n)]
    return head + [0.0] if zero_end else head + [0.05]


@pytest.mark.parametrize("zero_end", [True, False])
@pytest.mark.parametrize("n", STEPS)
@pytest.mark.parametrize("name", ONE_CALL)
def test_one_call_samplers(name, n, zero_end) -> None:
    assert expected_model_calls(name, _sigmas(n, zero_end)) == n


@pytest.mark.parametrize("n", STEPS)
@pytest.mark.parametrize("name", TWO_CALL)
def test_two_call_samplers_skip_the_last_correction_at_sigma_zero(name, n) -> None:
    assert expected_model_calls(name, _sigmas(n, True)) == 2 * n - 1
    assert expected_model_calls(name, _sigmas(n, False)) == 2 * n


@pytest.mark.parametrize("zero_end", [True, False])
@pytest.mark.parametrize("n,calls", [(1, 1), (2, 3), (5, 12), (10, 27)])
def test_heunpp2(n, calls, zero_end) -> None:
    assert expected_model_calls("sample_heunpp2", _sigmas(n, zero_end)) == calls


def test_accepts_the_function_and_bounds_unknown_samplers() -> None:
    def sample_heun():
        pass

    assert expected_model_calls(sample_heun, _sigmas(5, True)) == 9
    assert expected_model_calls("sample_custom", _sigmas(5, True)) == 15
    assert expected_model_calls(None, _sigmas(5, True)) == 15
    assert expected_model_calls("sample_euler", [1.0]) == 0
