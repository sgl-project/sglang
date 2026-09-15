import pytest
import triton
import triton.language as tl

from sglang.srt.utils.common import next_power_of_2
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


@triton.jit
def _kernel_using_power_of_two(size: tl.constexpr):
    block_size: tl.constexpr = triton.next_power_of_2(size)
    return block_size


def test_sglang_power_of_two_can_be_referenced_from_triton_jit():
    # Kernel hashing validates helper references before GPU compilation.
    # A plain Python replacement raises "Unsupported function referenced".
    assert _kernel_using_power_of_two.cache_key


@pytest.mark.parametrize("size, expected", [(0, 1), (1, 1), (3, 4), (16, 16), (31, 32)])
def test_power_of_two_keeps_host_call_semantics(size, expected):
    assert triton.next_power_of_2(size) == next_power_of_2(size) == expected
