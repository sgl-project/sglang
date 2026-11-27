import sys

import pytest

import sglang.kernels.ops.layernorm.hy4_ihc as hy4_ihc
from sglang.srt.environ import envs
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

OPS = ("fuse_ihc_pre", "fuse_ihc_post_pre", "fuse_ihc_post", "fuse_ihc_head")
HPC_OP = object()


@pytest.fixture
def policy(monkeypatch):
    def apply(value, *, hpc_available=True, threshold=128):
        monkeypatch.setattr(
            hy4_ihc, "_hpc_ihc_op", lambda *_: HPC_OP if hpc_available else None
        )
        monkeypatch.setenv(envs.SGLANG_HY4_IHC_DECODE_POLICY.name, value)
        monkeypatch.setenv(envs.SGLANG_HY4_IHC_PRE_T_THRESHOLD.name, str(threshold))
        hy4_ihc._ihc_policy.cache_clear()

    yield apply
    hy4_ihc._ihc_policy.cache_clear()


def _backends(num_tokens, is_decode):
    with hy4_ihc.hy4_ihc_forward_mode(is_decode):
        return tuple(
            "hpc"
            if hy4_ihc._selected_hpc_ihc_op(op, 4, 6144, num_tokens) is HPC_OP
            else "triton"
            for op in OPS
        )


ALL_HPC = ("hpc",) * 4
ALL_TRITON = ("triton",) * 4
TRITON_PRE = ("triton", "hpc", "hpc", "hpc")


@pytest.mark.parametrize(
    "value,num_tokens,is_decode,expected",
    [
        ("hpc", 1, True, ALL_HPC),
        ("hpc", 512, False, ALL_HPC),
        ("triton", 1, True, ALL_TRITON),
        ("triton", 512, False, ALL_TRITON),
        ("mode", 1, False, ALL_HPC),
        ("mode", 512, False, ALL_HPC),
        ("mode", 1, True, ALL_TRITON),
        ("mode", 512, True, ALL_TRITON),
        ("batch", 1, True, TRITON_PRE),
        ("batch", 1, False, TRITON_PRE),
        ("batch", 512, True, ALL_HPC),
        ("batch", 512, False, ALL_HPC),
    ],
)
def test_policy_selects_backend_per_op(policy, value, num_tokens, is_decode, expected):
    policy(value)
    assert _backends(num_tokens, is_decode) == expected


@pytest.mark.parametrize(
    "num_tokens,expected", [(127, TRITON_PRE), (128, ALL_HPC), (129, ALL_HPC)]
)
def test_batch_policy_switches_pre_to_hpc_at_the_threshold(
    policy, num_tokens, expected
):
    policy("batch")
    assert _backends(num_tokens, is_decode=True) == expected


@pytest.mark.parametrize("value", ["hpc", "triton", "mode", "batch"])
def test_every_policy_falls_back_without_hpc(policy, value):
    policy(value, hpc_available=False)
    assert _backends(1, is_decode=False) == ALL_TRITON
    assert _backends(512, is_decode=True) == ALL_TRITON


@pytest.mark.parametrize("hpc_available", [True, False])
def test_unknown_policy_raises_even_without_hpc(policy, hpc_available):
    policy("typo", hpc_available=hpc_available)
    with pytest.raises(ValueError, match="SGLANG_HY4_IHC_DECODE_POLICY='typo'"):
        hy4_ihc._selected_hpc_ihc_op("fuse_ihc_pre", 4, 6144, 1)


def test_forward_mode_is_restored_after_nesting_and_errors(policy):
    policy("mode")
    assert _backends(1, is_decode=False) == ALL_HPC
    with hy4_ihc.hy4_ihc_forward_mode(True):
        with hy4_ihc.hy4_ihc_forward_mode(False):
            assert hy4_ihc._ihc_decode_forward.get() is False
        assert hy4_ihc._ihc_decode_forward.get() is True
        with pytest.raises(RuntimeError):
            with hy4_ihc.hy4_ihc_forward_mode(False):
                raise RuntimeError
        assert hy4_ihc._ihc_decode_forward.get() is True
    assert hy4_ihc._ihc_decode_forward.get() is False


if __name__ == "__main__":
    sys.exit(pytest.main([__file__]))
