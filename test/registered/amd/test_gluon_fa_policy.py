"""Host policy and diffusion adapter checks for existing upstream AMD runners."""

import sys
from pathlib import Path

import pytest
import torch

from sglang.kernels.ops.attention.gluon.fa3.gfx1250.policy import (
    LaunchConfig,
    NumericalMode,
    resolve_execution_policy,
)
from sglang.test.ci.ci_register import register_amd_ci

register_amd_ci(est_time=30, suite="stage-a-test-1-gpu-small-amd")
register_amd_ci(est_time=30, suite="nightly-amd-1-gpu", nightly=True)


def test_persistent_launch_selection_cpu():
    from sglang.kernels.ops.attention.gluon.fa3.gfx1250.policy import (
        select_launch_config,
    )

    config = select_launch_config(
        2,
        40,
        176_400,
        seqlen_k=512,
        num_cus=1,
    )
    assert config.schedule == "persistent"
    assert (config.block_m, config.block_n, config.num_warps) == (128, 256, 4)
    assert config.tdm_output
    assert config.llvm_fn_attrs == "amdgpu-sched-strategy=iterative-ilp"

    non_wan = select_launch_config(
        1,
        40,
        32_768,
        seqlen_k=511,
        num_cus=256,
    )
    assert non_wan.schedule == "persistent"

    underfilled = select_launch_config(
        1,
        8,
        4_096,
        seqlen_k=512,
        num_cus=256,
    )
    assert underfilled.schedule != "persistent"

    for unsupported_k in (256, 513):
        unsupported = select_launch_config(
            2,
            40,
            176_400,
            seqlen_k=unsupported_k,
            num_cus=256,
        )
        assert unsupported.schedule != "persistent"

    stable_self_config = select_launch_config(
        2,
        40,
        176_400,
        seqlen_k=176_400,
        num_cus=256,
    )
    assert stable_self_config.schedule == "pingpong"
    single_video = select_launch_config(1, 40, 176_400, seqlen_k=176_400, num_cus=256)
    assert single_video.schedule == "pingpong"


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("head_size", [64, 128, 256])
def test_extended_launch_selection_cpu(dtype, head_size):
    from sglang.kernels.ops.attention.gluon.fa3.gfx1250.policy import (
        select_launch_config,
    )

    pipeline_scheduler = (
        "" if dtype == torch.float16 else "amdgpu-sched-strategy=max-ilp"
    )
    small_config = select_launch_config(
        2,
        3,
        129,
        seqlen_k=257,
        num_cus=256,
        head_size=head_size,
        dtype=dtype,
    )
    assert small_config.schedule == "pipeline"
    assert small_config.llvm_fn_attrs == pipeline_scheduler

    config = select_launch_config(
        1,
        40,
        32768,
        seqlen_k=511,
        num_cus=256,
        head_size=head_size,
        dtype=dtype,
    )
    if dtype == torch.bfloat16 and head_size == 128:
        assert config.schedule == "persistent"
    elif head_size == 256:
        assert config.schedule == "pipeline"
        assert not config.tdm_output
        assert config.llvm_fn_attrs == pipeline_scheduler
    else:
        assert config.schedule == "pingpong"
        assert config.tdm_output is (head_size == 128)


def _meta_inputs(sq=65536, sk=65536, head_size=128, dtype=torch.bfloat16):
    query = torch.empty(1, sq, 40, head_size, device="meta", dtype=dtype)
    key = torch.empty(1, sk, 40, head_size, device="meta", dtype=dtype)
    return query, key, torch.empty_like(key)


@pytest.mark.parametrize("operand", [0, 1, 2])
def test_resolver_reports_contiguity_fallback(operand):
    inputs = list(_meta_inputs(sq=32768, sk=511))
    original = inputs[operand]
    shape = list(original.shape)
    shape[1] *= 2
    inputs[operand] = torch.empty(shape, device="meta", dtype=original.dtype)[:, ::2]
    assert not inputs[operand].is_contiguous()
    policy = resolve_execution_policy(*inputs, num_cus=256)
    assert policy.launch_config.schedule == "pingpong"
    assert policy.launch_config.tdm_output
    assert policy.numerical_mode == NumericalMode.ONLINE


def test_resolver_reports_persistent_scheduler():
    inputs = _meta_inputs(sq=32768, sk=511)
    explicit = LaunchConfig("persistent", 128, 256, 4, True, "")
    for override in (None, explicit):
        policy = resolve_execution_policy(*inputs, launch_config=override, num_cus=256)
        assert policy.launch_config.schedule == "persistent"
        assert policy.numerical_mode == NumericalMode.PERSISTENT
        assert (
            policy.launch_config.llvm_fn_attrs == "amdgpu-sched-strategy=iterative-ilp"
        )


@pytest.mark.parametrize(
    "sq,sk,scale,forced,num_cus,expected",
    [
        (65536, 65536, None, False, 256, NumericalMode.GUARDED),
        (65535, 65536, None, False, 256, NumericalMode.ONLINE),
        (65536, 65535, None, False, 256, NumericalMode.ONLINE),
        (65536, 2**31 - 1, None, False, 256, NumericalMode.GUARDED),
        (65536, 2**31, None, False, 256, NumericalMode.ONLINE),
        (65536, 65536, 0.25, False, 256, NumericalMode.ONLINE),
        (65536, 65536, None, True, 256, NumericalMode.ONLINE),
        (65536, 65536, None, False, 65536, NumericalMode.ONLINE),
    ],
)
def test_resolver_guarded_boundaries(sq, sk, scale, forced, num_cus, expected):
    config = LaunchConfig("pingpong", 256, 64, 8, True, "") if forced else None
    policy = resolve_execution_policy(
        *_meta_inputs(sq=sq, sk=sk),
        softmax_scale=scale,
        launch_config=config,
        num_cus=num_cus,
    )
    assert policy.numerical_mode == expected
    assert policy.softmax_scale == (128**-0.5 if scale is None else scale)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("head_size", [64, 128, 256])
def test_resolver_guarded_dtype_and_head_contract(dtype, head_size):
    policy = resolve_execution_policy(
        *_meta_inputs(dtype=dtype, head_size=head_size), num_cus=256
    )
    expected = (
        NumericalMode.GUARDED
        if dtype == torch.bfloat16 and head_size == 128
        else NumericalMode.ONLINE
    )
    assert policy.numerical_mode == expected


def test_resolver_uses_metadata_without_device_queries(monkeypatch):
    from sglang.kernels.ops.attention.gluon.fa3.gfx1250 import policy as gfx1250_policy

    def unexpected_device_query(*args, **kwargs):
        raise AssertionError("metadata inspection must not query a GPU")

    monkeypatch.setattr(gfx1250_policy, "get_num_cus", unexpected_device_query)
    policy = resolve_execution_policy(*_meta_inputs(), num_cus=256)
    assert policy.numerical_mode == NumericalMode.GUARDED


@pytest.mark.parametrize("scale", [0.0, -1.0, float("inf"), float("nan")])
def test_resolver_rejects_invalid_scale(scale):
    with pytest.raises(ValueError, match="softmax_scale must be finite and positive"):
        resolve_execution_policy(*_meta_inputs(), softmax_scale=scale, num_cus=256)


@pytest.mark.parametrize(
    "config,message",
    [
        (
            LaunchConfig("unknown", 128, 64, 4, False, ""),
            "unsupported FAv3 schedule",
        ),
        (LaunchConfig("pipeline", 64, 64, 4, False, ""), "pipeline requires"),
        (LaunchConfig("pingpong", 128, 64, 8, True, ""), "pingpong requires"),
        (LaunchConfig("persistent", 128, 64, 4, True, ""), "persistent requires"),
        (LaunchConfig("persistent", 128, 256, 4, False, ""), "TDM output"),
    ],
)
def test_resolver_rejects_invalid_launch_geometry(config, message):
    with pytest.raises(ValueError, match=message):
        resolve_execution_policy(
            *_meta_inputs(sk=511), launch_config=config, num_cus=256
        )


if __name__ == "__main__":
    # The existing adapter tests mock kernel calls and run on any upstream AMD
    # runner. Keep them in this registered entry point rather than the package.
    backend_tests = (
        Path(__file__).resolve().parents[3]
        / "python/sglang/multimodal_gen/test/unit/test_gluon_fa_backend.py"
    )
    sys.exit(pytest.main([__file__, str(backend_tests), "-v"]))
