"""CPU-only tests for experimental_sgl_marlin's correctness contract."""

from __future__ import annotations

import types

import pytest

from sglang.srt.lora.marlin_lora_temp.policy import (
    use_post_reduce_down_delta,
    validate_experimental_sgl_marlin_contract,
    validate_experimental_sgl_marlin_server_args,
)
from sglang.srt.lora.trtllm_lora_temp.specialized_expand import _get_gated_a_half
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=5, stage="base-b", runner_config="1-gpu-small")


def _config(**overrides):
    values = dict(
        activation="silu",
        is_gated=True,
        gemm1_alpha=None,
        gemm1_clamp_limit=None,
        swiglu_limit=None,
        apply_router_weight_on_input=False,
        no_combine=False,
        num_experts=256,
        num_local_experts=256,
    )
    values.update(overrides)
    return types.SimpleNamespace(**values)


def _validate(config=None, **overrides):
    values = dict(
        runner_config=config or _config(),
        moe_ep_size=1,
        device_capability=(9, 0),
    )
    values.update(overrides)
    return validate_experimental_sgl_marlin_contract(**values)


def _validate_server(**overrides):
    ep_size = overrides.pop("ep_size", 4)
    moe_a2a_backend = overrides.pop("moe_a2a_backend", "none")
    values = dict(
        enable_lora=True,
        lora_paths=[],
        lora_use_virtual_experts=True,
        init_expert_location="trivial",
        ep_num_redundant_experts=0,
        enable_eplb=False,
        elastic_ep_backend=None,
        enable_elastic_expert_backup=False,
        elastic_ep_rejoin=False,
        experts_shared_outer_loras=False,
        max_lora_rank=64,
        lora_backend="triton",
    )
    values.update(overrides)
    return validate_experimental_sgl_marlin_server_args(
        types.SimpleNamespace(**values),
        types.SimpleNamespace(
            ep_size=ep_size,
            moe_a2a_backend=moe_a2a_backend,
        ),
    )


def test_supported_contract_passes():
    _validate()


def test_supported_ep_contract_passes():
    _validate(config=_config(num_local_experts=64), moe_ep_size=4)


@pytest.mark.parametrize(
    ("enable_lora", "lora_paths"),
    [(None, []), (False, []), (False, ["ignored=/tmp/adapter"])],
)
def test_base_only_ep_preserves_stock_marlin_placement_support(enable_lora, lora_paths):
    _validate_server(
        enable_lora=enable_lora,
        lora_paths=lora_paths,
        init_expert_location="random",
        ep_num_redundant_experts=1,
        enable_eplb=True,
        elastic_ep_backend="mooncake",
        enable_elastic_expert_backup=True,
        elastic_ep_rejoin=True,
    )


def test_base_only_ep_rejects_unsupported_a2a():
    with pytest.raises(ValueError, match="moe-a2a-backend none"):
        _validate_server(enable_lora=False, moe_a2a_backend="deepep")


def test_lora_rejects_non_triton_backend():
    with pytest.raises(ValueError, match="requires --lora-backend triton"):
        _validate_server(lora_backend="csgmv")
    # base-only servers are free to pick any dense backend
    _validate_server(enable_lora=False, lora_backend="csgmv")


@pytest.mark.parametrize("ep_size", [1, 4])
def test_lora_requires_virtual_experts(ep_size):
    with pytest.raises(ValueError, match="lora-use-virtual-experts"):
        _validate_server(ep_size=ep_size, lora_use_virtual_experts=False)


@pytest.mark.parametrize(
    "setting",
    [
        {"init_expert_location": "random"},
        {"ep_num_redundant_experts": 1},
        {"enable_eplb": True},
        {"elastic_ep_backend": "mooncake"},
        {"enable_elastic_expert_backup": True},
        {"elastic_ep_rejoin": True},
    ],
)
def test_lora_ep_rejects_nontrivial_placement_features(setting):
    with pytest.raises(ValueError, match="trivial expert placement"):
        _validate_server(**setting)


def test_adapter_paths_implicitly_enable_lora_ep_validation():
    with pytest.raises(ValueError, match="trivial expert placement"):
        _validate_server(
            enable_lora=None,
            lora_paths=["adapter=/tmp/adapter"],
            enable_eplb=True,
        )


def test_supported_lora_ep_passes():
    _validate_server(experts_shared_outer_loras=True, max_lora_rank=64)


@pytest.mark.parametrize("rank", [65, 128, 256])
def test_shared_outer_lora_ep_allows_generic_rank_fallback(rank):
    _validate_server(experts_shared_outer_loras=True, max_lora_rank=rank)


@pytest.mark.parametrize(
    ("config", "message"),
    [
        (_config(activation="relu2"), "activation must be 'silu'"),
        (_config(is_gated=False), "only gated SwiGLU"),
        (_config(gemm1_alpha=1.0), "gemm1_alpha"),
        (_config(gemm1_clamp_limit=7.0), "gemm1_clamp_limit"),
        (_config(swiglu_limit=7.0), "swiglu_limit"),
        (
            _config(apply_router_weight_on_input=True),
            "apply_router_weight_on_input",
        ),
        (_config(no_combine=True), "no_combine"),
    ],
)
def test_rejects_unimplemented_activation_semantics(config, message):
    with pytest.raises(ValueError, match=message):
        _validate(config)


@pytest.mark.parametrize(
    "overrides",
    [
        {"config": _config(num_local_experts=128)},
        {"config": _config(num_local_experts=63), "moe_ep_size": 4},
        {"config": _config(num_experts=255, num_local_experts=64), "moe_ep_size": 4},
        {"moe_ep_size": 0},
    ],
)
def test_rejects_incoherent_expert_parallelism(overrides):
    config = overrides.get("config")
    validation_overrides = {k: v for k, v in overrides.items() if k != "config"}
    with pytest.raises(ValueError, match="moe_ep_size|num_local_experts"):
        _validate(config, **validation_overrides)


def test_rejects_pre_hopper_gpu():
    with pytest.raises(ValueError, match="compute capability 9.0 or newer"):
        _validate(device_capability=(8, 0))


def test_direct_expand_always_splits_gated_gate_up_a():
    assert _get_gated_a_half(intermediate_width=64, rank=32, output_width=768) == 384
    assert _get_gated_a_half(intermediate_width=32, rank=32, output_width=6144) == 0


def test_direct_expand_rejects_invalid_intermediate_width():
    with pytest.raises(ValueError, match="intermediate width"):
        _get_gated_a_half(intermediate_width=48, rank=32, output_width=768)


@pytest.mark.parametrize(
    ("run_lora", "scale", "num_tokens", "expected"),
    [
        (True, 1.0, 1, True),
        (True, 1.0, 2048, True),
        (True, 1.0, 2049, False),
        (True, 0.5, 32, False),
        (False, 1.0, 32, False),
    ],
)
def test_post_reduce_down_policy(run_lora, scale, num_tokens, expected):
    assert (
        use_post_reduce_down_delta(
            run_lora=run_lora,
            routed_scaling_factor=scale,
            num_tokens=num_tokens,
        )
        is expected
    )
