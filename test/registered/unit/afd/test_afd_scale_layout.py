"""Independent CPU oracle for the K128 to K32 packed scale conversion."""

import pytest
import torch

from sglang.srt.layers.moe.mega_moe import _mega_moe_kernel_scale
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


@pytest.mark.parametrize("k", [128, 512, 1024])
def test_packed_scale_repeats_exponents_without_changing_weight_bytes(k):
    weight = torch.zeros((2, 3, k), dtype=torch.float8_e4m3fn)
    # Check padding bytes are ignored when K needs fewer than four exponents.
    count = max(4, k // 128)
    exponents = torch.arange(2 * 3 * count, dtype=torch.uint8).reshape(2, 3, count)
    scale = exponents.view(torch.int32)
    before = weight.view(torch.uint8).clone()
    actual = _mega_moe_kernel_scale(weight, scale)
    expected = torch.stack([exponents[..., j // 4] for j in range(k // 32)], dim=-1)
    torch.testing.assert_close(actual.contiguous().view(torch.uint8), expected)
    torch.testing.assert_close(weight.view(torch.uint8), before)
    assert actual.mT.is_contiguous()
    assert _mega_moe_kernel_scale(weight.to(torch.bfloat16), scale) is scale


def test_m2n_packed_scale_reuses_native_bytes_and_rejects_lossy_inputs():
    from sglang.srt.afd.contracts import AFDError
    from sglang.srt.afd.m2n import _packed_scale

    weight = torch.zeros(1, 128, 512, dtype=torch.float8_e4m3fn)
    packed = torch.zeros(1, 128, 1, dtype=torch.int32)
    packed.format_ue8m0 = True
    assert _packed_scale(weight, packed).data_ptr() == packed.data_ptr()
    for invalid in (torch.ones(1, 1, 4), torch.ones(1, 1, 4) * 1.5, torch.zeros(2)):
        if invalid.shape == (1, 1, 4) and invalid.max() == 1:
            invalid[0, 0, 0] = float("nan")
        with pytest.raises(
            AFDError, match="SCALE_LAYOUT_UNSUPPORTED|COMMON_UE8M0_RECIPE_REQUIRED"
        ):
            _packed_scale(weight, invalid)


@pytest.mark.parametrize("family", ["qwen", "glm5"])
def test_m2n_admission_pins_geometry_and_output_scale(family):
    from types import SimpleNamespace as NS

    from sglang.srt.afd.contracts import AFDError, AFDRole
    from sglang.srt.afd.m2n import joint_rank, m2n_model_spec

    config = NS(num_experts=128, moe_intermediate_size=1536, num_experts_per_tok=8)
    adapter = NS(inner=NS(config=config), hidden_size=4096, num_layers=94)
    if family == "glm5":
        config.__dict__.update(
            model_type="glm_moe_dsa",
            n_routed_experts=256,
            moe_intermediate_size=2048,
            first_k_dense_replace=3,
            moe_layer_freq=1,
            n_shared_experts=1,
            routed_scaling_factor=2.5,
        )
        adapter.hidden_size, adapter.num_layers = 6144, 78
    spec = m2n_model_spec(adapter)
    assert spec["identity"] == family
    assert spec["output_scale"] == (1.0 if family == "qwen" else 2.5)
    assert [joint_rank(role, 2) for role in (AFDRole.ATTENTION, AFDRole.FFN)] == [2, 10]
    config.num_experts_per_tok = 7
    with pytest.raises(AFDError, match="MODEL_LAYOUT_UNSUPPORTED"):
        m2n_model_spec(adapter)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v", "-x"]))
