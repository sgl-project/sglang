"""The down-B into-base fold stage matrix on the extra tier: fused activation
middles and into-base tails against the materialized reference on both
provider row domains, plus the plan-flag check that rejects the overlap
windows racing the base GEMM. The per-PR parity and CUDA-graph guards stay in
test_moe_lora_down_b_into_base.py."""

from __future__ import annotations

from dataclasses import replace

import pytest
import torch

from sglang.srt.lora.moe.plan import (
    ActFamily,
    DownOverlap,
)
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.kits.moe_lora_down_b_kit import (
    _INTO_BASE_TOLERANCE,
    _build_runner,
    _into_base_pair,
    _make_gpu_tensors,
    _make_plan,
    _run_once,
    _standalone_output_allocation,
    _token_lora_mapping,
    _workspace_buffer_names,
    cutedsl_cuda_only,
)

register_cuda_ci(est_time=30, stage="extra-a", runner_config="1-gpu-large")


class TestDownBIntoBasePlan:
    def test_flag_rejects_the_windows_that_race_the_base_gemm(self) -> None:
        overlapped = _make_plan(down_overlap=DownOverlap.DOWN_A_B)
        with pytest.raises(ValueError, match="down-B into-base"):
            replace(overlapped, down_b_into_base=True)


@cutedsl_cuda_only
@pytest.mark.parametrize("base_gemm_rows", ("expert_major", "route_major"))
@pytest.mark.parametrize(
    "act_family,into_base",
    (
        (ActFamily.MATERIALIZED, True),
        (ActFamily.B_ACTIVATION, False),
        (ActFamily.B_ACTIVATION, True),
    ),
    ids=("into-base", "b-act", "b-act-into-base"),
)
def test_runner_fused_stages_match_the_materialized_reference(
    base_gemm_rows: str,
    act_family: ActFamily,
    into_base: bool,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Compare fused middles and tails across both provider row domains."""
    from sglang.srt.lora.moe.runner import MoeLoraRunner

    device = torch.device("cuda")
    monkeypatch.setattr(
        MoeLoraRunner, "_allocate_output", _standalone_output_allocation
    )
    serial_plan, _, launch_config = _into_base_pair()
    fused_plan = _make_plan(act_family=act_family, down_b_into_base=into_base)
    num_tokens, num_experts = 64, 4
    gpu = _make_gpu_tensors(num_tokens, num_experts, device)
    reference_runner = _build_runner(
        serial_plan, launch_config, "expert_major", gpu, num_experts
    )
    fused_runner = _build_runner(
        fused_plan, launch_config, base_gemm_rows, gpu, num_experts
    )
    # B_ACTIVATION adds the delta in FP32, unlike the BF16 materialized bridge.
    tolerance = (
        {"atol": 1e-2, "rtol": 0.05}
        if act_family is ActFamily.B_ACTIVATION
        else _INTO_BASE_TOLERANCE
    )

    for traffic in ("active", "mixed", "base_only"):
        token_lora_mapping = _token_lora_mapping(traffic, num_tokens).to(device)
        reference = _run_once(reference_runner, gpu, token_lora_mapping).hidden_states
        fused = _run_once(fused_runner, gpu, token_lora_mapping).hidden_states
        torch.testing.assert_close(
            fused,
            reference,
            **tolerance,
            msg=f"{base_gemm_rows}: {act_family.value} into_base={into_base} {traffic}",
        )

    # The scatter path never allocates the pair-major delta buffer for down-B.
    # The shipped path always allocates it.
    assert "down_b:delta" in _workspace_buffer_names(reference_runner)
    assert ("down_b:delta" not in _workspace_buffer_names(fused_runner)) == into_base


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
