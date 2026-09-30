"""BCG prefill must not change decode or verification fusion eligibility."""

from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from sglang.srt.model_executor.forward_context import ForwardContext, forward_context
from sglang.srt.models.inkling_common.kernels import comm
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


@pytest.mark.parametrize("scope", ["eager", "breakable", "full"])
@pytest.mark.parametrize("mode", ["extend", "decode", "verify"])
@pytest.mark.parametrize("scattered", [False, True])
def test_fusion_gate_preserves_non_prefill_behavior(scope, mode, scattered):
    batch = SimpleNamespace(
        forward_mode=SimpleNamespace(
            is_draft_extend_v2=lambda: False,
            is_extend=lambda: mode != "decode",
            is_decode=lambda: mode == "decode",
            is_target_verify=lambda: mode == "verify",
            is_extend_without_speculative=lambda: mode == "extend",
        )
    )
    group = SimpleNamespace(
        world_size=4,
        torch_symm_mem_comm=SimpleNamespace(disabled=False, dtype=torch.bfloat16),
    )
    gate = (
        comm.scattered_ar_sconv_fusable
        if scattered
        else comm.fullwidth_ar_sconv_fusable
    )
    with (
        forward_context(ForwardContext(attn_backend=None, full_graph=scope == "full")),
        patch.object(comm, "is_cuda", return_value=True),
        patch.object(
            comm, "is_in_breakable_cuda_graph", return_value=scope == "breakable"
        ),
        patch.object(
            comm,
            "get_exec",
            return_value=SimpleNamespace(
                comm=SimpleNamespace(enable_scattered_sconv=scattered)
            ),
        ),
        patch.object(
            comm.envs.SGLANG_OPT_USE_INKLING_CUSTOM_AR, "get", return_value=True
        ),
        patch.object(
            comm.envs.SGLANG_OPT_USE_INKLING_FUSED_AR_SCONV, "get", return_value=True
        ),
        patch.object(
            comm,
            "_get_inkling_ar_resources",
            return_value=SimpleNamespace(ssconv_out=4096 * 128),
        ),
    ):
        expected = (scattered or mode == "extend") and not (
            scope == "breakable" and mode == "extend"
        )
        assert gate(group, batch, 4096, 128, torch.bfloat16) == expected
