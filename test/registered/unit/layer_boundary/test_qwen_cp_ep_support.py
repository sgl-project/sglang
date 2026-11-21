"""Qwen CP with whole-expert partitioning: topology and boundary regressions."""

from types import SimpleNamespace as NS
from unittest.mock import patch

import pytest
import test_token_order_attention as boundary_fixture

from sglang.srt.arg_groups import cp_tp_group_sharing_hook as hook
from sglang.srt.layers.layer_boundary import layout
from sglang.srt.layers.moe import utils as moe_utils
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=12, suite="base-a-test-cpu")


def resolve(ep=2, tp=2, a2a="none", mode="prefill"):
    cfg = NS(
        enable_prefill_cp=True,
        tp_size=tp,
        cp_strategy="zigzag",
        speculative_algorithm=None,
        enable_dsa_cache_layer_split=False,
        dcp_size=1,
        enable_mixed_chunk=False,
        moe_dp_size=1,
        enable_waterfill=False,
        pp_size=1,
        disaggregation_mode=mode,
        cuda_graph_config=None,
    )
    view = NS(attn_cp_size=tp, ep_size=ep, moe_a2a_backend=a2a)
    model = NS(hf_config=NS(architectures=["Qwen4ExpForConditionalGeneration"]))
    linear = NS(linear_num_key_heads=16, linear_num_value_heads=32)
    with (
        patch.object(hook, "resolving_view", return_value=cfg),
        patch.object(hook, "resolved_view", return_value=view),
        patch.object(hook, "mambaish_config", return_value=linear),
        patch.object(hook.envs.SGLANG_DISAGG_STAGING_BUFFER, "get", return_value=False),
        patch.object(hook, "with_phase", return_value=None),
        patch.object(hook, "declare_resolution") as declare,
    ):
        hook.resolve_cp_tp_group_sharing(object(), model)
        assert declare.call_args.kwargs["enable_cp_tp_group_sharing"]


@pytest.mark.parametrize("tp,ep", [(2, 1), (2, 2), (4, 1), (4, 4)])
def test_whole_expert_or_tensor_partition(tp, ep):
    resolve(ep=ep, tp=tp)


def test_mixed_expert_tensor_subgroups_rejected():
    with pytest.raises(ValueError, match="ep-size"):
        resolve(ep=2, tp=4)


@pytest.mark.parametrize("a2a", ["deepep", "nixl"])
def test_token_dispatch_backends_stay_rejected(a2a):
    with pytest.raises(ValueError, match="moe-a2a-backend"):
        resolve(a2a=a2a)


def test_decode_cp_stays_rejected():
    with pytest.raises(ValueError, match="disaggregation-mode decode"):
        resolve(mode="decode")


class TestWholeExpertCpBoundaries(boundary_fixture.TestTokenOrderAttention):
    """Run existing Qwen GDN and sparse-attention CP contracts with EP2."""

    def setUp(self):
        parallel = boundary_fixture.fixture.parallel_of(
            attn_dp=1,
            attn_tp=1,
            attn_cp=2,
            enable_prefill_cp=True,
            enable_cp_tp_group_sharing=True,
            moe_ep_size=2,
            moe_tp_size=1,
        )
        self.linear, self.sparse = boundary_fixture._build_layers(parallel)


def test_whole_expert_output_sum_uses_expert_group():
    ep_group, tp_group = object(), object()
    parallel = NS(
        moe_ep_size=2,
        moe_tp_size=1,
        moe_ep_group=ep_group,
        moe_tp_group=object(),
        tp_group=tp_group,
    )
    with (
        patch.object(moe_utils, "get_parallel", return_value=parallel),
        patch.object(
            moe_utils, "can_merge_post_experts_all_reduce", return_value=False
        ),
    ):
        assert moe_utils.post_experts_reduction_group() is ep_group
        assert layout._sum_group(layout.SumGroup.MOE_OUTPUT) is ep_group


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__]))
