"""Opt-in gfx942 regression: real TopK + Triton runner + model shared add.

Run in separate fresh processes after a GPU is explicitly reserved:
  SGLANG_TEST_DSV4_MOE_GPU=1 SGLANG_USE_AITER=0 python -m pytest <this file> -q
  SGLANG_TEST_DSV4_MOE_GPU=1 SGLANG_USE_AITER=1 python -m pytest <this file> -q

No checkpoint or distributed process group is needed. The expert adapter owns
an actual MoeRunner; it skips only weight loading/collectives and supplies
small deterministic BF16 matrices, keeping this independent of FP8 encoding.
Without the opt-in, collection performs no device discovery or GPU imports.
"""

import os
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import patch

import pytest

pytestmark = pytest.mark.skipif(
    os.environ.get("SGLANG_TEST_DSV4_MOE_GPU") != "1",
    reason="requires explicitly reserved gfx942 GPU and opt-in",
)


@pytest.mark.parametrize(
    "layer_id,is_nextn", [(0, False), (2, False), (3, False), (0, True)]
)
@pytest.mark.parametrize("hash_field", ["num_hash_layers", "n_hash_layers"])
@pytest.mark.parametrize("dual_stream", [False, True])
@pytest.mark.parametrize("fused_runner", [False, True])
@pytest.mark.parametrize("fused_hash", [False, True])
def test_dsv4_moe_scale_gpu(
    layer_id, is_nextn, hash_field, dual_stream, fused_runner, fused_hash
):
    import torch
    import torch.nn.functional as F

    from sglang.srt.utils import is_gfx942_supported

    if not is_gfx942_supported():
        pytest.skip("requires gfx942")

    import sglang.srt.layers.moe.moe_runner.triton_utils.fused_moe as fused_moe
    import sglang.srt.layers.moe.topk as topk_module
    import sglang.srt.models.deepseek_v2 as model
    from sglang.srt.environ import envs
    from sglang.srt.layers.moe.moe_runner.base import MoeRunnerConfig
    from sglang.srt.layers.moe.moe_runner.runner import MoeRunner
    from sglang.srt.layers.moe.moe_runner.triton import TritonMoeQuantInfo
    from sglang.srt.layers.moe.token_dispatcher.standard import StandardDispatchOutput
    from sglang.srt.layers.moe.utils import MoeA2ABackend, MoeRunnerBackend
    from sglang.srt.runtime_context import get_flags, get_parallel
    from sglang.srt.server_args import ServerArgs, set_global_server_args_for_scheduler

    set_global_server_args_for_scheduler(ServerArgs(model_path="dummy"))
    flags = get_flags().moe
    size, experts, rows, top_k = 128, 256, 7, 6
    config = SimpleNamespace(
        architectures=["DeepseekV4ForCausalLM"],
        n_routed_experts=experts,
        n_shared_experts=0,
        hidden_size=size,
        hidden_act="silu",
        topk_method="noaux_tc",
        scoring_func="sqrtsoftplus",
        num_experts_per_tok=top_k,
        vocab_size=16,
        moe_intermediate_size=size,
        norm_topk_prob=True,
        n_group=1,
        topk_group=1,
        routed_scaling_factor=1.5,
        swiglu_limit=10,
        **{hash_field: 3},
    )
    fixed_gate = torch.linspace(-1, 2, experts, device="cuda")
    fixed_up = torch.linspace(0.5, 2, experts, device="cuda")

    class SmallExperts:
        def __init__(self, **kwargs):
            self.should_fuse_routed_scaling_factor_in_topk = False
            self.quant_method = object()
            self.moe_runner_config = MoeRunnerConfig(
                num_experts=experts,
                num_local_experts=experts,
                top_k=top_k,
                inplace=False,
                routed_scaling_factor=kwargs["routed_scaling_factor"],
                swiglu_limit=10,
            )
            self.runner = MoeRunner(MoeRunnerBackend.TRITON, self.moe_runner_config)
            if not fused_runner:
                self.runner.fused_func = None
            w1 = torch.zeros(
                experts, 2 * size, size, device="cuda", dtype=torch.bfloat16
            )
            w1[:, :size, 0] = fixed_gate[:, None]
            w1[:, size:, 0] = fixed_up[:, None]
            w2 = torch.eye(size, device="cuda", dtype=torch.bfloat16).repeat(
                experts, 1, 1
            )
            self.quant_info = TritonMoeQuantInfo(w13_weight=w1, w2_weight=w2)
            self.reference = (
                (F.silu(w1[:, :size, 0].float()) * w1[:, size:, 0].float())
                .to(torch.bfloat16)
                .float()
            )

        def __call__(self, hidden, topk, **kwargs):
            return self.runner.run(
                StandardDispatchOutput(hidden, None, topk), self.quant_info
            ).hidden_states

    parallel = get_parallel()

    class _NoTpGroup:
        """use_symmetric_memory(get_parallel().tp_group, ...) resolves the group eagerly,
        which raises before distributed init; everything else reads the overridden context."""

        def __getattr__(self, name):
            return None if name == "tp_group" else getattr(parallel, name)

    def without_tp_group(module):
        # main reaches the group through get_parallel(); older trees (sglang-miles)
        # through get_tp_group().
        if hasattr(module, "get_tp_group"):
            return patch.object(module, "get_tp_group", return_value=None)
        return patch.object(module, "get_parallel", return_value=_NoTpGroup())

    with (
        parallel.override(
            moe_ep_size=1,
            moe_tp_size=1,
            moe_ep_rank=0,
            moe_tp_rank=0,
            tp_rank=0,
            attn_tp_size=1,
            attn_tp_rank=0,
            attn_dp_size=1,
            attn_dp_rank=0,
        ),
        patch.object(flags, "a2a_backend", MoeA2ABackend.NONE),
        patch.object(flags, "runner_backend", MoeRunnerBackend.TRITON),
        patch.object(model, "get_moe_impl_class", return_value=SmallExperts),
        patch.object(model, "is_shared_experts_fusion_disabled", return_value=True),
        without_tp_group(fused_moe),
        without_tp_group(topk_module),
        patch.object(
            topk_module,
            "use_symmetric_memory",
            side_effect=lambda *args, **kwargs: nullcontext(),
        ),
        envs.SGLANG_OPT_USE_FUSED_HASH_TOPK.override(fused_hash),
    ):
        moe = model.DeepseekV2MoE(
            config, layer_id, is_nextn=is_nextn, is_deepseek_v4=True
        ).to(device="cuda")
        logits = torch.linspace(-1, 2, experts, device="cuda").repeat(rows, 1)
        if not moe.is_hash:
            moe.gate.e_score_correction_bias.data.zero_()
        # Fixed logits isolate routing scale from GEMM precision. Keep the real MoEGate:
        # forward_normal reads gate.e_score_correction_bias_vl, which Identity lacks.
        moe.gate.forward = lambda *args: logits
        moe._maybe_quant_moe_input_once = lambda _: None
        shared = torch.full((rows, size), 0.25, device="cuda", dtype=torch.bfloat16)
        moe._forward_shared_experts = lambda *args, **kwargs: shared.clone()
        moe.alt_stream = torch.cuda.Stream()
        hidden = torch.zeros(rows, size, device="cuda", dtype=torch.bfloat16)
        hidden[:, 0] = 1
        input_ids = torch.arange(rows, device="cuda")
        scores = torch.log1p(torch.exp(logits.double())).sqrt()
        if moe.is_hash:
            selected = (
                input_ids[:, None] + torch.arange(top_k, device="cuda")
            ) % experts
        else:
            selected = scores.argsort(dim=1, descending=True)[:, :top_k]
        weights = scores.gather(1, selected)
        weights /= weights.sum(1, keepdim=True)
        expected = (
            moe.experts.reference.double()[selected] * weights.unsqueeze(-1)
        ).sum(1) * 1.5 + shared
        forward = moe.forward_normal_dual_stream if dual_stream else moe.forward_normal
        actual = forward(hidden, input_ids_global=input_ids)
        torch.cuda.synchronize()
        torch.testing.assert_close(actual.double(), expected, rtol=0.025, atol=0.025)
