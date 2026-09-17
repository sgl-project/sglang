"""TP4 attention and MoE reduction/post handoff and mutable graph replay on gfx950."""

import gc
import os
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import patch
import pytest
import torch
import torch.distributed as dist
from sglang.srt.utils import is_hip
from sglang.test.ci.ci_register import register_amd_ci
from sglang.test.kernels.utils import multigpu_pytest_main





register_amd_ci(est_time=60, suite="stage-c-kernel-test-4-gpu-amd-mi35x")
pytestmark = pytest.mark.skipif(
    not is_hip() or "LOCAL_RANK" not in os.environ,
    reason="run through the four-GPU entry point",
)


@pytest.fixture(scope="module")
def group():
    from sglang.srt.distributed import parallel_state as ps
    from sglang.srt.distributed.parallel_state import GroupCoordinator
    from sglang.srt.utils import is_gfx95_supported

    rank, world = int(os.environ["LOCAL_RANK"]), int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(rank)
    if not is_gfx95_supported() or world % 4:
        pytest.skip("requires gfx950 groups of four GPUs")
    dist.init_process_group("gloo")
    ps._WORLD = ps.init_world_group(list(range(world)), rank, backend="nccl")
    torch.cuda.set_stream(torch.cuda.Stream())
    g = GroupCoordinator(
        group_ranks=[list(range(start, start + 4)) for start in range(0, world, 4)],
        local_rank=rank,
        torch_distributed_backend="nccl",
        use_pynccl=True,
        use_custom_allreduce=True,
        use_pymscclpp=False,
        use_torch_symm_mem_all_reduce=False,
        use_hpu_communicator=False,
        use_xpu_communicator=False,
        use_npu_communicator=False,
        group_name="mhc_hip_test",
    )
    from sglang.srt.runtime_context import get_context

    with get_context().override_server_args(moe_runner_backend="aiter"):
        yield g
    if g.qr_comm is not None:
        g.qr_comm.close()
        g.qr_comm = None
    g.destroy()
    gc.collect()
    ps._WORLD.destroy()
    ps._WORLD = None
    dist.destroy_process_group()


class _Projection:
    reduce_results = True

    def __init__(self, group):
        self.group = group
        self.skip_reduction = []

    def __call__(self, x, *, skip_all_reduce):
        self.skip_reduction.append(skip_all_reduce)
        return (x if skip_all_reduce else self.group.all_reduce(x)), None


class _Attention:
    attn_tp_size = 4

    def __init__(self, group):
        self.wo_b = _Projection(group)
        self.rank = group.rank_in_group

    def maybe_use_decode_attn_tp(self, forward_batch):
        return nullcontext()

    def __call__(self, *, x, **kwargs):
        from sglang.srt.models.deepseek_v4 import MQALayer

        return MQALayer._project_wo_b(self, x * (self.rank + 1))


def _layer(group):
    from sglang.kernels.ops.layernorm.mhc_boundary_hip import rmsnorm_with_sinkhorn

    torch.manual_seed(39186)
    norm_weight = torch.ones(5120, device="cuda", dtype=torch.bfloat16)
    layer = SimpleNamespace(
        config=SimpleNamespace(model_type="deepseek_v41"),
        dsa_enable_prefill_cp=False,
        self_attn=_Attention(group),
        mlp=SimpleNamespace(tp_size=1),
        hc_mult=4,
        hc_sinkhorn_iters=20,
        rms_norm_eps=1e-6,
        hc_eps=1e-6,
        hc_attn_fn=torch.randn(24, 20480, device="cuda") * 0.01,
        hc_ffn_fn=torch.randn(24, 20480, device="cuda") * 0.01,
        hc_attn_scale=torch.ones(3, device="cuda"),
        hc_ffn_scale=torch.ones(3, device="cuda"),
        hc_attn_base=torch.zeros(24, device="cuda"),
        hc_ffn_base=torch.zeros(24, device="cuda"),
        post_attention_layernorm=SimpleNamespace(
            weight=norm_weight, variance_epsilon=1e-6
        ),
        _run_moe_ffn_dp_sync=lambda x, *args, **kwargs: x,
    )

    def input_norm(x, *, coefficients, **kwargs):
        _, normalized = rmsnorm_with_sinkhorn(
            x, norm_weight, 1e-6, coefficients, fake_quant=False
        )
        return normalized, None

    layer._input_norm = input_norm
    return layer


@pytest.mark.parametrize("rows,verify", [(1, False), (8, True)], ids=["decode", "verify"])
def test_model_handoff_and_graph_replay(group, rows, verify):
    from sglang.srt.environ import envs
    from sglang.srt.model_executor.forward_batch_info import ForwardMode
    from sglang.srt.models.deepseek_common.amd.deepseek_v4_fused_mhc import (
        forward_hc_pre_from_prev_fused_boundary,
    )
    from sglang.srt.runtime_context import get_forward, get_parallel

    layer = _layer(group)
    residual = torch.randn(rows, 4, 5120, device="cuda", dtype=torch.bfloat16)
    pre = torch.sigmoid(torch.randn(rows, 4, device="cuda"))
    batch = SimpleNamespace(
        forward_mode=ForwardMode.TARGET_VERIFY if verify else ForwardMode.DECODE
    )

    def run():
        _, next_pre, pending = forward_hc_pre_from_prev_fused_boundary(
            layer, None, residual, None, batch, None, pre, None, True
        )
        return next_pre, *pending

    with (
        get_parallel().override(tp_size=4, attn_dp_size=1),
        get_forward().scoped(sp_active=False),
        patch(
            "sglang.srt.distributed.parallel_state.get_attn_tp_group",
            return_value=group,
        ),
        patch(
            "sglang.srt.runtime_context.get_exec",
            return_value=SimpleNamespace(
                deterministic=SimpleNamespace(enable_deterministic_inference=False)
            ),
        ),
    ):
        graphs = []
        for enabled in (False, True):
            with envs.SGLANG_OPT_HIP_ALL_REDUCE_MHC.override(enabled):
                graph = torch.cuda.CUDAGraph()
                with group.graph_capture() as capture:
                    run()
                    with torch.cuda.graph(graph, stream=capture.stream):
                        output = run()
                assert layer.self_attn.wo_b.skip_reduction[-1] == enabled
                graphs.append((graph, output))
        for _ in range(3):
            residual.normal_()
            pre.uniform_()
            for graph, _ in graphs:
                graph.replay()
            torch.cuda.synchronize()
            for expected, actual in zip(graphs[0][1], graphs[1][1]):
                if actual.dtype == torch.bfloat16:
                    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
                else:
                    torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-5)


def _moe(group, dual, shared_tp1):
    from sglang.srt.layers.moe.topk import TopKOutputFormat
    from sglang.srt.models.deepseek_v2 import DeepseekV2MoE

    class Experts:
        quant_method = None
        moe_runner_config = SimpleNamespace(inplace=False)

        def __call__(self, x, *args, **kwargs):
            return x * (group.rank_in_group + 1)

    class Moe(DeepseekV2MoE):
        def __init__(self):
            torch.nn.Module.__init__(self)
            self.tp_size = 4
            self._shared_expert_tp1 = shared_tp1
            self.layer_id = 0
            self.is_nextn = False
            self.is_hash = False
            self._fuse_shared_experts_inside_sbo = False
            self._fuse_finalize_all_reduce = False
            self.num_fused_shared_experts = 0
            self.routed_scaling_factor = 1.0
            self.experts = Experts()
            self.alt_stream = torch.cuda.Stream()
            self.topk = lambda *a, **kw: SimpleNamespace(
                format=TopKOutputFormat.STANDARD
            )

        def _maybe_quant_moe_input_once(self, x):
            return None

        def _should_quant_routed_input_mxfp8(self, x):
            return False

        def _forward_gate(self, x, *args, **kwargs):
            return x, None

        def _forward_shared_experts(self, x, *args, **kwargs):
            return x * 0.5

        def forward(self, x, *args, **kwargs):
            return (self.forward_normal_dual_stream if dual else self.forward_normal)(x)

    return Moe()


@pytest.mark.parametrize("rows,dual,defer,shared_tp1", [(1, False, False, False), (8, False, True, False), (8, True, True, False), (8, False, True, True)], ids=["unfused", "deferred", "dual-stream", "shared-tp1"])
def test_moe_model_handoff(group, rows, dual, defer, shared_tp1):
    from sglang.srt.environ import envs
    from sglang.srt.layers.moe import MoeA2ABackend
    from sglang.srt.model_executor.forward_batch_info import ForwardMode
    from sglang.srt.models.deepseek_common.amd.deepseek_v4_fused_mhc import (
        forward_hc_pre_from_prev_fused_boundary,
    )
    from sglang.srt.models.deepseek_v4 import DeepseekV4DecoderLayer
    from sglang.srt.runtime_context import get_forward, get_parallel

    layer = _layer(group)
    layer.mlp = _moe(group, dual, shared_tp1)
    layer._run_moe_ffn_dp_sync = lambda *a, **kw: (
        DeepseekV4DecoderLayer._run_moe_ffn_dp_sync(layer, *a, **kw)
    )
    layer.hc_post = lambda *a: DeepseekV4DecoderLayer.hc_post(layer, *a)
    residual = torch.randn(rows, 4, 5120, device="cuda", dtype=torch.bfloat16)
    pre = torch.sigmoid(torch.randn(rows, 4, device="cuda"))
    batch = SimpleNamespace(forward_mode=ForwardMode.DECODE, num_token_non_padded=None)
    outcomes = []

    def run():
        hidden, next_pre, pending = forward_hc_pre_from_prev_fused_boundary(
            layer, None, residual, None, batch, None, pre, None, defer
        )
        outcomes.append(pending is not None)
        if pending is not None:
            hidden = layer.hc_post(*pending)
        return hidden, next_pre

    with (
        get_parallel().override(tp_size=4, attn_tp_size=4, attn_dp_size=1),
        get_forward().scoped(
            sp_active=False, fuse_mlp_allreduce=False, flashinfer_trtllm_bypass=False
        ),
        patch(
            "sglang.srt.distributed.parallel_state.get_attn_tp_group",
            return_value=group,
        ),
        patch("sglang.srt.distributed.parallel_state.get_tp_group", return_value=group),
        patch(
            "sglang.srt.models.deepseek_v2.tensor_model_parallel_all_reduce",
            side_effect=group.all_reduce,
        ) as original_reduce,
        patch(
            "sglang.srt.layers.moe.get_moe_a2a_backend", return_value=MoeA2ABackend.NONE
        ),
        patch(
            "sglang.srt.models.deepseek_v4.get_moe_a2a_backend",
            return_value=MoeA2ABackend.NONE,
        ),
        patch(
            "sglang.srt.runtime_context.get_exec",
            return_value=SimpleNamespace(
                deterministic=SimpleNamespace(enable_deterministic_inference=False)
            ),
        ),
        patch(
            "sglang.srt.models.deepseek_v2.get_exec",
            return_value=SimpleNamespace(moe=SimpleNamespace(enable_eplb=False)),
        ),
    ):
        graphs = []
        for enabled in (False, True):
            with envs.SGLANG_OPT_HIP_ALL_REDUCE_MHC.override(enabled):
                graph = torch.cuda.CUDAGraph()
                with group.graph_capture() as capture:
                    run()
                    original_reduce.reset_mock()
                    with torch.cuda.graph(graph, stream=capture.stream):
                        output = run()
                assert original_reduce.call_count == int(not enabled or shared_tp1)
                assert outcomes[-1] == (defer and (not enabled or shared_tp1))
                graphs.append((graph, output))
        for _ in range(3):
            residual.normal_()
            pre.uniform_()
            for graph, _ in graphs:
                graph.replay()
            torch.cuda.synchronize()
            torch.testing.assert_close(
                graphs[0][1][0], graphs[1][1][0], atol=0.01, rtol=0.01
            )
            torch.testing.assert_close(
                graphs[0][1][1], graphs[1][1][1], atol=1e-5, rtol=1e-5
            )


@pytest.mark.parametrize("dual", [False, True], ids=['False', 'True'])
@pytest.mark.parametrize("flag", ["mlp_reduce_scatter", "fuse_mlp_allreduce"], ids=["'mlp_reduce_scatter'", "'fuse_mlp_allreduce'"])
def test_moe_skipped_reduction_does_not_apply_post(group, dual, flag):
    from sglang.srt.layers.moe.mhc_post_fusion import MhcPostFusion, use_mhc_post_fusion
    from sglang.srt.runtime_context import get_forward

    moe = _moe(group, dual, False)
    x = torch.ones(1, 5120, device="cuda", dtype=torch.bfloat16)
    state = MhcPostFusion(None, None, None, None)
    with (
        get_forward().scoped(**{flag: True}, flashinfer_trtllm_bypass=False),
        use_mhc_post_fusion(state),
        patch(
            "sglang.srt.models.deepseek_v2.tensor_model_parallel_all_reduce"
        ) as reduction,
        patch(
            "sglang.srt.models.deepseek_v2.get_exec",
            return_value=SimpleNamespace(moe=SimpleNamespace(enable_eplb=False)),
        ),
    ):
        actual = moe(x)
        assert state.output is None
        reduction.assert_not_called()
        torch.testing.assert_close(
            actual, x * (group.rank_in_group + 1.5), atol=0, rtol=0
        )


if __name__ == "__main__":
    multigpu_pytest_main(__name__, __file__, num_gpus=(4,))
