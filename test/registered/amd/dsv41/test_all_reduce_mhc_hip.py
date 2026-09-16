"""TP4 attention reduction/post handoff and mutable graph replay on gfx950."""

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

register_amd_ci(est_time=120, suite="stage-c-test-large-8-gpu-amd-mi35x")
pytestmark = pytest.mark.skipif(
    not is_hip() or "LOCAL_RANK" not in os.environ,
    reason="run through the eight-GPU entry point",
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


@pytest.mark.parametrize("rows", [1, 2, 4, 8])
@pytest.mark.parametrize("verify", [False, True])
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


@pytest.mark.parametrize(
    "reason",
    ["rows", "tp8", "prefill", "cp", "invariant", "deterministic", "registration"],
)
def test_fallback_keeps_the_original_reduction(group, reason):
    from sglang.srt.environ import envs
    from sglang.srt.layers.moe.mhc_post_fusion import use_mhc_post_fusion
    from sglang.srt.model_executor.forward_batch_info import ForwardMode
    from sglang.srt.models.deepseek_common.amd.deepseek_v4_fused_mhc import (
        attention_mhc_fusion,
    )
    from sglang.srt.models.deepseek_v4 import MQALayer
    from sglang.srt.runtime_context import get_forward, get_parallel

    layer = _layer(group)
    rows = 9 if reason == "rows" else 1
    residual = torch.empty(rows, 4, 5120, device="cuda", dtype=torch.bfloat16)
    layer.dsa_enable_prefill_cp = reason == "cp"
    layer.self_attn.attn_tp_size = 8 if reason == "tp8" else 4
    batch = SimpleNamespace(
        forward_mode=ForwardMode.EXTEND if reason == "prefill" else ForwardMode.DECODE
    )
    with (
        envs.SGLANG_OPT_HIP_ALL_REDUCE_MHC.override(True),
        get_parallel().override(tp_size=layer.self_attn.attn_tp_size, attn_dp_size=1),
        get_forward().scoped(sp_active=False),
        patch(
            "sglang.srt.distributed.parallel_state.get_attn_tp_group",
            return_value=group,
        ),
        patch(
            "sglang.srt.batch_invariant_ops.is_batch_invariant_mode_enabled",
            return_value=reason == "invariant",
        ),
        patch(
            "sglang.srt.runtime_context.get_exec",
            return_value=SimpleNamespace(
                deterministic=SimpleNamespace(
                    enable_deterministic_inference=reason == "deterministic"
                )
            ),
        ),
        patch.object(
            group.ca_comm, "enable_register_for_capturing", reason != "registration"
        ),
    ):
        state = attention_mhc_fusion(layer, residual, None, batch)
        assert state is None
        with use_mhc_post_fusion(state):
            x = torch.full(
                (rows, 5120),
                group.rank_in_group + 1,
                device="cuda",
                dtype=torch.bfloat16,
            )
            actual = MQALayer._project_wo_b(layer.self_attn, x)
        assert layer.self_attn.wo_b.skip_reduction == [False]
        torch.testing.assert_close(actual, torch.full_like(actual, 10), atol=0, rtol=0)


if __name__ == "__main__":
    multigpu_pytest_main(__name__, __file__, num_gpus=(8,))
