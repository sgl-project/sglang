"""The DeepSeek-V4.1 TP4 collectives on gfx950 in one torchrun: the fused all-reduce +
mHC post handoff of attention and MoE under mutable graph replay, the sharded greedy
selection of the DSpark draft, and the bit-preserving Engram reconstruction."""

import os
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from sglang.kernels.ops.speculative.dspark.sharded_greedy import sharded_greedy_step
from sglang.srt.distributed.device_communicators.vocab_gather import (
    make_vocab_gather,
)
from sglang.srt.distributed.parallel_state import (
    destroy_distributed_environment,
    destroy_model_parallel,
    get_tp_group,
    init_distributed_environment,
    initialize_model_parallel,
)
from sglang.srt.environ import envs
from sglang.srt.layers.engram import EngramEmbedding
from sglang.srt.runtime_context import get_forward, get_parallel, reset_context
from sglang.srt.utils import is_gfx95_supported, is_hip
from sglang.test.ci.ci_register import register_amd_ci
from sglang.test.dsv4_moe_stub import make_dsv4_moe_stub
from sglang.test.kernels.utils import multigpu_pytest_main
from sglang.test.test_utils import publish_build_topology

register_amd_ci(est_time=60, suite="stage-c-kernel-test-4-gpu-amd-mi35x")
pytestmark = pytest.mark.skipif(
    not is_hip() or "LOCAL_RANK" not in os.environ,
    reason="run through the four-GPU entry point",
)


@pytest.fixture(scope="module")
def group():
    """The published TP4 topology and its real groups (pynccl + custom all-reduce)."""
    rank, world = int(os.environ["LOCAL_RANK"]), int(os.environ["WORLD_SIZE"])
    if not is_gfx95_supported() or world != 4:
        pytest.skip("requires a gfx950 group of four GPUs")
    torch.cuda.set_device(rank)
    torch.cuda.set_stream(torch.cuda.Stream())
    init_distributed_environment(
        world_size=world, rank=rank, local_rank=rank, backend="nccl"
    )
    publish_build_topology(tp_size=world, world_rank=rank, moe_runner_backend="aiter")
    initialize_model_parallel()
    yield get_tp_group()
    destroy_model_parallel()
    destroy_distributed_environment()
    reset_context()


# ---- fused all-reduce + mHC post ----


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
        hc_pre_from_prev_sublayer=False,
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


def _no_deterministic_inference():
    return patch(
        "sglang.srt.runtime_context.get_exec",
        return_value=SimpleNamespace(
            deterministic=SimpleNamespace(enable_deterministic_inference=False)
        ),
    )


_FUSED_MHC = "sglang.srt.models.deepseek_common.amd.deepseek_v4_fused_mhc"


def _replay_fused_and_unfused(group, run, check, *, residual, pre):
    """Capture run with the fused all-reduce off and on, then replay both on new
    inputs and compare through check."""
    graphs = []
    for enabled in (False, True):
        # the server hook turns the tilelang post off on ROCm at model load
        with (
            envs.SGLANG_OPT_USE_TILELANG_MHC_POST.override(False),
            nullcontext()
            if enabled
            else patch(f"{_FUSED_MHC}._can_fuse_mhc", lambda *_: False),
        ):
            graph = torch.cuda.CUDAGraph()
            with group.graph_capture() as capture:
                run()
                with torch.cuda.graph(graph, stream=capture.stream):
                    output = run()
            check(enabled)
            graphs.append((graph, output))
    for _ in range(3):
        residual.normal_()
        pre.uniform_()
        for graph, _ in graphs:
            graph.replay()
        torch.cuda.synchronize()
        yield graphs[0][1], graphs[1][1]


@pytest.mark.parametrize(
    "rows,verify", [(1, False), (8, True)], ids=["decode", "verify"]
)
def test_model_handoff_and_graph_replay(group, rows, verify):
    from sglang.srt.model_executor.forward_batch_info import ForwardMode
    from sglang.srt.models.deepseek_common.amd.deepseek_v4_fused_mhc import (
        forward_hc_pre_from_prev_fused_boundary,
    )

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

    def check(enabled):
        assert layer.self_attn.wo_b.skip_reduction[-1] == enabled

    with get_forward().scoped(sp_active=False), _no_deterministic_inference():
        for unfused, fused in _replay_fused_and_unfused(
            group, run, check, residual=residual, pre=pre
        ):
            for expected, actual in zip(unfused, fused):
                if actual.dtype == torch.bfloat16:
                    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
                else:
                    torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize(
    "rows,dual,defer,shared_tp1",
    [
        (1, False, False, False),
        (8, False, True, False),
        (8, True, True, False),
        (8, False, True, True),
    ],
    ids=["unfused", "deferred", "dual-stream", "shared-tp1"],
)
def test_moe_model_handoff(group, rows, dual, defer, shared_tp1):
    from sglang.srt.layers.moe import MoeA2ABackend
    from sglang.srt.model_executor.forward_batch_info import ForwardMode
    from sglang.srt.models.deepseek_common.amd.deepseek_v4_fused_mhc import (
        forward_hc_pre_from_prev_fused_boundary,
    )
    from sglang.srt.models.deepseek_v4 import DeepseekV4DecoderLayer

    layer = _layer(group)
    layer.mlp = make_dsv4_moe_stub(
        group.rank_in_group, dual=dual, shared_tp1=shared_tp1
    )
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
        get_forward().scoped(
            sp_active=False, fuse_mlp_allreduce=False, flashinfer_trtllm_bypass=False
        ),
        patch(
            "sglang.srt.models.deepseek_common.amd.deepseek_v2_hip_moe.post_experts_all_reduce",
            side_effect=group.all_reduce,
        ) as original_reduce,
        patch(
            "sglang.srt.layers.moe.get_moe_a2a_backend", return_value=MoeA2ABackend.NONE
        ),
        patch(
            "sglang.srt.models.deepseek_v4.get_moe_a2a_backend",
            return_value=MoeA2ABackend.NONE,
        ),
        _no_deterministic_inference(),
        patch(
            "sglang.srt.models.deepseek_v2.get_exec",
            return_value=SimpleNamespace(moe=SimpleNamespace(enable_eplb=False)),
        ),
    ):

        def run_counting():
            original_reduce.reset_mock()
            return run()

        def check(enabled):
            # the fused kernel replaces the reduction unless TP1 shared experts
            # need the plain one; the post stays pending only when it was deferred
            # and not fused
            assert original_reduce.call_count == int(not enabled or shared_tp1)
            assert outcomes[-1] == (defer and (not enabled or shared_tp1))

        for unfused, fused in _replay_fused_and_unfused(
            group, run_counting, check, residual=residual, pre=pre
        ):
            torch.testing.assert_close(unfused[0], fused[0], atol=0.01, rtol=0.01)
            torch.testing.assert_close(unfused[1], fused[1], atol=1e-5, rtol=1e-5)


# ---- DSpark sharded greedy ----


def _replay_and_check(group, m, width, last, perturb):
    """Capture the selection graph, then replay it under perturb."""
    transport = make_vocab_gather(
        group, local_width=width, prefer_nvlink=False, symm_rows=0
    )
    rank = group.rank_in_group
    real = last if rank == group.world_size - 1 else width
    # A slice of a block's logits has a non-contiguous row stride.
    storage = torch.randn(m, 5, width, device="cuda")
    base = storage[:, 2]
    bias = torch.randn(m, real, device="cuda", dtype=torch.bfloat16)

    def candidate():
        return sharded_greedy_step(
            bias,
            base,
            group=group,
            vocab_start=rank * width,
            gather=transport.gather_stacked,
        )

    graph = torch.cuda.CUDAGraph()
    # Warm the same communicator and stream that the runner uses for capture.
    with group.graph_capture() as capture:
        for _ in range(3):
            candidate()
        with torch.cuda.graph(graph, stream=capture.stream):
            out = candidate()
    for replay in range(4):
        storage.normal_()
        bias.normal_()
        perturb(replay, storage, base, bias, real)
        graph.replay()
        # Independent reference: gather complete, correctly padded FP32 logits.
        local = torch.full((m, width), -float("inf"), device="cuda")
        local[:, :real] = base[:, :real] + bias.float()
        full = group.all_gather(local, dim=-1)
        ref = full[:, : (group.world_size - 1) * width + last].argmax(-1)
        torch.cuda.synchronize()
        assert torch.equal(out, ref)


def _random(replay, storage, base, bias, real):
    pass


def _tie(replay, storage, base, bias, real):
    storage.zero_()
    bias.zero_()


def _nan(replay, storage, base, bias, real):
    if real:
        base[:, replay % real] = float("nan")


def _inf(replay, storage, base, bias, real):
    storage.fill_(-float("inf"))
    if replay % 2 and real:
        base[:, replay % real] = float("inf")


@pytest.mark.parametrize(
    "m,width,last",
    [(1, 32320, 32320), (4, 8192, 17), (64, 8, 0)],
    ids=["full", "partial", "empty"],
)
def test_greedy_random_logits(group, m, width, last):
    _replay_and_check(group, m, width, last, _random)


def test_greedy_ties_resolve_to_the_lowest_index(group):
    _replay_and_check(group, 4, 8192, 17, _tie)


def test_greedy_nan_propagates(group):
    _replay_and_check(group, 4, 8192, 17, _nan)


def test_greedy_infinities(group):
    _replay_and_check(group, 4, 8192, 17, _inf)


# ---- sharded Engram ----


def _engram_table(rows):
    weight = torch.ones(rows, 128, dtype=torch.float32)
    weight[:, 1::4] = -1
    weight[:, 2::4] = 0.5
    weight[:, 3::4] = -0.0
    weight = weight.to(torch.float8_e4m3fn)
    scale = torch.tensor([0, 1, 120, 140], dtype=torch.uint8).repeat(rows, 1)
    reference = (
        (
            weight.float().reshape(rows, 4, 32)
            * scale.view(torch.float8_e8m0fnu).float().unsqueeze(-1)
        )
        .flatten(1)
        .to(torch.bfloat16)
    )
    with (
        envs.SGLANG_ENABLE_DSV41_ENGRAM_HOST_TABLE.override(False),
        torch.device("cuda"),
    ):
        embed = EngramEmbedding(rows, 128, layer_id=1)
    embed.weight.weight_loader(embed.weight, weight)
    embed.scale.weight_loader(embed.scale, scale.view(torch.float8_e8m0fnu))
    embed.finish_load(label="test")
    return embed, reference


@pytest.mark.parametrize("rows", [1, 17], ids=["1", "17"])
def test_engram_eager_and_graph_reconstruction(group, rows):
    """Reconstructing a sharded row must retain signed zero and BF16 subnormals."""
    embed, reference = _engram_table(rows)
    ids = torch.arange(16, device="cuda", dtype=torch.int64).view(-1, 1) % rows
    # plain TP (attn_dp_size == 1): the all-reduce path
    eager = embed(ids)
    assert torch.equal(
        eager.cpu().view(torch.int16), reference[ids.cpu()].view(torch.int16)
    )
    graph = torch.cuda.CUDAGraph()
    with group.graph_capture() as capture:
        embed(ids)
        with torch.cuda.graph(graph, stream=capture.stream):
            output = embed(ids)
    for shift in (1, 3):
        ids.add_(shift).remainder_(rows)
        graph.replay()
        assert torch.equal(
            output.cpu().view(torch.int16), reference[ids.cpu()].view(torch.int16)
        )


@pytest.mark.parametrize("scatter", [False, True], ids=["False", "True"])
def test_engram_dp_shard_reconstruction(group, scatter):
    """DP distribution must preserve the owning shard's BF16 payload bits."""
    embed, reference = _engram_table(17)
    rank, world = group.rank_in_group, group.world_size
    ids = torch.tensor([[rank]], device="cuda", dtype=torch.int64)
    batch = SimpleNamespace(dp_padding_mode=SimpleNamespace(is_max_len=lambda: scatter))

    # Supply a fixed one-token-per-rank DP schedule; exercise real collectives.
    def gather(dst, src, _batch):
        group.all_gather_into_tensor(dst, src)

    # Present the plain TP groups to the layer and the DP helpers as a TP-wide
    # attention-DP layout (one attention rank per GPU).
    real = get_parallel()
    dp_layout = SimpleNamespace(
        tp_size=world,
        tp_rank=rank,
        tp_group=real.tp_group,
        attn_tp_group=real.tp_group,
        attn_dp_size=world,
        attn_dp_rank=rank,
        attn_tp_size=1,
        attn_tp_rank=0,
        attn_cp_size=1,
        attn_cp_rank=0,
    )
    with (
        patch("sglang.srt.layers.engram.get_global_dp_buffer_len", return_value=world),
        patch("sglang.srt.layers.engram.get_parallel", return_value=dp_layout),
        patch("sglang.srt.layers.dp_attention.get_parallel", return_value=dp_layout),
        patch("sglang.srt.layers.engram.dp_gather_replicate", side_effect=gather),
        patch(
            "sglang.srt.layers.dp_attention.get_dp_local_info",
            return_value=(
                torch.tensor(rank, device="cuda", dtype=torch.int32),
                torch.tensor(1, device="cuda", dtype=torch.int32),
            ),
        ),
    ):
        output = embed._dp_sharded_lookup(ids, batch)
        assert torch.equal(
            output.cpu().view(torch.int16), reference[ids.cpu()].view(torch.int16)
        )


if __name__ == "__main__":
    multigpu_pytest_main(__name__, __file__, num_gpus=(4,))
