"""Real Triton compute and CUDA Graph on one GPU; no native EP is needed."""

import sys
from dataclasses import replace
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

from nccl_ep_test.triton_compute import (
    check_expert_output,
    configure_compute,
    make_compute_fixture,
)

from sglang.srt.layers.moe.moe_runner.nccl_ep_triton import (
    prepare_expert_slots,
    run_nccl_ep_triton,
)
from sglang.srt.runtime_context import get_context
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=40, stage="base-b", runner_config="1-gpu-small")

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


@pytest.fixture(autouse=True)
def compute_context():
    with get_context().preserve_config():
        configure_compute()
        yield


@pytest.mark.parametrize("counts", [[0, 0, 0, 0], [8, 0, 0, 0], [1, 3, 5, 8]])
def test_masked_slots_do_not_read_nan_tails(counts):
    dispatched, quant, config = make_compute_fixture()
    dispatched.masked_m.copy_(torch.tensor(counts, device="cuda"))
    # Fill both inactive payload and scales with NaN. A zero times an unmasked
    # NaN scale would contaminate the prepared row.
    poisoned = dispatched.hidden_states.float()
    for expert, count in enumerate(counts):
        poisoned[expert, count:] = float("nan")
        dispatched.hidden_states_scale[expert, count:] = float("nan")
    dispatched.hidden_states.copy_(poisoned.to(torch.float8_e4m3fn))
    x, ids, weights = prepare_expert_slots(
        dispatched.hidden_states, dispatched.hidden_states_scale, dispatched.masked_m
    )
    assert torch.isfinite(x).all()
    for expert, count in enumerate(counts):
        torch.testing.assert_close(
            x.view(4, 8, -1)[expert, count:],
            torch.zeros_like(x.view(4, 8, -1)[expert, count:]),
            rtol=0,
            atol=0,
        )
        assert ids.view(4, 8)[expert, :count].eq(expert).all()
        assert ids.view(4, 8)[expert, count:].eq(-1).all()
    assert weights.eq(1).all()
    actual = run_nccl_ep_triton(dispatched, quant, config)
    assert actual.topk_ids is dispatched.topk_ids
    assert actual.topk_weights is dispatched.topk_weights
    check_expert_output(actual.hidden_states, dispatched, quant)


@pytest.mark.parametrize("hidden,intermediate", [(256, 128), (2048, 1408)])
def test_real_experts_match_cpu_and_dynamic_graph(hidden, intermediate):
    dispatched, quant, config = make_compute_fixture(
        hidden=hidden, intermediate=intermediate
    )
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            run_nccl_ep_triton(dispatched, quant, config)
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        captured = run_nccl_ep_triton(dispatched, quant, config)
    addresses = tuple(
        t.data_ptr()
        for t in (
            dispatched.hidden_states,
            dispatched.hidden_states_scale,
            dispatched.masked_m,
        )
    )
    for counts in ([0, 0, 0, 0], [8, 0, 0, 0], [1, 3, 5, 8], [0, 8, 0, 2]):
        dispatched.masked_m.copy_(torch.tensor(counts, device="cuda"))
        eager = run_nccl_ep_triton(dispatched, quant, config)
        graph.replay()
        torch.cuda.synchronize()
        for expert, count in enumerate(counts):
            torch.testing.assert_close(
                captured.hidden_states[expert, :count],
                eager.hidden_states[expert, :count],
                rtol=0,
                atol=0,
            )
        check_expert_output(captured.hidden_states, dispatched, quant)
    assert addresses == tuple(
        t.data_ptr()
        for t in (
            dispatched.hidden_states,
            dispatched.hidden_states_scale,
            dispatched.masked_m,
        )
    )


@pytest.mark.parametrize(
    "change", [{"use_mxfp8": True}, {"block_shape": [64, 128]}, {"use_fp8_w8a8": False}]
)
def test_unsupported_quantization_is_explicit(change):
    dispatched, quant, config = make_compute_fixture()
    with pytest.raises(ValueError, match="block-128 FP8"):
        run_nccl_ep_triton(dispatched, replace(quant, **change), config)


def test_high_gain_matches_independent_unpadded_triton_experts():
    """Retain a high-gain stress case where two FP8 stages amplify rounding.

    Compare to existing single-expert GEMMs, independently constructing their
    input/routing. The CPU test uses fan-in-normalized weights and fixed tolerance.
    """
    from sglang.srt.layers.moe.moe_runner.triton_utils.fused_moe import (
        fused_experts_impl,
    )

    dispatched, quant, config = make_compute_fixture(hidden=2048, intermediate=1408)
    quant.w13_scale.fill_(0.0625)
    quant.w2_scale.fill_(0.0625)
    actual = run_nccl_ep_triton(dispatched, quant, config).hidden_states
    for expert, count in enumerate(dispatched.masked_m.cpu().tolist()):
        source = (
            dispatched.hidden_states[expert, :count].float()
            * dispatched.hidden_states_scale[expert, :count].repeat_interleave(128, -1)
        ).bfloat16()
        expected = fused_experts_impl(
            source,
            quant.w13_weight[expert : expert + 1],
            quant.w2_weight[expert : expert + 1],
            torch.ones(count, 1, device="cuda"),
            torch.zeros(count, 1, device="cuda", dtype=torch.int32),
            use_fp8_w8a8=True,
            w1_scale=quant.w13_scale[expert : expert + 1],
            w2_scale=quant.w2_scale[expert : expert + 1],
            block_shape=[128, 128],
            no_combine=True,
        ).view(count, -1)
        torch.testing.assert_close(
            actual[expert, :count], expected, rtol=0.02, atol=0.02
        )


def test_public_runner_selects_the_nccl_ep_adapter():
    from sglang.srt.layers.moe.moe_runner.runner import MoeRunner
    from sglang.srt.layers.moe.utils import MoeA2ABackend, MoeRunnerBackend
    from sglang.srt.runtime_context import get_flags

    dispatched, quant, config = make_compute_fixture()
    with get_flags().moe.override(a2a_backend=MoeA2ABackend.NCCL_EP):
        runner = MoeRunner(MoeRunnerBackend.TRITON, config)
        actual = runner.run(dispatched, quant)
        check_expert_output(actual.hidden_states, dispatched, quant)
        with pytest.raises(ValueError, match="no LoRA"):
            MoeRunner(MoeRunnerBackend.TRITON, config, lora_enabled=True)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
