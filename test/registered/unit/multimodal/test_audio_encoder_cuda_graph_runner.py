import sys

import pytest
import torch

from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=20, stage="base-b", runner_config="1-gpu-small")

from sglang.kernels.ops.attention.prefill_attention import context_attention_fwd
from sglang.srt.multimodal.audio_encoder_cuda_graph_runner import (
    AudioEncoderCudaGraphRunner,
)

HIDDEN = 64
HEADS = 2
WINDOW = 16

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


def _forward_transformer(hidden_states, metadata):
    """Windowed attention plus per-token ops, like one encoder layer."""
    x = hidden_states.view(-1, HEADS, HIDDEN // HEADS)
    out = torch.empty_like(x)
    context_attention_fwd(
        x,
        x,
        x,
        out,
        metadata.cu_seqlens,
        metadata.seq_lens,
        metadata.max_seqlen,
        is_causal=False,
    )
    out = out.view(-1, HIDDEN) + hidden_states
    return torch.nn.functional.gelu(out) * 2


def _eager(hidden_states, cu_seqlens):
    from sglang.srt.layers.attention.vision import prepare_vision_attention_metadata

    metadata = prepare_vision_attention_metadata(
        torch.tensor(cu_seqlens, dtype=torch.int32), hidden_states.device
    )
    return _forward_transformer(hidden_states, metadata)


def _windows(lengths):
    cu = [0]
    for n in lengths:
        for size in [WINDOW] * (n // WINDOW) + ([n % WINDOW] if n % WINDOW else []):
            cu.append(cu[-1] + size)
    return cu


def _runner():
    return AudioEncoderCudaGraphRunner(
        _forward_transformer,
        hidden_size=HIDDEN,
        window_size=WINDOW,
        dtype=torch.float32,
        device=torch.device("cuda"),
        token_budgets=(32, 64, 256),
        max_items=4,
    )


def _run(runner, lengths):
    cu = _windows(lengths)
    hidden = torch.randn(cu[-1], HIDDEN, device="cuda")
    budget = runner.select_budget(hidden.shape[0], len(cu) - 1)
    assert budget is not None
    return runner.run(hidden, cu, budget), _eager(hidden, cu)


@pytest.mark.parametrize("lengths", [[5], [16, 3, 40], [100, 100, 7, 1]])
def test_replay_matches_eager(lengths):
    runner = _runner()
    graph_out, eager_out = _run(runner, lengths)
    torch.testing.assert_close(graph_out, eager_out)
    assert len(runner.graphs) == 3


def test_no_stale_rows_after_larger_batch():
    runner = _runner()
    _run(runner, [200, 50])
    graph_out, eager_out = _run(runner, [9, 2])
    torch.testing.assert_close(graph_out, eager_out)


def test_oversized_batches_fall_back():
    runner = _runner()
    assert runner.select_budget(257, 17) is None
    # 256 tokens fit, but 40 one-token items need more windows than it holds.
    assert runner.select_budget(40, 40) is None


if __name__ == "__main__":
    sys.exit(pytest.main([__file__]))
