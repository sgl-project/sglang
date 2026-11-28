"""Changing packed window boundaries must remain valid under FA3 graph replay."""

import sys

import pytest
import torch

from sglang.kernels.ops.attention.flash_attention import flash_attn_varlen_func
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=30, stage="base-b", runner_config="1-gpu-large")

from sglang.srt.layers.attention.vision import (
    VisionFlash3Attention,
    prepare_vision_attention_metadata,
)
from sglang.srt.multimodal.audio_encoder_cuda_graph_runner import (
    AudioEncoderCudaGraphRunner,
)

pytestmark = pytest.mark.skipif(
    torch.version.cuda is None
    or not torch.cuda.is_available()
    or torch.cuda.get_device_capability()[0] != 9,
    reason="FA3 graph regression requires Hopper",
)


@torch.inference_mode()
def test_fa3_replay_with_changing_windows():
    torch.manual_seed(52148)
    heads, head_dim, window = 14, 64, 104
    hidden = heads * head_dim
    attention = VisionFlash3Attention()
    projection = torch.nn.Linear(
        hidden, hidden * 3, device="cuda", dtype=torch.bfloat16
    )

    def forward(x, metadata):
        q, k, v = projection(x).view(-1, 3, heads, head_dim).unbind(1)
        output = attention(
            q,
            k,
            v,
            metadata.cu_seqlens,
            1,
            x.shape[0],
            forward_metadata=metadata,
        )
        return output.reshape(-1, hidden) + x

    runner = AudioEncoderCudaGraphRunner(
        forward,
        hidden_size=hidden,
        window_size=window,
        dtype=torch.bfloat16,
        device=torch.device("cuda"),
        token_budgets=(128, 256, 512),
        max_items=4,
    )
    for cu in ([0, 104, 208, 299], [0, 5], [0, 100, 104, 127], [0, 1, 2, 3, 4]):
        x = torch.randn(cu[-1], hidden, device="cuda", dtype=torch.bfloat16)
        budget = runner.select_budget(x.shape[0], len(cu) - 1)
        output = runner.run(x, list(cu), budget)
        assert output is not None
        # Keep exactly the captured shapes; only valid window boundaries and
        # input contents change, as in variable-length audio batches.
        entry = runner.graphs[budget]
        padded = torch.zeros_like(entry.hidden_states)
        padded[: x.shape[0]].copy_(x)
        reference_metadata = prepare_vision_attention_metadata(
            torch.tensor(cu, dtype=torch.int32, device="cuda"),
            x.device,
            max_seqlen=window,
        )
        expected = forward(padded, reference_metadata)[: x.shape[0]]
        torch.testing.assert_close(output, expected, rtol=0, atol=0)
    assert len(runner.graphs) == 3


@torch.inference_mode()
def test_fa3_padding_rows_do_not_poison_valid_tokens():
    """Unwritten padding can turn valid outputs into NaNs in the next layer."""
    torch.manual_seed(52148)
    heads, dim, window = 14, 64, 104
    hidden = heads * dim
    projection = torch.nn.Linear(
        hidden, hidden * 3, device="cuda", dtype=torch.bfloat16
    )

    def forward(x, metadata):
        for _ in range(2):
            q, k, v = projection(x).view(-1, 3, heads, dim).unbind(1)
            # Model the NaN bytes a reused graph-pool allocation can contain.
            # FA3 must write every row before another layer consumes them.
            target = torch.full(
                (x.shape[0], heads, dim), float("nan"), dtype=x.dtype, device=x.device
            )
            output = flash_attn_varlen_func(
                q,
                k,
                v,
                metadata.cu_seqlens,
                metadata.cu_seqlens,
                max_seqlen_q=window,
                max_seqlen_k=window,
                out=target,
            )
            assert output.data_ptr() == target.data_ptr()
            x = x + output.reshape(-1, hidden)
        return x

    runner = AudioEncoderCudaGraphRunner(
        forward,
        hidden_size=hidden,
        window_size=window,
        dtype=torch.bfloat16,
        device=torch.device("cuda"),
        token_budgets=(128, 256),
        max_items=4,
    )
    x = torch.randn(73, hidden, device="cuda", dtype=torch.bfloat16)
    cu = [0, 73]
    output = runner.run(x, cu, 128)
    reference_metadata = prepare_vision_attention_metadata(
        torch.tensor(cu, dtype=torch.int32, device="cuda"), x.device, max_seqlen=window
    )
    expected = forward(x, reference_metadata)
    assert torch.isfinite(output).all()
    # Padding changes the projection's GEMM shape, allowing bf16 rounding.
    torch.testing.assert_close(output, expected, rtol=0.02, atol=0.02)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__]))
