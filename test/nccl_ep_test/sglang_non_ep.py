"""Real SGLang routing-mask and masked FP8 kernels on an ordinary CUDA Graph."""

import torch

from .oracle import dequantize_fp8


def exercise_topk_fp8(*, routing_dtype=torch.int32):
    from sglang.kernels.ops.quantization.fp8_kernel import (
        sglang_per_token_group_quant_fp8,
    )
    from sglang.srt.layers.moe.topk import TopKConfig, select_experts

    hidden, rows = 2048, 8
    payload = torch.ones(2, rows, hidden, dtype=torch.bfloat16, device="cuda")
    ids = torch.tensor([[0, 2]] * rows, dtype=routing_dtype, device="cuda")
    weights = torch.tensor([[0.25, 0.75]] * rows, device="cuda")
    valid = torch.tensor([5], dtype=torch.int32, device="cuda")
    counts = torch.tensor([5, 5], dtype=torch.int32, device="cuda")
    logits = torch.zeros(rows, 4, device="cuda")

    def router(**kwargs):
        # A synthetic router is deliberately padding-unaware. The actual
        # select_experts postprocessing must mask its nonzero padded routes.
        return weights.clone(), ids.clone()

    config = TopKConfig(
        top_k=2,
        custom_routing_function=router,
        allow_routed_experts_capture=False,
    )

    def forward():
        topk = select_experts(payload[0], logits, config, num_token_non_padded=valid)
        quantized, scales = sglang_per_token_group_quant_fp8(
            payload, group_size=128, masked_m=counts
        )
        return topk, dequantize_fp8(quantized, scales), scales

    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(2):
            forward()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            topk, reconstructed, scales = forward()
    torch.cuda.current_stream().wait_stream(stream)
    observed = []
    try:
        for step, count in ((0, 5), (1, 0), (0, 5)):
            # Distinct block scales exercise all H/128 scale positions.
            block = torch.arange(hidden, device="cuda") // 128 % 4
            exact = (2.0 ** (block + step)).to(torch.bfloat16)
            payload.copy_(exact.expand_as(payload))
            ids.copy_(
                torch.tensor([[0, 2] if step == 0 else [1, 3]] * rows, device="cuda")
            )
            weights.copy_(
                torch.tensor(
                    [[0.25, 0.75] if step == 0 else [0.75, 0.25]] * rows, device="cuda"
                )
            )
            valid.fill_(count)
            counts.fill_(count)
            graph.replay()
            wanted_ids = torch.tensor(
                [[0, 2] if step == 0 else [1, 3]] * rows, dtype=routing_dtype
            )
            wanted_ids[count:] = -1
            torch.testing.assert_close(topk.topk_ids.cpu(), wanted_ids, rtol=0, atol=0)
            # Masking must work even though the real CUDA path retains weights.
            torch.testing.assert_close(topk.topk_weights, weights, rtol=0, atol=0)
            torch.testing.assert_close(
                reconstructed[:, :count], payload[:, :count], rtol=0, atol=0
            )
            if count:
                assert (scales[:, :count] > 0).all()
                assert torch.unique(scales[:, :count]).numel() == 4
            observed.append(count)
        return {
            "valid_rows": observed,
            "scales_applied": True,
            "hidden": hidden,
            "routing_dtype": str(routing_dtype),
            "ep_tested": False,
            "operations": [
                "select_experts CUDA masking",
                "masked 3D FP8 quant/dequant",
            ],
        }
    finally:
        torch.cuda.synchronize()
        graph.reset()
