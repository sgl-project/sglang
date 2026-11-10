import pytest
import torch

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() != (7, 0),
    reason="The specialized NVFP4 MoE decode kernel requires an NVIDIA V100",
)


@pytest.mark.parametrize("batch_size", [1, 2, 4, 6, 8, 16, 32])
def test_sm70_topk10_softmax_matches_torch(batch_size):
    from sglang.kernels.ops.moe.sm70_nvfp4_moe_decode import sm70_topk10_softmax

    torch.manual_seed(100 + batch_size)
    logits = torch.randn(
        batch_size, 512, dtype=torch.float16, device="cuda"
    )
    weights, ids = sm70_topk10_softmax(logits)

    probabilities = torch.softmax(logits.float(), dim=-1)
    _, reference_ids = torch.topk(probabilities, 10, dim=-1)
    selected = torch.gather(probabilities, 1, ids.long())
    reference_weights = selected / selected.sum(dim=-1, keepdim=True)

    torch.testing.assert_close(
        torch.sort(ids).values,
        torch.sort(reference_ids.int()).values,
        rtol=0,
        atol=0,
    )
    torch.testing.assert_close(weights, reference_weights, rtol=2e-6, atol=3e-8)


@pytest.mark.parametrize("rows", [4, 8, 32])
def test_sm70_nvfp4_moe_wide_rows_match_individual_rows(rows):
    from sglang.kernels.ops.moe.sm70_nvfp4_moe_decode import sm70_nvfp4_moe_decode

    torch.manual_seed(11)
    # Ten experts are sufficient for this route-layout test. The serving path
    # supplies all 512; keeping the fixture small avoids a >300 MB allocation.
    w13 = torch.empty(
        (10, 160, 640), dtype=torch.int32, device="cuda"
    ).random_()
    w2 = torch.empty(
        (10, 10, 5120), dtype=torch.int32, device="cuda"
    ).random_()
    w13_scales = torch.full(
        (10, 160, 320), 0x70, dtype=torch.uint8, device="cuda"
    )
    w2_scales = torch.full(
        (10, 10, 2560), 0x70, dtype=torch.uint8, device="cuda"
    )
    w13_global = torch.full((10,), 100.0, dtype=torch.float32, device="cuda")
    w2_global = torch.full((10,), 100.0, dtype=torch.float32, device="cuda")
    hidden_states = torch.randn((rows, 2560), dtype=torch.float16, device="cuda")
    original_input = hidden_states.clone()
    ids = torch.arange(10, dtype=torch.int32, device="cuda").repeat(rows, 1)
    weights = torch.softmax(torch.randn((rows, 10), device="cuda"), dim=-1)

    batched = sm70_nvfp4_moe_decode(
        hidden_states,
        w13,
        w2,
        w13_scales,
        w2_scales,
        w13_global,
        w2_global,
        ids.flatten(),
        weights.flatten(),
    )
    individual = torch.cat(
        [
            sm70_nvfp4_moe_decode(
                hidden_states[row : row + 1],
                w13,
                w2,
                w13_scales,
                w2_scales,
                w13_global,
                w2_global,
                ids[row],
                weights[row],
            )
            for row in range(rows)
        ]
    )

    assert torch.isfinite(batched).all()
    assert batched.data_ptr() != hidden_states.data_ptr()
    torch.testing.assert_close(hidden_states, original_input, rtol=0, atol=0)
    assert batched.abs().max() > 0
    torch.testing.assert_close(batched, individual, rtol=0, atol=0)


@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
def test_sm70_topk10_softmax_ties_and_signed_zero(dtype):
    from sglang.kernels.ops.moe.sm70_nvfp4_moe_decode import sm70_topk10_softmax

    logits = torch.zeros((4, 512), device="cuda", dtype=dtype)
    logits[1, ::2] = -0.0
    logits[2].fill_(-5)
    logits[2, [19, 31, 57]] = 4
    logits[3] = torch.arange(512, device="cuda", dtype=dtype) % 7
    weights, ids = sm70_topk10_softmax(logits)
    expected_ids = torch.argsort(logits.float(), descending=True, stable=True)[:, :10]
    expected_weights = torch.softmax(logits.float().gather(1, expected_ids), dim=1)
    torch.testing.assert_close(ids.long(), expected_ids, rtol=0, atol=0)
    torch.testing.assert_close(weights, expected_weights, rtol=2e-6, atol=3e-8)


@pytest.mark.parametrize("batch_size", [1, 4, 8, 32])
def test_sm70_nvfp4_moe_matches_dequantized_weights(batch_size):
    from sglang.kernels.ops.moe.sm70_nvfp4_moe_decode import sm70_nvfp4_moe_decode
    from sglang.srt.hardware_backend.gpu.quantization.gptq_kernels import gptq_marlin_moe_repack
    from sglang.srt.layers.quantization.marlin_utils import (
        sm70_nvfp4_marlin_process_global_scale,
        sm70_nvfp4_marlin_process_scales,
    )

    torch.manual_seed(412 + batch_size)
    device = "cuda"
    experts = 12
    levels = torch.tensor(
        [0, 0.5, 1, 1.5, 2, 3, 4, 6, 0, -0.5, -1, -1.5, -2, -3, -4, -6],
        device=device,
    )

    def make_weight(n, k):
        raw = torch.randint(256, (experts, n, k // 2), dtype=torch.uint8, device=device)
        scales = (torch.rand(experts, n, k // 16, device=device) + 0.5).to(
            torch.float8_e4m3fn
        )
        global_scale = torch.rand(experts, device=device) * 0.01 + 0.01
        codes = torch.stack((raw & 15, raw >> 4), dim=-1).reshape(experts, n, k)
        dequant = levels[codes.long()] * scales.float().repeat_interleave(16, dim=-1)
        dequant *= global_scale[:, None, None]
        packed = gptq_marlin_moe_repack(
            raw.view(torch.int32).transpose(1, 2).contiguous(),
            torch.empty((experts, 0), device=device, dtype=torch.int32),
            k,
            n,
            4,
        )
        encoded, factor = sm70_nvfp4_marlin_process_scales(
            scales.transpose(1, 2).contiguous(), torch.float16
        )
        processed_global = (
            sm70_nvfp4_marlin_process_global_scale(global_scale, torch.float16) / factor
        )
        return packed, encoded, processed_global, dequant

    w13, s13, g13, reference13 = make_weight(320, 2560)
    w2, s2, g2, reference2 = make_weight(2560, 160)
    x = torch.randn(batch_size, 2560, device=device, dtype=torch.float16)
    ids = torch.rand(batch_size, experts, device=device).topk(10, dim=-1).indices.int()
    ids[:, -1] = -1  # Masked routes must not read stale partials.
    weights = torch.softmax(torch.randn(batch_size, 10, device=device), dim=-1)
    output = sm70_nvfp4_moe_decode(x, w13, w2, s13, s2, g13, g2, ids, weights)
    reference = torch.zeros_like(x, dtype=torch.float32)
    for row in range(batch_size):
        for route in range(9):
            expert = int(ids[row, route])
            gate, up = (reference13[expert] @ x[row].float()).chunk(2)
            activated = (torch.nn.functional.silu(gate) * up).half().float()
            reference[row] += (reference2[expert] @ activated) * weights[row, route]

    assert torch.isfinite(output).all()
    # The existing decode contract accumulates each 64/32-element partial in
    # FP16, unlike the independent FP32 reference. Check normalized error so
    # cancellations near zero do not require a loose relative tolerance.
    relative_l2 = (output.float() - reference).norm() / reference.norm()
    assert relative_l2 < 0.003
    torch.testing.assert_close(output.float(), reference, rtol=0.02, atol=0.01)
