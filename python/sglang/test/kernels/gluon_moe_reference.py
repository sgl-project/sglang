"""Independent, dense PyTorch oracle for the GLM MXFP4 Gluon MoE tests.

Repeat a small bank of random tokens for large-M tests. MoE has no cross-token
semantics, so every output can be checked without evaluating a dense reference
for all 32K rows. The kernel still executes the full prefill shape and schedule.
"""

import torch


def dequantize_weight(packed, scales):
    codes = torch.stack((packed & 15, packed >> 4), dim=-1).flatten(-2).long()
    levels = torch.tensor(
        [0, 0.5, 1, 1.5, 2, 3, 4, 6, 0, -0.5, -1, -1.5, -2, -3, -4, -6],
        device=packed.device,
    )
    return levels[codes] * torch.exp2(scales.float() - 127).repeat_interleave(32, -1)


def quantize_dequantize(value, *, shared=False):
    groups = value.float().reshape(*value.shape[:-1], -1, 32)
    peak = groups.abs().amax(dim=-1, keepdim=True)
    if shared:
        floor = torch.floor(torch.log2(peak))
        exponent = floor - 2 + (peak >= 1.75 * torch.exp2(floor))
    else:
        exponent = torch.ceil(torch.log2(peak / 6))
    scale = torch.exp2(exponent.clamp(-127, 127))
    magnitude = (groups / scale).abs().contiguous()
    boundaries = torch.tensor(
        [0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5], device=value.device
    )
    codes = torch.bucketize(magnitude, boundaries)
    # E2M1 round-to-nearest-even: odd lower codes round upward on ties.
    codes += ((magnitude == 0.75) | (magnitude == 1.75) | (magnitude == 3.5)).long()
    levels = torch.tensor([0, 0.5, 1, 1.5, 2, 3, 4, 6], device=value.device)
    return (levels[codes] * groups.sign() * scale).reshape(value.shape)


class GluonMoeFixture:
    """Nonzero random weights with varying E8M0 scales and sparse exact router."""

    def __init__(self, *, local_experts, intermediate, expert_start, shared):
        from sglang.srt.layers.moe.glm_mxfp4_gluon import _pack_scale, _pack_weight

        self.expert_start = expert_start
        self.local_experts = local_experts
        self.shared = shared
        generator = torch.Generator(device="cuda").manual_seed(20261029)
        # A prime period prevents modulo-16/32/64 indexing bugs from passing
        # accidentally when large inputs repeat the reference token bank.
        self.tokens = (
            torch.randn(
                (37, 6144), device="cuda", dtype=torch.bfloat16, generator=generator
            )
            * 0.25
        )
        self.tokens[-1].zero_()
        self.router = torch.zeros((256, 6144), device="cuda", dtype=torch.bfloat16)
        experts = torch.arange(256, device="cuda")
        self.router[experts, experts * 23] = ((experts % 2 + 1) / 2).bfloat16()
        self.bias = ((experts * 73 % 256).float() / 4096).bfloat16()
        self.raw = []
        self.packed = []
        for n, k in ((2 * intermediate, 6144), (6144, intermediate)):
            weight = torch.randint(
                0,
                256,
                (local_experts + 1, n, k // 2),
                device="cuda",
                dtype=torch.uint8,
                generator=generator,
            )
            scales = torch.randint(
                119,
                122,
                (local_experts + 1, n, k // 32),
                device="cuda",
                dtype=torch.uint8,
                generator=generator,
            )
            if not shared:
                weight[-1].zero_()
                scales[-1].fill_(127)
            self.raw.extend((weight, scales))
            self.packed.extend((_pack_weight(weight), _pack_scale(scales)))

    def reference(self, *, routed_scaling_factor):
        """Dense GEMMs, sigmoid top-8, SwiGLU, and both MXFP4 quantization steps."""
        x = self.tokens.float()
        probability = (x @ self.router.float().T).bfloat16().float().sigmoid()
        ids = (probability + self.bias.float()).argsort(
            descending=True, stable=True, dim=-1
        )[:, :8]
        weights = probability.gather(1, ids)
        weights = weights / weights.sum(dim=1, keepdim=True) * routed_scaling_factor
        quantized = quantize_dequantize(x)
        result = torch.zeros_like(x)
        w13, s13, w2, s2 = self.raw
        for expert in ids.unique().tolist():
            local = expert - self.expert_start
            if not 0 <= local < self.local_experts:
                continue
            rows, ranks = torch.where(ids == expert)
            gu = quantized[rows] @ dequantize_weight(w13[local], s13[local]).T
            gate, up = gu.chunk(2, dim=-1)
            activated = (gate * gate.sigmoid() * up).bfloat16().float()
            down = (
                quantize_dequantize(activated)
                @ dequantize_weight(w2[local], s2[local]).T
            )
            result[rows] += down * weights[rows, ranks, None]
        if self.shared:
            gu = (
                quantize_dequantize(x, shared=True)
                @ dequantize_weight(w13[-1], s13[-1]).T
            )
            gate, up = gu.bfloat16().float().chunk(2, dim=-1)
            activated = (gate * gate.sigmoid() * up).bfloat16().float()
            down = (
                quantize_dequantize(activated, shared=True)
                @ dequantize_weight(w2[-1], s2[-1]).T
            )
            result += down.bfloat16().float()
        return result.bfloat16()

    def inputs(self, m):
        period = self.tokens.shape[0]
        return self.tokens.repeat(((m + period - 1) // period, 1))[:m].contiguous()
