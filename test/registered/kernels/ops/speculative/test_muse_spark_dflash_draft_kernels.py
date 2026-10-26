"""DFlash grouped conv kernels (gfx950): the input conv fed by the projection's split-K partials and the output
conv fused with the residual add + RMSNorm, against the eager formula."""

import pytest
import torch
import torch.nn.functional as F

from sglang.srt.utils import is_gfx95_supported, is_hip
from sglang.test.ci.ci_register import register_amd_ci

register_amd_ci(est_time=60, suite="stage-b-test-1-gpu-small-amd-mi35x")

pytestmark = pytest.mark.skipif(
    not (is_hip() and is_gfx95_supported()), reason="Tuned for MI355X (gfx950)."
)


def _close(a, b):
    # BF16 roundings of FP32 sums taken in different orders: a few ULPs of the largest operand at worst
    # (sums that cancel), about one rounding (2^-9) of relative L2 error overall.
    a, b = a.float(), b.float()
    err = (a - b).abs()
    assert err.max() <= 2**-6 * b.abs().max(), err.max()
    assert err.norm() <= 4e-3 * b.norm(), err.norm() / b.norm()


def _eager_add_rmsnorm(x, residual, weight, eps):
    s = x.float() + residual.float()
    y = s * torch.rsqrt(s.pow(2).mean(-1, keepdim=True) + eps)
    return (y * weight.float()).to(x.dtype), s.to(x.dtype)


# (hidden, groups, taps, block): the TP4 Muse Spark draft, and a 3-tap conv on a block size that is not a power
# of two (positions are taken modulo the block, so taps must not reach into the previous request's block).
@pytest.mark.parametrize("h,groups,taps,block", [(8192, 512, 2, 8), (1024, 64, 3, 5)])
@pytest.mark.parametrize("rows", [24, 40, 64, 128])
def test_conv_kernels_match_eager(h, groups, taps, block, rows):
    from sglang.kernels.ops.speculative.dflash_grouped_conv import (
        conv_finish_add_rmsnorm,
        conv_prepare,
    )
    from sglang.srt.models.dflash import _grouped_conv

    torch.manual_seed(rows)
    x = torch.randn(rows, h, device="cuda").to(torch.bfloat16)
    y = torch.randn(rows, h, device="cuda").to(torch.bfloat16)
    residual = torch.randn(rows, h, device="cuda").to(torch.bfloat16)
    weight = (1 + 0.1 * torch.randn(h, device="cuda")).to(torch.bfloat16)
    base = (0.1 * torch.randn(2, taps, h, device="cuda")).to(torch.bfloat16)
    base[:, 0] += 1
    w = (0.01 * torch.randn(2 * taps * groups, h, device="cuda")).to(torch.bfloat16)
    projection = F.linear(x, w)
    delta = projection.view(rows, 2, taps, groups)
    gs = h // groups

    def conv(inp, side):
        # The eager formula in FP32: the kernels round once, the BF16 eager formula after every op.
        return _grouped_conv.__wrapped__(
            inp.float(),
            delta[:, side].float(),
            base[side].float(),
            block,
            groups,
            gs,
            taps,
        )

    # Split-K partials (summed in the kernel) and the BF16 product give the same deltas.
    partials = torch.stack([F.linear(x[:, i::2], w[:, i::2]).float() for i in range(2)])
    for source in (projection, partials):
        out, d1 = conv_prepare(
            source, x, base[0], block_size=block, taps=taps, num_groups=groups
        )
        _close(out, conv(x, 0))
        _close(d1, delta[:, 1])

    ref = _eager_add_rmsnorm(conv(y, 1).to(torch.bfloat16), residual, weight, 1e-5)
    got = conv_finish_add_rmsnorm(
        y, delta[:, 1], base[1], residual, weight, 1e-5, block_size=block
    )
    for g, r in zip(got, ref):
        _close(g, r)


@pytest.mark.parametrize("rows", [8, 16, 24, 64, 128])
def test_fused_grouped_conv_module_matches_unfused(rows):
    """The fused path (row-major partials from 17 rows, the BF16 projection below; deltas handed between the two
    kernels) against the unfused module on the same weights, through prepare and finish_add_norm."""
    from sglang.srt.environ import envs
    from sglang.srt.layers.layernorm import RMSNorm
    from sglang.srt.models.dflash import DFlashGroupedConv

    torch.manual_seed(rows)
    h = 8192
    convs = {}
    for fused in (True, False):
        with envs.SGLANG_MUSE_DFLASH_FUSED_CONV.override(fused):
            convs[fused] = DFlashGroupedConv(h, block_size=8, taps=2, group_size=16)
    assert convs[True].fused and not convs[False].fused
    with torch.no_grad():
        convs[True].base_kernel.add_(0.1 * torch.randn_like(convs[True].base_kernel))
        convs[True].kernel_projection.weight.mul_(0.1)
    convs[False].load_state_dict(convs[True].state_dict())
    for conv in convs.values():
        conv.to(device="cuda", dtype=torch.bfloat16)
    norm = RMSNorm(h, eps=1e-5).to(device="cuda", dtype=torch.bfloat16)
    with torch.no_grad():
        norm.weight.add_(0.1 * torch.randn_like(norm.weight))

    x = torch.randn(rows, h, device="cuda").to(torch.bfloat16)
    y = torch.randn(rows, h, device="cuda").to(torch.bfloat16)
    residual = torch.randn(rows, h, device="cuda").to(torch.bfloat16)
    (out_f, k_f), (out_u, k_u) = convs[True].prepare(x), convs[False].prepare(x)
    _close(out_f, out_u)
    got = convs[True].finish_add_norm(y, k_f, residual, norm)
    ref = convs[False].finish_add_norm(y, k_u, residual, norm)
    for g, r in zip(got, ref):
        _close(g, r)


@pytest.mark.parametrize("tp", [2, 4, 8])
@pytest.mark.parametrize("softcap", [20.0, None])
def test_merged_candidates_match_topk_of_full_vocab(tp, softcap):
    """Per-shard packed top-k (one all-gather) merged in one kernel gives the full-vocabulary top-k: global ids
    offset by each shard's start and exact in FP32, values descending with the unary transform; a shard's
    winners may straddle its 1024-column chunks and the ragged last chunk."""
    from sglang.kernels.ops.speculative.dflash_candidates import (
        merge_candidates,
        topk_packed,
    )

    torch.manual_seed(tp)
    rows, shard, k, multiplier = 64, 50512, 16, 0.1767766952966369
    logits = torch.randn(rows, tp * shard, device="cuda").to(torch.bfloat16)
    # Distinct winners so the expected order has no ties; some rows pack them into one shard's last chunk.
    winners = torch.rand(rows, tp * shard, device="cuda").argsort(-1)[:, : 4 * k]
    winners[::4] = shard - 1 - torch.arange(4 * k, device="cuda")
    logits.scatter_(
        1,
        winners,
        (10 + torch.arange(4 * k, device="cuda") / 8)
        .to(torch.bfloat16)
        .expand(rows, -1)
        .contiguous(),
    )
    packed = [
        topk_packed(logits[:, r * shard : (r + 1) * shard], k, r * shard)
        for r in range(tp)
    ]
    ids, vals = merge_candidates(
        torch.cat(packed, -1), k, multiplier=multiplier, softcap=softcap
    )
    top, want_ids = torch.topk(logits.float(), k, dim=-1)
    want = top * multiplier
    if softcap is not None:
        want = torch.tanh(want / softcap) * softcap
    assert torch.equal(ids, want_ids)
    torch.testing.assert_close(vals, want, rtol=1e-6, atol=1e-6)


class _LinearAttention(torch.nn.Module):
    def __init__(self, h):
        super().__init__()
        self.proj = torch.nn.Linear(h, h, bias=False)

    def forward(self, positions, hidden_states, forward_batch):
        return self.proj(hidden_states)


@pytest.mark.parametrize("hidden,rows", [(8192, 64), (1024, 40)])
def test_fused_conv_layer_stack_matches_unfused(hidden, rows):
    """Through DFlashDraftModel.forward: each layer's output conv is deferred into the next layer's input norm and
    the last one into the final norm; the stack must match the unfused layers (none dropped or applied twice)."""
    from torch import nn

    from sglang.srt.environ import envs
    from sglang.srt.layers.layernorm import RMSNorm
    from sglang.srt.models.dflash import (
        DFlashDecoderLayer,
        DFlashDraftModel,
        DFlashGroupedConv,
    )

    def build(fused):
        torch.manual_seed(0)
        layers = []
        for _ in range(3):
            layer = DFlashDecoderLayer.__new__(DFlashDecoderLayer)
            nn.Module.__init__(layer)
            layer.input_layernorm = RMSNorm(hidden, eps=1e-5)
            layer.post_attention_layernorm = RMSNorm(hidden, eps=1e-5)
            layer.self_attn = _LinearAttention(hidden)
            layer.mlp = nn.Linear(hidden, hidden, bias=False)
            with envs.SGLANG_MUSE_DFLASH_FUSED_CONV.override(fused):
                layer.attention_conv = DFlashGroupedConv(hidden, 8, 2, 16)
                layer.mlp_conv = DFlashGroupedConv(hidden, 8, 2, 16)
            for conv in (layer.attention_conv, layer.mlp_conv):
                with torch.no_grad():
                    conv.base_kernel.add_(0.1 * torch.randn_like(conv.base_kernel))
                    conv.kernel_projection.weight.mul_(0.1)
            layers.append(layer)
        model = DFlashDraftModel.__new__(DFlashDraftModel)
        nn.Module.__init__(model)
        model.layers = nn.ModuleList(layers)
        model.norm = RMSNorm(hidden, eps=1e-5)
        model.embed_tokens = None
        return model.to(device="cuda", dtype=torch.bfloat16)

    x = torch.randn(rows, hidden, device="cuda").to(torch.bfloat16)
    positions = torch.arange(rows, device="cuda")
    out = {
        fused: DFlashDraftModel.forward(
            build(fused), None, positions, None, input_embeds=x
        ).hidden_states
        for fused in (True, False)
    }
    _close(out[True], out[False])
