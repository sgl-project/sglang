"""AITER MegaMoE-TP on 2 GPUs: the integration path against a pure-PyTorch MXFP4 reference.

Drives ``register`` / ``finalize`` / ``maybe_forward`` on two hosted layers with synthetic
MXFP4 weights at TP 2, with the unfused shared expert appended to the routed experts and
fp8 communication, at a row count that needs no pad and one that needs a pad row. One
process per rank. Needs ROCm gfx950, AITER with ``aiter.ops.flydsl.mega_moe_tp`` and 2 GPUs.

The kernel is lossy by design (MXFP4 activations, fp8 communication), so the reference
applies the same activation quantization.
"""

from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist

import sglang.srt.layers.moe.aiter_mega_moe_tp as mmtp
from sglang.srt.environ import envs
from sglang.srt.layers.moe.utils import MoeRunnerBackend
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.test.ci.ci_register import register_amd_ci
from sglang.test.quant_ref_utils import break_fp4_bytes
from sglang.test.test_utils import run_distributed_test

# gfx950 only; the test uses 2 of the lane's 8 GPUs.
register_amd_ci(est_time=90, stage="stage-c", runner_config="large-8-gpu-amd-mi35x")

HIDDEN = 512
INTER = 256  # per rank
ROUTED = 8
TOPK = 2
MAX_TOKENS = 512  # the window under test
TOKENS = (16, 17)  # ar mode pads to a multiple of tp: none, and one row
WORLD_SIZE = 2
NUM_LAYERS = 2
# Measured at this shape on MI355X: rel L2 up to 0.050, row cosine down to 0.997.
REL_L2 = 0.065
MIN_COSINE = 0.98


def _kernel_available() -> bool:
    if not (torch.cuda.is_available() and torch.version.hip):
        return False
    try:
        from aiter.ops.flydsl.mega_moe_tp import mega_moe_tp_supported
    except ImportError:
        return False
    return mega_moe_tp_supported()


# --- reference: dequantized MXFP4 weights, fp32 ------------------------------------


def _dequant_mxfp4(packed, scale):
    """packed uint8 [..., K // 2] (two E2M1 codes, low nibble first), scale uint8
    [..., K // 32] (E8M0, bias 127) -> fp32 [..., K]."""
    values = break_fp4_bytes(packed.flatten(0, -2)).view(*packed.shape[:-1], -1)
    scales = torch.exp2(scale.float() - 127).unsqueeze(-1)
    return (values.unflatten(-1, (-1, 32)) * scales).flatten(-2)


def _quant_dequant_mxfp4(x):
    """Activations as the kernel quantizes them: per 32 values an E8M0 scale of
    ceil_pow2(amax / 6) and E2M1 codes rounded to nearest even."""
    blocks = x.float().unflatten(-1, (-1, 32))
    amax = blocks.abs().amax(dim=-1, keepdim=True)
    scale = torch.exp2(torch.ceil(torch.log2(amax / 6)).clamp(-127, 127))
    scaled = (blocks / scale).abs()
    # E2M1 spacing: 0.5 below 2, 1 in [2, 4), 2 from 4 up to the maximum 6.
    step = 0.5 * (1 + (scaled >= 2).float() + 2 * (scaled >= 4).float())
    quantized = (torch.round(scaled / step) * step).clamp(max=6)
    return (quantized.copysign(blocks) * scale).flatten(-2)


def _reference_partial(x, ids, weights, w13, w2):
    """sum_k w_k * down(silu(gate) * up) over one rank's slice of the intermediate dim.
    w13 [E, 2 * inter, H] and w2 [E, H, inter] are dequantized."""
    quantized_x = _quant_dequant_mxfp4(x)
    out = torch.zeros(x.shape[0], w2.shape[1], dtype=torch.float32, device=x.device)
    for expert in range(w13.shape[0]):
        token, slot = (ids == expert).nonzero(as_tuple=True)
        if token.numel() == 0:
            continue
        gate, up = (quantized_x[token] @ w13[expert].T).chunk(2, dim=-1)
        activated = _quant_dequant_mxfp4(torch.nn.functional.silu(gate) * up)
        weight = weights[token, slot].unsqueeze(-1)
        out.index_add_(0, token, (activated @ w2[expert].T) * weight)
    return out


# --- synthetic weights --------------------------------------------------------------


def _shard(layer, rank, device):
    """Raw (unshuffled) MXFP4 weights of one layer and rank: ROUTED experts and, at
    index ROUTED, the shared expert. Random codes; scales of magnitude about 2**-6."""
    gen = torch.Generator().manual_seed(1000 * layer + rank)

    def stack(rows, cols, exponent):
        codes = torch.randint(
            0, 256, (ROUTED + 1, rows, cols // 2), dtype=torch.uint8, generator=gen
        )
        scales = torch.randint(
            exponent,
            exponent + 3,
            (ROUTED + 1, rows, cols // 32),
            dtype=torch.uint8,
            generator=gen,
        )
        return codes.to(device), scales.to(device)

    w13, w13_scale = stack(2 * INTER, HIDDEN, 120)
    w2, w2_scale = stack(HIDDEN, INTER, 121)
    return dict(w13=w13, w13_scale=w13_scale, w2=w2, w2_scale=w2_scale)


def _block(layer, raw):
    """A hosted MoE block as the loader leaves it: the routed experts preshuffled, the
    shared expert in its dense layers' checkpoint layout."""
    from aiter.ops.shuffle import shuffle_weight
    from aiter.utility.fp4_utils import e8m0_shuffle

    def param(tensor):
        return torch.nn.Parameter(tensor, requires_grad=False)

    def shuffled(weight, scale):
        rows = scale.shape[1]
        flat = e8m0_shuffle(scale[:ROUTED].reshape(ROUTED * rows, -1))
        return (
            param(shuffle_weight(weight[:ROUTED].contiguous(), (16, 16))),
            param(flat.view(ROUTED, rows, -1)),
        )

    experts = SimpleNamespace(
        runner_backend=MoeRunnerBackend.AITER,
        with_bias=False,
        moe_runner_config=SimpleNamespace(
            activation="silu", is_gated=True, top_k=TOPK, swiglu_limit=None
        ),
    )
    experts.w13_weight, experts.w13_weight_scale = shuffled(
        raw["w13"], raw["w13_scale"]
    )
    experts.w2_weight, experts.w2_weight_scale = shuffled(raw["w2"], raw["w2_scale"])
    experts.w13_weight.is_shuffled = True

    block = torch.nn.Module()
    block.layer_id = layer
    block.experts = experts
    block.num_fused_shared_experts = 0
    block.is_hash = False
    block.routed_scaling_factor_fused_in_topk = True
    block._shared_expert_tp1 = False
    block.gate = SimpleNamespace(e_score_correction_bias_vl=None)
    block.shared_experts = SimpleNamespace(
        gate_up_proj=SimpleNamespace(
            weight=param(raw["w13"][ROUTED].clone()),
            weight_scale=param(raw["w13_scale"][ROUTED].clone()),
        ),
        down_proj=SimpleNamespace(
            weight=param(raw["w2"][ROUTED].clone()),
            weight_scale=param(raw["w2_scale"][ROUTED].clone()),
        ),
    )
    return block


class _Router:
    """The block's top-k: returns the routing the test staged."""

    def __init__(self):
        self.enable_waterfill, self.staged = False, None

    def stage(self, ids, weights):
        self.staged = SimpleNamespace(topk_ids=ids, topk_weights=weights)

    def __call__(self, hidden_states, router_logits, **kwargs):
        return self.staged


def _routing(tokens, layer, device):
    """Replicated x, routed ids and weights (broadcast from rank 0)."""
    gen = torch.Generator().manual_seed(7 * tokens + layer)
    x = torch.randn(tokens, HIDDEN, generator=gen).to(device, torch.bfloat16)
    ids = torch.rand(tokens, ROUTED, generator=gen).topk(TOPK).indices.to(device)
    weights = torch.softmax(torch.randn(tokens, TOPK, generator=gen), dim=-1).to(device)
    for tensor in (x, ids, weights):
        dist.broadcast(tensor, src=0)
    return x, ids, weights


# --- one rank ---------------------------------------------------------------------


def _check(rank):
    device = torch.device("cuda", rank)
    cpu_group = dist.new_group(backend="gloo")
    mmtp.get_exec = lambda: SimpleNamespace(
        moe=SimpleNamespace(enable_aiter_mega_moe_tp=True)
    )
    mmtp.get_parallel = lambda: SimpleNamespace(
        tp_group=SimpleNamespace(
            rank_in_group=rank, world_size=WORLD_SIZE, cpu_group=cpu_group
        )
    )
    envs.SGLANG_AITER_MEGA_MOE_TP_MAX_TOKENS.set(MAX_TOKENS)
    envs.SGLANG_AITER_MEGA_MOE_TP_COMM_DTYPE.set("fp8")
    import sglang.srt.models.deepseek_common.amd.deepseek_v2_hip_moe as hip_moe

    # Routing is staged by the test: only the kernel path is under test.
    hip_moe.forward_gate = lambda moe, hidden, allocator, *, fused_gate: (None, None)

    # Every rank's raw shards: the reference sums the partial of each rank's slice.
    shards = [
        [_shard(layer, r, device) for r in range(WORLD_SIZE)]
        for layer in range(NUM_LAYERS)
    ]
    dequantized = [
        [
            (
                _dequant_mxfp4(s["w13"], s["w13_scale"]),
                _dequant_mxfp4(s["w2"], s["w2_scale"]),
            )
            for s in layer_shards
        ]
        for layer_shards in shards
    ]
    blocks = [_block(layer, shards[layer][rank]) for layer in range(NUM_LAYERS)]
    routers = [_Router() for _ in blocks]
    for block, router in zip(blocks, routers, strict=True):
        block.topk = router
    model = torch.nn.Sequential(*blocks)
    stock_w13 = [b.experts.w13_weight.data.clone() for b in blocks]

    for block in blocks:
        mmtp.register(block)
    mmtp.finalize(model)

    # The stock path keeps reading the routed experts it always read.
    for block, before in zip(blocks, stock_w13, strict=True):
        assert block.experts.w13_weight.shape[0] == ROUTED
        assert torch.equal(block.experts.w13_weight.data, before)

    batch = SimpleNamespace(forward_mode=ForwardMode.EXTEND)
    for tokens in TOKENS:
        for layer, (block, router) in enumerate(zip(blocks, routers, strict=True)):
            x, ids, weights = _routing(tokens, layer, device)
            router.stage(ids, weights)
            out = mmtp.maybe_forward(block, x, batch)
            assert out is not None, f"{tokens} tokens took the stock path"
            assert out.shape == x.shape and out.dtype == torch.bfloat16

            full_ids = torch.cat([ids, torch.full_like(ids[:, :1], ROUTED)], dim=1)
            full_weights = torch.cat([weights, torch.ones_like(weights[:, :1])], dim=1)
            reference = sum(
                _reference_partial(x, full_ids, full_weights, *dequantized[layer][r])
                for r in range(WORLD_SIZE)
            )
            error = (out.float() - reference).norm() / reference.norm()
            cosine = torch.nn.functional.cosine_similarity(
                out.float(), reference, dim=-1
            ).min()
            assert error <= REL_L2, (
                f"{tokens} tokens layer {layer}: rel L2 {error:.4f} > {REL_L2}"
            )
            assert cosine >= MIN_COSINE, (
                f"{tokens} tokens layer {layer}: row cosine {cosine:.4f} < {MIN_COSINE}"
            )

            replicas = [torch.empty_like(out) for _ in range(WORLD_SIZE)]
            dist.all_gather(replicas, out)
            assert all(torch.equal(replicas[0], r) for r in replicas), "ranks differ"
    torch.cuda.synchronize(device)


@pytest.mark.skipif(
    not _kernel_available(),
    reason="needs ROCm gfx950 with aiter.ops.flydsl.mega_moe_tp",
)
def test_mega_moe_tp_matches_the_mxfp4_reference():
    if torch.cuda.device_count() < WORLD_SIZE:
        pytest.skip(f"needs {WORLD_SIZE} GPUs")
    run_distributed_test(_check, world_size=WORLD_SIZE)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
