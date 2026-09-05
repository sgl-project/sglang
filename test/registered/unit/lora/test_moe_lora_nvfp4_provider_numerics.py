"""Marlin NVFP4 W4A16 provider against an exactly-dequantized reference.

Weights are quantized here in the modelopt checkpoint format (packed e2m1
pairs, low nibble = even index; e4m3 group-16 scales; fp32 per-expert global
scale), repacked with the same load-path helper serving uses, and compared
pair-major through the provider's identity ``pair_to_row``.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from sglang.srt.lora.moe.base_gemm_provider import select_provider_cls
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=60, stage="base-b", runner_config="1-gpu-large")

_E = 4
_H = 256
_I = 128
_TOPK = 2
_T = 32

_E2M1_VALUES = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])


def _skip_unless_supported():
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    if torch.cuda.get_device_capability()[0] < 9:
        pytest.skip("Marlin NVFP4 needs SM90+ here")


def _quant_nvfp4(weight: torch.Tensor):
    """[E, N, K] bf16 -> (packed uint8 [E,N,K/2], e4m3 [E,N,K/16], fp32 [E])
    plus the exact dequantized weights."""
    e, n, k = weight.shape
    w = weight.float()
    global_scale = (w.abs().amax(dim=(1, 2)) / (448.0 * 6.0)).clamp(min=1e-12)
    groups = w.view(e, n, k // 16, 16)
    amax16 = groups.abs().amax(dim=-1)
    block_scale = (
        (amax16 / (6.0 * global_scale[:, None, None]))
        .clamp(min=2**-6)
        .to(torch.float8_e4m3fn)
    )
    denom = block_scale.float() * global_scale[:, None, None]
    scaled = groups / denom[..., None]
    table = _E2M1_VALUES.to(weight.device)
    idx = (scaled.abs().unsqueeze(-1) - table).abs().argmin(dim=-1)
    sign = scaled < 0
    codes = (idx | (sign.long() << 3)).to(torch.uint8).view(e, n, k)
    packed = codes[..., 0::2] | (codes[..., 1::2] << 4)
    dequant = (torch.where(sign, -table[idx], table[idx]) * denom[..., None]).view(
        e, n, k
    )
    return packed, block_scale, global_scale, dequant


def _fake_layer(w13, w2):
    w13_q, w13_bs, w13_gs, w13_eff = _quant_nvfp4(w13)
    w2_q, w2_bs, w2_gs, w2_eff = _quant_nvfp4(w2)
    layer = SimpleNamespace(
        w13_weight=SimpleNamespace(data=w13_q),
        w2_weight=SimpleNamespace(data=w2_q),
        w13_weight_scale=SimpleNamespace(data=w13_bs),
        w2_weight_scale=SimpleNamespace(data=w2_bs),
        w13_weight_scale_2=SimpleNamespace(data=w13_gs),
        w2_weight_scale_2=SimpleNamespace(data=w2_gs),
        quant_config=SimpleNamespace(group_size=16),
        moe_runner_config=SimpleNamespace(is_gated=True),
        num_local_experts=_E,
        intermediate_size_per_partition=_I,
        hidden_size=_H,
        params_dtype=torch.bfloat16,
    )
    return layer, w13_eff, w2_eff


def _marlin_quant_info(layer):
    from sglang.srt.layers.quantization.marlin_utils_fp4 import (
        prepare_moe_nvfp4_layer_for_marlin,
    )
    from sglang.srt.lora.moe.quant_info import MoeLoraNvFp4MarlinQuantInfo

    prepare_moe_nvfp4_layer_for_marlin(layer)
    # The fake layer has no quant_method: the payload comes off the repacked
    # tensors alone, not through the legacy MarlinMoeQuantInfo accessor.
    return MoeLoraNvFp4MarlinQuantInfo.from_layer(layer)


def _rel_l2(a, b):
    return (a - b).norm() / b.norm().clamp(min=1e-12)


def test_nvfp4_marlin_gateup_and_down_match_reference():
    _skip_unless_supported()
    device = torch.device("cuda")
    g = torch.Generator(device="cpu").manual_seed(11)

    def rand(*shape):
        return (torch.randn(*shape, generator=g) * 0.05).to(torch.bfloat16).to(device)

    w13 = rand(_E, 2 * _I, _H)
    w2 = rand(_E, _H, _I)
    hidden = rand(_T, _H)
    topk_ids = torch.stack(
        [torch.randperm(_E, generator=g)[:_TOPK] for _ in range(_T)]
    ).to(device=device, dtype=torch.int32)

    layer, w13_eff, w2_eff = _fake_layer(w13, w2)
    quant_info = _marlin_quant_info(layer)

    from sglang.srt.runtime_context import get_context

    provider_cls = select_provider_cls("route_major", "nvfp4", "marlin")
    assert provider_cls is select_provider_cls(
        "expert_major", "nvfp4", "marlin"
    ), "decode must run the same route-major marlin provider"
    provider = provider_cls(quant_info)

    with get_context().override_server_args():
        state = provider.prepare(hidden, topk_ids, _TOPK)
        gateup = torch.zeros(
            provider.gateup_out_shape(state), dtype=torch.bfloat16, device=device
        )
        provider.gateup(state, gateup)

        flat_ids = topk_ids.flatten()
        pair_idx = torch.nonzero(flat_ids >= 0, as_tuple=True)[0]
        experts = flat_ids[pair_idx].long()
        tokens = (pair_idx // _TOPK).long()

        ref_gateup = torch.einsum(
            "ph,pnh->pn", hidden[tokens].float(), w13_eff[experts]
        )
        got_gateup = gateup[pair_idx].float()
        assert _rel_l2(got_gateup, ref_gateup) < 2e-2

        act_pairs = (torch.randn_like(ref_gateup[:, :_I]) * 0.1).to(torch.bfloat16)
        act = torch.zeros(
            provider.act_out_shape(state), dtype=torch.bfloat16, device=device
        )
        act[pair_idx] = act_pairs
        down = torch.zeros(
            provider.down_out_shape(state), dtype=torch.bfloat16, device=device
        )
        provider.down(state, act, down)

        ref_down = torch.einsum("pi,phi->ph", act_pairs.float(), w2_eff[experts])
        got_down = down[pair_idx].float()
        assert _rel_l2(got_down, ref_down) < 2e-2


def test_single_token_align_matches_general_path_with_invalid_ids():
    """The bs=1 one-program alignment must emit exactly what the general path
    emits with ignore_invalid_expert=True: EP maps non-local experts to -1,
    and those pairs get no block, no slot, and no share of num_post. With
    every id valid it is also a drop-in for the upstream CUDA warp kernel."""
    _skip_unless_supported()
    from sglang.kernels.ops.moe.moe_align_single_token import (
        moe_align_single_token as moe_align_single_token_cuda,
    )
    from sglang.srt.layers.moe.moe_runner.triton_utils.moe_align_block_size import (
        moe_align_block_size,
    )
    from sglang.srt.lora.moe.kernels.align_rows import (
        moe_align_single_token,
    )

    device = torch.device("cuda")
    g = torch.Generator(device="cpu").manual_seed(7)
    num_experts, topk = 64, 8
    for block_size in (8, 16, 48, 256):
        for n_invalid in (0, 3, topk):
            ids = torch.randperm(num_experts, generator=g)[:topk]
            if n_invalid:
                ids[torch.randperm(topk, generator=g)[:n_invalid]] = -1
            ids = ids.view(1, topk).to(device=device, dtype=torch.int32)

            sorted_fast, experts_fast, post_fast = moe_align_single_token(
                ids, block_size
            )
            sorted_ref, experts_ref, post_ref = moe_align_block_size(
                ids, block_size, num_experts, ignore_invalid_expert=True
            )

            post = int(post_ref.item())
            assert int(post_fast.item()) == post
            assert torch.equal(sorted_fast[:post], sorted_ref[:post])
            blocks = post // block_size
            assert torch.equal(experts_fast[:blocks], experts_ref[:blocks])
            if n_invalid == 0:
                sorted_cuda, experts_cuda, post_cuda = moe_align_single_token_cuda(
                    ids, block_size
                )
                assert torch.equal(post_fast, post_cuda)
                assert torch.equal(sorted_fast, sorted_cuda)
                assert torch.equal(experts_fast, experts_cuda)


def test_align_rows_routes_by_token_count():
    """The shared provider entry takes the one-program path for one token and
    the general path otherwise; both must match
    moe_align_block_size(ignore_invalid_expert=True), invalid ids included."""
    _skip_unless_supported()
    from sglang.srt.layers.moe.moe_runner.triton_utils.moe_align_block_size import (
        moe_align_block_size,
    )
    from sglang.srt.lora.moe.kernels.align_rows import align_rows

    device = torch.device("cuda")
    g = torch.Generator(device="cpu").manual_seed(3)
    num_experts, topk, block_size = 64, 8, 16
    for tokens in (1, 2):
        ids = torch.stack(
            [torch.randperm(num_experts, generator=g)[:topk] for _ in range(tokens)]
        )
        ids[0, 0] = -1
        ids = ids.to(device=device, dtype=torch.int32)
        sorted_got, experts_got, post_got = align_rows(ids, block_size, num_experts)
        sorted_ref, experts_ref, post_ref = moe_align_block_size(
            ids, block_size, num_experts, ignore_invalid_expert=True
        )
        post = int(post_ref.item())
        assert int(post_got.item()) == post
        assert torch.equal(sorted_got[:post], sorted_ref[:post])
        blocks = post // block_size
        assert torch.equal(experts_got[:blocks], experts_ref[:blocks])


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
