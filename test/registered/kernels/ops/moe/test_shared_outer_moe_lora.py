"""Shared-outer decode MoE-LoRA (SGLANG_OPT_LORA_SHARED_OUTER_MOE_DECODE) through
the LoRA hooks the MoE runners call, at the per-rank decode shapes of a
896-expert, top-16 MoE at TP16, against a reference with the per-expert
fused_moe_lora path's rounding points."""

import pytest
import torch

from sglang.srt.lora import lora_moe_runners
from sglang.srt.lora.lora_moe_runners import LoRAInfo, build_lora_hooks
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=60, stage="base-b-kernel-unit", runner_config="1-gpu-large")

# experts, hidden, intermediate per rank (Marlin-padded), LoRA rank, top-k
E, H, N, N_PAD, R, TOP_K = 896, 3584, 192, 256, 16, 16


def _bf16_ulps(ref: torch.Tensor, got: torch.Tensor) -> torch.Tensor:
    """|got - ref| in units of |ref| * 2^-7 (one bf16 ulp up to a factor 2)."""
    ref, got = ref.float(), got.float()
    ulp = torch.where(ref == 0, torch.full_like(ref, 2.0**-133), ref.abs() * 2.0**-7)
    return (ref - got).abs() / ulp


def _make_case(m: int, seed: int, no_lora_rows: int = 0):
    """Decode-sized MoE-LoRA inputs; the first ``no_lora_rows`` tokens have no adapter."""
    g = torch.Generator(device="cuda").manual_seed(seed)

    def rnd(*shape, scale=1.0):
        return (torch.randn(*shape, generator=g, device="cuda") * scale).bfloat16()

    topk_ids = torch.stack(
        [
            torch.randperm(E, generator=torch.Generator().manual_seed(seed * 1000 + t))[
                :TOP_K
            ]
            for t in range(m)
        ]
    ).to("cuda", torch.int32)
    topk_weights = torch.rand(m, TOP_K, generator=g, device="cuda")
    topk_weights = topk_weights / topk_weights.sum(1, keepdim=True)
    i32 = dict(device="cuda", dtype=torch.int32)
    token_lora_mapping = torch.zeros(m, **i32)
    token_lora_mapping[:no_lora_rows] = -1
    if no_lora_rows:
        seg_indptr = torch.tensor([0, no_lora_rows, m], **i32)
        req_to_lora = torch.tensor([-1, 0], **i32)
    else:
        seg_indptr = torch.tensor([0, m], **i32)
        req_to_lora = torch.zeros(1, **i32)
    info = LoRAInfo(
        gate_up_lora_a_weights=rnd(1, 1, 2 * R, H, scale=0.02),
        gate_up_lora_b_weights=rnd(1, E, 2 * N, R, scale=0.05),
        down_lora_a_weights=rnd(1, E, R, N, scale=0.05),
        down_lora_b_weights=rnd(1, 1, H, R, scale=0.02),
        seg_indptr=seg_indptr,
        req_to_lora=req_to_lora,
        lora_ranks=torch.tensor([R], **i32),
        adapter_enabled=torch.ones(1, **i32),
        token_lora_mapping=token_lora_mapping,
        max_lora_rank=R,
        num_experts=E,
        experts_shared_outer_loras=True,
    )
    return dict(
        info=info,
        x=rnd(m, H),
        topk_ids=topk_ids,
        topk_weights=topk_weights,
        cache1=rnd(m, TOP_K, 2 * N_PAD),
        inter=rnd(m * TOP_K, N_PAD),
        cache3=rnd(m, TOP_K, H),
    )


def _run_hooks(case):
    """(gate_up output, down output) of the hooks applied to copies of the caches."""
    cache1, cache3 = case["cache1"].clone(), case["cache3"].clone()
    hooks = build_lora_hooks(case["x"], case["info"], case["topk_ids"])
    hooks.after_gate_up(case["x"], cache1, case["topk_weights"], case["topk_ids"])
    hooks.after_down(case["inter"], cache3, case["topk_weights"], case["topk_ids"])
    return cache1, cache3


def _run(monkeypatch, case, *, shared_outer: bool):
    monkeypatch.setattr(lora_moe_runners, "_SHARED_OUTER_DECODE", shared_outer)
    return _run_hooks(case)


def _reference(case):
    """fp64 products rounded where the per-expert path rounds: bf16 shrink,
    bf16 delta (after the routing weight for down), bf16 add to the base."""
    info, ids = case["info"], case["topk_ids"].long()
    mapping = info.token_lora_mapping.tolist()
    a_gu = info.gate_up_lora_a_weights[0, 0].double()
    b_gu = info.gate_up_lora_b_weights[0].double()
    a_dn = info.down_lora_a_weights[0].double()
    b_dn = info.down_lora_b_weights[0, 0].double()
    cache1, cache3 = case["cache1"].clone(), case["cache3"].clone()
    shrink = (case["x"].double() @ a_gu.t()).bfloat16().double()
    inter = case["inter"].double()
    weights = case["topk_weights"].double()
    for t in range(ids.shape[0]):
        if mapping[t] < 0:
            continue
        for k in range(TOP_K):
            e, pair = ids[t, k], t * TOP_K + k
            cache1[t, k, :N] += (b_gu[e, :N] @ shrink[t, :R]).bfloat16()
            cache1[t, k, N_PAD : N_PAD + N] += (b_gu[e, N:] @ shrink[t, R:]).bfloat16()
            t_down = (a_dn[e] @ inter[pair, :N]).bfloat16().double()
            cache3[t, k] += (weights[t, k] * (b_dn @ t_down)).bfloat16()
    return cache1, cache3


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("m", [1, 2, 4, 8, 16, 32])
@pytest.mark.parametrize("half_without_lora", [False, True])
def test_matches_reference(monkeypatch, m: int, half_without_lora: bool):
    no_lora_rows = m // 2 if half_without_lora else 0
    case = _make_case(m, seed=m, no_lora_rows=no_lora_rows)
    ref = _reference(case)
    got = _run(monkeypatch, case, shared_outer=True)
    for name, r, o in zip(("gate_up", "down"), ref, got):
        ulps = _bf16_ulps(r, o)
        print(
            f"m={m} no_lora_rows={no_lora_rows} {name}: max {ulps.max().item():.2f} "
            f"ulp, bit-equal {(r == o).float().mean().item():.5f}"
        )
        assert ulps.max().item() <= 2.0, name
    # tokens without an adapter keep the base output bit for bit
    assert torch.equal(got[0][:no_lora_rows], case["cache1"][:no_lora_rows])
    assert torch.equal(got[1][:no_lora_rows], case["cache3"][:no_lora_rows])
    # the shrink partials are summed in a fixed order: run-to-run identical
    again = _run(monkeypatch, case, shared_outer=True)
    assert torch.equal(got[0], again[0]) and torch.equal(got[1], again[1])


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_path_selection(monkeypatch):
    """Batches of up to MAX_PAIRS (token, expert) pairs take the shared-outer
    path; larger ones and the disabled flag keep the per-expert path."""
    case = _make_case(8, seed=7)
    info, topk_ids = case["info"], case["topk_ids"]
    monkeypatch.setattr(lora_moe_runners, "_SHARED_OUTER_DECODE", True)
    monkeypatch.setattr(lora_moe_runners, "_SHARED_OUTER_DECODE_MAX_PAIRS", 8 * TOP_K)
    assert lora_moe_runners._use_shared_outer_decode_path(info, topk_ids)
    monkeypatch.setattr(
        lora_moe_runners, "_SHARED_OUTER_DECODE_MAX_PAIRS", 8 * TOP_K - 1
    )
    assert not lora_moe_runners._use_shared_outer_decode_path(info, topk_ids)
    monkeypatch.setattr(lora_moe_runners, "_SHARED_OUTER_DECODE", False)
    monkeypatch.setattr(lora_moe_runners, "_SHARED_OUTER_DECODE_MAX_PAIRS", 8 * TOP_K)
    assert not lora_moe_runners._use_shared_outer_decode_path(info, topk_ids)


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v", "-s"]))
