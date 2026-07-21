"""Recall tests for the LiteTopk fused DSA indexer top-k (SM100).

Validates ``dsa_litetopk_indexer`` (fused fp8 MQA scoring + bucketed gate +
compact exact top-k, no materialized logits) against an independent fp32
torch reference of the DSA indexer score:

    score[r, j] = sum_h weights[r, h] * relu(q[r, h, :] . k[j, :]) * kv_scale[j]
    for j in [ks[r], ke[r])

The fused kernel guarantees the exact top-k SET by construction (conservative
gate + exact select over candidates); tie-breaking at the k-th value and the
within-row output order are unspecified. The checks are therefore
tie-tolerant: every selected index must score >= (k-th value - eps), and every
index scoring > (k-th value + eps) must be selected.

Requires SM100 (Blackwell) + deep_gemm; skipped elsewhere via function-level
skipif (module-level pytest.skip would abort non-zero under ``python3 file.py``,
the CI runner's invocation).
"""

import pytest
import torch

from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=120, stage="base-b-kernel-unit", runner_config="4-gpu-b200")

_LITETOPK_OK = torch.cuda.is_available() and torch.cuda.get_device_capability()[0] == 10
requires_sm100 = pytest.mark.skipif(
    not _LITETOPK_OK, reason="LiteTopk requires CUDA SM100 (Blackwell)."
)

NUM_HEADS = 32
HEAD_DIM = 128


def _ceil_to_ue8m0(x: torch.Tensor) -> torch.Tensor:
    return torch.pow(2.0, torch.ceil(torch.log2(x.abs())))


def _make_case(req_specs, seed=0, score_mode="randn"):
    """Build a multi-request ragged case.

    req_specs: list of (num_rows, kv_len). Rows of request g attend the
    request's gathered KV span with causal per-row ends (last ``num_rows``
    positions of the span are the new tokens).
    Returns inputs for dsa_litetopk_indexer + the fp32 reference pieces.
    """
    torch.manual_seed(seed)
    dev = "cuda"
    total_rows = sum(r for r, _ in req_specs)
    total_kv = sum(k for _, k in req_specs)

    if score_mode == "randn":
        q_bf16 = torch.randn(total_rows, NUM_HEADS, HEAD_DIM, device=dev) * 0.5
        k_bf16 = torch.randn(total_kv, HEAD_DIM, device=dev) * 0.5
    elif score_mode == "flat":
        # Adversarial: identical K rows -> every position scores the same ->
        # massive k-th-value ties; exercises the threshold-bucket boundary and
        # the candidate-buffer pressure path.
        q_bf16 = torch.randn(total_rows, NUM_HEADS, HEAD_DIM, device=dev) * 0.5
        k_bf16 = torch.randn(1, HEAD_DIM, device=dev).expand(total_kv, HEAD_DIM)
        k_bf16 = k_bf16.contiguous() * 0.5
    else:
        raise ValueError(score_mode)

    q_fp8 = q_bf16.to(torch.float8_e4m3fn).contiguous()
    kv_amax = k_bf16.abs().float().amax(dim=-1).clamp(1e-4)
    kv_scale = _ceil_to_ue8m0(kv_amax / 448.0)
    kv_fp8 = (k_bf16 / kv_scale.unsqueeze(-1)).to(torch.float8_e4m3fn).contiguous()
    weights = (torch.rand(total_rows, NUM_HEADS, device=dev) + 0.1).float()

    ks_list, ke_list, req_bounds = [], [], []
    row0 = kv0 = 0
    for rows, kv_len in req_specs:
        assert kv_len >= rows
        prefix = kv_len - rows  # cached tokens before the first new token
        for i in range(rows):
            ks_list.append(kv0)
            ke_list.append(kv0 + prefix + i + 1)
        req_bounds.append((row0, row0 + rows, kv0, kv0 + kv_len))
        row0 += rows
        kv0 += kv_len
    ks = torch.tensor(ks_list, dtype=torch.int32, device=dev)
    ke = torch.tensor(ke_list, dtype=torch.int32, device=dev)

    return dict(
        q_fp8=q_fp8,
        kv_fp8=kv_fp8,
        kv_scale=kv_scale.float(),
        weights=weights,
        ks=ks,
        ke=ke,
        req_bounds=req_bounds,
    )


def _ref_scores(case):
    """fp32 torch reference of the indexer score, full [rows, total_kv]."""
    q = case["q_fp8"].float()  # [R, H, D]
    k = case["kv_fp8"].float() * case["kv_scale"].unsqueeze(-1)  # [S, D]
    qk = torch.einsum("rhd,sd->rhs", q, k).relu()
    scores = torch.einsum("rhs,rh->rs", qk, case["weights"])
    cols = torch.arange(k.shape[0], device=q.device)[None, :]
    valid = (cols >= case["ks"][:, None]) & (cols < case["ke"][:, None])
    return scores.masked_fill(~valid, float("-inf")), valid


def _check_topk_indices(out_idx, scores, valid, topk, eps_rel=1e-3):
    """Tie-tolerant exactness check per row."""
    rows = out_idx.shape[0]
    for r in range(rows):
        n_valid = int(valid[r].sum().item())
        expect = min(topk, n_valid)
        sel = out_idx[r]
        sel_valid = sel[sel >= 0]
        # count: exactly min(topk, n_valid) selected, rest -1 padded
        assert (
            sel_valid.numel() == expect
        ), f"row {r}: selected {sel_valid.numel()}, expected {expect}"
        assert (sel[expect:] == -1).all(), f"row {r}: padding must be -1"
        # no duplicates, all causally valid
        uniq = torch.unique(sel_valid)
        assert uniq.numel() == sel_valid.numel(), f"row {r}: duplicate indices"
        assert valid[r][sel_valid.long()].all(), f"row {r}: out-of-range index"
        if expect == 0:
            continue
        row_scores = scores[r]
        kth = torch.topk(row_scores, expect).values[-1].item()
        eps = eps_rel * max(abs(kth), 1.0)
        # every selected index scores >= kth - eps
        sel_scores = row_scores[sel_valid.long()]
        assert (sel_scores >= kth - eps).all(), (
            f"row {r}: selected score below k-th value: "
            f"min={sel_scores.min().item():.6f} kth={kth:.6f}"
        )
        # every index strictly above kth + eps is selected
        must = torch.nonzero(row_scores > kth + eps).flatten()
        missing = ~torch.isin(must, sel_valid.long())
        assert (
            not missing.any()
        ), f"row {r}: {int(missing.sum())} strictly-above-threshold indices missing"


@requires_sm100
@pytest.mark.parametrize(
    "req_specs,topk",
    [
        ([(32, 32768)], 2048),  # single long request
        ([(16, 8192), (16, 24576), (8, 4096)], 2048),  # multi-request ragged
        ([(32, 32768)], 512),
        ([(8, 1024)], 2048),  # short rows: valid < topk -> -1 padding
        # Alignment regression: kv_len < sample_len and not a multiple of 4
        # gives the calibration sample logits an odd row stride, which the
        # seed kernel's float4 row loads only tolerate via the -inf width
        # padding; 9+5 rows also exercises the ragged final q-block padding.
        ([(9, 4099), (5, 1023)], 512),
    ],
)
def test_litetopk_matches_reference(req_specs, topk):
    from sglang.kernels.ops.attention.dsa.litetopk import dsa_litetopk_indexer

    case = _make_case(req_specs, seed=len(req_specs) * 7 + topk)
    out = dsa_litetopk_indexer(
        case["q_fp8"],
        case["kv_fp8"],
        case["kv_scale"],
        case["weights"],
        case["ks"],
        case["ke"],
        topk,
        req_bounds=case["req_bounds"],
    )
    torch.cuda.synchronize()
    scores, valid = _ref_scores(case)
    _check_topk_indices(out, scores, valid, topk)


@requires_sm100
def test_litetopk_flat_scores_adversarial():
    """All-identical K rows: every position ties at the k-th value. The gate
    threshold bucket then holds ~the entire row -- worst case for candidate
    buffer pressure. The exact-set property degenerates to count + validity."""
    from sglang.kernels.ops.attention.dsa.litetopk import dsa_litetopk_indexer

    topk = 2048
    case = _make_case([(8, 16384)], seed=3, score_mode="flat")
    out = dsa_litetopk_indexer(
        case["q_fp8"],
        case["kv_fp8"],
        case["kv_scale"],
        case["weights"],
        case["ks"],
        case["ke"],
        topk,
        req_bounds=case["req_bounds"],
    )
    torch.cuda.synchronize()
    _, valid = _ref_scores(case)
    for r in range(out.shape[0]):
        expect = min(topk, int(valid[r].sum().item()))
        sel = out[r]
        sel_valid = sel[sel >= 0]
        assert sel_valid.numel() == expect
        assert valid[r][sel_valid.long()].all()
        assert torch.unique(sel_valid).numel() == sel_valid.numel()


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v"]))
