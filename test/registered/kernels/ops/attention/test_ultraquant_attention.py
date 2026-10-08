"""UltraQuant 4-bit KV cache: decode and extend attention correctness.

Both kernels are compared against exact fp32 attention over the *dequantized*
cache with the E4M3 rotated query they consume, rather than the original bf16
K/V and query, which separates kernel correctness from quantization error.
"""

import itertools
import sys

import pytest
import torch

from sglang.kernels.ops.attention.ultraquant_decode_attention import (
    decode_attention_fwd_ultraquant,
)
from sglang.kernels.ops.attention.ultraquant_extend_attention import (
    extend_attention_fwd_ultraquant,
)
from sglang.kernels.ops.kvcache.ultraquant import (
    ultraquant_gather_dequant,
    ultraquant_rotate,
    ultraquant_store,
)
from sglang.srt.layers.quantization.ultraquant_tensor import (
    UltraQuantKVQuantizeUtil,
    code_bytes,
    hadamard_matrix,
    n_groups,
)
from sglang.srt.utils import is_hip
from sglang.test.ci.ci_register import register_amd_ci

register_amd_ci(est_time=20, stage="stage-b", runner_config="1-gpu-small-amd-mi35x")

pytestmark = pytest.mark.skipif(
    not (torch.cuda.is_available() and is_hip()),
    reason="UltraQuant KV cache currently targets ROCm",
)

# bf16 round-to-nearest carries ~2e-3 relative error, and the kernels emit
# bf16, so this bounds representation error rather than kernel error.
TOLERANCE = 6e-3


def _build_cache(num_slots, kv_head_num, head_dim, seed):
    torch.manual_seed(seed)
    key = torch.randn(
        num_slots, kv_head_num, head_dim, device="cuda", dtype=torch.bfloat16
    )
    value = torch.randn(
        num_slots, kv_head_num, head_dim, device="cuda", dtype=torch.bfloat16
    )

    k_codes = torch.zeros(
        (num_slots, kv_head_num, code_bytes(head_dim)), dtype=torch.uint8, device="cuda"
    )
    v_codes = torch.zeros_like(k_codes)
    k_scales = torch.zeros(
        (num_slots, kv_head_num, n_groups(head_dim)), dtype=torch.uint8, device="cuda"
    )
    v_scales = torch.zeros_like(k_scales)

    slots = torch.arange(num_slots, device="cuda", dtype=torch.int64)
    ultraquant_store(key, value, k_codes, k_scales, v_codes, v_scales, slots)
    return k_codes, k_scales, v_codes, v_scales


def _rotate(raw_q, head_dim):
    rotation = hadamard_matrix(head_dim, raw_q.device)
    return (raw_q.float() @ rotation.T).to(torch.bfloat16)


def _e4m3_query(raw_q):
    """The rotated query as the Triton kernels consume it.

    Rounding to E4M3 alone moves the output by several 1e-3, so the reference
    starts from the same rounded query.
    """
    out = torch.empty(raw_q.shape, dtype=torch.float8_e4m3fn, device=raw_q.device)
    return ultraquant_rotate(raw_q, out).float()


def _dequantized(codes, scales, indices):
    return UltraQuantKVQuantizeUtil.batched_dequantize(
        codes[indices], scales[indices], dtype=torch.float32
    )


def _decode_reference(
    q, k_codes, k_scales, v_codes, v_scales, kv_indptr, kv_indices, sm_scale
):
    """Exact fp32 attention of each sequence's query over its dequantized KV."""
    batch, q_head_num = q.shape[:2]
    group = q_head_num // k_codes.shape[1]
    reference = torch.empty(q.shape, dtype=torch.float32, device=q.device)
    for b in range(batch):
        indices = kv_indices[kv_indptr[b] : kv_indptr[b + 1]].long()
        k_deq = _dequantized(k_codes, k_scales, indices)
        v_deq = _dequantized(v_codes, v_scales, indices)
        for h in range(q_head_num):
            scores = (q[b, h].float() @ k_deq[:, h // group].T) * sm_scale
            reference[b, h] = torch.softmax(scores, dim=-1) @ v_deq[:, h // group]
    return reference


@pytest.mark.parametrize("head_dim", (64, 128, 256))
@pytest.mark.parametrize("q_head_num,kv_head_num", ((8, 8), (32, 8), (48, 8), (6, 1)))
# Fewer splits than the grid launches is what production runs: the tail
# programs must leave their partials untouched and stage 2 must ignore them.
@pytest.mark.parametrize("splits", (1, 3, 8))
def test_decode_matches_dequantized_reference(
    head_dim, q_head_num, kv_head_num, splits
):
    seq_lens = [17, 128, 301]
    batch, max_kv_splits = len(seq_lens), 8
    num_slots = sum(seq_lens) + 64
    sm_scale = head_dim**-0.5

    k_codes, k_scales, v_codes, v_scales = _build_cache(
        num_slots, kv_head_num, head_dim, seed=head_dim + q_head_num
    )

    # Shuffled indices catch address math that only works for ordered slots.
    perm = torch.randperm(num_slots, device="cuda")
    kv_indptr = torch.zeros(batch + 1, dtype=torch.int32, device="cuda")
    kv_indptr[1:] = torch.tensor(seq_lens, device="cuda", dtype=torch.int32).cumsum(0)
    kv_indices = torch.cat([perm[:length] for length in seq_lens]).to(torch.int32)

    raw_q = torch.randn(
        batch, q_head_num, head_dim, device="cuda", dtype=torch.bfloat16
    )
    q = _e4m3_query(raw_q)
    o = torch.empty_like(raw_q)
    attn_logits = torch.empty(
        batch, q_head_num, max_kv_splits, head_dim, dtype=torch.float32, device="cuda"
    )
    attn_lse = torch.empty(
        batch, q_head_num, max_kv_splits, dtype=torch.float32, device="cuda"
    )
    num_kv_splits = torch.full((batch,), splits, dtype=torch.int32, device="cuda")

    decode_attention_fwd_ultraquant(
        raw_q,
        k_codes,
        k_scales,
        v_codes,
        v_scales,
        o,
        kv_indptr,
        kv_indices,
        attn_logits,
        attn_lse,
        num_kv_splits,
        max_kv_splits,
        sm_scale,
    )

    reference = _decode_reference(
        q, k_codes, k_scales, v_codes, v_scales, kv_indptr, kv_indices, sm_scale
    )
    error = (o.float() - reference).norm() / reference.norm()
    assert error < TOLERANCE, f"relative error {error:.2e}"


def test_decode_single_token_is_exact():
    """One key means softmax is 1.0, so the output is that value verbatim.

    E2M1 levels carry at most one mantissa bit and scales are powers of two,
    so the dequantized value is exact in bf16 and the kernel must reproduce it
    bit for bit.
    """
    head_dim, q_head_num, kv_head_num = 256, 8, 8
    k_codes, k_scales, v_codes, v_scales = _build_cache(
        8, kv_head_num, head_dim, seed=3
    )

    kv_indptr = torch.tensor([0, 1], dtype=torch.int32, device="cuda")
    kv_indices = torch.tensor([5], dtype=torch.int32, device="cuda")
    q = torch.randn(1, q_head_num, head_dim, device="cuda", dtype=torch.bfloat16)
    o = torch.empty_like(q)
    attn_logits = torch.empty(
        1, q_head_num, 4, head_dim, dtype=torch.float32, device="cuda"
    )
    attn_lse = torch.empty(1, q_head_num, 4, dtype=torch.float32, device="cuda")
    num_kv_splits = torch.full((1,), 4, dtype=torch.int32, device="cuda")

    decode_attention_fwd_ultraquant(
        q,
        k_codes,
        k_scales,
        v_codes,
        v_scales,
        o,
        kv_indptr,
        kv_indices,
        attn_logits,
        attn_lse,
        num_kv_splits,
        4,
        head_dim**-0.5,
    )

    expected = _dequantized(v_codes, v_scales, torch.tensor([5], device="cuda"))
    torch.testing.assert_close(o.float(), expected.expand_as(o.float()), rtol=0, atol=0)


@pytest.mark.parametrize("head_dim", (64, 128, 256))
@pytest.mark.parametrize("q_head_num,kv_head_num", ((8, 8), (32, 8), (48, 8), (16, 1)))
@pytest.mark.parametrize(
    "prefix_lens,extend_lens", (([0, 0], [16, 33]), ([37, 128], [16, 64]))
)
def test_extend_matches_dequantized_reference(
    head_dim, q_head_num, kv_head_num, prefix_lens, extend_lens
):
    batch = len(prefix_lens)
    seq_lens = [p + e for p, e in zip(prefix_lens, extend_lens)]
    num_slots = sum(seq_lens) + 32
    sm_scale = head_dim**-0.5

    k_codes, k_scales, v_codes, v_scales = _build_cache(
        num_slots, kv_head_num, head_dim, seed=head_dim + sum(extend_lens)
    )

    perm = torch.randperm(num_slots, device="cuda")
    kv_indptr = torch.zeros(batch + 1, dtype=torch.int32, device="cuda")
    kv_indptr[1:] = torch.tensor(seq_lens, device="cuda", dtype=torch.int32).cumsum(0)
    qo_indptr = torch.zeros(batch + 1, dtype=torch.int32, device="cuda")
    qo_indptr[1:] = torch.tensor(extend_lens, device="cuda", dtype=torch.int32).cumsum(
        0
    )
    prefix = torch.tensor(prefix_lens, dtype=torch.int32, device="cuda")
    kv_indices = perm[: sum(seq_lens)].to(torch.int32)

    raw_q = torch.randn(
        sum(extend_lens), q_head_num, head_dim, device="cuda", dtype=torch.bfloat16
    )
    q = _e4m3_query(raw_q)
    o = torch.zeros_like(raw_q)

    extend_attention_fwd_ultraquant(
        raw_q,
        o,
        k_codes,
        k_scales,
        v_codes,
        v_scales,
        qo_indptr,
        kv_indptr,
        kv_indices,
        prefix,
        max(extend_lens),
        sm_scale,
    )

    group = q_head_num // kv_head_num
    reference = torch.zeros_like(o, dtype=torch.float32)
    for b in range(batch):
        indices = kv_indices[int(kv_indptr[b]) : int(kv_indptr[b + 1])].long()
        k_deq = _dequantized(k_codes, k_scales, indices)
        v_deq = _dequantized(v_codes, v_scales, indices)
        q_start, prefix_len = int(qo_indptr[b]), prefix_lens[b]
        for i in range(extend_lens[b]):
            limit = prefix_len + i + 1  # causal within the extend region
            for h in range(q_head_num):
                scores = (q[q_start + i, h] @ k_deq[:limit, h // group].T) * sm_scale
                reference[q_start + i, h] = (
                    torch.softmax(scores, dim=-1) @ v_deq[:limit, h // group]
                )

    error = (o.float() - reference).norm() / reference.norm()
    assert error < TOLERANCE, f"relative error {error:.2e}"


@pytest.mark.parametrize(
    "seq_lens,max_kv_splits,splits",
    (
        ((300,), 64, 16),
        ((4097,), 16, 16),
        ((16385,), 64, 64),
        ((4097,), 64, 32),
        ((300, 4097), 64, 24),
        ((1, 300, 4097, 16385, 2000), 64, 64),
        ((70000, 300), 512, 512),
        # More sequences than lanes, so each lane scans several of them.
        (tuple(range(1, 7 * 130, 7)), 16, 16),
    ),
)
@pytest.mark.parametrize("block_kv", (128, 256))
@pytest.mark.parametrize("work_budget", (0, 9, 64, 2048))
def test_flydsl_decode_matches_dequantized_reference(
    seq_lens, max_kv_splits, splits, block_kv, work_budget
):
    """FlyDSL decode plus its reducer, run the way the backend does.

    With no work budget the kernel deals ``block_kv``-token blocks to its
    splits round robin, so a length just past a split boundary leaves data in
    splits the contiguous-split mask would skip. Launching fewer splits than
    the buffer holds is the per-batch split count; a non-power-of-two count
    exercises the reducer's padding. A work budget instead cuts the batch into
    chunks, whose rows the reducer must derive the same way.
    """
    from sglang.kernels.ops.attention.flydsl.ultraquant_decode import (
        flydsl_ultraquant_decode,
        is_flydsl_ultraquant_decode_supported,
        ultraquant_decode_reduce,
    )

    head_dim, q_head_num, kv_head_num = 256, 12, 2
    if not is_flydsl_ultraquant_decode_supported(
        head_dim, q_head_num // kv_head_num, torch.bfloat16
    ):
        pytest.skip("FlyDSL UltraQuant decode targets gfx950")
    sm_scale = head_dim**-0.5
    bs, total = len(seq_lens), sum(seq_lens)
    if work_budget and bs >= work_budget:
        pytest.skip("a work budget must exceed the batch size")
    num_slots = total + 64
    k_codes, k_scales, v_codes, v_scales = _build_cache(
        num_slots, kv_head_num, head_dim, seed=total
    )
    kv_indices = torch.randperm(num_slots, device="cuda")[:total]
    kv_indptr = torch.tensor(
        [0, *itertools.accumulate(seq_lens)], dtype=torch.int32, device="cuda"
    )

    raw_q = torch.randn(bs, q_head_num, head_dim, device="cuda", dtype=torch.bfloat16)
    o = torch.empty_like(raw_q)
    # Chunks fill the buffers' storage as flat rows, past the batch rows.
    width = max(max_kv_splits, -(-(bs + work_budget) // bs))
    # NaN-filled so reading a split this launch did not write shows up.
    attn_logits = torch.full(
        (bs, q_head_num, width, head_dim), float("nan"), device="cuda"
    )
    attn_lse = torch.full((bs, q_head_num, width), float("nan"), device="cuda")

    flydsl_ultraquant_decode(
        raw_q,
        k_codes,
        k_scales,
        v_codes,
        v_scales,
        attn_logits,
        attn_lse,
        kv_indptr,
        kv_indices,
        sm_scale,
        num_splits=None if work_budget else splits,
        block_kv=block_kv,
        work_budget=work_budget,
    )
    if work_budget:
        chunk = max(-(-total // ((work_budget - bs) * block_kv)) * block_kv, block_kv)
        expected = set()
        for s, (start, n) in enumerate(zip(kv_indptr.tolist(), seq_lens)):
            first = start // chunk + s
            expected.update(range(first, first + max(-(-n // chunk), 1)))
        flat_lse = attn_lse.view(-1, q_head_num)
        written = set((~flat_lse.isnan()).all(dim=1).nonzero().flatten().tolist())
        assert written == expected, "chunk rows written differ from the layout"
    else:
        written = (~attn_lse.isnan()).all(dim=1).sum(dim=1).tolist()
        for n, w in zip(seq_lens, written):
            assert w == splits, f"seq_len {n}: {w} splits written, expected {splits}"
    ultraquant_decode_reduce(
        attn_logits,
        attn_lse,
        kv_indptr,
        o,
        None if work_budget else splits,
        block_kv,
        work_budget,
    )

    reference = _decode_reference(
        _rotate(raw_q, head_dim),
        k_codes,
        k_scales,
        v_codes,
        v_scales,
        kv_indptr,
        kv_indices,
        sm_scale,
    )

    # The scaled QK MFMA consumes Q as E4M3, hence the looser bound. Dropping
    # one split's block lands near 0.3.
    error = (o.float() - reference).norm() / reference.norm()
    assert error < 5e-2, f"relative error {error:.2e}"


def test_flydsl_decode_launch_config():
    """Chunks for batches the lane search covers, strided splits past that."""
    from sglang.kernels.ops.attention.flydsl.ultraquant_decode import (
        ultraquant_decode_launch_config as config,
    )

    # 8 workgroups per CU on 256 CUs; the last argument is the buffer rows.
    assert config(1, 1, 1, 256, 256, 8192) == (None, 128, 2048)
    assert config(8, 2, 1, 256, 256, 8192) == (None, 128, 1024)
    assert config(512, 1, 1, 256, 256, 1 << 17) == (None, 128, 2048)
    # Too many sequences to search, too few buffer rows, or a budget the batch
    # fills on its own.
    assert config(513, 1, 1, 256, 256, 1 << 17) == (4, 256, 0)
    assert config(8, 1, 1, 256, 256, 2055) == (256, 256, 0)
    assert config(256, 8, 1, 256, 256, 1 << 16) == (1, 256, 0)
    # Deterministic inference pins the split count, and with it the block size.
    assert config(1, 1, 256, 256, 256, 8192) == (256, 256, 0)
    assert config(64, 1, 256, 256, 256, 1 << 14) == (256, 256, 0)


def test_decode_rejects_packed_width_attn_logits():
    """A split workspace sized to the packed buffer width must fail loudly.

    Stage 1 writes head_dim floats per split and derives the LSE index by
    dividing that flat offset by head_dim. Sizing the workspace to
    ``head_dim // 2`` (the packed FP4 row) instead of the logical head dim
    silently corrupts neighbouring splits.
    """
    head_dim, q_head_num, kv_head_num, max_kv_splits = 256, 8, 8, 4
    k_codes, k_scales, v_codes, v_scales = _build_cache(
        16, kv_head_num, head_dim, seed=11
    )
    q = torch.randn(1, q_head_num, head_dim, device="cuda", dtype=torch.bfloat16)
    o = torch.empty_like(q)
    kv_indptr = torch.tensor([0, 8], dtype=torch.int32, device="cuda")
    kv_indices = torch.arange(8, dtype=torch.int32, device="cuda")
    attn_lse = torch.empty(
        1, q_head_num, max_kv_splits, dtype=torch.float32, device="cuda"
    )
    num_kv_splits = torch.full((1,), max_kv_splits, dtype=torch.int32, device="cuda")
    packed_width_logits = torch.empty(
        1,
        q_head_num,
        max_kv_splits,
        code_bytes(head_dim),
        dtype=torch.float32,
        device="cuda",
    )

    with pytest.raises(ValueError, match="must equal the logical head_dim"):
        decode_attention_fwd_ultraquant(
            q,
            k_codes,
            k_scales,
            v_codes,
            v_scales,
            o,
            kv_indptr,
            kv_indices,
            packed_width_logits,
            attn_lse,
            num_kv_splits,
            max_kv_splits,
            head_dim**-0.5,
        )


def test_kernels_reject_mismatched_buffers():
    head_dim, kv_head_num = 256, 8
    k_codes, k_scales, v_codes, v_scales = _build_cache(
        16, kv_head_num, head_dim, seed=1
    )
    q = torch.randn(1, 8, head_dim, device="cuda", dtype=torch.bfloat16)
    o = torch.empty_like(q)
    kv_indptr = torch.tensor([0, 4], dtype=torch.int32, device="cuda")
    kv_indices = torch.arange(4, dtype=torch.int32, device="cuda")
    attn_logits = torch.empty(1, 8, 2, head_dim, dtype=torch.float32, device="cuda")
    attn_lse = torch.empty(1, 8, 2, dtype=torch.float32, device="cuda")
    num_kv_splits = torch.full((1,), 2, dtype=torch.int32, device="cuda")

    # Passing the scale buffer where the code buffer belongs must fail loudly
    # rather than silently reading the wrong bytes.
    with pytest.raises(ValueError, match="k_code_buffer must be"):
        decode_attention_fwd_ultraquant(
            q,
            k_scales,
            k_scales,
            v_codes,
            v_scales,
            o,
            kv_indptr,
            kv_indices,
            attn_logits,
            attn_lse,
            num_kv_splits,
            2,
            1.0,
        )


@pytest.mark.parametrize(
    "prefix_lens,extend_lens", (([0, 0], [64, 96]), ([512, 33], [64, 128]))
)
def test_gather_dequant_feeds_correct_prefill_scores(prefix_lens, extend_lens):
    """Dequantizing the run then attending densely must match the extend kernel.

    This is the prefill path's premise: keys stay in the rotated basis, so
    rotated queries score against them unchanged, and a continuation chunk
    must still see its whole prefix.
    """
    head_dim, q_head_num, kv_head_num = 256, 8, 2
    total_kv = sum(p + e for p, e in zip(prefix_lens, extend_lens))
    k_codes, k_scales, v_codes, v_scales = _build_cache(
        total_kv, kv_head_num, head_dim, seed=7
    )

    kv_indptr = torch.tensor(
        [0]
        + list(
            torch.cumsum(
                torch.tensor([p + e for p, e in zip(prefix_lens, extend_lens)]), 0
            )
        ),
        dtype=torch.int32,
        device="cuda",
    )
    qo_indptr = torch.tensor(
        [0] + list(torch.cumsum(torch.tensor(extend_lens), 0)),
        dtype=torch.int32,
        device="cuda",
    )
    kv_indices = torch.arange(total_kv, dtype=torch.int64, device="cuda")
    prefix = torch.tensor(prefix_lens, dtype=torch.int32, device="cuda")

    torch.manual_seed(7)
    raw_q = torch.randn(
        sum(extend_lens), q_head_num, head_dim, device="cuda", dtype=torch.bfloat16
    )
    q = _e4m3_query(raw_q)
    sm_scale = head_dim**-0.5

    o_kernel = torch.empty_like(raw_q)
    extend_attention_fwd_ultraquant(
        raw_q,
        o_kernel,
        k_codes,
        k_scales,
        v_codes,
        v_scales,
        qo_indptr,
        kv_indptr,
        kv_indices,
        prefix,
        max(extend_lens),
        sm_scale,
    )

    k_deq = torch.empty(
        total_kv, kv_head_num, head_dim, device="cuda", dtype=torch.bfloat16
    )
    v_deq = torch.empty_like(k_deq)
    ultraquant_gather_dequant(
        k_codes, k_scales, v_codes, v_scales, kv_indices, k_deq, v_deq
    )

    # Reference attention over the dequantized run, with the lower-right
    # causal alignment a continuation chunk requires.
    reps = q_head_num // kv_head_num
    for i, (plen, elen) in enumerate(zip(prefix_lens, extend_lens)):
        ks, ke = int(kv_indptr[i]), int(kv_indptr[i + 1])
        qs, qe = int(qo_indptr[i]), int(qo_indptr[i + 1])
        kk = k_deq[ks:ke].float().repeat_interleave(reps, dim=1)
        vv = v_deq[ks:ke].float().repeat_interleave(reps, dim=1)
        scores = torch.einsum("qhd,khd->hqk", q[qs:qe], kk) * sm_scale
        pos_q = torch.arange(elen, device="cuda") + plen
        pos_k = torch.arange(ke - ks, device="cuda")
        scores = scores.masked_fill(
            pos_k[None, None, :] > pos_q[None, :, None], -torch.inf
        )
        ref = torch.einsum("hqk,khd->qhd", scores.softmax(-1), vv)
        torch.testing.assert_close(
            o_kernel[qs:qe].float(), ref, atol=TOLERANCE, rtol=TOLERANCE
        )


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
