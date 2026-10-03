"""UltraQuant 4-bit KV cache: decode and extend attention correctness.

Both kernels are compared against exact fp32 attention over the *dequantized*
cache rather than the original bf16 K/V, which separates kernel correctness
from FP4 quantization error.
"""

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


def _dequantized(codes, scales, indices):
    return UltraQuantKVQuantizeUtil.batched_dequantize(
        codes[indices], scales[indices], dtype=torch.float32
    )


@pytest.mark.parametrize("head_dim", (64, 128, 256))
@pytest.mark.parametrize("q_head_num,kv_head_num", ((8, 8), (48, 8), (6, 1)))
def test_decode_matches_dequantized_reference(head_dim, q_head_num, kv_head_num):
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

    q = _rotate(
        torch.randn(batch, q_head_num, head_dim, device="cuda", dtype=torch.bfloat16),
        head_dim,
    )
    o = torch.empty_like(q)
    attn_logits = torch.empty(
        batch, q_head_num, max_kv_splits, head_dim, dtype=torch.float32, device="cuda"
    )
    attn_lse = torch.empty(
        batch, q_head_num, max_kv_splits, dtype=torch.float32, device="cuda"
    )
    num_kv_splits = torch.full(
        (batch,), max_kv_splits, dtype=torch.int32, device="cuda"
    )

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
        max_kv_splits,
        sm_scale,
    )

    group = q_head_num // kv_head_num
    reference = torch.empty_like(o, dtype=torch.float32)
    for b in range(batch):
        indices = kv_indices[int(kv_indptr[b]) : int(kv_indptr[b + 1])].long()
        k_deq = _dequantized(k_codes, k_scales, indices)
        v_deq = _dequantized(v_codes, v_scales, indices)
        for h in range(q_head_num):
            scores = (q[b, h].float() @ k_deq[:, h // group].T) * sm_scale
            reference[b, h] = torch.softmax(scores, dim=-1) @ v_deq[:, h // group]

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
    q = _rotate(
        torch.randn(1, q_head_num, head_dim, device="cuda", dtype=torch.bfloat16),
        head_dim,
    )
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
@pytest.mark.parametrize(
    "prefix_lens,extend_lens", (([0, 0], [16, 33]), ([37, 128], [16, 64]))
)
def test_extend_matches_dequantized_reference(head_dim, prefix_lens, extend_lens):
    q_head_num, kv_head_num = 48, 8
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

    q = _rotate(
        torch.randn(
            sum(extend_lens), q_head_num, head_dim, device="cuda", dtype=torch.bfloat16
        ),
        head_dim,
    )
    o = torch.zeros_like(q)

    extend_attention_fwd_ultraquant(
        q,
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
        window_start_pos=torch.zeros(batch, dtype=torch.int32, device="cuda"),
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
                scores = (
                    q[q_start + i, h].float() @ k_deq[:limit, h // group].T
                ) * sm_scale
                reference[q_start + i, h] = (
                    torch.softmax(scores, dim=-1) @ v_deq[:limit, h // group]
                )

    error = (o.float() - reference).norm() / reference.norm()
    assert error < TOLERANCE, f"relative error {error:.2e}"


@pytest.mark.parametrize("splits", (1, 2, 3, 7))
def test_decode_partial_kv_splits(splits):
    """num_kv_splits below max_kv_splits is the shape production actually uses.

    SGLang picks a per-request split count and launches the grid over the max,
    so the tail programs must leave their partials untouched and stage 2 must
    ignore them.
    """
    head_dim, q_head_num, kv_head_num, max_kv_splits = 256, 12, 2, 8
    seq_len, sm_scale = 300, 256**-0.5
    num_slots = seq_len + 16

    k_codes, k_scales, v_codes, v_scales = _build_cache(
        num_slots, kv_head_num, head_dim, seed=splits
    )
    perm = torch.randperm(num_slots, device="cuda")
    kv_indices = perm[:seq_len].to(torch.int32)
    kv_indptr = torch.tensor([0, seq_len], dtype=torch.int32, device="cuda")

    q = _rotate(
        torch.randn(1, q_head_num, head_dim, device="cuda", dtype=torch.bfloat16),
        head_dim,
    )
    o = torch.empty_like(q)
    attn_logits = torch.empty(
        1, q_head_num, max_kv_splits, head_dim, dtype=torch.float32, device="cuda"
    )
    attn_lse = torch.empty(
        1, q_head_num, max_kv_splits, dtype=torch.float32, device="cuda"
    )
    num_kv_splits = torch.full((1,), splits, dtype=torch.int32, device="cuda")

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
        max_kv_splits,
        sm_scale,
    )

    group = q_head_num // kv_head_num
    indices = kv_indices.long()
    k_deq = _dequantized(k_codes, k_scales, indices)
    v_deq = _dequantized(v_codes, v_scales, indices)
    reference = torch.empty_like(o, dtype=torch.float32)
    for h in range(q_head_num):
        scores = (q[0, h].float() @ k_deq[:, h // group].T) * sm_scale
        reference[0, h] = torch.softmax(scores, dim=-1) @ v_deq[:, h // group]

    error = (o.float() - reference).norm() / reference.norm()
    assert error < TOLERANCE, f"relative error {error:.2e} at splits={splits}"


@pytest.mark.parametrize(
    "seq_len,max_kv_splits,splits",
    ((300, 64, 16), (4097, 16, 16), (16385, 64, 64), (4097, 64, 32)),
)
def test_flydsl_decode_matches_dequantized_reference(seq_len, max_kv_splits, splits):
    """FlyDSL stage 1 plus the shared reducer, run the way the backend does.

    The kernel deals 256-token blocks to its splits round robin, so a length
    just past a split boundary leaves data in splits the contiguous-split mask
    would skip. Launching fewer splits than the buffer holds is the per-batch
    split count.
    """
    from sglang.kernels.ops.attention.decode_attention import (
        _decode_softmax_reducev_fwd,
    )
    from sglang.kernels.ops.attention.flydsl.ultraquant_decode import (
        ULTRAQUANT_DECODE_BLOCK_KV,
        flydsl_ultraquant_decode,
        is_flydsl_ultraquant_decode_supported,
    )

    head_dim, q_head_num, kv_head_num = 256, 12, 2
    if not is_flydsl_ultraquant_decode_supported(
        head_dim, q_head_num // kv_head_num, torch.bfloat16
    ):
        pytest.skip("FlyDSL UltraQuant decode targets gfx950")
    sm_scale = head_dim**-0.5
    num_slots = seq_len + 64
    k_codes, k_scales, v_codes, v_scales = _build_cache(
        num_slots, kv_head_num, head_dim, seed=seq_len
    )
    kv_indices = torch.randperm(num_slots, device="cuda")[:seq_len]
    kv_indptr = torch.tensor([0, seq_len], dtype=torch.int32, device="cuda")

    raw_q = torch.randn(1, q_head_num, head_dim, device="cuda", dtype=torch.bfloat16)
    o = torch.empty_like(raw_q)
    # NaN-filled so reading a split this launch did not write shows up.
    attn_logits = torch.full(
        (1, q_head_num, max_kv_splits, head_dim), float("nan"), device="cuda"
    )
    attn_lse = torch.full((1, q_head_num, max_kv_splits), float("nan"), device="cuda")
    num_kv_splits = torch.full((1,), splits, dtype=torch.int32, device="cuda")

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
        num_splits=splits,
    )
    _decode_softmax_reducev_fwd(
        attn_logits,
        attn_lse,
        raw_q,
        o,
        1.0,
        v_codes,
        kv_indptr,
        num_kv_splits,
        max_kv_splits,
        v_head_dim=head_dim,
        strided_block_kv=ULTRAQUANT_DECODE_BLOCK_KV,
    )

    q = _rotate(raw_q, head_dim)
    group = q_head_num // kv_head_num
    k_deq = _dequantized(k_codes, k_scales, kv_indices)
    v_deq = _dequantized(v_codes, v_scales, kv_indices)
    reference = torch.empty_like(o, dtype=torch.float32)
    for h in range(q_head_num):
        scores = (q[0, h].float() @ k_deq[:, h // group].T) * sm_scale
        reference[0, h] = torch.softmax(scores, dim=-1) @ v_deq[:, h // group]

    # The scaled QK MFMA consumes Q as E4M3, hence the looser bound. Dropping
    # one split's block lands near 0.3.
    error = (o.float() - reference).norm() / reference.norm()
    assert error < 5e-2, f"relative error {error:.2e}"


def test_decode_rejects_packed_width_attn_logits():
    """A split workspace sized to the packed buffer width must fail loudly.

    Stage 1 writes head_dim floats per split and derives the LSE index by
    dividing that flat offset by head_dim. Sizing the workspace to
    ``head_dim // 2`` (the packed FP4 row) instead of the logical head dim
    silently corrupts neighbouring splits -- it cost an 80-point GSM8K
    regression before this guard existed.
    """
    head_dim, q_head_num, kv_head_num, max_kv_splits = 256, 8, 8, 4
    k_codes, k_scales, v_codes, v_scales = _build_cache(
        16, kv_head_num, head_dim, seed=11
    )
    q = _rotate(
        torch.randn(1, q_head_num, head_dim, device="cuda", dtype=torch.bfloat16),
        head_dim,
    )
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
    q = _rotate(
        torch.randn(sum(extend_lens), q_head_num, head_dim, device="cuda"), head_dim
    )
    sm_scale = head_dim**-0.5

    o_kernel = torch.empty_like(q)
    extend_attention_fwd_ultraquant(
        q,
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

    k_deq = torch.empty(total_kv, kv_head_num, head_dim, device="cuda", dtype=q.dtype)
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
        scores = torch.einsum("qhd,khd->hqk", q[qs:qe].float(), kk) * sm_scale
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
