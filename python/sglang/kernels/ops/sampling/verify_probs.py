import torch
import triton
import triton.language as tl


@triton.jit
def _topk_mass(
    logits,
    top,
    temperatures,
    ks,
    probs,
    mass,
    VOCAB: tl.constexpr,
    CAP: tl.constexpr,
    PARTS: tl.constexpr,
    WIDTH: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row, part = tl.program_id(0), tl.program_id(1)
    cols = part * BLOCK + tl.arange(0, BLOCK)
    temperature = tl.load(temperatures + row // WIDTH)
    k = tl.minimum(tl.maximum(tl.load(ks + row // WIDTH), 1), CAP)
    cutoff = tl.load(top + row * CAP + k - 1)
    maximum = tl.load(top + row * CAP) / temperature
    values = tl.load(logits + row * VOCAB + cols, cols < VOCAB, other=-float("inf"))
    weights = tl.where(
        (cols < VOCAB) & (values >= cutoff), tl.exp(values / temperature - maximum), 0.0
    )
    tl.store(probs + row * VOCAB + cols, weights, cols < VOCAB)
    tl.store(mass + row * PARTS + part, tl.sum(weights, 0))


@triton.jit
def _top_p_cutoff(
    top,
    temperatures,
    ks,
    ps,
    mass,
    cutoff_out,
    norm_out,
    CAP: tl.constexpr,
    PARTS: tl.constexpr,
    WIDTH: tl.constexpr,
    TOP_BLOCK: tl.constexpr,
    MASS_BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    idx = tl.arange(0, TOP_BLOCK)
    parts = tl.arange(0, MASS_BLOCK)
    total = tl.sum(tl.load(mass + row * PARTS + parts, parts < PARTS, other=0.0), 0)
    temperature = tl.load(temperatures + row // WIDTH)
    k = tl.minimum(tl.maximum(tl.load(ks + row // WIDTH), 1), CAP)
    kth = tl.load(top + row * CAP + k - 1)
    values = tl.load(top + row * CAP + idx, idx < CAP, other=-float("inf"))
    maximum = tl.load(top + row * CAP) / temperature
    weights = tl.where(
        (idx < CAP) & (values >= kth), tl.exp(values / temperature - maximum), 0.0
    )
    p = tl.load(ps + row // WIDTH)
    first = tl.min(
        tl.where((idx < CAP) & (tl.cumsum(weights) >= p * total), idx, CAP), 0
    )
    # All values above kth fit in CAP; an overflowing tie group is kept in full.
    cutoff = tl.load(top + row * CAP + first, first < CAP, other=-float("inf"))
    cutoff = tl.where(p >= 1.0, kth, tl.maximum(cutoff, kth))
    kept = tl.where(
        cutoff <= kth, total, tl.sum(tl.where(values >= cutoff, weights, 0.0), 0)
    )
    tl.store(cutoff_out + row, cutoff)
    tl.store(norm_out + row, 1.0 / kept)


@triton.jit
def _apply_cutoff(
    logits, probs, cutoff, norm, VOCAB: tl.constexpr, BLOCK: tl.constexpr
):
    row, part = tl.program_id(0), tl.program_id(1)
    cols = part * BLOCK + tl.arange(0, BLOCK)
    values = tl.load(logits + row * VOCAB + cols, cols < VOCAB, other=-float("inf"))
    weights = tl.load(probs + row * VOCAB + cols, cols < VOCAB, other=0.0)
    result = tl.where(
        values >= tl.load(cutoff + row), weights * tl.load(norm + row), 0.0
    )
    tl.store(probs + row * VOCAB + cols, result, cols < VOCAB)


def sparse_target_probs(logits, sampling_info, width):
    assert logits.dtype == torch.float32 and logits.is_contiguous()
    rows, vocab = logits.shape
    cap, block = min(64, vocab), 2048
    parts = triton.cdiv(vocab, block)
    top = torch.topk(logits, cap, dim=-1).values
    probs = torch.empty_like(logits)
    mass = torch.empty((rows, parts), dtype=torch.float32, device=logits.device)
    cutoff = torch.empty(rows, dtype=torch.float32, device=logits.device)
    norm = torch.empty_like(cutoff)
    _topk_mass[(rows, parts)](
        logits,
        top,
        sampling_info.temperatures,
        sampling_info.top_ks,
        probs,
        mass,
        vocab,
        cap,
        parts,
        width,
        block,
    )
    _top_p_cutoff[(rows,)](
        top,
        sampling_info.temperatures,
        sampling_info.top_ks,
        sampling_info.top_ps,
        mass,
        cutoff,
        norm,
        cap,
        parts,
        width,
        triton.next_power_of_2(cap),
        triton.next_power_of_2(parts),
    )
    _apply_cutoff[(rows, parts)](logits, probs, cutoff, norm, vocab, block)
    return probs.view(rows // width, width, vocab)
