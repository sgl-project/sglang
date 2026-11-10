import torch
import triton
import triton.language as tl


@triton.jit
def _selector_walk_kernel(
    scores_ptr,
    candidate_ptr,
    uniforms_ptr,
    temperatures_ptr,
    greedy_ptr,
    tokens_ptr,
    q_ptr,
    path_state_ptr,
    slots: tl.constexpr,
    walk_slots: tl.constexpr,
    top_k: tl.constexpr,
):
    """Walk one DFlash2 candidate lattice per program.

    A slot's candidate scores remain in registers and the dependent path walk is
    performed in one kernel, avoiding a launch and global-memory round trip per
    draft position.
    """
    row = tl.program_id(0)
    offsets = tl.arange(0, top_k)
    temperature = tl.load(temperatures_ptr + row)
    greedy = tl.load(greedy_ptr + row) != 0
    previous = 0
    for slot in range(walk_slots):
        base = (row * slots + slot) * top_k
        scores = tl.load(scores_ptr + (base + previous) * top_k + offsets).to(
            tl.float32
        )
        if greedy:
            best = tl.max(scores, axis=0)
            index = tl.min(tl.where(scores == best, offsets, top_k), axis=0)
            probabilities = tl.where(offsets == index, 1.0, 0.0)
        else:
            scaled = scores / temperature
            exponentials = tl.exp(scaled - tl.max(scaled, axis=0))
            probabilities = exponentials / tl.sum(exponentials, axis=0)
            uniform = tl.load(uniforms_ptr + row * slots + slot)
            index = tl.sum(
                tl.where(uniform >= tl.cumsum(probabilities, axis=0), 1, 0), axis=0
            )
            index = tl.minimum(index, top_k - 1)
        tl.store(q_ptr + base + offsets, probabilities)
        tl.store(tokens_ptr + row * slots + slot, tl.load(candidate_ptr + base + index))
        previous = index
    tl.store(path_state_ptr + row, previous)


@triton.jit
def _selector_walk_tail_kernel(
    scores_ptr,
    candidate_ptr,
    uniforms_ptr,
    temperatures_ptr,
    greedy_ptr,
    tokens_ptr,
    q_ptr,
    path_state_ptr,
    slots: tl.constexpr,
    top_k: tl.constexpr,
):
    """Finish the final dependent slot separately on SM70.

    Triton/PyTorch 2.9 can drop the seventh store of the fully unrolled walk on
    Volta when its inputs come from a dynamic compiled region.  Keeping the
    dependency state in one tiny buffer avoids that compiler fault while the
    expensive six-slot prefix still stays in one kernel.
    """
    row = tl.program_id(0)
    offsets = tl.arange(0, top_k)
    slot: tl.constexpr = slots - 1
    previous = tl.load(path_state_ptr + row)
    base = (row * slots + slot) * top_k
    scores = tl.load(scores_ptr + (base + previous) * top_k + offsets).to(tl.float32)
    greedy = tl.load(greedy_ptr + row) != 0
    if greedy:
        best = tl.max(scores, axis=0)
        index = tl.min(tl.where(scores == best, offsets, top_k), axis=0)
        probabilities = tl.where(offsets == index, 1.0, 0.0)
    else:
        temperature = tl.load(temperatures_ptr + row)
        scaled = scores / temperature
        exponentials = tl.exp(scaled - tl.max(scaled, axis=0))
        probabilities = exponentials / tl.sum(exponentials, axis=0)
        uniform = tl.load(uniforms_ptr + row * slots + slot)
        index = tl.sum(
            tl.where(uniform >= tl.cumsum(probabilities, axis=0), 1, 0), axis=0
        )
        index = tl.minimum(index, top_k - 1)
    tl.store(q_ptr + base + offsets, probabilities)
    tl.store(tokens_ptr + row * slots + slot, tl.load(candidate_ptr + base + index))


def selector_walk_triton(
    *,
    candidate_ids: torch.Tensor,
    scores: torch.Tensor,
    uniforms: torch.Tensor,
    temperatures: torch.Tensor,
    greedy_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    batch, slots, top_k = map(int, candidate_ids.shape)
    tokens = torch.empty((batch, slots), dtype=torch.int64, device=scores.device)
    q_rows = torch.empty(
        (batch, slots, top_k), dtype=torch.float32, device=scores.device
    )
    path_state = torch.empty((batch,), dtype=torch.int32, device=scores.device)
    scores = scores.contiguous()
    candidate_ids = candidate_ids.contiguous()
    uniforms = uniforms.contiguous()
    temperatures = temperatures.contiguous()
    greedy_mask = greedy_mask.contiguous()
    use_sm70_tail = (
        torch.cuda.get_device_capability(scores.device)[0] == 7 and slots > 1
    )
    walk_slots = slots - 1 if use_sm70_tail else slots
    _selector_walk_kernel[(batch,)](
        scores,
        candidate_ids,
        uniforms,
        temperatures,
        greedy_mask,
        tokens,
        q_rows,
        path_state,
        slots=slots,
        walk_slots=walk_slots,
        top_k=top_k,
        num_warps=1,
    )
    if use_sm70_tail:
        _selector_walk_tail_kernel[(batch,)](
            scores,
            candidate_ids,
            uniforms,
            temperatures,
            greedy_mask,
            tokens,
            q_rows,
            path_state,
            slots=slots,
            top_k=top_k,
            num_warps=1,
        )
    return tokens, q_rows
