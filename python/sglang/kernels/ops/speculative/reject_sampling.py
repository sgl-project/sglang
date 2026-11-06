import threading

import torch
import triton
import triton.language as tl

# The final-token pass of ``speculative_sampling_classic_kernel`` scans the whole
# vocabulary twice inside one program per request, so a launch costs about as much at
# one request as at many. ``_split_*`` below run that same arithmetic on a
# (requests x vocabulary tiles) grid instead. The launcher picks it only when the
# one-program-per-request grid cannot fill the device; see
# ``_should_split_final_sampling``.
_SPLIT_SCRATCH_LOCK = threading.Lock()
_SPLIT_SCRATCH: dict = {}


@triton.jit
def speculative_sampling_classic_kernel(
    # Pointers
    Predicts,
    AcceptIndex,
    AcceptTokenNum,
    Candidates,
    RetriveIndex,
    UniformSamples,
    UniformSamplesFinal,
    TargetProbs,
    DraftProbs,
    # Strides
    stride_cand_b,
    stride_cand_s,
    stride_idx_b,
    stride_idx_s,
    stride_uni_b,
    stride_uni_s,
    stride_tp_b,
    stride_tp_s,
    stride_tp_v,
    stride_dp_b,
    stride_dp_s,
    stride_dp_v,
    # Constants
    NUM_SLOTS: tl.constexpr,
    VOCAB_SIZE: tl.constexpr,
    BLOCK_V: tl.constexpr,
    BLOCK_VERIFICATION: tl.constexpr = False,
    BLOCK_REDUCE: tl.constexpr = 4096,
):
    pid = tl.program_id(0)
    cur_prob_row = 0

    cand_ptr_base = Candidates + pid * stride_cand_b
    idx_ptr_base = RetriveIndex + pid * stride_idx_b
    uni_ptr_base = UniformSamples + pid * stride_uni_b

    root_global_idx = tl.load(idx_ptr_base + 0 * stride_idx_s)
    tl.store(AcceptIndex + pid * stride_idx_b + 0 * stride_idx_s, root_global_idx)
    last_accepted_global_idx = root_global_idx

    num_accept = 0
    residual_scale = 1.0
    norm_sum = -1.0

    if BLOCK_VERIFICATION:
        # Algorithm 2: https://arxiv.org/abs/2403.10444
        # Keep the short recurrence in registers; only the longest accepted prefix matters.
        prefix_prob = 1.0
        prefix_probs = ()
        for step in tl.static_range(1, NUM_SLOTS):
            draft_token = tl.load(cand_ptr_base + step * stride_cand_s)
            p = tl.load(
                TargetProbs
                + pid * stride_tp_b
                + (step - 1) * stride_tp_s
                + draft_token * stride_tp_v
            )
            q = tl.load(
                DraftProbs
                + pid * stride_dp_b
                + (step - 1) * stride_dp_s
                + draft_token * stride_dp_v
            )
            q_is_prob = (q > 0.0) & (q <= 1.0)
            prefix_prob = tl.where(q_is_prob, tl.minimum(prefix_prob * p / q, 1.0), 0.0)
            prefix_probs += (prefix_prob,)

        for step in tl.static_range(NUM_SLOTS - 1, 0, -1):
            if num_accept == 0:
                prefix_prob = prefix_probs[step - 1]
                coin = tl.load(uni_ptr_base + (step - 1) * stride_uni_s)
                # For normalized distributions h_i <= r_i; reject before scanning.
                if coin < prefix_prob:
                    accept_prob = prefix_prob
                    residual_mass = -1.0
                    if (
                        (step < NUM_SLOTS - 1)
                        & (prefix_prob > 0.0)
                        & (prefix_prob < 1.0)
                    ):
                        residual_mass = 0.0
                        for v_start in range(0, VOCAB_SIZE, BLOCK_REDUCE):
                            v_offsets = v_start + tl.arange(0, BLOCK_REDUCE)
                            mask = v_offsets < VOCAB_SIZE
                            p_next = tl.load(
                                TargetProbs
                                + pid * stride_tp_b
                                + step * stride_tp_s
                                + v_offsets * stride_tp_v,
                                mask=mask,
                                other=0.0,
                            )
                            q_next = tl.load(
                                DraftProbs
                                + pid * stride_dp_b
                                + step * stride_dp_s
                                + v_offsets * stride_dp_v,
                                mask=mask,
                                other=0.0,
                            )
                            residual_mass += tl.sum(
                                tl.maximum(prefix_prob * p_next - q_next, 0.0)
                            )
                        accept_prob = residual_mass / (
                            residual_mass + 1.0 - prefix_prob
                        )
                    if coin < accept_prob:
                        num_accept = step
                        residual_scale = prefix_prob
                        norm_sum = residual_mass

        cur_prob_row = num_accept
        for step in range(1, num_accept + 1):
            draft_token = tl.load(cand_ptr_base + step * stride_cand_s)
            tl.store(Predicts + last_accepted_global_idx, draft_token)
            last_accepted_global_idx = tl.load(idx_ptr_base + step * stride_idx_s)
            tl.store(
                AcceptIndex + pid * stride_idx_b + step * stride_idx_s,
                last_accepted_global_idx,
            )
    else:
        step = 1
        continue_verifying = 1
        # At each token-verification loop entry, cur_prob_row == num_accept == step - 1.
        while (step < NUM_SLOTS) and (continue_verifying == 1):
            draft_token = tl.load(cand_ptr_base + step * stride_cand_s)
            p = tl.load(
                TargetProbs
                + pid * stride_tp_b
                + cur_prob_row * stride_tp_s
                + draft_token * stride_tp_v
            )
            q = tl.load(
                DraftProbs
                + pid * stride_dp_b
                + cur_prob_row * stride_dp_s
                + draft_token * stride_dp_v
            )
            coin = tl.load(uni_ptr_base + (step - 1) * stride_uni_s)
            # A proposal token must have positive probability under its draft distribution.
            q_is_prob = (q > 0.0) & (q <= 1.0)
            if q_is_prob & (coin * q < p):
                num_accept += 1
                cur_prob_row = step
                tl.store(Predicts + last_accepted_global_idx, draft_token)
                last_accepted_global_idx = tl.load(idx_ptr_base + step * stride_idx_s)
                tl.store(
                    AcceptIndex + pid * stride_idx_b + num_accept * stride_idx_s,
                    last_accepted_global_idx,
                )
                step += 1
            else:
                continue_verifying = 0

    tl.store(AcceptTokenNum + pid, num_accept)

    # Final Sampling
    all_drafts_accepted = num_accept == NUM_SLOTS - 1
    coin_final = tl.load(UniformSamplesFinal + pid)

    tp_base_ptr = TargetProbs + (pid * stride_tp_b) + (cur_prob_row * stride_tp_s)
    # DraftProbs has only num_steps rows (TargetProbs has num_steps + 1). When
    # all drafts are accepted cur_prob_row == num_steps is out of bounds for
    # DraftProbs, but the all-accepted branch samples pure target p and never
    # dereferences this pointer; on rejection cur_prob_row <= num_steps - 1.
    dp_base_ptr_safe = DraftProbs + (pid * stride_dp_b) + (cur_prob_row * stride_dp_s)

    # Reuse the selected prefix mass as the correction normalizer.
    if norm_sum < 0.0:
        norm_sum = 0.0
        # Pass 1: Sum
        for v_start in range(0, VOCAB_SIZE, BLOCK_V):
            v_offsets = v_start + tl.arange(0, BLOCK_V)
            mask = v_offsets < VOCAB_SIZE

            p_ptr = tp_base_ptr + v_offsets * stride_tp_v
            p_val = tl.load(p_ptr, mask=mask, other=0.0)

            if all_drafts_accepted:
                val = p_val
            else:
                q_ptr = dp_base_ptr_safe + v_offsets * stride_dp_v
                q_val = tl.load(q_ptr, mask=mask, other=0.0)
                # Treat any non-probability q (NaN, +-inf, negative) as 0: the
                # residual falls back to p. A comparison against NaN is false, so
                # the range test rejects it along with the infinities.
                q_val = tl.where((q_val >= 0.0) & (q_val <= 1.0), q_val, 0.0)
                diff = residual_scale * p_val - q_val
                val = tl.where(diff > 0.0, diff, 0.0)

            norm_sum += tl.sum(val)

    # Pass 2: CDF. Degenerate residual (norm_sum == 0, i.e. p == q everywhere on
    # rejection) leaves the cumsum at 0 <= target_u, so final_token falls back to
    # VOCAB_SIZE - 1; acceptable since this case is numerically near-impossible.
    target_u = coin_final * norm_sum
    cum_sum = 0.0
    final_token = VOCAB_SIZE - 1
    found = 0

    for v_start in range(0, VOCAB_SIZE, BLOCK_V):
        if found == 0:
            v_offsets = v_start + tl.arange(0, BLOCK_V)
            mask = v_offsets < VOCAB_SIZE

            p_ptr = tp_base_ptr + v_offsets * stride_tp_v
            p_val = tl.load(p_ptr, mask=mask, other=0.0)

            if all_drafts_accepted:
                val = p_val
            else:
                q_ptr = dp_base_ptr_safe + v_offsets * stride_dp_v
                q_val = tl.load(q_ptr, mask=mask, other=0.0)
                # Same guard as pass 1.
                q_val = tl.where((q_val >= 0.0) & (q_val <= 1.0), q_val, 0.0)
                diff = residual_scale * p_val - q_val
                val = tl.where(diff > 0.0, diff, 0.0)

            block_cumsum = tl.cumsum(val, axis=0)
            total_cumsum = cum_sum + block_cumsum

            candidates_mask = total_cumsum > target_u
            has_match = tl.max(candidates_mask, axis=0)

            if has_match:
                match_idx = tl.argmax(candidates_mask.to(tl.int32), axis=0)
                final_token = v_start + match_idx
                found = 1

            cum_sum += tl.sum(val)

    tl.store(Predicts + last_accepted_global_idx, final_token)


@triton.jit
def _split_chain_walk(
    Predicts,
    AcceptIndex,
    Candidates,
    RetriveIndex,
    UniformSamples,
    TargetProbs,
    DraftProbs,
    pid,
    do_store,
    stride_cand_b,
    stride_cand_s,
    stride_idx_b,
    stride_idx_s,
    stride_uni_b,
    stride_uni_s,
    stride_tp_b,
    stride_tp_s,
    stride_tp_v,
    stride_dp_b,
    stride_dp_s,
    stride_dp_v,
    NUM_SLOTS: tl.constexpr,
):
    """The chain walk of ``speculative_sampling_classic_kernel`` (non-block-verification
    branch), with the stores made only by the program that owns them."""
    cur_prob_row = 0
    cand_ptr_base = Candidates + pid * stride_cand_b
    idx_ptr_base = RetriveIndex + pid * stride_idx_b
    uni_ptr_base = UniformSamples + pid * stride_uni_b

    root_global_idx = tl.load(idx_ptr_base + 0 * stride_idx_s)
    if do_store:
        tl.store(AcceptIndex + pid * stride_idx_b + 0 * stride_idx_s, root_global_idx)
    last_accepted_global_idx = root_global_idx

    num_accept = 0
    step = 1
    continue_verifying = 1
    while (step < NUM_SLOTS) and (continue_verifying == 1):
        draft_token = tl.load(cand_ptr_base + step * stride_cand_s)
        p = tl.load(
            TargetProbs
            + pid * stride_tp_b
            + cur_prob_row * stride_tp_s
            + draft_token * stride_tp_v
        )
        q = tl.load(
            DraftProbs
            + pid * stride_dp_b
            + cur_prob_row * stride_dp_s
            + draft_token * stride_dp_v
        )
        coin = tl.load(uni_ptr_base + (step - 1) * stride_uni_s)
        q_is_prob = (q > 0.0) & (q <= 1.0)
        if q_is_prob & (coin * q < p):
            num_accept += 1
            cur_prob_row = step
            if do_store:
                tl.store(Predicts + last_accepted_global_idx, draft_token)
            last_accepted_global_idx = tl.load(idx_ptr_base + step * stride_idx_s)
            if do_store:
                tl.store(
                    AcceptIndex + pid * stride_idx_b + num_accept * stride_idx_s,
                    last_accepted_global_idx,
                )
            step += 1
        else:
            continue_verifying = 0
    return num_accept, cur_prob_row, last_accepted_global_idx


@triton.jit
def _split_residual_tile(
    TargetProbs,
    DraftProbs,
    pid,
    cur_prob_row,
    all_drafts_accepted,
    v_start,
    stride_tp_b,
    stride_tp_s,
    stride_tp_v,
    stride_dp_b,
    stride_dp_s,
    stride_dp_v,
    VOCAB_SIZE: tl.constexpr,
    BLOCK_V: tl.constexpr,
):
    """One tile of the residual distribution, identical to the tile body of both
    full-vocabulary passes above, including the non-probability guard on q."""
    residual_scale = 1.0
    tp_base_ptr = TargetProbs + (pid * stride_tp_b) + (cur_prob_row * stride_tp_s)
    dp_base_ptr_safe = DraftProbs + (pid * stride_dp_b) + (cur_prob_row * stride_dp_s)
    v_offsets = v_start + tl.arange(0, BLOCK_V)
    mask = v_offsets < VOCAB_SIZE
    p_val = tl.load(tp_base_ptr + v_offsets * stride_tp_v, mask=mask, other=0.0)
    if all_drafts_accepted:
        val = p_val
    else:
        q_val = tl.load(
            dp_base_ptr_safe + v_offsets * stride_dp_v, mask=mask, other=0.0
        )
        q_val = tl.where((q_val >= 0.0) & (q_val <= 1.0), q_val, 0.0)
        diff = residual_scale * p_val - q_val
        val = tl.where(diff > 0.0, diff, 0.0)
    return val


@triton.jit
def _split_sum_kernel(
    Predicts,
    AcceptIndex,
    AcceptTokenNum,
    Candidates,
    RetriveIndex,
    UniformSamples,
    TargetProbs,
    DraftProbs,
    TileSums,
    Result,
    stride_cand_b,
    stride_cand_s,
    stride_idx_b,
    stride_idx_s,
    stride_uni_b,
    stride_uni_s,
    stride_tp_b,
    stride_tp_s,
    stride_tp_v,
    stride_dp_b,
    stride_dp_s,
    stride_dp_v,
    NUM_SLOTS: tl.constexpr,
    VOCAB_SIZE: tl.constexpr,
    BLOCK_V: tl.constexpr,
    NUM_TILES: tl.constexpr,
):
    """Pass 1, split: every program repeats the chain walk and reduces one tile.
    Tile 0 additionally performs the stores the stock kernel makes along the walk."""
    pid = tl.program_id(0)
    tile = tl.program_id(1)
    num_accept, cur_prob_row, _ = _split_chain_walk(
        Predicts,
        AcceptIndex,
        Candidates,
        RetriveIndex,
        UniformSamples,
        TargetProbs,
        DraftProbs,
        pid,
        tile == 0,
        stride_cand_b,
        stride_cand_s,
        stride_idx_b,
        stride_idx_s,
        stride_uni_b,
        stride_uni_s,
        stride_tp_b,
        stride_tp_s,
        stride_tp_v,
        stride_dp_b,
        stride_dp_s,
        stride_dp_v,
        NUM_SLOTS,
    )
    if tile == 0:
        tl.store(AcceptTokenNum + pid, num_accept)
        # The stock fallback when no tile matches (degenerate residual).
        tl.store(Result + pid, VOCAB_SIZE - 1)
    all_drafts_accepted = num_accept == NUM_SLOTS - 1
    val = _split_residual_tile(
        TargetProbs,
        DraftProbs,
        pid,
        cur_prob_row,
        all_drafts_accepted,
        tile * BLOCK_V,
        stride_tp_b,
        stride_tp_s,
        stride_tp_v,
        stride_dp_b,
        stride_dp_s,
        stride_dp_v,
        VOCAB_SIZE,
        BLOCK_V,
    )
    tl.store(TileSums + pid * NUM_TILES + tile, tl.sum(val))


@triton.jit
def _split_cdf_kernel(
    Predicts,
    Candidates,
    RetriveIndex,
    UniformSamples,
    UniformSamplesFinal,
    TargetProbs,
    DraftProbs,
    TileSums,
    Result,
    Arrived,
    stride_cand_b,
    stride_cand_s,
    stride_idx_b,
    stride_idx_s,
    stride_uni_b,
    stride_uni_s,
    stride_tp_b,
    stride_tp_s,
    stride_tp_v,
    stride_dp_b,
    stride_dp_s,
    stride_dp_v,
    NUM_SLOTS: tl.constexpr,
    VOCAB_SIZE: tl.constexpr,
    BLOCK_V: tl.constexpr,
    NUM_TILES: tl.constexpr,
):
    """Pass 2, split. Each program adds the tile sums in tile order, which gives both
    the normalizer and the running sum in front of its own tile, then applies the stock
    tile step. The stock loop takes the first tile with a match and its first matching
    index, which is the smallest matching index of the row, so keeping the minimum over
    all tiles with an atomic reproduces the stock token."""
    pid = tl.program_id(0)
    tile = tl.program_id(1)
    num_accept, cur_prob_row, last_idx = _split_chain_walk(
        Predicts,
        Predicts,  # AcceptIndex is unused here: do_store is False
        Candidates,
        RetriveIndex,
        UniformSamples,
        TargetProbs,
        DraftProbs,
        pid,
        False,
        stride_cand_b,
        stride_cand_s,
        stride_idx_b,
        stride_idx_s,
        stride_uni_b,
        stride_uni_s,
        stride_tp_b,
        stride_tp_s,
        stride_tp_v,
        stride_dp_b,
        stride_dp_s,
        stride_dp_v,
        NUM_SLOTS,
    )
    all_drafts_accepted = num_accept == NUM_SLOTS - 1

    # Accumulated in tile order from 0.0, as the stock pass-1 loop accumulates them.
    norm_sum = 0.0
    cum_sum = 0.0
    for t in tl.static_range(NUM_TILES):
        if t == tile:
            cum_sum = norm_sum
        norm_sum += tl.load(TileSums + pid * NUM_TILES + t)

    target_u = tl.load(UniformSamplesFinal + pid) * norm_sum
    v_start = tile * BLOCK_V
    val = _split_residual_tile(
        TargetProbs,
        DraftProbs,
        pid,
        cur_prob_row,
        all_drafts_accepted,
        v_start,
        stride_tp_b,
        stride_tp_s,
        stride_tp_v,
        stride_dp_b,
        stride_dp_s,
        stride_dp_v,
        VOCAB_SIZE,
        BLOCK_V,
    )
    total_cumsum = cum_sum + tl.cumsum(val, axis=0)
    candidates_mask = total_cumsum > target_u
    if tl.max(candidates_mask, axis=0):
        match_idx = tl.argmax(candidates_mask.to(tl.int32), axis=0)
        tl.atomic_min(Result + pid, v_start + match_idx, sem="relaxed", scope="gpu")

    # Arrival counter (acq_rel RMW, gpu scope): each program's atomic minimum above is
    # ordered before its arrival, and the last arrival acquires every earlier one
    # through the RMW chain on the counter, so its read below sees all of them. The
    # counter is reset here, so the buffer is reusable in stream order and the launch
    # is safe to capture in a CUDA graph.
    tl.debug_barrier()
    arrived = tl.atomic_add(Arrived + pid, 1, sem="acq_rel", scope="gpu")
    if arrived == NUM_TILES - 1:
        final_token = tl.atomic_add(Result + pid, 0, sem="relaxed", scope="gpu")
        tl.store(Predicts + last_idx, final_token)
        tl.atomic_xchg(Arrived + pid, 0, sem="relaxed", scope="gpu")


def _split_scratch(device, batch_size, num_tiles):
    """Tile sums, sampled token and arrival counters, one set per (device, stream, tile
    count). Returns ``None`` while capturing if the cached set is too small, because
    allocating during capture would bake a pointer that the next launch will not use."""
    stream = torch._C._cuda_getCurrentRawStream(device.index)
    key = (device.index, stream, num_tiles)
    s = _SPLIT_SCRATCH.get(key)
    if s is not None and s[0] >= batch_size:
        return s[1], s[2], s[3]
    if torch.cuda.is_current_stream_capturing():
        return None
    with _SPLIT_SCRATCH_LOCK:
        s = _SPLIT_SCRATCH.get(key)
        if s is None or s[0] < batch_size:
            cap = max(batch_size, 0 if s is None else 2 * s[0])
            s = (
                cap,
                torch.empty((cap, num_tiles), dtype=torch.float32, device=device),
                torch.empty((cap,), dtype=torch.int32, device=device),
                # Left at zero; _split_cdf_kernel resets each counter it completes.
                torch.zeros((cap,), dtype=torch.int32, device=device),
            )
            _SPLIT_SCRATCH[key] = s
    return s[1], s[2], s[3]


def _should_split_final_sampling(batch_size, num_tiles, device):
    """Split only when the stock grid leaves the device idle. One program per request
    means ``batch_size`` programs; below one per multiprocessor the final-token pass is
    latency-bound on a few programs each scanning the whole vocabulary, and splitting it
    into ``batch_size * num_tiles`` programs is strictly more parallelism for the same
    arithmetic. At or above that point the stock grid already fills the device and the
    split only adds a second launch and the tile-sum round trip."""
    if num_tiles < 2:
        return False
    return batch_size < torch.cuda.get_device_properties(device).multi_processor_count


def _chain_speculative_sampling_split(
    predicts,
    accept_index,
    accept_token_num,
    candidates,
    retrive_index,
    uniform_samples,
    uniform_samples_for_final_sampling,
    target_probs,
    draft_probs,
    *,
    batch_size,
    num_slots,
    vocab_size,
    num_tiles,
    block_v,
    num_warps,
):
    """Same outputs as ``speculative_sampling_classic_kernel`` on the
    non-block-verification path. Returns False if the split path cannot run now."""
    scratch = _split_scratch(target_probs.device, batch_size, num_tiles)
    if scratch is None:
        return False
    tile_sums, result, arrived = scratch
    strides = (
        candidates.stride(0),
        candidates.stride(1),
        retrive_index.stride(0),
        retrive_index.stride(1),
        uniform_samples.stride(0),
        uniform_samples.stride(1),
        target_probs.stride(0),
        target_probs.stride(1),
        target_probs.stride(2),
        draft_probs.stride(0),
        draft_probs.stride(1),
        draft_probs.stride(2),
    )
    consts = dict(
        NUM_SLOTS=num_slots,
        VOCAB_SIZE=vocab_size,
        BLOCK_V=block_v,
        NUM_TILES=num_tiles,
    )
    grid = (batch_size, num_tiles)
    _split_sum_kernel[grid](
        predicts,
        accept_index,
        accept_token_num,
        candidates,
        retrive_index,
        uniform_samples,
        target_probs,
        draft_probs,
        tile_sums,
        result,
        *strides,
        **consts,
        num_warps=num_warps,
    )
    _split_cdf_kernel[grid](
        predicts,
        candidates,
        retrive_index,
        uniform_samples,
        uniform_samples_for_final_sampling,
        target_probs,
        draft_probs,
        tile_sums,
        result,
        arrived,
        *strides,
        **consts,
        num_warps=num_warps,
    )
    return True


def chain_speculative_sampling_triton(
    predicts,
    accept_index,
    accept_token_num,
    candidates,
    retrive_index,
    retrive_next_token,
    retrive_next_sibling,  # not used in chain verification
    uniform_samples,
    uniform_samples_for_final_sampling,
    target_probs,
    draft_probs,
    threshold_single,
    threshold_acc,
    deterministic,  # not used
    *,
    block_verification: bool = False,
):
    batch_size, num_slots = candidates.shape
    vocab_size = target_probs.shape[-1]
    # Keep CDF tiles small, and cap multi-tile reductions to avoid register spills.
    block_reduce = min(triton.next_power_of_2(vocab_size), 32768)
    num_warps = 8 if block_verification and block_reduce > 4096 else 4
    block_v = 4096

    if not block_verification:
        num_tiles = triton.cdiv(vocab_size, block_v)
        if _should_split_final_sampling(
            batch_size, num_tiles, target_probs.device
        ) and _chain_speculative_sampling_split(
            predicts,
            accept_index,
            accept_token_num,
            candidates,
            retrive_index,
            uniform_samples,
            uniform_samples_for_final_sampling,
            target_probs,
            draft_probs,
            batch_size=batch_size,
            num_slots=num_slots,
            vocab_size=vocab_size,
            num_tiles=num_tiles,
            block_v=block_v,
            num_warps=num_warps,
        ):
            return

    grid = (batch_size,)
    speculative_sampling_classic_kernel[grid](
        predicts,
        accept_index,
        accept_token_num,
        candidates,
        retrive_index,
        uniform_samples,
        uniform_samples_for_final_sampling,
        target_probs,
        draft_probs,
        candidates.stride(0),
        candidates.stride(1),
        retrive_index.stride(0),
        retrive_index.stride(1),
        uniform_samples.stride(0),
        uniform_samples.stride(1),
        target_probs.stride(0),
        target_probs.stride(1),
        target_probs.stride(2),
        draft_probs.stride(0),
        draft_probs.stride(1),
        draft_probs.stride(2),
        NUM_SLOTS=num_slots,
        VOCAB_SIZE=vocab_size,
        BLOCK_V=block_v,
        BLOCK_VERIFICATION=block_verification,
        BLOCK_REDUCE=block_reduce,
        num_warps=num_warps,
    )
