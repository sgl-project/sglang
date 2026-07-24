"""CuTe DSL device kernel KDA conv-MTP.

conv enabled, no bias, no output norm, lower_bound gate, Q/K L2 norm,
beta sigmoid, ILP=2, W=4.  TILE_V is 64 with either a serial two-tile loop
(large grids) or grid-z tile parallelism (SPLIT_V, DSpARK N<=16).
"""

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
from cutlass._mlir.dialects import llvm
from cutlass.cute.typing import Int64
from cutlass.cutlass_dsl import T, dsl_user_op

NUM_THREADS = 256
TILE_K = 128


@dsl_user_op
def read_globaltimer(*, loc=None, ip=None) -> Int64:
    """Read the SM global timer for optional in-kernel stage profiling."""
    return Int64(
        llvm.inline_asm(
            T.i64(),
            [],
            "mov.u64 $0, %globaltimer;",
            "=l",
            has_side_effects=True,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


# Arg reference (K = V = head_dim = 128, KERNEL_WIDTH = short_conv_kernel_size
# = 4, lower_bound = gate_lower_bound = -5.0 per kimi-k3-config.json's
# text_config.linear_attn_config; HV = per-TP-rank head count, e.g. 12 for
# TP8; H == HV, one query head per value head, no GQA in KDA; T_total = total
# decode tokens across the batch; num_slots = radix-cache KDA state pool
# size; this kernel is specialized for USE_ZERO_ACCEPTED=True, i.e. every
# spec round has 0 tokens accepted from the prior round):
#   h0            - input recurrent state (read before the update loop).
#                   [num_slots, HV, V, K], or flat [num_slots*HV, V, K] if
#                   USE_FLAT_LAYOUT.
#   x_q, x_k      - new-token Q/K conv input. [1, T_total, H, K] (or
#                   [1, T_total, H*K] flat).
#   x_v           - new-token V conv input. [1, T_total, HV, V] (or flat).
#   w_q, w_k      - depthwise causal-conv weight per Q/K channel.
#                   [H*K, KERNEL_WIDTH].
#   w_v           - same for V. [HV*V, KERNEL_WIDTH].
#   cs_q, cs_k    - persistent conv-tap cache, extended with NUM_SPEC extra
#                   slots to stage speculative-round history (see the
#                   causal-conv1d-update-in-place memory: don't reuse this
#                   buffer across calls without cloning).
#                   [num_slots, H*K, KERNEL_WIDTH - 1 + NUM_SPEC].
#   cs_v          - same for V. [num_slots, HV*V, KERNEL_WIDTH - 1 + NUM_SPEC].
#   A_log         - per-head log decay rate ("A" in KDA). [H].
#   g             - raw forget-gate logits (pre dt_bias, pre decay calc).
#                   [1, T_total, HV, K].
#   dt_bias       - bias added to g before the decay nonlinearity. [H*K].
#   beta          - per-(token, value-head) delta-rule mixing coefficient.
#                   [1, T_total, HV].
#   o             - fused kernel output. [1, T_total, HV, V].
#   ht            - output recurrent state after this call (same layout as
#                   h0).
#   qkg_cache     - per-slot cache of conv-normed q/k and lowered gate g for
#                   not-yet-committed speculative tokens (channel 0=q,
#                   1=k, 2=g), so a later "replay" of an already-processed
#                   token skips recompute. [num_slots, NUM_SPEC, 3, H*K].
#   v_cache       - same idea for the conv-normed V. [num_slots, NUM_SPEC, HV*V].
#   beta_cache    - same idea for sigmoid(beta). [num_slots, NUM_SPEC, HV].
#   smem_qk_layout - static shared-memory tile layout for sQ/sK/sG (not a
#                   tensor; a cute.Layout literal baked in at trace time).
#   ssm_state_indices - per-request slot index into the state/cache tensors
#                   above (radix-cache page mapping). [B].
#   cu_seqlens    - cumulative seqlens per request (bos/eos), used only when
#                   USE_REGULAR_METADATA is False. [B + 1].
#   num_accepted_tokens - tokens accepted from the prior spec round per
#                   request (commit_len), used only when USE_ZERO_ACCEPTED is
#                   False. [B].
#   precompute_control - single runtime flag gating the precompute stage,
#                   used only when RUNTIME_PRECOMPUTE_FLAG is True. [1].
#   TILE_V        - V-dim register tile width for the main update loop.
#                   Benchmarked contract uses 64.
#   scale         - Q scale, 1/sqrt(head_dim) = 1/sqrt(128).
#   HV            - num value heads (== H here). Sweep values 2/12/32 in the
#                   bench correspond to TP-sharded slices of the model's 96.
#   K             - Q/K head dim. head_dim = 128.
#   V             - V head dim. head_dim = 128.
#   NUM_SPEC      - num speculative draft tokens per verify step (bonus token
#                   not included) == --speculative-dspark-block-size; a
#                   spec-decode runtime setting, not a model config value.
#                   Supported: 1..8 (shared-memory bound).
#   KERNEL_WIDTH  - short-conv kernel width. short_conv_kernel_size = 4.
#   lower_bound   - decay gate lower bound. gate_lower_bound = -5.0.
#   USE_FLAT_LAYOUT - selects flattened-last-dim vs nested tensor indexing for
#                   x_q/x_k/x_v/h0/ht; no shape/semantic change.
#   USE_SETMAXREG - toggles explicit warpgroup register alloc/dealloc around
#                   the precompute stage (occupancy tuning only).
#   USE_REGULAR_METADATA - True: derive bos/eos/slot from a fixed per-request
#                   stride (2*NUM_SPEC+1) instead of cu_seqlens/
#                   ssm_state_indices.
#   USE_REG_Q_WEIGHTS - keeps the Q conv weights resident in registers
#                   (r_wq) instead of shared memory (sConvW).
#   USE_ZERO_ACCEPTED - specializes the loop for the benchmarked contract
#                   (commit_len == 0, T_loop == 1 + NUM_SPEC) vs the general
#                   replay-from-cache path.
#   FUSE_PRECOMPUTE - fuses the conv/gate precompute stage into this kernel
#                   launch instead of a separate prior kernel.
#   RUNTIME_PRECOMPUTE_FLAG - reads precompute_control to decide, at runtime,
#                   whether to run the (already fused) precompute stage.
#   stage_timing  - optional per-block stage-duration output (globaltimer
#                   deltas), used only when PROFILE_STAGES is True.
#                   [grid_n * B * 4].
#   PROFILE_STAGES - enables read_globaltimer() instrumentation and writes to
#                   stage_timing.
@cute.kernel
def kda_decode_mtp_kernel(
    h0: cute.Tensor,
    x_q: cute.Tensor,
    x_k: cute.Tensor,
    x_v: cute.Tensor,
    w_q: cute.Tensor,
    w_k: cute.Tensor,
    w_v: cute.Tensor,
    cs_q: cute.Tensor,
    cs_k: cute.Tensor,
    cs_v: cute.Tensor,
    A_log: cute.Tensor,
    g: cute.Tensor,
    dt_bias: cute.Tensor,
    beta: cute.Tensor,
    o: cute.Tensor,
    ht: cute.Tensor,
    intermediate_state_indices: cute.Tensor,
    intermediate_conv_q: cute.Tensor,
    intermediate_conv_k: cute.Tensor,
    intermediate_conv_v: cute.Tensor,
    qkg_cache: cute.Tensor,
    v_cache: cute.Tensor,
    beta_cache: cute.Tensor,
    smem_qk_layout: cute.Layout,
    ssm_state_indices: cute.Tensor,
    cu_seqlens: cute.Tensor,
    num_accepted_tokens: cute.Tensor,
    precompute_control: cute.Tensor,
    TILE_V: cutlass.Constexpr[int],
    scale: cutlass.Constexpr[float],
    HV: cutlass.Constexpr[int],
    K: cutlass.Constexpr[int],
    V: cutlass.Constexpr[int],
    NUM_SPEC: cutlass.Constexpr[int],
    KERNEL_WIDTH: cutlass.Constexpr[int],
    lower_bound: cutlass.Constexpr[float],
    USE_FLAT_LAYOUT: cutlass.Constexpr[bool],
    USE_SETMAXREG: cutlass.Constexpr[bool],
    USE_REGULAR_METADATA: cutlass.Constexpr[bool],
    USE_REG_Q_WEIGHTS: cutlass.Constexpr[bool],
    USE_ZERO_ACCEPTED: cutlass.Constexpr[bool],
    FUSE_PRECOMPUTE: cutlass.Constexpr[bool],
    RUNTIME_PRECOMPUTE_FLAG: cutlass.Constexpr[bool],
    DSPARK_SCRATCH: cutlass.Constexpr[bool],
    SPLIT_V: cutlass.Constexpr[bool],
    ENABLE_PDL: cutlass.Constexpr[bool],
    stage_timing: cute.Tensor,
    PROFILE_STAGES: cutlass.Constexpr[bool],
):
    """KDA MTP decode — SMEM pre-compute + register-resident state.

    SPLIT_V distributes the V // TILE_V state tiles over grid z-blocks (each
    z-block runs the sequential token chain once, duplicating the smem
    precompute) instead of looping tiles serially within one block.

    ENABLE_PDL (launch with use_pdl=True): griddepcontrol-wait before the
    first global read, launch-dependents once every predecessor-written
    input (x_*/g/beta/metadata) has been consumed — the remaining h0 reads
    are ordered by our own wait, and dependents' waits order our writes.
    """
    tidx, _, _ = cute.arch.thread_idx()
    in_warp_tid = tidx % 32
    warp_idx = cute.arch.warp_idx()
    warp_idx = cute.arch.make_warp_uniform(warp_idx)
    i_hv, i_n, i_z = cute.arch.block_idx()
    if cutlass.const_expr(ENABLE_PDL):
        cute.arch.griddepcontrol_wait()
    i_h = i_hv
    if cutlass.const_expr(PROFILE_STAGES):
        t_stage0 = read_globaltimer()
    if cutlass.const_expr(USE_REGULAR_METADATA):
        bos = i_n * (2 * NUM_SPEC + 1)
        eos = bos + (2 * NUM_SPEC + 1)
        slot = i_n
    else:
        bos = cu_seqlens[i_n]
        eos = cu_seqlens[i_n + 1]
        slot = ssm_state_indices[i_n]
    scratch_row = intermediate_state_indices[i_n]
    h0_idx = slot * HV + i_hv
    hk_off = i_h * K
    hv_off = i_hv * V
    if cutlass.const_expr(USE_ZERO_ACCEPTED):
        commit_len = 0
    else:
        commit_len = num_accepted_tokens[i_n]
    if cutlass.const_expr(USE_ZERO_ACCEPTED):
        T_loop = 1 + NUM_SPEC
        t_max = 1 + NUM_SPEC
    else:
        T_loop = commit_len + 1 + NUM_SPEC
        t_max = 2 * NUM_SPEC + 1
    vec_size = TILE_K // 32
    num_v_tiles = V // TILE_V
    NUM_V_ROWS = TILE_V // (NUM_THREADS // 32)
    if cutlass.const_expr(USE_REG_Q_WEIGHTS):
        q_weight_elems = 0
    else:
        q_weight_elems = KERNEL_WIDTH * K
    v_weight_elems = KERNEL_WIDTH * V
    k_weight_base = q_weight_elems
    v_weight_base = q_weight_elems + KERNEL_WIDTH * K
    conv_weight_elems = q_weight_elems + KERNEL_WIDTH * K + v_weight_elems
    smem = cutlass.utils.SmemAllocator()
    sQ = smem.allocate_tensor(cutlass.Float32, smem_qk_layout, 16)
    sK = smem.allocate_tensor(cutlass.Float32, smem_qk_layout, 16)
    sG = smem.allocate_tensor(cutlass.Float32, smem_qk_layout, 16)
    sBeta = smem.allocate_tensor(cutlass.Float32, cute.make_layout((t_max,)), 16)
    # Preserve the original shared-memory offsets after sBeta.  The removed
    # output-norm path used these 8 floats; shifting later buffers changed
    # bank mapping in earlier experiments.
    sWarpSum = smem.allocate_tensor(cutlass.Float32, cute.make_layout((8,)), 16)
    sVall = smem.allocate_tensor(cutlass.Float32, cute.make_layout((t_max * V,)), 16)
    sConvW = smem.allocate_tensor(
        cutlass.Float32,
        cute.make_layout((conv_weight_elems,)),
        16,
    )
    r_q = cute.make_rmem_tensor(
        cute.make_layout((vec_size,), stride=(1,)), cutlass.Float32
    )
    r_k = cute.make_rmem_tensor(
        cute.make_layout((vec_size,), stride=(1,)), cutlass.Float32
    )
    r_decay = cute.make_rmem_tensor(
        cute.make_layout((vec_size,), stride=(1,)), cutlass.Float32
    )
    r_bk = cute.make_rmem_tensor(
        cute.make_layout((vec_size,), stride=(1,)), cutlass.Float32
    )
    r_state = cute.make_rmem_tensor(
        cute.make_layout((NUM_V_ROWS * vec_size,), stride=(1,)), cutlass.Float32
    )
    if cutlass.const_expr(USE_REG_Q_WEIGHTS):
        r_wq = cute.make_rmem_tensor(
            cute.make_layout((KERNEL_WIDTH * vec_size,), stride=(1,)), cutlass.Float32
        )
    r_exp_A = cutlass.Float32(0.0)
    if cutlass.const_expr(USE_REGULAR_METADATA) or eos > bos:
        if cutlass.const_expr(FUSE_PRECOMPUTE or RUNTIME_PRECOMPUTE_FLAG):
            if cutlass.const_expr(RUNTIME_PRECOMPUTE_FLAG):
                run_precompute = precompute_control[0] != 0
            else:
                run_precompute = True
            if run_precompute:
                for i in range(vec_size):
                    k_idx = i * 32 + in_warp_tid
                    for w in range(KERNEL_WIDTH - 1):
                        r_state[w * vec_size + i] = cutlass.Float32(
                            cs_q[slot, hk_off + k_idx, w]
                        )
                        r_state[(KERNEL_WIDTH - 1) * vec_size + w * vec_size + i] = (
                            cutlass.Float32(cs_k[slot, hk_off + k_idx, w])
                        )
                for w in range(KERNEL_WIDTH):
                    if tidx < K:
                        if cutlass.const_expr(not USE_REG_Q_WEIGHTS):
                            sConvW[w * K + tidx] = cutlass.Float32(
                                w_q[hk_off + tidx, w]
                            )
                        sConvW[k_weight_base + w * K + tidx] = cutlass.Float32(
                            w_k[hk_off + tidx, w]
                        )
                for ld in range(V * KERNEL_WIDTH // NUM_THREADS):
                    flat = ld * NUM_THREADS + tidx
                    sConvW[v_weight_base + flat] = cutlass.Float32(
                        w_v[hv_off + flat % V, flat // V]
                    )
                if cutlass.const_expr(USE_REG_Q_WEIGHTS):
                    if warp_idx == 0:
                        for _w in range(KERNEL_WIDTH):
                            for _i in range(vec_size):
                                r_wq[_w * vec_size + _i] = cutlass.Float32(
                                    w_q[hk_off + _i * 32 + in_warp_tid, _w]
                                )
                cute.arch.barrier()
                if cutlass.const_expr(USE_SETMAXREG):
                    cute.arch.warpgroup_reg_dealloc(64)
                if warp_idx < 3:
                    if warp_idx == 2:
                        r_exp_A = cute.math.exp(
                            cutlass.Float32(A_log[i_h]), fastmath=True
                        )
                    i_t = 0
                    while i_t < T_loop:
                        if cutlass.const_expr(USE_ZERO_ACCEPTED):
                            replay_from_cache = False
                        else:
                            replay_from_cache = i_t < commit_len
                        if replay_from_cache:
                            if warp_idx == 0:
                                for i in range(vec_size):
                                    k_idx = i * 32 + in_warp_tid
                                    sQ[i_t, k_idx] = cutlass.Float32(
                                        qkg_cache[slot, i_t, 0, hk_off + k_idx]
                                    )
                                for i in range(vec_size):
                                    k_idx = i * 32 + in_warp_tid
                                    r_xq_raw = cutlass.Float32(
                                        cs_q[
                                            slot, hk_off + k_idx, KERNEL_WIDTH - 1 + i_t
                                        ]
                                    )
                                    for w in range(KERNEL_WIDTH - 2):
                                        r_state[w * vec_size + i] = r_state[
                                            (w + 1) * vec_size + i
                                        ]
                                    r_state[(KERNEL_WIDTH - 2) * vec_size + i] = (
                                        r_xq_raw
                                    )
                            elif warp_idx == 1:
                                for i in range(vec_size):
                                    k_idx = i * 32 + in_warp_tid
                                    sK[i_t, k_idx] = cutlass.Float32(
                                        qkg_cache[slot, i_t, 1, hk_off + k_idx]
                                    )
                                if in_warp_tid == 0:
                                    sBeta[i_t] = cutlass.Float32(
                                        beta_cache[slot, i_t, i_hv]
                                    )
                                for i in range(vec_size):
                                    k_idx = i * 32 + in_warp_tid
                                    r_xk_raw = cutlass.Float32(
                                        cs_k[
                                            slot, hk_off + k_idx, KERNEL_WIDTH - 1 + i_t
                                        ]
                                    )
                                    for w in range(KERNEL_WIDTH - 2):
                                        r_state[
                                            (KERNEL_WIDTH - 1) * vec_size
                                            + w * vec_size
                                            + i
                                        ] = r_state[
                                            (KERNEL_WIDTH - 1) * vec_size
                                            + (w + 1) * vec_size
                                            + i
                                        ]
                                    r_state[
                                        (KERNEL_WIDTH - 1) * vec_size
                                        + (KERNEL_WIDTH - 2) * vec_size
                                        + i
                                    ] = r_xk_raw
                            else:
                                for i in range(vec_size):
                                    k_idx = i * 32 + in_warp_tid
                                    r_gk_c2 = cutlass.Float32(
                                        qkg_cache[slot, i_t, 2, hk_off + k_idx]
                                    )
                                    sG[i_t, k_idx] = cute.math.exp(
                                        r_gk_c2, fastmath=True
                                    )
                        else:
                            token = bos + i_t
                            if warp_idx == 0:
                                for i_pair in range(vec_size // 2):
                                    i0 = i_pair * 2
                                    i1 = i_pair * 2 + 1
                                    k_idx0 = i0 * 32 + in_warp_tid
                                    k_idx1 = i1 * 32 + in_warp_tid
                                    r_conv_0 = 0.0
                                    r_conv_1 = 0.0
                                    if cutlass.const_expr(USE_REG_Q_WEIGHTS):
                                        _cwq_0 = r_wq[0 * vec_size + i0]
                                        _cwq_1 = r_wq[0 * vec_size + i1]
                                        r_conv_0 += r_state[0 * vec_size + i0] * _cwq_0
                                        r_conv_1 += r_state[0 * vec_size + i1] * _cwq_1
                                        _cwq_0 = r_wq[1 * vec_size + i0]
                                        _cwq_1 = r_wq[1 * vec_size + i1]
                                        r_conv_0 += r_state[1 * vec_size + i0] * _cwq_0
                                        r_conv_1 += r_state[1 * vec_size + i1] * _cwq_1
                                        _cwq_0 = r_wq[2 * vec_size + i0]
                                        _cwq_1 = r_wq[2 * vec_size + i1]
                                        r_conv_0 += r_state[2 * vec_size + i0] * _cwq_0
                                        r_conv_1 += r_state[2 * vec_size + i1] * _cwq_1
                                    else:
                                        _cwq_0 = cutlass.Float32(0.0)
                                        _cwq_1 = cutlass.Float32(0.0)
                                        for w in range(KERNEL_WIDTH - 1):
                                            _cwq_0 = sConvW[
                                                w * K + i0 * 32 + in_warp_tid
                                            ]
                                            _cwq_1 = sConvW[
                                                w * K + i1 * 32 + in_warp_tid
                                            ]
                                            r_conv_0 += (
                                                r_state[w * vec_size + i0] * _cwq_0
                                            )
                                            r_conv_1 += (
                                                r_state[w * vec_size + i1] * _cwq_1
                                            )
                                    if cutlass.const_expr(USE_FLAT_LAYOUT):
                                        r_xq_0 = cutlass.Float32(
                                            x_q[0, token, hk_off + k_idx0]
                                        )
                                        r_xq_1 = cutlass.Float32(
                                            x_q[0, token, hk_off + k_idx1]
                                        )
                                    else:
                                        r_xq_0 = cutlass.Float32(
                                            x_q[0, token, i_h, k_idx0]
                                        )
                                        r_xq_1 = cutlass.Float32(
                                            x_q[0, token, i_h, k_idx1]
                                        )
                                    if cutlass.const_expr(USE_REG_Q_WEIGHTS):
                                        _cwq_last_0 = r_wq[
                                            (KERNEL_WIDTH - 1) * vec_size + i0
                                        ]
                                        _cwq_last_1 = r_wq[
                                            (KERNEL_WIDTH - 1) * vec_size + i1
                                        ]
                                    else:
                                        _cwq_last_0 = sConvW[
                                            (KERNEL_WIDTH - 1) * K
                                            + i0 * 32
                                            + in_warp_tid
                                        ]
                                        _cwq_last_1 = sConvW[
                                            (KERNEL_WIDTH - 1) * K
                                            + i1 * 32
                                            + in_warp_tid
                                        ]
                                    r_conv_0 += r_xq_0 * _cwq_last_0
                                    r_conv_1 += r_xq_1 * _cwq_last_1
                                    e0 = cute.math.exp(-r_conv_0, fastmath=True)
                                    e1 = cute.math.exp(-r_conv_1, fastmath=True)
                                    sig_0 = cute.arch.rcp_approx(
                                        cutlass.Float32(1.0) + e0
                                    )
                                    sig_1 = cute.arch.rcp_approx(
                                        cutlass.Float32(1.0) + e1
                                    )
                                    r_q[i0] = r_conv_0 * sig_0
                                    r_q[i1] = r_conv_1 * sig_1
                                    r_state[0 * vec_size + i0] = r_state[
                                        1 * vec_size + i0
                                    ]
                                    r_state[0 * vec_size + i1] = r_state[
                                        1 * vec_size + i1
                                    ]
                                    r_state[1 * vec_size + i0] = r_state[
                                        2 * vec_size + i0
                                    ]
                                    r_state[1 * vec_size + i1] = r_state[
                                        2 * vec_size + i1
                                    ]
                                    r_state[2 * vec_size + i0] = r_xq_0
                                    r_state[2 * vec_size + i1] = r_xq_1
                                sum_q = 0.0
                                for i in range(vec_size):
                                    sum_q += r_q[i] * r_q[i]
                                for offset in [16, 8, 4, 2, 1]:
                                    sum_q += cute.arch.shuffle_sync_bfly(
                                        sum_q, offset=offset, mask=-1, mask_and_clamp=31
                                    )
                                rnorm_q_scaled = (
                                    cute.math.rsqrt(sum_q + 1e-06, fastmath=True)
                                    * scale
                                )
                                for i in range(vec_size):
                                    r_q[i] = r_q[i] * rnorm_q_scaled
                                for i in range(vec_size):
                                    k_idx = i * 32 + in_warp_tid
                                    sQ[i_t, k_idx] = r_q[i]
                            elif warp_idx == 1:
                                r_b_raw = cutlass.Float32(0.0)
                                if in_warp_tid == 0:
                                    r_b_raw = cutlass.Float32(beta[0, token, i_hv])
                                for i in range(vec_size):
                                    k_idx = i * 32 + in_warp_tid
                                    r_conv = (
                                        r_state[
                                            (KERNEL_WIDTH - 1) * vec_size
                                            + 0 * vec_size
                                            + i
                                        ]
                                        * sConvW[
                                            k_weight_base + 0 * K + i * 32 + in_warp_tid
                                        ]
                                    )
                                    r_conv += (
                                        r_state[
                                            (KERNEL_WIDTH - 1) * vec_size
                                            + 1 * vec_size
                                            + i
                                        ]
                                        * sConvW[
                                            k_weight_base + 1 * K + i * 32 + in_warp_tid
                                        ]
                                    )
                                    r_conv += (
                                        r_state[
                                            (KERNEL_WIDTH - 1) * vec_size
                                            + 2 * vec_size
                                            + i
                                        ]
                                        * sConvW[
                                            k_weight_base + 2 * K + i * 32 + in_warp_tid
                                        ]
                                    )
                                    if cutlass.const_expr(USE_FLAT_LAYOUT):
                                        r_xk = cutlass.Float32(
                                            x_k[0, token, hk_off + k_idx]
                                        )
                                    else:
                                        r_xk = cutlass.Float32(
                                            x_k[0, token, i_h, k_idx]
                                        )
                                    r_conv += (
                                        r_xk
                                        * sConvW[
                                            k_weight_base
                                            + (KERNEL_WIDTH - 1) * K
                                            + i * 32
                                            + in_warp_tid
                                        ]
                                    )
                                    r_conv = r_conv * cute.arch.rcp_approx(
                                        cutlass.Float32(1.0)
                                        + cute.math.exp(-r_conv, fastmath=True)
                                    )
                                    r_k[i] = r_conv
                                    r_state[
                                        (KERNEL_WIDTH - 1) * vec_size + 0 * vec_size + i
                                    ] = r_state[
                                        (KERNEL_WIDTH - 1) * vec_size + 1 * vec_size + i
                                    ]
                                    r_state[
                                        (KERNEL_WIDTH - 1) * vec_size + 1 * vec_size + i
                                    ] = r_state[
                                        (KERNEL_WIDTH - 1) * vec_size + 2 * vec_size + i
                                    ]
                                    r_state[
                                        (KERNEL_WIDTH - 1) * vec_size + 2 * vec_size + i
                                    ] = r_xk
                                sum_k = 0.0
                                for i in range(vec_size):
                                    sum_k += r_k[i] * r_k[i]
                                for offset in [16, 8, 4, 2, 1]:
                                    sum_k += cute.arch.shuffle_sync_bfly(
                                        sum_k, offset=offset, mask=-1, mask_and_clamp=31
                                    )
                                rnorm_k = cute.math.rsqrt(sum_k + 1e-06, fastmath=True)
                                for i in range(vec_size):
                                    r_k[i] = r_k[i] * rnorm_k
                                for i in range(vec_size):
                                    k_idx = i * 32 + in_warp_tid
                                    sK[i_t, k_idx] = r_k[i]
                                if in_warp_tid == 0:
                                    sBeta[i_t] = cute.arch.rcp_approx(
                                        cutlass.Float32(1.0)
                                        + cute.math.exp(-r_b_raw, fastmath=True)
                                    )
                            else:
                                for i in range(vec_size):
                                    k_idx = i * 32 + in_warp_tid
                                    r_g_raw = cutlass.Float32(g[0, token, i_hv, k_idx])
                                    r_g_raw = r_g_raw + cutlass.Float32(
                                        dt_bias[i_h * K + k_idx]
                                    )
                                    exp_A_x = r_exp_A * r_g_raw
                                    sigmoid_val = cute.arch.rcp_approx(
                                        cutlass.Float32(1.0)
                                        + cute.math.exp(-exp_A_x, fastmath=True)
                                    )
                                    r_gk = lower_bound * sigmoid_val
                                    sG[i_t, k_idx] = cute.math.exp(r_gk, fastmath=True)
                                    r_decay[i] = r_gk
                                if cutlass.const_expr(not DSPARK_SCRATCH):
                                    if i_t > commit_len:
                                        cache_pos = i_t - commit_len - 1
                                        for i in range(vec_size):
                                            k_idx = i * 32 + in_warp_tid
                                            qkg_cache[
                                                slot, cache_pos, 2, hk_off + k_idx
                                            ] = cutlass.BFloat16(r_decay[i])
                            if cutlass.const_expr(DSPARK_SCRATCH):
                                if warp_idx == 0:
                                    for i in range(vec_size):
                                        k_idx = i * 32 + in_warp_tid
                                        for w in range(KERNEL_WIDTH - 1):
                                            intermediate_conv_q[
                                                scratch_row, i_t, hk_off + k_idx, w
                                            ] = cutlass.BFloat16(r_state[w * vec_size + i])
                                elif warp_idx == 1:
                                    for i in range(vec_size):
                                        k_idx = i * 32 + in_warp_tid
                                        for w in range(KERNEL_WIDTH - 1):
                                            intermediate_conv_k[
                                                scratch_row, i_t, hk_off + k_idx, w
                                            ] = cutlass.BFloat16(r_state[
                                                (KERNEL_WIDTH - 1) * vec_size
                                                + w * vec_size
                                                + i
                                            ])
                            elif i_t == commit_len:
                                if warp_idx == 0:
                                    for i in range(vec_size):
                                        k_idx = i * 32 + in_warp_tid
                                        for w in range(KERNEL_WIDTH - 1):
                                            cs_q[slot, hk_off + k_idx, w] = r_state[
                                                w * vec_size + i
                                            ]
                                elif warp_idx == 1:
                                    for i in range(vec_size):
                                        k_idx = i * 32 + in_warp_tid
                                        for w in range(KERNEL_WIDTH - 1):
                                            cs_k[slot, hk_off + k_idx, w] = r_state[
                                                (KERNEL_WIDTH - 1) * vec_size
                                                + w * vec_size
                                                + i
                                            ]
                            if cutlass.const_expr(not DSPARK_SCRATCH):
                                if i_t > commit_len:
                                    cache_pos = i_t - commit_len - 1
                                    if warp_idx == 0:
                                        for i in range(vec_size):
                                            k_idx = i * 32 + in_warp_tid
                                            qkg_cache[
                                                slot, cache_pos, 0, hk_off + k_idx
                                            ] = cutlass.BFloat16(sQ[i_t, k_idx])
                                        for i in range(vec_size):
                                            k_idx = i * 32 + in_warp_tid
                                            cs_q[
                                                slot,
                                                hk_off + k_idx,
                                                KERNEL_WIDTH - 1 + cache_pos,
                                            ] = r_state[
                                                (KERNEL_WIDTH - 2) * vec_size + i
                                            ]
                                    elif warp_idx == 1:
                                        for i in range(vec_size):
                                            k_idx = i * 32 + in_warp_tid
                                            qkg_cache[
                                                slot, cache_pos, 1, hk_off + k_idx
                                            ] = cutlass.BFloat16(sK[i_t, k_idx])
                                        if in_warp_tid == 0:
                                            beta_cache[slot, cache_pos, i_hv] = (
                                                cutlass.BFloat16(sBeta[i_t])
                                            )
                                        for i in range(vec_size):
                                            k_idx = i * 32 + in_warp_tid
                                            cs_k[
                                                slot,
                                                hk_off + k_idx,
                                                KERNEL_WIDTH - 1 + cache_pos,
                                            ] = r_state[
                                                (KERNEL_WIDTH - 1) * vec_size
                                                + (KERNEL_WIDTH - 2) * vec_size
                                                + i
                                            ]
                        i_t = i_t + 1
                else:
                    _v_idx = tidx - 96
                    if _v_idx < V:
                        _csv0 = cutlass.Float32(cs_v[slot, hv_off + _v_idx, 0])
                        _csv1 = cutlass.Float32(cs_v[slot, hv_off + _v_idx, 1])
                        _csv2 = cutlass.Float32(cs_v[slot, hv_off + _v_idx, 2])
                        if cutlass.const_expr(USE_ZERO_ACCEPTED):
                            _wv = [
                                sConvW[v_weight_base + w * V + _v_idx]
                                for w in range(KERNEL_WIDTH)
                            ]
                            # Sliding conv window, oldest -> newest: KERNEL_WIDTH-1
                            # cached taps followed by the 1 + NUM_SPEC new tokens.
                            # Trace-time list; every loop below unrolls.
                            _win = [_csv0, _csv1, _csv2]
                            for _t in cutlass.range_constexpr(1 + NUM_SPEC):
                                if cutlass.const_expr(USE_FLAT_LAYOUT):
                                    _win.append(
                                        cutlass.Float32(
                                            x_v[0, bos + _t, hv_off + _v_idx]
                                        )
                                    )
                                else:
                                    _win.append(
                                        cutlass.Float32(x_v[0, bos + _t, i_hv, _v_idx])
                                    )
                            for _t in cutlass.range_constexpr(1 + NUM_SPEC):
                                _vconv = _win[_t] * _wv[0]
                                for _w in cutlass.range_constexpr(1, KERNEL_WIDTH):
                                    _vconv += _win[_t + _w] * _wv[_w]
                                _vconv = _vconv * cute.arch.rcp_approx(
                                    cutlass.Float32(1.0)
                                    + cute.math.exp(-_vconv, fastmath=True)
                                )
                                sVall[_t * V + _v_idx] = _vconv
                                if cutlass.const_expr(DSPARK_SCRATCH):
                                    for _w in cutlass.range_constexpr(KERNEL_WIDTH - 1):
                                        intermediate_conv_v[
                                            scratch_row, _t, hv_off + _v_idx, _w
                                        ] = cutlass.BFloat16(_win[_t + 1 + _w])
                                elif cutlass.const_expr(_t == 0):
                                    for _w in cutlass.range_constexpr(KERNEL_WIDTH - 1):
                                        cs_v[slot, hv_off + _v_idx, _w] = _win[1 + _w]
                                else:
                                    v_cache[slot, _t - 1, hv_off + _v_idx] = _vconv
                                    cs_v[
                                        slot, hv_off + _v_idx, KERNEL_WIDTH - 2 + _t
                                    ] = _win[KERNEL_WIDTH - 1 + _t]
                        else:
                            _i_t = 0
                            while _i_t < T_loop:
                                if _i_t < commit_len:
                                    sVall[_i_t * V + _v_idx] = cutlass.Float32(
                                        v_cache[slot, _i_t, hv_off + _v_idx]
                                    )
                                    _xv_replay = cutlass.Float32(
                                        cs_v[
                                            slot,
                                            hv_off + _v_idx,
                                            KERNEL_WIDTH - 1 + _i_t,
                                        ]
                                    )
                                    _csv0 = _csv1
                                    _csv1 = _csv2
                                    _csv2 = _xv_replay
                                else:
                                    _token_v = bos + _i_t
                                    _v_conv = 0.0
                                    _v_conv += (
                                        _csv0 * sConvW[v_weight_base + 0 * V + _v_idx]
                                    )
                                    _v_conv += (
                                        _csv1 * sConvW[v_weight_base + 1 * V + _v_idx]
                                    )
                                    _v_conv += (
                                        _csv2 * sConvW[v_weight_base + 2 * V + _v_idx]
                                    )
                                    if cutlass.const_expr(USE_FLAT_LAYOUT):
                                        _xv = cutlass.Float32(
                                            x_v[0, _token_v, hv_off + _v_idx]
                                        )
                                    else:
                                        _xv = cutlass.Float32(
                                            x_v[0, _token_v, i_hv, _v_idx]
                                        )
                                    _v_conv += (
                                        _xv
                                        * sConvW[
                                            v_weight_base
                                            + (KERNEL_WIDTH - 1) * V
                                            + _v_idx
                                        ]
                                    )
                                    _v_conv = _v_conv * cute.arch.rcp_approx(
                                        cutlass.Float32(1.0)
                                        + cute.math.exp(-_v_conv, fastmath=True)
                                    )
                                    sVall[_i_t * V + _v_idx] = _v_conv
                                    _csv0 = _csv1
                                    _csv1 = _csv2
                                    _csv2 = _xv
                                    if cutlass.const_expr(DSPARK_SCRATCH):
                                        intermediate_conv_v[
                                            scratch_row, _i_t, hv_off + _v_idx, 0
                                        ] = cutlass.BFloat16(_csv0)
                                        intermediate_conv_v[
                                            scratch_row, _i_t, hv_off + _v_idx, 1
                                        ] = cutlass.BFloat16(_csv1)
                                        intermediate_conv_v[
                                            scratch_row, _i_t, hv_off + _v_idx, 2
                                        ] = cutlass.BFloat16(_csv2)
                                    elif _i_t == commit_len:
                                        cs_v[slot, hv_off + _v_idx, 0] = _csv0
                                        cs_v[slot, hv_off + _v_idx, 1] = _csv1
                                        cs_v[slot, hv_off + _v_idx, 2] = _csv2
                                    if (
                                        cutlass.const_expr(not DSPARK_SCRATCH)
                                        and _i_t > commit_len
                                    ):
                                        _cp = _i_t - commit_len - 1
                                        v_cache[slot, _cp, hv_off + _v_idx] = _v_conv
                                        cs_v[
                                            slot,
                                            hv_off + _v_idx,
                                            KERNEL_WIDTH - 1 + _cp,
                                        ] = _xv
                                _i_t = _i_t + 1
                if cutlass.const_expr(PROFILE_STAGES):
                    cute.arch.barrier()
                    t_stage1 = read_globaltimer()
            else:
                cute.arch.barrier()
                if cutlass.const_expr(USE_SETMAXREG):
                    cute.arch.warpgroup_reg_dealloc(64)
                if cutlass.const_expr(PROFILE_STAGES):
                    cute.arch.barrier()
                    t_stage1 = read_globaltimer()
        else:
            cute.arch.barrier()
            if cutlass.const_expr(USE_SETMAXREG):
                cute.arch.warpgroup_reg_dealloc(64)
            if cutlass.const_expr(PROFILE_STAGES):
                cute.arch.barrier()
                t_stage1 = read_globaltimer()
        if cutlass.const_expr(SPLIT_V):
            v_split_base = i_z * TILE_V
        else:
            v_split_base = 0
        for row in range(NUM_V_ROWS):
            v_row = v_split_base + warp_idx * NUM_V_ROWS + row
            for i in range(vec_size):
                if cutlass.const_expr(USE_FLAT_LAYOUT):
                    r_state[row * vec_size + i] = cutlass.Float32(
                        h0[h0_idx, v_row, i * 32 + in_warp_tid]
                    )
                else:
                    r_state[row * vec_size + i] = cutlass.Float32(
                        h0[slot, i_hv, v_row, i * 32 + in_warp_tid]
                    )
        if cutlass.const_expr(USE_SETMAXREG):
            cute.arch.warpgroup_reg_alloc(72)
        cute.arch.barrier()
        if cutlass.const_expr(ENABLE_PDL):
            cute.arch.griddepcontrol_launch_dependents()
        for i_v in range(1 if cutlass.const_expr(SPLIT_V) else num_v_tiles):
            if cutlass.const_expr(SPLIT_V):
                v_base = v_split_base
            else:
                v_base = i_v * TILE_V
            if i_v > 0:
                for row in range(NUM_V_ROWS):
                    v_row = warp_idx * NUM_V_ROWS + row
                    for i in range(vec_size):
                        if cutlass.const_expr(USE_FLAT_LAYOUT):
                            r_state[row * vec_size + i] = cutlass.Float32(
                                h0[h0_idx, v_base + v_row, i * 32 + in_warp_tid]
                            )
                        else:
                            r_state[row * vec_size + i] = cutlass.Float32(
                                h0[slot, i_hv, v_base + v_row, i * 32 + in_warp_tid]
                            )
            r_v_val = cutlass.Float32(0.0)
            i_t = 0
            while i_t < T_loop:
                if in_warp_tid < NUM_V_ROWS:
                    v_idx = v_base + warp_idx * NUM_V_ROWS + in_warp_tid
                    r_v_val = sVall[i_t * V + v_idx]
                r_beta_val = sBeta[i_t]
                for i_pair in range(vec_size // 2):
                    i0 = i_pair * 2
                    i1 = i_pair * 2 + 1
                    k_idx0 = i0 * 32 + in_warp_tid
                    k_idx1 = i1 * 32 + in_warp_tid
                    r_q[i0] = sQ[i_t, k_idx0]
                    r_q[i1] = sQ[i_t, k_idx1]
                    _k0 = sK[i_t, k_idx0]
                    _k1 = sK[i_t, k_idx1]
                    r_decay[i0] = sG[i_t, k_idx0]
                    r_decay[i1] = sG[i_t, k_idx1]
                    r_bk[i0], r_bk[i1] = cute.arch.mul_packed_f32x2(
                        (r_beta_val, r_beta_val), (_k0, _k1)
                    )
                    r_k[i0], r_k[i1] = cute.arch.mul_packed_f32x2(
                        (r_decay[i0], r_decay[i1]), (_k0, _k1)
                    )
                for row_pair in range(NUM_V_ROWS // 2):
                    ra = row_pair * 2
                    rb = row_pair * 2 + 1
                    r_va = cute.arch.shuffle_sync(r_v_val, ra)
                    r_vb = cute.arch.shuffle_sync(r_v_val, rb)
                    shk_a1 = 0.0
                    shk_a2 = 0.0
                    shk_b1 = 0.0
                    shk_b2 = 0.0
                    for _pi in range(vec_size // 2):
                        _p = _pi * 2
                        shk_a1, shk_a2 = cute.arch.fma_packed_f32x2(
                            src_a=(
                                r_state[ra * vec_size + _p],
                                r_state[ra * vec_size + _p + 1],
                            ),
                            src_b=(r_k[_p], r_k[_p + 1]),
                            src_c=(shk_a1, shk_a2),
                        )
                        shk_b1, shk_b2 = cute.arch.fma_packed_f32x2(
                            src_a=(
                                r_state[rb * vec_size + _p],
                                r_state[rb * vec_size + _p + 1],
                            ),
                            src_b=(r_k[_p], r_k[_p + 1]),
                            src_c=(shk_b1, shk_b2),
                        )
                    shk_a = shk_a1 + shk_a2
                    shk_b = shk_b1 + shk_b2
                    for offset in [16, 8, 4, 2, 1]:
                        shk_a += cute.arch.shuffle_sync_bfly(
                            shk_a, offset=offset, mask=-1, mask_and_clamp=31
                        )
                        shk_b += cute.arch.shuffle_sync_bfly(
                            shk_b, offset=offset, mask=-1, mask_and_clamp=31
                        )
                    vn_a = r_va - shk_a
                    vn_b = r_vb - shk_b
                    shq_a1 = 0.0
                    shq_a2 = 0.0
                    shq_b1 = 0.0
                    shq_b2 = 0.0
                    for _pi in range(vec_size // 2):
                        _p = _pi * 2
                        vnbk_a0, vnbk_a1 = cute.arch.mul_packed_f32x2(
                            (vn_a, vn_a), (r_bk[_p], r_bk[_p + 1])
                        )
                        vnbk_b0, vnbk_b1 = cute.arch.mul_packed_f32x2(
                            (vn_b, vn_b), (r_bk[_p], r_bk[_p + 1])
                        )
                        r_state[ra * vec_size + _p], r_state[ra * vec_size + _p + 1] = (
                            cute.arch.fma_packed_f32x2(
                                src_a=(r_decay[_p], r_decay[_p + 1]),
                                src_b=(
                                    r_state[ra * vec_size + _p],
                                    r_state[ra * vec_size + _p + 1],
                                ),
                                src_c=(vnbk_a0, vnbk_a1),
                            )
                        )
                        r_state[rb * vec_size + _p], r_state[rb * vec_size + _p + 1] = (
                            cute.arch.fma_packed_f32x2(
                                src_a=(r_decay[_p], r_decay[_p + 1]),
                                src_b=(
                                    r_state[rb * vec_size + _p],
                                    r_state[rb * vec_size + _p + 1],
                                ),
                                src_c=(vnbk_b0, vnbk_b1),
                            )
                        )
                        shq_a1, shq_a2 = cute.arch.fma_packed_f32x2(
                            src_a=(
                                r_state[ra * vec_size + _p],
                                r_state[ra * vec_size + _p + 1],
                            ),
                            src_b=(r_q[_p], r_q[_p + 1]),
                            src_c=(shq_a1, shq_a2),
                        )
                        shq_b1, shq_b2 = cute.arch.fma_packed_f32x2(
                            src_a=(
                                r_state[rb * vec_size + _p],
                                r_state[rb * vec_size + _p + 1],
                            ),
                            src_b=(r_q[_p], r_q[_p + 1]),
                            src_c=(shq_b1, shq_b2),
                        )
                    shq_a = shq_a1 + shq_a2
                    shq_b = shq_b1 + shq_b2
                    for offset in [16, 8, 4, 2, 1]:
                        shq_a += cute.arch.shuffle_sync_bfly(
                            shq_a, offset=offset, mask=-1, mask_and_clamp=31
                        )
                        shq_b += cute.arch.shuffle_sync_bfly(
                            shq_b, offset=offset, mask=-1, mask_and_clamp=31
                        )
                    if in_warp_tid == 0:
                        v_row_a = warp_idx * NUM_V_ROWS + ra
                        v_row_b = warp_idx * NUM_V_ROWS + rb
                        if cutlass.const_expr(USE_ZERO_ACCEPTED):
                            o[0, bos + i_t, i_hv, v_base + v_row_a] = cutlass.BFloat16(
                                shq_a
                            )
                            o[0, bos + i_t, i_hv, v_base + v_row_b] = cutlass.BFloat16(
                                shq_b
                            )
                        elif i_t >= commit_len:
                            o[0, bos + i_t, i_hv, v_base + v_row_a] = cutlass.BFloat16(
                                shq_a
                            )
                            o[0, bos + i_t, i_hv, v_base + v_row_b] = cutlass.BFloat16(
                                shq_b
                            )
                if cutlass.const_expr(DSPARK_SCRATCH):
                    for row in range(NUM_V_ROWS):
                        v_row = warp_idx * NUM_V_ROWS + row
                        for i in range(vec_size):
                            ht[
                                scratch_row,
                                i_t,
                                i_hv,
                                v_base + v_row,
                                i * 32 + in_warp_tid,
                            ] = r_state[row * vec_size + i]
                elif i_t == commit_len:
                    for row in range(NUM_V_ROWS):
                        v_row = warp_idx * NUM_V_ROWS + row
                        for i in range(vec_size):
                            if cutlass.const_expr(USE_FLAT_LAYOUT):
                                ht[h0_idx, v_base + v_row, i * 32 + in_warp_tid] = (
                                    r_state[row * vec_size + i]
                                )
                            else:
                                ht[slot, i_hv, v_base + v_row, i * 32 + in_warp_tid] = (
                                    r_state[row * vec_size + i]
                                )
                i_t = i_t + 1
        if cutlass.const_expr(PROFILE_STAGES):
            cute.arch.barrier()
            t_stage2 = read_globaltimer()
            if tidx == 0:
                _, grid_n, _ = cute.arch.grid_dim()
                timing_base = (i_hv * grid_n + i_n) * 4
                stage_timing[timing_base + 0] = t_stage1 - t_stage0
                stage_timing[timing_base + 1] = t_stage2 - t_stage1
                stage_timing[timing_base + 2] = t_stage2 - t_stage0
                stage_timing[timing_base + 3] = t_stage0


@cute.jit
def _run_kda_decode_mtp_dspark(
    h0: cute.Tensor,
    x_q: cute.Tensor,
    x_k: cute.Tensor,
    x_v: cute.Tensor,
    w_q: cute.Tensor,
    w_k: cute.Tensor,
    w_v: cute.Tensor,
    cs_q: cute.Tensor,
    cs_k: cute.Tensor,
    cs_v: cute.Tensor,
    A_log: cute.Tensor,
    g: cute.Tensor,
    dt_bias: cute.Tensor,
    beta: cute.Tensor,
    out: cute.Tensor,
    intermediate_ssm: cute.Tensor,
    intermediate_state_indices: cute.Tensor,
    intermediate_conv_q: cute.Tensor,
    intermediate_conv_k: cute.Tensor,
    intermediate_conv_v: cute.Tensor,
    ssm_state_indices: cute.Tensor,
    cu_seqlens: cute.Tensor,
    unused_qkg_cache: cute.Tensor,
    unused_v_cache: cute.Tensor,
    unused_beta_cache: cute.Tensor,
    scale: cutlass.Constexpr[float],
    H: cutlass.Constexpr[int],
    N: cutlass.Constexpr[int],
    NUM_SPEC: cutlass.Constexpr[int],
    TILE_V: cutlass.Constexpr[int],
    SPLIT_V: cutlass.Constexpr[bool],
    ENABLE_PDL: cutlass.Constexpr[bool],
    lower_bound: cutlass.Constexpr[float],
    USE_SETMAXREG: cutlass.Constexpr[bool],
    stream: cuda.CUstream,
):
    """Launch the fixed Kimi-K3/DSpARK bonus + NUM_SPEC-draft specialization."""
    smem_qk_layout = cute.make_layout((1 + NUM_SPEC, TILE_K), stride=(TILE_K, 1))
    kda_decode_mtp_kernel(
        h0,
        x_q,
        x_k,
        x_v,
        w_q,
        w_k,
        w_v,
        cs_q,
        cs_k,
        cs_v,
        A_log,
        g,
        dt_bias,
        beta,
        out,
        intermediate_ssm,
        intermediate_state_indices,
        intermediate_conv_q,
        intermediate_conv_k,
        intermediate_conv_v,
        # These replay caches are dead when USE_ZERO_ACCEPTED=True, but CuTe
        # still type-checks their indexed branches.
        unused_qkg_cache,
        unused_v_cache,
        unused_beta_cache,
        smem_qk_layout,
        ssm_state_indices,
        cu_seqlens,
        ssm_state_indices,
        ssm_state_indices,
        TILE_V,
        scale,
        H,
        TILE_K,
        TILE_K,
        NUM_SPEC,
        4,
        lower_bound,
        False,
        USE_SETMAXREG,
        False,
        True,
        True,
        True,
        False,
        True,
        SPLIT_V,
        ENABLE_PDL,
        out,
        False,
    ).launch(
        grid=(H, N, TILE_K // TILE_V if cutlass.const_expr(SPLIT_V) else 1),
        block=[NUM_THREADS, 1, 1],
        stream=stream,
        use_pdl=ENABLE_PDL,
    )


_DSPARK_COMPILED = {}


def _tensor_layout_key(tensor):
    return (tensor.device, tensor.dtype, tuple(tensor.shape), tuple(tensor.stride()))


def _fits_32bit_stride(tensor):
    max_offset = int(tensor.storage_offset())
    for size, stride in zip(tensor.shape, tensor.stride()):
        if abs(int(stride)) > 2**31 - 1:
            return False
        if size:
            max_offset += (int(size) - 1) * abs(int(stride))
            if max_offset > 2**31 - 1:
                return False
    return True


def _cute_tensor(tensor, *, dynamic=True):
    from cutlass.cute.runtime import from_dlpack

    if tensor.requires_grad:
        tensor = tensor.detach()
    value = from_dlpack(
        tensor,
        assumed_align=16,
        use_32bit_stride=_fits_32bit_stride(tensor),
    )
    leading_dim = next(
        (dim for dim, stride in enumerate(tensor.stride()) if stride == 1), None
    )
    if not dynamic or leading_dim is None:
        return value
    return value.mark_layout_dynamic(leading_dim)


def fused_kda_decode_mtp_dspark(
    *,
    x_q,
    x_k,
    x_v,
    w_q,
    w_k,
    w_v,
    cs_q,
    cs_k,
    cs_v,
    g,
    beta,
    A_log,
    dt_bias,
    recurrent_state,
    intermediate_ssm,
    intermediate_state_indices,
    intermediate_conv_q,
    intermediate_conv_k,
    intermediate_conv_v,
    ssm_state_indices,
    cu_seqlens,
    lower_bound,
    scale=None,
    split_v=None,
):
    """Run Kimi-K3 KDA verify while preserving DSpARK rollback semantics.

    Persistent recurrent/conv states are read-only.  Every post-token state and
    convolution window is written to DSpARK's existing intermediate buffers.
    The caller is responsible for enforcing the fixed dense token contract:
    every request contributes exactly 1 + num_spec tokens (num_spec ==
    --speculative-dspark-block-size), inferred here from T // N - 1.
    """
    import torch

    H = x_q.shape[2]
    N = cu_seqlens.numel() - 1
    T = x_q.shape[1]
    expected_q = (1, T, H, TILE_K)
    expected_v = (1, T, H, TILE_K)
    if tuple(x_q.shape) != expected_q or tuple(x_k.shape) != expected_q:
        raise ValueError(f"expected q/k shape {expected_q}")
    if tuple(x_v.shape) != expected_v or tuple(g.shape) != expected_q:
        raise ValueError(f"expected v/g shape {expected_v}")
    if tuple(beta.shape) != (1, T, H):
        raise ValueError(f"expected beta shape {(1, T, H)}")
    if N <= 0 or T % N != 0 or T // N < 2:
        raise ValueError(
            f"DSpARK KDA MTP requires a fixed 1 + num_spec dense tokens per "
            f"request; got T={T}, N={N}"
        )
    num_spec = T // N - 1
    if recurrent_state.shape[1:] != (H, TILE_K, TILE_K):
        raise ValueError("expected recurrent state layout [pool, H, V=128, K=128]")
    if (
        intermediate_ssm.shape[1] < 1 + num_spec
        or intermediate_ssm.shape[2:5] != (H, TILE_K, TILE_K)
    ):
        raise ValueError(
            f"expected intermediate SSM layout [scratch, >={1 + num_spec}, H, "
            f"V=128, K=128]"
        )
    if scale is None:
        scale = TILE_K**-0.5
    # Small grids (decode bs=1) are latency-bound: SPLIT_V spreads the two
    # V tiles over grid z-blocks so the sequential token chain runs once per
    # block.  Safe only under this scratch contract — persistent conv taps
    # are read-only, and both z-blocks' duplicated scratch writes carry
    # identical values.  Larger grids keep the serial two-tile loop for
    # occupancy.
    if split_v is None:
        # Measured crossover on GB300 H=12 M=5: split wins -37%..-17% for
        # N<=16 (latency-bound grid) and loses ~+8% from N=32 up, where the
        # duplicated per-z-block precompute outweighs the halved token chain.
        split_v = N <= 16
    tile_v = 64
    # PDL: the decode-graph neighbors (tgv gemm, flashinfer norm) are
    # PDL-launched, so waiting on the predecessor and releasing dependents
    # early hides launch latency on both edges.
    enable_pdl = True
    use_setmaxreg = N in (1, 32, 128, 512) and H in (2, 12, 32)
    # empty is safe: the dense contract (checked above) writes every
    # [token, head, v] element, both z-blocks covering their V tiles.
    out = torch.empty_like(x_v)
    args = (
        recurrent_state,
        x_q,
        x_k,
        x_v,
        w_q,
        w_k,
        w_v,
        cs_q,
        cs_k,
        cs_v,
        A_log,
        g,
        dt_bias,
        beta,
        out,
        intermediate_ssm,
        intermediate_state_indices,
        intermediate_conv_q,
        intermediate_conv_k,
        intermediate_conv_v,
        ssm_state_indices,
        cu_seqlens,
        # Reuse float32 scratch storage as correctly-ranked placeholders for
        # dead replay-cache branches, avoiding allocations in the decode path.
        intermediate_conv_q,
        intermediate_conv_v[..., 0],
        intermediate_conv_q[..., 0],
    )
    key = (
        H,
        N,
        num_spec,
        tile_v,
        split_v,
        enable_pdl,
        float(scale),
        float(lower_bound),
        use_setmaxreg,
        *(_tensor_layout_key(tensor) for tensor in args),
    )
    stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
    compiled = _DSPARK_COMPILED.get(key)
    if compiled is None:
        cute_args = tuple(_cute_tensor(tensor, dynamic=False) for tensor in args)
        compiled = cute.compile(
            _run_kda_decode_mtp_dspark,
            *cute_args,
            scale=float(scale),
            H=H,
            N=N,
            NUM_SPEC=num_spec,
            TILE_V=tile_v,
            SPLIT_V=split_v,
            ENABLE_PDL=enable_pdl,
            lower_bound=float(lower_bound),
            USE_SETMAXREG=use_setmaxreg,
            stream=stream,
        )
        _DSPARK_COMPILED[key] = compiled
    compiled(
        *(_cute_tensor(tensor) for tensor in args),
        stream,
    )
    return out
