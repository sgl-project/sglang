# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Where the mono layer kernel (K4) puts its tasks and data, per decode token count.

Plain Python, evaluated at build time: which CTA runs each stage's tasks, the
scratch and peer-symmetric regions, and the LDS row pitches. Kept free of FlyDSL
so it is testable without a GPU.
"""

from sglang.kernels.ops.mono.gfx950.minimax_m3.config import (
    BLOCKS,
    HEAD_DIM,
    HIDDEN,
    INTER,
    LOCAL_INDEX_HEADS,
    LOCAL_KV_HEADS,
    LOCAL_Q_HEADS,
    MAX_INDEX_BLOCKS,
    MAX_SPARSE_KEYS,
    MAX_TOKENS,
    MOE_SLOTS,
    N_ROUTED,
    PAGE16,
    SPARSE_BLOCK,
    THREADS,
    TOP_K,
    TOPK_BLOCKS,
    TOTAL_INDEX_HEADS,
    TP,
    WAVES,
)

H = LOCAL_Q_HEADS
SPLIT_KEYS = 256  # keys per split task (one paged-attention context partition)
N_SPLIT = MAX_SPARSE_KEYS // SPLIT_KEYS
O_K = H * HEAD_DIM
# The split stage runs one pass per KV head: a pass attends that head's GQA
# group (H // LOCAL_KV_HEADS q heads) to its own K and V. The model is 64 q over
# 4 KV heads, so a group is 16 q heads at every TP -- which is also exactly what
# the stage's thread map covers (the MFMA A operand is 16 rows, a lane holds
# heads 4 * (lane // 16) + e, and the partial reduction walks O_K_SUB with
# THREADS * 4 = 2048 elements). The assert below is what ties the two together.
#
# At TP4 and TP8 a rank holds one KV head, so SPLIT_PASSES is 1, h0 is 0, and
# every expression is the original value -- those builds are unchanged.
#
# This is also what keeps K4 inside the 160 KB LDS at TP2: the PV partials are
# WAVES * O_K, which at 32 q heads alone is 131 072 B of a 163 840 B budget.
# Only ``opart`` is sized for a pass; every other region stays whole-H and is
# indexed by absolute head.
H_SUB = H // LOCAL_KV_HEADS  # q heads sharing one KV head
O_K_SUB = H_SUB * HEAD_DIM
SPLIT_PASSES = LOCAL_KV_HEADS
assert H % LOCAL_KV_HEADS == 0, f"{H} q heads do not divide over {LOCAL_KV_HEADS} KV heads"
assert H_SUB <= 16, f"GQA group {H_SUB} exceeds the split stage's 16-row thread map"

# A page-16 of ONE KV head (``pre_attn.PAGE_BYTES``). The cache is
# [page16][kv head][...] -- aiter's sparse views are k.view(-1, heads, 8, 16, 16)
# and v.view(-1, heads, 1, 128, 16), both 2048 B a head -- so stepping one page
# means stepping over every head, and KV head h of page p starts at
# (p * LOCAL_KV_HEADS + h) * PAGE_BYTES.
KV_PAGE_BYTES = LOCAL_KV_HEADS * PAGE16 * HEAD_DIM

# A token's indexer score region, sized for whichever mode needs more: own mode
# gives each of the rank's heads every block, CP gives all TOTAL_INDEX_HEADS a
# 1/TP stripe each. The two are equal wherever a rank has one index head, so
# TP4 and TP8 keep the original MAX_INDEX_BLOCKS; TP2 doubles it. Scratch is
# global memory, not LDS -- this does not touch the 160 KB budget.
ISCORE_BLOCKS = max(
    LOCAL_INDEX_HEADS * MAX_INDEX_BLOCKS,
    TOTAL_INDEX_HEADS * (MAX_INDEX_BLOCKS // TP),
)
O_ROWS = 32  # hidden rows per o_proj / attention-reduce task
# from this many tokens the attention all-reduce is a reduce-scatter to the row
# group's owner rank (t % npes) plus an all-gather of the sums: half the bytes over
# each link for one more hop (smaller batches are latency-bound: one hop wins)
AG_RS_FROM = 12
N_O = HIDDEN // O_ROWS
PAGES_PER_BLOCK = SPARSE_BLOCK // PAGE16
N_ROUTER = N_ROUTED // WAVES  # one expert per wave
UG_ROWS = 16
UG_PER_SLOT = INTER // UG_ROWS
SH_ROWS = 8  # mid rows per shared-expert up / gate task (a half task)
SH_TASKS = INTER // SH_ROWS
DN_ROWS = 32
N_DN = HIDDEN // DN_ROWS
# The router logits are written SCORE_COPIES times and CTA b polls copy
# b % SCORE_COPIES: hundreds of expert CTAs spinning on one 1 KB region serialize
# on its cache lines.
SCORE_COPIES = 8


def _layout(items, align=256):
    """Byte offsets of (value, tag) pair regions, 8 bytes per pair."""
    off, out = 0, {}
    for name, pairs in items:
        out[name] = off
        off += (pairs * 8 + align - 1) // align * align
    out["_bytes"] = off
    return out


# a context past THREADS blocks, one-head build: each split task's TOPK_BLOCKS
# best (key, block) (indexer context parallelism: the peers' sel_cp)
SEL_LONG_PAIRS = N_SPLIT * TOPK_BLOCKS * 2

# every region holds MAX_TOKENS rows (token k at k * the one-token size)
SCRATCH = _layout(
    [
        # indexer block scores, token k's head h block b at pair
        # k * ISCORE_BLOCKS + h * heads.score_blocks + b (``IndexHeads``)
        ("iscore", ISCORE_BLOCKS * MAX_TOKENS),
        ("sp_o", N_SPLIT * H * HEAD_DIM // 2 * MAX_TOKENS),  # packed bf16 pairs
        ("sp_m", N_SPLIT * H * MAX_TOKENS),
        ("sp_l", N_SPLIT * H * MAX_TOKENS),
        ("attnq", O_K // 4 * MAX_TOKENS // 2),  # merged attention, plain FP8 words
        ("attnq_s", MAX_TOKENS),  # and its per-token scale
        # the finished attention all-reduce + residual, bf16 pairs (o tasks -> router)
        ("a", HIDDEN // 2 * MAX_TOKENS),
        ("xn8", HIDDEN // 4 * MAX_TOKENS),  # MXFP8 xn, 4 fp8 per word
        ("xsc", HIDDEN // 32 * MAX_TOKENS),  # and its per-1x32 E8M0 scales
        # copy c, token k at pairs [(c * MAX_TOKENS + k) * N_ROUTED, ..)
        # per expert (routing key, sigmoid) pairs, see route_key
        ("scores", 2 * N_ROUTED * MAX_TOKENS * SCORE_COPIES),
        ("mid", MOE_SLOTS * INTER * MAX_TOKENS),
        # token k's split task p's candidates for index head h: rank r at pairs
        # 2 (((h MAX_TOKENS + k) N_SPLIT + p) TOPK_BLOCKS + r) + (0 key, 1 block)
        #
        # The head term is load-bearing in own mode. A share's candidates are
        # gathered ACROSS CTAs -- every split task polls every other task's
        # entries -- and the mailbox tag is the step's, the same for both passes.
        # Without a per-head range a CTA entering pass 1 would overwrite slots a
        # slower peer is still reading for pass 0. LOCAL_INDEX_HEADS is 1 at TP4
        # and TP8, where this is the original size.
        ("sel_long", SEL_LONG_PAIRS * MAX_TOKENS * LOCAL_INDEX_HEADS),
        # wide: the MXFP8 xn and scales as plain words (every token's row), and a
        # done flag per router task
        ("xn8p", HIDDEN // 4 * MAX_TOKENS // 2),
        ("xscp", HIDDEN // 32 * MAX_TOKENS // 2),
        ("xndone", N_ROUTER * MAX_TOKENS),
        # and the mid rows as MXFP8 words, their E8M0 scales a word each (the down
        # stage's LDS layout), a done flag per up / gate task pair
        ("mid8p", MOE_SLOTS * INTER * MAX_TOKENS // 8),
        ("midsp", MOE_SLOTS * INTER // 32 * MAX_TOKENS // 2),
        ("ugdone", (TOP_K * MAX_TOKENS + 1) * UG_PER_SLOT // 2),
    ]
)
SCRATCH_BYTES = SCRATCH["_bytes"]


def sym_layout(npes: int):
    """Per-rank symmetric buffer: attn / ffn partial regions, [src rank][token][HIDDEN],
    and the peers' index-selection candidates (``sel_cp``)."""
    rows = npes * MAX_TOKENS * HIDDEN // 2
    # a_ag / ffn_ag: the finished attention / FFN all-reduce rows, each written by
    # its owner rank (the reduce-scatter + all-gather form, S >= AG_RS_FROM)
    ag = MAX_TOKENS * HIDDEN // 2
    return _layout(
        [
            ("attn", rows),
            ("ffn", rows),
            ("a_ag", ag),
            ("ffn_ag", ag),
            # indexer context parallelism: token k's candidates for this rank's
            # index head, source rank s's half h's r-th best at pairs
            # 2 ((k N_SPLIT + 2 s + h) TOPK_BLOCKS + r) + (0 key, 1 block)
            ("sel_cp", SEL_LONG_PAIRS * MAX_TOKENS),
        ]
    )


def stage_bases(tokens: int):
    """Task t of a stage runs on CTA (base + t) % BLOCKS; the merge (S > 1), o and
    down tasks start past the tokens' split CTAs."""
    n_split = N_SPLIT * tokens
    n_merge = tokens if tokens > 1 else 0
    router = (n_split + n_merge + N_O) % BLOCKS
    return {
        "split": 0,
        "merge": n_split,
        "o": n_split + n_merge,
        "router": router,
        "shared": _shared_base(tokens, router),
        "ug": 0,
        "down": n_split,
    }


def ug_ctas_per_token(tokens: int):
    """CTAs per token of the routed up / gate tasks, token k on CTAs k C .. (k + 1)
    C - 1, so a CTA stages one token's xn, not all. None: round-robin, for S < 4
    or S not dividing the grid."""
    return BLOCKS // tokens if tokens >= 4 and BLOCKS % tokens == 0 else None


def ug_tasks_of(cta: int, tokens: int) -> int:
    """Routed up / gate tasks CTA ``cta`` runs."""
    per_token = TOP_K * UG_PER_SLOT
    cpt = ug_ctas_per_token(tokens)
    if cpt is None:
        return (per_token * tokens - cta + BLOCKS - 1) // BLOCKS
    return (per_token - cta % cpt + cpt - 1) // cpt


def _shared_base(tokens: int, router: int) -> int:
    """First of the SH_TASKS consecutive CTAs that run the shared expert's up /
    gate before the routing lands. Ranked by the most routed up / gate tasks a
    window CTA carries (they queue behind it); then, only if that is the most any
    CTA carries, by its router CTAs (they start it late and would delay the last
    routed tasks); then by its down CTAs; then by its router CTAs anyway."""
    n_router = N_ROUTER * tokens
    n_split = N_SPLIT * tokens
    most = max(ug_tasks_of(c, tokens) for c in range(BLOCKS))

    def cost(base):
        ctas = [(base + j) % BLOCKS for j in range(SH_TASKS)]
        busiest = max(ug_tasks_of(c, tokens) for c in ctas)
        in_router = sum((c - router) % BLOCKS < n_router for c in ctas)
        in_down = sum(n_split <= c < n_split + N_DN for c in ctas)
        return busiest, in_router if busiest == most else 0, in_down, in_router

    return min(range(BLOCKS), key=cost)


def shared_after_routing(tokens: int) -> bool:
    """Every CTA carries the same number of routed up / gate tasks: a
    shared-expert CTA routes first and runs its shared half task while its first
    routed task's weights are in flight, instead of holding its routing back."""
    loads = [ug_tasks_of(c, tokens) for c in range(BLOCKS)]
    return min(loads) == max(loads)


# Token counts from here run the expert-grouped MoE. The narrow path keeps every
# token's mids three ways at once (see pool_words), so its pool is the budget's
# swing term and the width at which it stops fitting depends on INTER -- i.e. on
# TP. Measured by COMPILE_ONLY builds against the 160 KB gfx950 limit: TP2
# narrow is clean at S=1-2 but needs 168,416 B at S=3 and 192,416 B at S=4,
# against 163,840. TP4 narrow fits all of S=1-4 (its mids are half the size), so
# it keeps the original threshold. Set this from a compile, never from a mirror:
# the LDS overflow is a non-raising diagnostic.
WIDE_FROM = 3 if TP == 2 else 5


def wide_moe(tokens: int) -> bool:
    """The MoE stage groups its work by expert (every distinct expert's weights
    read once, a B column per token) instead of by (token, slot): past four tokens
    the per-token tasks re-read the experts the tokens share, and their operands
    no longer fit in LDS."""
    return tokens >= WIDE_FROM


# LDS rows of a token's MXFP8 xn and of its E8M0 scales, padded by 16 B: at the
# natural 1536 / 192-word rows the 16 tokens a MFMA B operand gathers sit in one
# bank group (16-way conflicts)
XN8_ROW = HIDDEN // 4 + 4
XSC_ROW = HIDDEN // 32 + 4
# and the down stage's MXFP8 mid rows / their scales (16 lanes, 16 rows): 196 and
# 28 words put the rows' 16 B reads, and 4 B scale reads, in distinct banks
MID_ROW = INTER // 4 + 4
MIDSC_ROW = INTER // 32 + 4
# and the o stage's FP8 attention rows (512 words: the same 16-way conflict)
O_ROW = O_K // 4 + 4

# Thread maps whose extent is set by TP, each walked in whole THREADS-wide
# rounds. Every one of these was originally written as a single round, which
# was correct only because at TP4 the work happens to come to exactly THREADS
# slots; at TP2 the dimension doubles and one round covers half of it, leaving
# the rest of the region unwritten for the next stage to read as NaN bits.
#
# The merge: a thread holds 4 dims of one head, so the CTA needs
# H * HEAD_DIM / 4 slots -- 512 at TP4, 1024 at TP2.
MERGE_SLOTS = H * (HEAD_DIM // 4)
MERGE_PASSES = -(-MERGE_SLOTS // THREADS)
# Whether those rounds come out whole. At TP8 (H = 8) the merge needs only half
# a CTA, so its tail round has to be masked out of both the amax reduction and
# the store, or it writes past O_K / 4 words and corrupts the pool. The refusal
# lives with the merge stage in post_attn, not here: this module only states
# the map, and TP8's other constants are perfectly well defined.
MERGE_ROUNDS_WHOLE = MERGE_SLOTS == MERGE_PASSES * THREADS
# The narrow MoE path's two mid passes, over a MOE_SLOTS * INTER region: 3840
# at TP4, 7680 at TP2, 1920 at TP8. (The wide path, from WIDE_FROM tokens up,
# derives its own counts from the row count it is given.) The copy moves two
# elements a thread; the MXFP8 pass takes eight.
MID_COPY_ROUNDS = -(-(MOE_SLOTS * INTER) // (2 * THREADS))
MID8_SLOTS = MOE_SLOTS * INTER // 8
MID8_ROUNDS = -(-MID8_SLOTS // THREADS)
# The MXFP8 pass's scale is an xor-reduction over 4-lane groups, so a round's
# active slots must not end inside one. THREADS is a multiple of 64, so a round
# offset never moves a lane out of its wave, and this keeps the tail aligned.
assert MID8_SLOTS % 4 == 0, (
    f"the MXFP8 mid pass leaves a partial 4-lane scale group: INTER = {INTER}"
)


def pool_words(tokens: int) -> int:
    """LDS words of the activation pool ``x``, the largest of the stages that take
    turns in it. Wide: the largest of the split's PV partials, the up / gate
    stage's MXFP8 xn + scales + up partials + router sigmoids, and the down
    stage's MXFP8 mids + scales + a zero row."""
    if not wide_moe(tokens):
        # The narrow down stage keeps every token's mids three ways at once --
        # bf16 as they land, then MXFP8 plus E8M0 scales -- and that, not the
        # bf16 xn, is the largest narrow operand once INTER passes 983.
        # MOE_SLOTS * INTER * 25/32 is 3000 at TP4, which slips under HIDDEN / 2
        # = 3072 by 2.4% and is why the xn was mistaken for the maximum; it is
        # 6000 at TP2. Sizing this from HIDDEN put the mid tail and every scale
        # word outside the declared array.
        mids = MOE_SLOTS * INTER // 2  # bf16, as the mids land
        mid8 = MOE_SLOTS * INTER // 4  # the MXFP8 copy the down GEMV reads
        midsc = MOE_SLOTS * INTER // 32  # and its E8M0 scales
        return tokens * max(HIDDEN // 2, mids + mid8 + midsc)
    rows = MOE_SLOTS * tokens + 1
    return max(
        # the split's PV partials, one pass' worth (see H_SUB above)
        WAVES * O_K_SUB,
        tokens * (XN8_ROW + XSC_ROW + N_ROUTED) + WAVES * 64 * 4,
        rows * (MID_ROW + MIDSC_ROW),
    )
