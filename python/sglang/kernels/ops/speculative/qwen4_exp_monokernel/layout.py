# SPDX-License-Identifier: Apache-2.0
"""Compile-time geometry and scratch (mailbox) layout of the Qwen3.8 multi-request speculative-verify monokernel.

A launch serves R requests x L verify tokens = S rows (S <= 32). All activation mailboxes are row-major: [S][n],
row t = request t // L, draft step t % L."""

from sglang.kernels.ops.speculative.qwen4_exp_monokernel.weights import (
    GDN_HV,
    GDN_QKVZ,
    HC_DIM,
    HC_LOWRANK,
    HIDDEN,
    MOE_INTER,
    N_EXPERTS,
    TOPK,
)

BLOCKS = 256  # one CTA per MI355X CU; all CTAs must be co-resident
THREADS = 512
WAVES = THREADS // 64
CM_DEV = 16  # sc1: device-coherent buffer load/store
MAX_T = 32  # rows (requests x verify tokens) per launch
MAX_R = 8  # requests per launch
MOE_PAIR = TOPK + 1  # (row, expert) pairs per row: top-10 routed + the shared expert

# hyper-connection mix
MIX_KSPLIT = 10  # down GEMV split-K across CTAs (1024-wide K slices)
MIX_DOWN_RG = HC_LOWRANK // 16 + 1  # 20 row tiles + the combine-gate tile
MIX_DOWN_TASKS = MIX_DOWN_RG * MIX_KSPLIT  # 210
MIX_UP_TASKS = HIDDEN // 16  # 160 (each covers 16 outputs x 4 branches)
NPART = (
    HIDDEN // 16
)  # sum-of-squares partials per (token, branch) published with a residual stream (row groups)

# GDN
IP_ROWS = GDN_QKVZ + 2 * GDN_HV  # 16480: [q k v | z | b | a]
GDN_QKVZ_ROWS = GDN_QKVZ  # packed FP8 in_proj rows [q k v | z]
IP_RG_MAIN = (
    GDN_QKVZ // 16
)  # 1024 = 4 per CTA; the b/a rows are computed by the core tasks
MAX_CORE_SPLIT = 8


def core_split(T):
    """V-row splits per value head of the GDN core: more CTAs for more tokens (the per-step state writes
    of a task scale with T and are bound by the CTA's store bandwidth)."""
    return 4  # 8 would need 384 CTAs


def core_geom(split):
    rows = 128 // split
    rpt = 512 // rows  # threads per V row
    kpt = 128 // rpt  # K elements (state words) per thread
    red_offs = tuple(
        rpt >> k for k in range(1, rpt.bit_length())
    )  # xor offsets of a row reduce
    return rows, GDN_HV * split, rpt, kpt, red_offs


OUT_RG = HIDDEN // 16  # 160

# MoE
SHARED_EXPERT = N_EXPERTS  # index of the shared expert in the stacked arrays
ROUTER_ROWS = N_EXPERTS + 16  # 512 routed logits + shared gate + 15 zero rows
ROUTER_RG = ROUTER_ROWS // 16  # 33
# MXFP4 experts in sglang's AITER layout. get_moe_weight_sizes(is_aiter_moe) pads the intermediate dim to the AITER
# alignment (w2 row 320 -> 384 bytes): FusedMoE w13 [513, 2 * 768, 1280] (gate rows 0..639, zero rows to 767, up rows
# 768..1407, zero rows to 1535), w2 [513, 2560, 384] (K 640..767 zero); after process_weights_after_loading
# shuffle_weight(w, (16, 16)) and scales e8m0_shuffle'd over the flattened [E * rows, groups] (groups padded to 8).
# The kernel reads only the real rows / K (gate tiles 0..39, up tiles 48..87, down K chunks 0..4).
MOE_INTER_PAD = 768
GU_PAIRS = (
    MOE_INTER // 16
)  # 40 MXFP4 gate/up tasks per expert: gate tile n + up tile n + UP_TILE0
UP_TILE0 = MOE_INTER_PAD // 16  # 48: first up row tile of w13
DN_RG = HIDDEN // 16  # 160
GU_BYTES = 2 * MOE_INTER_PAD * HIDDEN // 2  # fp4 bytes per expert (w13)
GU_SGROUPS = HIDDEN // 32  # 80
GU_SBYTES = 2 * MOE_INTER_PAD * GU_SGROUPS
DN_BYTES = HIDDEN * MOE_INTER_PAD // 2  # w2: 384-byte rows
DN_SGROUPS = (MOE_INTER_PAD // 32 + 7) // 8 * 8  # 24
DN_SBYTES = HIDDEN * DN_SGROUPS
DN_SG = 4  # expert slots per down K-group (partials summed by the combine)


def max_slots(T):
    """Union of routed experts of T tokens + the shared expert."""
    return min(TOPK * T, N_EXPERTS) + 1


def max_groups(T, sg=DN_SG):
    return (max_slots(T) + sg - 1) // sg


def kphases(T):
    """K phases of the K=2560 activation GEMMs: [T][2560] bf16 rows fit the LDS for T <= 16; above, the CTA stages
    the activation in two K halves (accumulators of all its row tiles kept across the halves)."""
    return 1 if T <= 16 else 2


def gu_kphases(T):
    """K halves of the MoE gate/up: the staged [T][2560] bf16 rows + the route tables fit the LDS for T <= 28; above,
    a CTA stages one K half and publishes f32 gate/up partials (summed by the down staging)."""
    return 1 if T <= 28 else 2


def dn_sg(T):
    """Expert slots per MoE down K-group: the staged [T][SG * 640] bf16 mid of a group must fit the LDS."""
    return 4 if T <= 28 else 3


# QSA (full attention) layer
QP_ROWS = (
    24 * 512 + 512 + 512 + 640
)  # [q|gate per head | k | v | index q (4x128) + index k (128)]
QP_RG = QP_ROWS // 16  # 872
QP_K_OFF = 24 * 512
QP_V_OFF = QP_K_OFF + 512
QP_IDX_OFF = QP_V_OFF + 512
SEG_BLOCKS = 8192  # top-512 selection: exact per-segment top-512 (16 blocks / thread), then a merge
NSEG_MAX = 9
MAX_BLOCKS = SEG_BLOCKS * NSEG_MAX  # 73728 compressed blocks
MAX_CTX = MAX_BLOCKS * 4  # 294912 >= max_position_embeddings (262144) + verify tokens
ATT_CHUNK = 64  # tokens per attention split
ATT_CHUNKS = 2048 // ATT_CHUNK + 1  # 32 block chunks + 1 tail chunk
ATT_TASKS = ATT_CHUNKS  # per token; one task covers both kv heads
PART_STRIDE = 24 * 256 + 2 * 24  # per task: o[24][256], m[24], l[24]
MERGE_TASKS = 24  # one per (token, head)
PREP_Q_TASKS = 24  # CTAs [0, 24): one attention q head each (all tokens)
PREP_K_CTA = 24  # CTAs 24, 25: k head (norm+rope) + v -> KV cache
PREP_IDX_CTA0 = 26  # CTAs [26, 26 + R): index q + index k ring / compress of request r (its tokens, in order)


def sel2_cta0(T):
    """CTAs [256 - T, 256): merge of the segment top-512s of row t (long contexts)."""
    return 256 - T


SEL_STRIDE = (
    1026  # pairs per token: (block, loc) pairs as 2 pairs each, count at pair 1024
)


def cnew_slots(L):
    """Compressed blocks completed inside one request's L verify tokens."""
    return L // 4 + 2


# per-layer pointer table slots
P_HCA_NORM, P_HCA_DOWN, P_HCA_UP, P_HCA_INJ = 0, 1, 2, 3
P_IP, P_CONV_W, P_NEG_A, P_DT_BIAS, P_NORM_W, P_OUT = 4, 5, 6, 7, 8, 9
P_CONV_STATE, P_SSM_STATE = 10, 11
P_HCM_NORM, P_HCM_DOWN, P_HCM_UP, P_HCM_INJ = 12, 13, 14, 15
P_ROUTER, P_GU, P_GU_S, P_DN, P_DN_S = 16, 17, 18, 19, 20
# QSA reuses P_IP (fused q/k/v/index projection) and P_OUT (o_proj)
P_QNORM, P_KNORM, P_IQNORM, P_IKNORM = 5, 6, 7, 8
P_KBUF, P_VBUF, P_RING, P_COMP = 10, 11, 21, 22
P_BA = 23  # GDN: row-major [b (48) ; a (48)] x HIDDEN bf16
# PLE (layer 1 only): fused into the start of the launch whose first layer carries it
P_PLE_W, P_PLE_NK, P_PLE_NQ, P_PLE_NC, P_PLE_CW, P_PLE_STATE = 24, 25, 26, 27, 28, 29
P_ISSM, P_ICONV = (
    30,
    31,
)  # GDN verify scratch: per-draft-step SSM states / conv windows (batch row 0)
P_PLE_ISTATE = 21  # PLE verify scratch: per-draft-step short-conv windows (GDN rows do not use P_RING)
PLE_ROWS = HC_DIM + HIDDEN  # [key_proj (10240) ; value_proj (2560)]
PLE_RG = PLE_ROWS // 16  # 800
PLE_STATE_LEN = 9
PLE_STATE_SLOT_BYTES = HC_DIM * PLE_STATE_LEN * 2
PLE_CH_PER_CTA = HC_DIM // 256  # 40
PLE_PARTS = HIDDEN // PLE_CH_PER_CTA  # 64 ple_h producers per branch
N_PTR = 40
# FP8 projections: F32 per-output-channel weight scales
P_IP_S, P_OUT_S, P_BA_S = 32, 33, 34
# QSA: sglang keeps qkv_proj and indexer.index_qk_proj as separate FP8 params (rows [0, 832 * 16) / the rest of QP)
P_IP2, P_IP2_S = 35, 36
QP_RG_QKV = (24 * 512 + 512 + 512) // 16  # 832
BA_ROWS = (
    128  # sglang pads in_proj_ba (96 rows) to 128 for AITER's CK GEMM (quark_w8a8_fp8)
)
# global row (index n_layers of the table)
(
    G_POS,
    G_LOC,
    G_REQ,
    G_SEQ,
    G_RTT,
    G_COS,
    G_ROPEPOS,
    G_MAMBA_MAP,
    G_TIMELINE,
    G_PLE_EMB,
) = 0, 1, 2, 3, 4, 5, 6, 7, 8, 9
G_FIN_OUT = 10  # final mixer output x [T, HIDDEN] bf16 (final_mix launches)
G_HIN_SSQ, G_HOUT_SSQ = (
    11,
    12,
)  # f32 [T, 4, NPART] sum-of-squares partials of h_in / h_out
G_INPUT_IDS, G_EMBED_W, G_NGRAM_W, G_NGRAM_SCALE, G_NGRAM_CTX, G_NGRAM_META = (
    13,
    14,
    15,
    16,
    17,
    18,
)
G_NGRAM_ICTX = 19  # int64 [T, 2] per-draft-step n-gram history (batch row 0 of intermediate_context)
# MTP draft launches (first_src "mtp"): input fusion of the draft model
G_MTP_HIN = 20  # bf16 [T, HC_DIM] target (or previous draft step) hyper-connection hidden states
G_MTP_PE, G_MTP_PH = (
    21,
    22,
)  # pre_fc_norm_embedding [HIDDEN], pre_fc_norm_hidden [HC_DIM] (bf16)
G_MTP_FCE, G_MTP_FCH = 23, 24  # packed bf16 fc_embedding / fc_hidden [HIDDEN, HIDDEN]
NGRAM_HEADS, NGRAM_DIM, HEADS_PER_NGRAM = 16, 160, 8
PRELUDE_CTA0 = 224  # CTAs [224, 224 + T): PLE n-gram rows of verify row t
TL_STAMPS = 160  # timeline stamps per CTA (debug builds)
CONV_SLOT_BYTES = 10240 * 3 * 2  # bf16 [10240, 3] per mamba slot
SSM_SLOT_BYTES = 48 * 128 * 128 * 4  # f32 [48, 128, 128] per mamba slot
SSM_STEP_WORDS = 48 * 128 * 128
LAUNCH_SLOTS = 64  # tag = step * LAUNCH_SLOTS + launch_id + 1


def _align(n, a=256):
    return (n + a - 1) // a * a


def scratch_layout(n_layers: int, T: int, R: int = 1):
    """Byte offsets of every tagged-pair mailbox; one region per (layer, stage). T = rows, R = requests."""
    regions = {}
    off = 0

    def add(name, pairs):
        nonlocal off
        regions[name] = off
        off += _align(pairs * 8)

    for layer in range(n_layers):
        p = f"L{layer}."
        add(p + "qp", T * QP_ROWS // 2)
        add(p + "q", T * 24 * 256 // 2)
        add(p + "kvnew", T * 1024 // 2)
        add(p + "qi", T * 512 // 2)
        add(p + "cnew", R * cnew_slots(T // R) * 64)  # [R][CN][64]
        add(p + "ilog", T * MAX_BLOCKS)
        add(p + "sel", T * SEL_STRIDE)
        add(
            p + "cand_k", T * NSEG_MAX * 512
        )  # per (token, segment): key bits of the segment's top-512
        add(p + "cand_bl", T * NSEG_MAX * 512 * 2)  # (block, KV loc)
        add(p + "part", T * ATT_TASKS * PART_STRIDE)
        add(p + "og", T * 24 * 256 // 2)
        add(p + "ple_kv", T * PLE_ROWS // 2)
        add(p + "ple_h", T * HC_DIM // 2)
        add(p + "ple_ssq", T * 4 * PLE_PARTS)
        for hc in ("a", "m"):
            add(p + hc + ".mixd", MIX_KSPLIT * T * HC_LOWRANK)
            add(p + hc + ".mixx", T * HIDDEN // 2)  # bf16 pairs
            add(p + hc + ".injp", MIX_KSPLIT * T * 4)
        add(p + "ip", T * IP_ROWS // 2)  # bf16 pairs
        add(
            p + "ssq", T * GDN_HV * MAX_CORE_SPLIT
        )  # per (token, head, split) sum of squares of the GDN output
        add(p + "og_amax", T * GDN_HV * MAX_CORE_SPLIT)
        add(p + "dbg_out", T * HIDDEN)
        add(
            p + "dbg_asc", T
        )  # debug builds: attention / GDN block output before the combine  # per (row, producer task) max |og| (FP8 activation scale)
        add(p + "h1", T * HC_DIM // 2)  # bf16 pairs
        add(p + "h1ssq", T * 4 * NPART)
        add(p + "logit", T * ROUTER_ROWS)  # f32
        # (row, expert) pairs p = t * 11 + rank: bf16 h [p][640]; gu_kphases 2: + f32 K-half-0 partials [p][1280]
        add(
            p + "mid", MOE_PAIR * T * MOE_INTER // 2
        )  # bf16 h = silu(g) * u of the routed pairs
        if gu_kphases(T) == 2:
            add(
                p + "gupart", MOE_PAIR * T * 2 * MOE_INTER
            )  # f32 K-half-0 gate/up partials
        add(p + "dpart", max_groups(T, dn_sg(T)) * T * HIDDEN)  # f32 [group][T][2560]
        add(p + "h2", T * HC_DIM // 2)  # bf16 pairs
        add(p + "h2ssq", T * 4 * NPART)
        add(
            p + "mtp_h", T * HC_DIM // 2
        )  # MTP input fusion output (layer-0 input of a draft launch)
        add(p + "mtp_ssq", T * 4 * NPART)
    return regions, off
