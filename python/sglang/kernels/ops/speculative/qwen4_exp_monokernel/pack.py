# SPDX-License-Identifier: Apache-2.0
"""Host-side weight packing + pointer tables for the Qwen3.8 monokernel.

The MXFP4 experts and the FP8 attention / GDN projections are sglang's own params (AITER layouts), read in place;
only the small bf16 tensors sglang does not keep in the kernel's tile layout (hyper-connection mixers, router, PLE,
MTP input fusion) are packed here."""

import torch

from sglang.kernels.ops.speculative.qwen4_exp_monokernel import layout as LY
from sglang.kernels.ops.speculative.qwen4_exp_monokernel.weights import (
    HIDDEN,
    N_EXPERTS,
)


def pack_bf16(w: torch.Tensor) -> torch.Tensor:
    """Pack BF16 ``[N, K]`` for the kernel's 16-row, 64-K MFMA tiles."""
    rows, k = w.shape
    if rows % 16 or k % 64:
        raise ValueError(
            f"BF16 matrix dimensions must be divisible by (16, 64), got {(rows, k)}"
        )
    w16 = w.view(torch.int16).reshape(rows // 16, 16, k // 64, 2, 4, 8)
    return w16.permute(0, 2, 3, 4, 1, 5).contiguous().view(-1)


def check_moe_params(m):
    """Shape / dtype / layout asserts on the (sglang) MXFP4 MoE params the kernel reads."""
    E = N_EXPERTS + 1
    assert m["gu"].dtype == torch.uint8 and tuple(m["gu"].shape) == (
        E,
        2 * LY.MOE_INTER_PAD,
        HIDDEN // 2,
    ), m["gu"].shape
    assert m["dn"].dtype == torch.uint8 and tuple(m["dn"].shape) == (
        E,
        HIDDEN,
        LY.MOE_INTER_PAD // 2,
    ), m["dn"].shape
    assert tuple(m["gu_s"].shape) == (E, 2 * LY.MOE_INTER_PAD, LY.GU_SGROUPS), m[
        "gu_s"
    ].shape
    assert tuple(m["dn_s"].shape) == (E, HIDDEN, LY.DN_SGROUPS), m["dn_s"].shape
    for k in ("gu", "dn", "gu_s", "dn_s"):
        assert m[k].is_contiguous() and m[k].element_size() == 1, k
    assert m["gu"].numel() == E * LY.GU_BYTES and m["dn"].numel() == E * LY.DN_BYTES
    assert (
        m["gu_s"].numel() == E * LY.GU_SBYTES and m["dn_s"].numel() == E * LY.DN_SBYTES
    )


def pack_hc_down(w_down, w_inj=None):
    """Mix down weights [320, HC_DIM] + one extra 16-row tile whose rows 0..3 are the combine-gate (inject)
    rows: the down GEMM of a K-slice also yields the gate partials."""
    extra = torch.zeros(16, w_down.shape[1], dtype=w_down.dtype, device=w_down.device)
    if w_inj is not None:
        extra[: w_inj.shape[0]] = w_inj
    return pack_bf16(torch.cat([w_down, extra], dim=0).contiguous())


def pack_hc(lw, name):
    p = f"{name}."
    return {
        "norm": lw[p + "hc_norm.weight"].contiguous(),
        "down": pack_hc_down(
            lw[p + "input_mix_weight_down.weight"], lw[p + "block_inject_weight.weight"]
        ),
        "up": pack_bf16(lw[p + "input_mix_weight_up.weight"].contiguous()),
        "inj": lw[p + "block_inject_weight.weight"].contiguous(),
    }


def pack_gdn_layer(lw, moe, proj):
    """moe / proj: sglang's FusedMoE params / FP8 projection params {"ip", "out", "ba": (weight, scale)} of the
    layer (AITER layouts, read in place)."""
    a = "linear_attn."
    (ip, ip_s), (out, out_s), (ba, ba_s) = proj["ip"], proj["out"], proj["ba"]
    router = torch.cat(
        [
            lw["mlp.gate.weight"],
            lw["mlp.shared_expert_gate.weight"],
            torch.zeros(15, HIDDEN, dtype=torch.bfloat16, device=ip.device),
        ],
        dim=0,
    )
    t = {
        "hca": pack_hc(lw, "attn_hyper_connection"),
        "hcm": pack_hc(lw, "mlp_hyper_connection"),
        "ip": ip,
        "ip_s": ip_s,
        "conv_w": lw[a + "conv1d.weight"].squeeze(1).contiguous(),
        "neg_a": (-torch.exp(lw[a + "A_log"].float())).contiguous(),
        "dt_bias": lw[a + "dt_bias"].float().contiguous(),
        "norm_w": lw[a + "norm.weight"].float().contiguous(),
        "out": out,
        "out_s": out_s,
        # b/a rows stay row-major FP8 (the GDN core dots them with the per-token FP8 activation) + their scales
        "ba": ba,
        "ba_s": ba_s,
        "router": pack_bf16(router.contiguous()),
        "moe": _moe(moe),
    }
    return t


def gdn_ptr_row(t, conv_state, ssm_state):
    """Pointer table row for one GDN layer; conv/ssm state = base of the [slots, ...] pools of this layer."""
    row = [0] * LY.N_PTR
    row[LY.P_HCA_NORM] = t["hca"]["norm"].data_ptr()
    row[LY.P_HCA_DOWN] = t["hca"]["down"].data_ptr()
    row[LY.P_HCA_UP] = t["hca"]["up"].data_ptr()
    row[LY.P_HCA_INJ] = t["hca"]["inj"].data_ptr()
    row[LY.P_IP] = t["ip"].data_ptr()
    row[LY.P_CONV_W] = t["conv_w"].data_ptr()
    row[LY.P_NEG_A] = t["neg_a"].data_ptr()
    row[LY.P_DT_BIAS] = t["dt_bias"].data_ptr()
    row[LY.P_NORM_W] = t["norm_w"].data_ptr()
    row[LY.P_OUT] = t["out"].data_ptr()
    row[LY.P_CONV_STATE] = conv_state.data_ptr()
    row[LY.P_SSM_STATE] = ssm_state.data_ptr()
    row[LY.P_HCM_NORM] = t["hcm"]["norm"].data_ptr()
    row[LY.P_HCM_DOWN] = t["hcm"]["down"].data_ptr()
    row[LY.P_HCM_UP] = t["hcm"]["up"].data_ptr()
    row[LY.P_HCM_INJ] = t["hcm"]["inj"].data_ptr()
    row[LY.P_ROUTER] = t["router"].data_ptr()
    row[LY.P_GU] = t["moe"]["gu"].data_ptr()
    row[LY.P_GU_S] = t["moe"]["gu_s"].data_ptr()
    row[LY.P_DN] = t["moe"]["dn"].data_ptr()
    row[LY.P_DN_S] = t["moe"]["dn_s"].data_ptr()
    row[LY.P_BA] = t["ba"].data_ptr()
    row[LY.P_IP_S] = t["ip_s"].data_ptr()
    row[LY.P_OUT_S] = t["out_s"].data_ptr()
    row[LY.P_BA_S] = t["ba_s"].data_ptr()
    return row


def pack_qsa_layer(lw, moe, proj):
    """proj: sglang's FP8 params {"qp": qkv_proj, "qp2": indexer.index_qk_proj, "out": o_proj} as (weight, scale)."""
    a = "self_attn."
    (qp, qp_s), (qp2, qp2_s), (out, out_s) = proj["qp"], proj["qp2"], proj["out"]
    router = torch.cat(
        [
            lw["mlp.gate.weight"],
            lw["mlp.shared_expert_gate.weight"],
            torch.zeros(15, HIDDEN, dtype=torch.bfloat16, device=qp.device),
        ],
        dim=0,
    )
    return {
        "hca": pack_hc(lw, "attn_hyper_connection"),
        "hcm": pack_hc(lw, "mlp_hyper_connection"),
        "qp": qp,
        "qp_s": qp_s,
        "qp2": qp2,
        "qp2_s": qp2_s,
        "out_s": out_s,
        "qnorm": lw[a + "q_norm.weight"].contiguous(),
        "knorm": lw[a + "k_norm.weight"].contiguous(),
        "iqnorm": lw[a + "indexer.q_layernorm.weight"].contiguous(),
        "iknorm": lw[a + "indexer.k_layernorm.weight"].contiguous(),
        "out": out,
        "router": pack_bf16(router.contiguous()),
        "moe": _moe(moe),
    }


def _moe(moe):
    check_moe_params(moe)
    return moe


def _common_rows(row, t):
    row[LY.P_HCA_NORM] = t["hca"]["norm"].data_ptr()
    row[LY.P_HCA_DOWN] = t["hca"]["down"].data_ptr()
    row[LY.P_HCA_UP] = t["hca"]["up"].data_ptr()
    row[LY.P_HCA_INJ] = t["hca"]["inj"].data_ptr()
    row[LY.P_HCM_NORM] = t["hcm"]["norm"].data_ptr()
    row[LY.P_HCM_DOWN] = t["hcm"]["down"].data_ptr()
    row[LY.P_HCM_UP] = t["hcm"]["up"].data_ptr()
    row[LY.P_HCM_INJ] = t["hcm"]["inj"].data_ptr()
    row[LY.P_ROUTER] = t["router"].data_ptr()
    row[LY.P_GU] = t["moe"]["gu"].data_ptr()
    row[LY.P_GU_S] = t["moe"]["gu_s"].data_ptr()
    row[LY.P_DN] = t["moe"]["dn"].data_ptr()
    row[LY.P_DN_S] = t["moe"]["dn_s"].data_ptr()


def qsa_ptr_row(t, k_buf, v_buf, ring, comp):
    """Pointer table row for one QSA layer (this layer's KV / index-ring / compressed buffers)."""
    for b in (k_buf, v_buf):  # addressed with 64-bit pointers (any pool size)
        assert b.is_contiguous() and b.shape[-2:] == (2, 256)
    for b in (ring, comp):
        assert b.is_contiguous() and b.numel() * b.element_size() < 2**32, (
            "buffer offsets are 32-bit"
        )
    row = [0] * LY.N_PTR
    _common_rows(row, t)
    row[LY.P_IP] = t["qp"].data_ptr()
    row[LY.P_OUT] = t["out"].data_ptr()
    row[LY.P_IP_S] = t["qp_s"].data_ptr()
    row[LY.P_IP2] = t["qp2"].data_ptr()
    row[LY.P_IP2_S] = t["qp2_s"].data_ptr()
    row[LY.P_OUT_S] = t["out_s"].data_ptr()
    row[LY.P_QNORM] = t["qnorm"].data_ptr()
    row[LY.P_KNORM] = t["knorm"].data_ptr()
    row[LY.P_IQNORM] = t["iqnorm"].data_ptr()
    row[LY.P_IKNORM] = t["iknorm"].data_ptr()
    row[LY.P_KBUF] = k_buf.data_ptr()
    row[LY.P_VBUF] = v_buf.data_ptr()
    row[LY.P_RING] = ring.data_ptr()
    row[LY.P_COMP] = comp.data_ptr()
    return row


def global_ptr_row(
    positions,
    out_cache_loc,
    req_pool_idx,
    seq_lens,
    req_to_token,
    cos_sin,
    rope_pos,
    mamba_map,
):
    assert cos_sin.dtype == torch.bfloat16 and req_to_token.dtype == torch.int32
    row = [0] * LY.N_PTR
    row[LY.G_MAMBA_MAP] = mamba_map.data_ptr()
    row[LY.G_POS] = positions.data_ptr()
    row[LY.G_LOC] = out_cache_loc.data_ptr()
    row[LY.G_REQ] = req_pool_idx.data_ptr()
    row[LY.G_SEQ] = seq_lens.data_ptr()
    row[LY.G_RTT] = req_to_token.data_ptr()
    row[LY.G_COS] = cos_sin.data_ptr()
    row[LY.G_ROPEPOS] = rope_pos.data_ptr()
    return row


def pack_ple(lw):
    """PLE weights of layer 1: packed [key_proj; value_proj] + norms + dilated conv weight."""
    p = "ple."
    w = torch.cat(
        [lw[p + "key_proj.weight"], lw[p + "value_proj.weight"]], dim=0
    ).contiguous()
    return {
        "w": pack_bf16(w),
        "nk": lw[p + "norm_key.weight"].contiguous(),
        "nq": lw[p + "norm_query.weight"].contiguous(),
        "nc": lw[p + "norm_conv.weight"].contiguous(),
        "cw": lw[p + "conv1d.weight"].squeeze(1).contiguous(),
    }


def add_ple_ptrs(row, t, state):
    """Add PLE pointers (and the [slots, 10240, 9] short-conv state base) to a layer row."""
    assert (
        state.dtype == torch.bfloat16
        and state.is_contiguous()
        and state[0].numel() * 2 == LY.PLE_STATE_SLOT_BYTES
    )
    row[LY.P_PLE_W] = t["w"].data_ptr()
    row[LY.P_PLE_NK] = t["nk"].data_ptr()
    row[LY.P_PLE_NQ] = t["nq"].data_ptr()
    row[LY.P_PLE_NC] = t["nc"].data_ptr()
    row[LY.P_PLE_CW] = t["cw"].data_ptr()
    row[LY.P_PLE_STATE] = state.data_ptr()
    return row


def pack_mtp(lw, moe, proj):
    """MTP draft head: QSA layer 0 (FP8 projections, MXFP4 routed + shared experts), input fusion, final mixer."""
    p = "layers.0."
    a = p + "self_attn."
    (qp, qp_s), (qp2, qp2_s), (out, out_s) = proj["qp"], proj["qp2"], proj["out"]
    router = torch.cat(
        [
            lw[p + "mlp.gate.weight"],
            lw[p + "mlp.shared_expert_gate.weight"],
            torch.zeros(15, HIDDEN, dtype=torch.bfloat16, device=qp.device),
        ],
        dim=0,
    )
    sub = {k[len(p) :]: v for k, v in lw.items() if k.startswith(p)}
    layer = {
        "hca": pack_hc(sub, "attn_hyper_connection"),
        "hcm": pack_hc(sub, "mlp_hyper_connection"),
        "qp": qp,
        "qp_s": qp_s,
        "qp2": qp2,
        "qp2_s": qp2_s,
        "qnorm": lw[a + "q_norm.weight"].contiguous(),
        "knorm": lw[a + "k_norm.weight"].contiguous(),
        "iqnorm": lw[a + "indexer.q_layernorm.weight"].contiguous(),
        "iknorm": lw[a + "indexer.k_layernorm.weight"].contiguous(),
        "out": out,
        "out_s": out_s,
        "router": pack_bf16(router.contiguous()),
        "moe": _moe(moe),
    }
    return {
        "layer": layer,
        "pe": lw["pre_fc_norm_embedding.weight"].contiguous(),
        "ph": lw["pre_fc_norm_hidden.weight"].contiguous(),
        "fce": pack_bf16(lw["fc_embedding.weight"].contiguous()),
        "fch": pack_bf16(lw["fc_hidden.weight"].contiguous()),
        "final": {
            "norm": lw["hyper_connection_mixer.hc_norm.weight"].contiguous(),
            "down": pack_hc_down(
                lw["hyper_connection_mixer.input_mix_weight_down.weight"]
            ),
            "up": pack_bf16(
                lw["hyper_connection_mixer.input_mix_weight_up.weight"].contiguous()
            ),
        },
    }


def mtp_rows(m, k_buf, v_buf, ring, comp, grow):
    """Pointer rows of a draft launch: [QSA layer row, global row (+ fusion pointers), final-mixer row]."""
    row = qsa_ptr_row(m["layer"], k_buf, v_buf, ring, comp)
    grow[LY.G_MTP_PE] = m["pe"].data_ptr()
    grow[LY.G_MTP_PH] = m["ph"].data_ptr()
    grow[LY.G_MTP_FCE] = m["fce"].data_ptr()
    grow[LY.G_MTP_FCH] = m["fch"].data_ptr()
    frow = [0] * LY.N_PTR
    frow[LY.P_HCA_NORM] = m["final"]["norm"].data_ptr()
    frow[LY.P_HCA_DOWN] = m["final"]["down"].data_ptr()
    frow[LY.P_HCA_UP] = m["final"]["up"].data_ptr()
    frow[LY.P_HCA_INJ] = m["layer"]["hca"]["inj"].data_ptr()  # combine gate unused
    return [row, grow, frow]


def unshuffle16(w):
    """aiter shuffle_weight(w, (16, 16)) inverse for 1-byte elements: [N, K] storage -> row-major [N, K] bytes."""
    w = w.view(torch.uint8)
    rows, kb = w.shape
    return (
        w.reshape(rows // 16, kb // 32, 2, 16, 16)
        .permute(0, 3, 1, 2, 4)
        .reshape(rows, kb)
    )


def unshuffle_moe_expert(m, e):
    """Expert e of the AITER-layout MoE params back to the checkpoint layout, real rows / K only:
    -> (gate [640, 1280], up [640, 1280], down [2560, 320], gate_s [640, 80], up_s [640, 80], down_s [2560, 20])."""
    I, IP = LY.MOE_INTER, LY.MOE_INTER_PAD

    w_un = unshuffle16

    def s_un(
        s,
    ):  # e8m0_shuffle inverse: [a, d, f, c, e, b] -> [32 a + 16 b + c, 8 d + 4 e + f]
        rows, g = s.shape
        return (
            s.reshape(rows // 32, g // 8, 4, 16, 2, 2)
            .permute(0, 5, 3, 1, 4, 2)
            .reshape(rows, g)
        )

    gu, gus = w_un(m["gu"][e]), s_un(m["gu_s"][e])
    dn, dns = w_un(m["dn"][e]), s_un(m["dn_s"][e])
    return (
        gu[:I],
        gu[IP : IP + I],
        dn[:, : I // 2],
        gus[:I, :],
        gus[IP : IP + I, :],
        dns[:, : I // 32],
    )
