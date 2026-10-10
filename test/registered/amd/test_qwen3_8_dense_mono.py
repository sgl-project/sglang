"""Bit-exactness of the Qwen3.8-27B dense mono decode kernels (ROCm / FlyDSL,
gfx950) against the stock decode path, on random MXFP4 weights.

``DenseFfnTest`` (K2, ``dense_ffn``): out_proj / o_proj + the residual add +
post_attention_layernorm + gate_up, SiLU-mul and down, against dynamic_mxfp4_quant
+ Triton ``gemm_afp4wfp4``, the ROCm Triton Gemma norm and SiluAndMul.

``DenseGdnTest`` (K1, ``dense_gdn``): the residual add + input_layernorm ->
in_proj_qkvz / in_proj_ba -> conv1d -> the gated delta rule -> the gated norm,
against the ROCm Triton Gemma norm, the MXFP4 linears, ``causal_conv1d_update``,
``fused_recurrent_gated_delta_rule_packed_decode`` and RMSNormGated. The core,
the residual and both states must match, on the first layer (no residual) and
with pad rows (slot -1); other slots stay untouched.

Widths 1..8, each with its own weights and Mailbox layer tag, over many input
draws: a norm reduced in another order than the stock kernel's still matches on
most rows, and shows on about one row in 640. Run directly with
``python test/registered/amd/test_qwen3_8_dense_mono.py``.
"""

import unittest

import torch

from sglang.test.ci.ci_register import register_amd_ci
from sglang.test.test_utils import CustomTestCase

register_amd_ci(est_time=600, suite="stage-b-test-1-gpu-small-amd-mi35x")

BF = torch.bfloat16
EPS = 1e-6
WIDTHS = range(1, 9)
FFN_DRAWS = 512
GDN_DRAWS = 512
# Mailbox tag slots: the layer index a launch passes as ``layer``.
LAYER_TAGS = (0, 17, 63)


def _gfx950() -> bool:
    return (
        torch.cuda.is_available()
        and torch.version.hip is not None
        and "gfx950" in torch.cuda.get_device_properties(0).gcnArchName
    )


def _mxfp4(g, n, k):
    """Random MXFP4 [n, k]: fp4 pairs and e8m0 scales 2^-7 .. 2^-1."""
    w = torch.randint(
        0, 256, (n, k // 2), dtype=torch.uint8, device="cuda", generator=g
    )
    ws = torch.randint(
        120, 127, (n, k // 32), dtype=torch.uint8, device="cuda", generator=g
    )
    return w, ws


def _rnd(g, *shape, scale=1.0, dtype=BF):
    return (torch.randn(*shape, generator=g, device="cuda") * scale).to(dtype)


def _linear(x, w, ws):
    from sglang.srt.layers.quantization.quark.schemes import quark_w4a4_mxfp4 as q

    xq, xs = q.dynamic_mxfp4_quant(x)
    y = torch.empty(x.size(0), w.size(0), dtype=BF, device=x.device)
    q.gemm_afp4wfp4(xq, w, xs, ws, BF, y)
    return y


def _gemma_norm(x, residual, w):
    """GemmaRMSNorm as the stock path runs it on ROCm with aiter -> x, residual."""
    from sglang.kernels.ops.layernorm import minimax_m3_rmsnorm as M

    if residual is None:
        return M.gemma_rmsnorm(x, w, EPS), x
    return M.gemma_fused_add_rmsnorm(x, residual, w, EPS)


@unittest.skipUnless(_gfx950(), "needs FlyDSL and aiter on gfx950")
class DenseFfnTest(CustomTestCase):
    def test_bit_exact(self):
        from sglang.kernels.ops.moe.qwen3_5_mono_flydsl import dense_ffn as D
        from sglang.srt.layers.activation import SiluAndMul

        act = SiluAndMul()
        for s in WIDTHS:
            g = torch.Generator(device="cuda").manual_seed(s)
            layer = LAYER_TAGS[s % len(LAYER_TAGS)]
            w_o, w_os = _mxfp4(g, D.HIDDEN, D.CORE)
            w13, w13s = _mxfp4(g, 2 * D.INTER, D.HIDDEN)
            w2, w2s = _mxfp4(g, D.HIDDEN, D.INTER)
            ln_w = _rnd(g, D.HIDDEN, scale=0.1)
            weights = dict(
                w_o=D.shuffle_w(w_o),
                w_os=w_os,
                ln_w=ln_w,
                w13=D.shuffle_w(w13),
                w13s=w13s,
                w2=D.shuffle_w(w2),
                w2s=w2s,
            )
            key = D.FfnBuild(tokens=s, eps=EPS, shuffled=True, oproj=True)
            out = torch.empty(s, D.HIDDEN, dtype=BF, device="cuda")
            res_out = torch.empty_like(out)
            epoch = torch.zeros(1, dtype=torch.int32, device="cuda")
            # One scratch for every launch: the epoch tags must keep it valid.
            scratch = torch.zeros(
                D.scratch_bytes(key), dtype=torch.uint8, device="cuda"
            )
            bad = []
            for draw in range(FFN_DRAWS):
                core = _rnd(g, s, D.CORE, scale=0.5)
                residual = _rnd(g, s, D.HIDDEN, scale=2.0)
                x, res_ref = _gemma_norm(_linear(core, w_o, w_os), residual, ln_w)
                out_ref = _linear(act(_linear(x, w13, w13s)), w2, w2s)
                epoch.add_(1)
                D.dense_ffn(
                    key,
                    hidden=core,
                    residual=residual,
                    out=out,
                    res_out=res_out,
                    scratch=scratch,
                    epoch=epoch,
                    layer=layer,
                    **weights,
                )
                if not (torch.equal(res_out, res_ref) and torch.equal(out, out_ref)):
                    bad.append(draw)
            with self.subTest(s=s, layer=layer):
                self.assertEqual(bad, [])


def _gdn_weights(g, K):
    from sglang.kernels.ops.attention.fla.layernorm_gated import RMSNorm as RMSNormGated

    w = {
        "ln_w": _rnd(g, K.HIDDEN, scale=0.1),
        "conv_w": _rnd(g, K.CONV, 1, K.CONV_W, scale=0.4),
        "a_log": _rnd(g, K.NV, scale=0.5, dtype=torch.float32),
        "dt_bias": _rnd(g, K.NV, scale=0.5),
        "norm_w": (1 + _rnd(g, K.HD, scale=0.1, dtype=torch.float32)).to(BF),
    }
    w["w_qkvz"], w["w_qkvzs"] = _mxfp4(g, K.QKVZ, K.HIDDEN)
    w["w_ba"], w["w_bas"] = _mxfp4(g, K.BA, K.HIDDEN)
    gn = RMSNormGated(
        K.HD, eps=EPS, group_size=None, norm_before_gate=True, dtype=BF
    ).cuda()
    gn.weight.data.copy_(w["norm_w"])
    w["gated_norm"] = gn
    return w


def _stock_gdn(K, w, h, residual, conv, ssm, idx):
    """-> core, residual out; ``conv`` / ``ssm`` updated in place."""
    from sglang.kernels.ops.attention.fla.fused_recurrent import (
        fused_recurrent_gated_delta_rule_packed_decode,
    )
    from sglang.kernels.ops.mamba.causal_conv1d_triton import causal_conv1d_update

    x, res = _gemma_norm(h, residual, w["ln_w"])
    qkvz = _linear(x, w["w_qkvz"], w["w_qkvzs"])
    ba = _linear(x, w["w_ba"], w["w_bas"])
    q, k, v, z = qkvz.split([K.KEY, K.KEY, K.VAL, K.VAL], dim=-1)
    b, a = ba.split([K.NV, K.NV], dim=-1)
    mixed = causal_conv1d_update(
        torch.cat([q, k, v], dim=-1),
        conv,
        w["conv_w"].view(K.CONV, K.CONV_W),
        None,
        "silu",
        conv_state_indices=idx,
    )
    o = torch.empty(h.size(0), 1, K.NV, K.HD, dtype=BF, device="cuda")
    fused_recurrent_gated_delta_rule_packed_decode(
        mixed_qkv=mixed,
        a=a.contiguous(),
        b=b.contiguous(),
        A_log=w["a_log"],
        dt_bias=w["dt_bias"],
        scale=K.HD**-0.5,
        initial_state=ssm,
        out=o,
        ssm_state_indices=idx,
        use_qk_l2norm_in_kernel=True,
    )
    core = w["gated_norm"](o.view(-1, K.HD), z.reshape(-1, K.HD))
    return core.view(h.size(0), K.CORE), res


@unittest.skipUnless(_gfx950(), "needs FlyDSL and aiter on gfx950")
class DenseGdnTest(CustomTestCase):
    SLOTS = 12

    def test_bit_exact(self):
        from sglang.kernels.ops.moe.qwen3_5_mono_flydsl import dense_ffn as D
        from sglang.kernels.ops.moe.qwen3_5_mono_flydsl import dense_gdn as K

        for s in WIDTHS:
            first = s in (1, 5)
            g = torch.Generator(device="cuda").manual_seed(100 + s)
            layer = LAYER_TAGS[s % len(LAYER_TAGS)]
            w = _gdn_weights(g, K)
            w_qkvz, w_ba = D.shuffle_w(w["w_qkvz"]), D.shuffle_w(w["w_ba"])
            key = K.K1Build(tokens=s, first=first, eps=EPS, norm_eps=EPS)
            core = torch.empty(s, K.CORE, dtype=BF, device="cuda")
            res_out = torch.empty(s, K.HIDDEN, dtype=BF, device="cuda")
            epoch = torch.zeros(1, dtype=torch.int32, device="cuda")
            scratch = torch.zeros(
                K.scratch_bytes(key), dtype=torch.uint8, device="cuda"
            )
            idx = torch.arange(2, 2 + s, dtype=torch.int32, device="cuda")
            if s > 1:
                idx[1] = -1  # a pad row
            bad = []
            for draw in range(GDN_DRAWS):
                h = _rnd(g, s, K.HIDDEN, scale=0.5)
                residual = None if first else _rnd(g, s, K.HIDDEN, scale=2.0)
                # The mamba pool's layout: conv (slots, dim, 3) dim-first, SSM fp32.
                conv = _rnd(g, self.SLOTS, K.CONV, K.SL)
                ssm = _rnd(
                    g, self.SLOTS, K.NV, K.HD, K.HD, scale=0.05, dtype=torch.float32
                )
                conv_ref, ssm_ref = conv.clone(), ssm.clone()
                core_ref, res_ref = _stock_gdn(
                    K, w, h, residual, conv_ref, ssm_ref, idx
                )
                core.fill_(7.0)
                epoch.add_(1)
                K.dense_gdn_pre(
                    key,
                    hidden=h,
                    residual=residual,
                    res_out=res_out,
                    ln_w=w["ln_w"],
                    w_qkvz=w_qkvz,
                    w_qkvzs=w["w_qkvzs"],
                    w_ba=w_ba,
                    w_bas=w["w_bas"],
                    conv_w=w["conv_w"],
                    conv_state=conv,
                    a_log=w["a_log"],
                    dt_bias=w["dt_bias"],
                    norm_w=w["norm_w"],
                    rstate=ssm,
                    st_idx=idx,
                    core=core,
                    scratch=scratch,
                    epoch=epoch,
                    layer=layer,
                )
                exact = (
                    torch.equal(res_out, res_ref)
                    and torch.equal(core, core_ref)
                    and torch.equal(conv, conv_ref)
                    and torch.equal(ssm, ssm_ref)
                )
                if s > 1:
                    exact = exact and not core[1].any()
                if not exact:
                    bad.append(draw)
            with self.subTest(s=s, first=first, layer=layer):
                self.assertEqual(bad, [])


if __name__ == "__main__":
    unittest.main()
