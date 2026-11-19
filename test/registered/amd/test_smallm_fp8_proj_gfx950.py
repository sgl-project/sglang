"""gfx950 small-M W8A8 FP8 GEMM and fused output-side FP8 quant, Qwen3.5 AttnFP8 TP4 and TP2 shapes,
and the same quant fusion for Qwen3.8-Flash-Next's 48-head GDN and 24-head attention at TP1 and TP2."""

import contextlib
import itertools
import os
import types
import unittest
from unittest.mock import patch

import torch

from sglang.srt.utils import is_gfx95_supported
from sglang.test.ci.ci_register import register_amd_ci
from sglang.test.test_utils import CustomTestCase

register_amd_ci(est_time=60, suite="stage-b-test-1-gpu-small-amd-mi35x")


def _off():
    """The kernel as it is off gfx950 (or after a failed build): not loaded."""
    from sglang.kernels.ops.gemm import smallm_fp8_gfx950

    return patch.object(smallm_fp8_gfx950, "_mod", False)


def _per_token_scheme():
    from sglang.srt.layers.quantization.quark.schemes.quark_w8a8_fp8 import (
        QuarkW8A8Fp8,
    )

    return QuarkW8A8Fp8(
        {"qscheme": "per_channel"}, {"qscheme": "per_channel", "is_dynamic": True}
    )


@contextlib.contextmanager
def _single_rank():
    from sglang.srt.runtime_context import get_context, get_parallel
    from sglang.test.layer_ut_utils import init_single_process_dist

    init_single_process_dist()
    with (
        get_parallel().override(tp_size=1, tp_rank=0),
        get_context().override_server_args(),
    ):
        yield


def _quark_row_linear(in_features, out_features):
    """A RowParallelLinear of the Quark per-token FP8 scheme, loaded the way Quark checkpoints are (in _single_rank)."""
    from sglang.srt.layers.linear import RowParallelLinear
    from sglang.srt.layers.quantization.quark.quark import QuarkLinearMethod

    class _QuarkPerToken:
        def get_quant_method(self, layer, prefix):
            layer.scheme = _per_token_scheme()
            return QuarkLinearMethod(self)

    proj = RowParallelLinear(
        in_features,
        out_features,
        bias=False,
        reduce_results=False,
        params_dtype=torch.bfloat16,
        quant_config=_QuarkPerToken(),
    ).cuda()
    proj.weight.data.copy_(
        (torch.randn(out_features, in_features, device="cuda") * 0.05).to(
            torch.float8_e4m3fn
        )
    )
    proj.weight_scale.data.copy_(torch.rand(out_features, device="cuda") * 0.01 + 1e-3)
    proj.quant_method.process_weights_after_loading(proj)
    return proj


@unittest.skipUnless(is_gfx95_supported(), "gfx950 (MI35x) only")
class TestSmallMFp8ProjGfx950(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        os.environ.setdefault("SGLANG_USE_AITER", "1")
        from aiter.ops.shuffle import shuffle_weight

        from sglang.kernels.ops.quantization.fp8_kernel import (
            per_token_group_quant_fp8,
        )
        from sglang.srt.layers.quantization import fp8_utils

        cls.fp8_utils, cls.quant = fp8_utils, staticmethod(per_token_group_quant_fp8)
        torch.manual_seed(0)
        # packed GDN in_proj, attention qkv_proj, GDN out_proj / attention o_proj at TP4 then TP2; max M sent to
        # the new kernel
        cls.shapes = []
        for n, k, max_m in (
            (5184, 4096, 28),
            (4608, 4096, 28),
            (4096, 2048, 36),
            (10304, 4096, 32),
            (8704, 4096, 16),
            (4096, 4096, 40),
        ):
            w = (torch.randn(n, k, device="cuda") * 0.05).to(torch.float8_e4m3fn)
            s = torch.rand(n, 1, device="cuda") * 0.01 + 1e-3
            cls.shapes.append((shuffle_weight(w, (16, 16)).t(), s, max_m))

    def linear(self, x, w, s):
        x, xs = x if isinstance(x, tuple) else (x, None)
        return self.fp8_utils.apply_fp8_linear(
            x, w, s, input_scale=xs, use_per_token_if_dynamic=True
        )

    def run_counted(self, *thunks):
        calls, real = [], self.fp8_utils.smallm_fp8_gemm
        with patch.object(
            self.fp8_utils,
            "smallm_fp8_gemm",
            lambda *a, **k: calls.append(1) or real(*a, **k),
        ):
            return [f() for f in thunks], len(calls)

    def cases(self):
        """(n, m, w, s, in range, x, q, xs, aiter ref) on both sides of each shape's max M."""
        from aiter import gemm_a8w8_bpreshuffle

        for w, s, max_m in self.shapes:
            for m in (1, 4, 8, 9, 16, 17, 24, 29, 33, 36, 41):
                x = torch.randn(m, w.shape[0], device="cuda", dtype=torch.bfloat16)
                q, xs = self.quant(x, group_size=x.shape[1])
                ref = gemm_a8w8_bpreshuffle(q, w.t(), xs, s, None, torch.bfloat16)
                yield w.shape[1], m, w, s, m <= max_m, x, q, xs, ref.float()

    def test_gemm_matches_aiter(self):
        for n, m, w, s, in_range, x, _, _, ref in self.cases():
            with self.subTest(n=n, m=m):
                (got,), calls = self.run_counted(lambda: self.linear(x, w, s))
                self.assertEqual(calls, int(in_range))
                ulp = torch.exp2(torch.floor(torch.log2(ref.abs() + 1e-30)) - 7)
                err = (got.float() - ref).abs()
                self.assertTrue((err <= ulp + 1e-5 * ref.abs().max()).all())

    def test_tuple_input_matches_bf16(self):
        for n, m, w, s, in_range, x, q, xs, _ in self.cases():
            with self.subTest(n=n, m=m):
                (got, got_tuple), calls = self.run_counted(
                    lambda: self.linear(x, w, s), lambda: self.linear((q, xs), w, s)
                )
                self.assertEqual(calls, 2 * in_range)
                self.assertTrue(torch.equal(got, got_tuple))

    def test_falls_back_without_kernel(self):
        for n, m, w, s, _, x, _, _, ref in self.cases():
            with self.subTest(n=n, m=m), _off():
                (off,), calls = self.run_counted(lambda: self.linear(x, w, s))
                self.assertEqual(calls, 0)
                self.assertTrue(torch.equal(off.float(), ref))

    def test_producer_guard(self):
        from sglang.srt.models import qwen3_5

        scheme = _per_token_scheme()
        fwd = types.SimpleNamespace(sp_active=False)
        # weight is [K, N] after Quark loading; Qwen3.8 TP1 o_proj (N, K) = (2560, 6144), outside SHAPES
        o_proj = types.SimpleNamespace(
            scheme=scheme, weight=torch.empty(6144, 2560), weight_scale=torch.empty(1)
        )
        dequantized = types.SimpleNamespace(
            scheme=scheme, weight=torch.empty(6144, 12), weight_scale=None
        )
        with patch.object(qwen3_5, "get_forward", return_value=fwd):
            fp8_in, bf16 = qwen3_5._fp8_tuple_input, torch.bfloat16
            self.assertTrue(fp8_in(o_proj, 1, bf16))
            self.assertTrue(fp8_in(o_proj, 256, bf16))
            self.assertFalse(fp8_in(o_proj, 257, bf16))  # past the token cap
            self.assertFalse(fp8_in(o_proj, 4, torch.float16))  # the GEMM returns BF16
            self.assertFalse(fp8_in(dequantized, 4, bf16))  # served dequantized
            self.assertFalse(fp8_in(torch.nn.Linear(8, 8), 4, bf16))  # BF16 MTP layer
            with _off():
                self.assertTrue(fp8_in(o_proj, 4, bf16))  # no small-M kernel needed
            with patch.object(qwen3_5, "_is_gfx95", False):
                self.assertFalse(
                    fp8_in(o_proj, 4, bf16)
                )  # the launches are checked on gfx950

    def producers(self, t, trial=0, tp=4, heads=(64, 32), activation="swish"):
        from sglang.kernels.ops.attention.fla.layernorm_gated import (
            _layer_norm_fwd,
            rms_norm_gated,
        )
        from sglang.kernels.ops.elementwise.elementwise import fused_sigmoid_mul

        # per rank: GDN RMSNormGated over heads[0] / tp heads x 128; attention gate over heads[1] / tp heads x 256
        nv, na = heads[0] // tp, heads[1] // tp
        x = torch.randn(t * nv, 128, device="cuda", dtype=torch.bfloat16) * (1 + trial)
        z = torch.randn_like(x) * 3
        w = torch.randn(128, device="cuda", dtype=torch.bfloat16) * 0.5 + 1
        a = torch.randn(t, na * 256, device="cuda", dtype=torch.bfloat16) * (1 + trial)
        g = torch.randn(t, na, 512, device="cuda", dtype=torch.bfloat16)[:, :, 256:]
        fused = (
            lambda: _layer_norm_fwd(
                x,
                w,
                None,
                1e-6,
                z=z,
                is_rms_norm=True,
                activation=activation,
                quant_heads=nv,
            )[0],
            lambda: fused_sigmoid_mul(a, g, quant=True),
        )
        norm = rms_norm_gated(
            x=x,
            weight=w,
            bias=None,
            z=z,
            eps=1e-6,
            is_rms_norm=True,
            activation=activation,
        )
        plain = (norm.view(t, nv * 128), fused_sigmoid_mul(a.clone(), g))
        unfused = tuple(self.quant(p, group_size=p.shape[1]) for p in plain)
        return fused, unfused, plain

    def test_fused_quant_bit_exact(self):
        """The codes and scales the producers hand the GEMM are the separate quant's, for Qwen3.5-397B (swish
        gate, 8 to 32 GDN heads a rank) and Qwen3.8-Flash-Next (sigmoid gate, 6 to 48 heads a rank: 48 in a
        64-row tile, 12 in a 16-row one), up to the served 176 tokens. The norm is fused from the token counts
        of norm_floor, for 256 CUs."""
        # (value heads, attention heads), the TP sizes, the GDN norm's gate
        models = (((64, 32), (8, 4, 2), "swish"), ((48, 24), (8, 4, 2, 1), "sigmoid"))
        # value heads per rank: the first token count at which the norm fuses
        norm_floor = {6: 1, 8: 1, 12: 1, 16: 1, 24: 22, 32: 17, 48: 22}
        for (heads, tps, activation), t in itertools.product(
            models, (1, 4, 16, 21, 22, 33, 40, 48, 176)
        ):
            for tp in tps:
                nv = heads[0] // tp
                norm_fuses = t >= norm_floor[nv]
                for trial in range(5):
                    fused, unfused, _ = self.producers(t, trial, tp, heads, activation)
                    for i, (f, (rq, rs)) in enumerate(zip(fused, unfused)):
                        if i == 0 and not norm_fuses:
                            continue
                        with self.subTest(heads=heads, tp=tp, t=t, trial=trial, i=i):
                            q, s = f()
                            self.assertTrue(
                                torch.equal(q.view(torch.uint8), rq.view(torch.uint8))
                            )
                            self.assertTrue(torch.equal(s, rs))

    def test_quark_row_linear_takes_producer_tuple(self):
        with _single_rank():
            # TP4 GDN out_proj / attention o_proj
            proj = _quark_row_linear(2048, 4096)
            for t in (1, 4, 36):
                fused, _, plain = self.producers(t)
                for f, p in zip(fused, plain):
                    with self.subTest(t=t):
                        (got, ref), n = self.run_counted(
                            lambda: proj(f())[0], lambda: proj(p)[0]
                        )
                        self.assertEqual(n, 2)
                        self.assertTrue(torch.equal(got, ref))
            # without the small-M kernel the tuple goes to the aiter GEMM the BF16 input takes
            with _off():
                (got, ref), n = self.run_counted(
                    lambda: proj(fused[0]())[0], lambda: proj(plain[0])[0]
                )
                self.assertEqual(n, 0)
                self.assertTrue(torch.equal(got, ref))

    def quant_launches(self, thunk):
        """thunk's result and how many per-token quants apply_fp8_linear ran for it."""
        calls, real = [], self.fp8_utils.per_token_group_quant_fp8
        with patch.object(
            self.fp8_utils,
            "per_token_group_quant_fp8",
            lambda *a, **k: calls.append(1) or real(*a, **k),
        ):
            return thunk(), len(calls)

    def test_qwen4_attention_gate_hands_o_proj_the_fp8_pair(self):
        """Qwen4ExpAttentionDecoderLayer.self_attention: o_proj takes the gate's (fp8, scale), not a BF16 tensor
        it quantizes itself, and its output is the BF16 path's; past decode sizes it is the BF16 path."""
        from sglang.srt.models import qwen3_5, qwen4_exp

        n_heads, head_dim, hidden = 24, 256, 2560  # Qwen3.8-Flash-Next TP1
        no_sp = types.SimpleNamespace(sp_active=False)
        with _single_rank(), patch.object(qwen3_5, "get_forward", return_value=no_sp):
            layer = types.SimpleNamespace(
                is_qsa=False,
                alt_stream=None,
                o_proj=_quark_row_linear(n_heads * head_dim, hidden),
            )
            layer.self_attention = types.MethodType(
                qwen4_exp.Qwen4ExpAttentionDecoderLayer.self_attention, layer
            )
            for t in (1, 48, 176, 256, 257):
                attn = (
                    torch.randn(t, n_heads * head_dim, device="cuda") * 0.5
                ).bfloat16()
                gate = torch.randn(t, n_heads, 2 * head_dim, device="cuda").bfloat16()
                layer._prepare_qkv_gate = lambda **_: (
                    None,
                    None,
                    None,
                    gate[..., head_dim:],
                )
                layer.attn = lambda *_: attn.clone()
                run = lambda: layer.self_attention(None, None, None)
                # _fp8_tuple_input, which qwen4_exp calls, says no without aiter
                with patch.object(qwen3_5, "_use_aiter", False):
                    ref, ref_quants = self.quant_launches(run)
                got, got_quants = self.quant_launches(run)
                with self.subTest(t=t):
                    self.assertEqual((ref_quants, got_quants), (1, int(t > 256)))
                    self.assertTrue(torch.equal(got, ref))

    def test_gdn_forward_fuses_the_norm_quant_above_the_floor(self):
        """Qwen3_5GatedDeltaNet.forward: out_proj takes the norm's (fp8, scale) where the norm's launch rounds
        as the unfused one does (at 48 heads from 22 tokens, rows_per_block 4), or where the small-M GEMM takes
        the shape (32 heads, Qwen3.5-397B TP2, up to 40 tokens); otherwise the norm and the quant stay two
        launches. The output is the BF16 path's wherever the launch rounds as the unfused one does."""
        from sglang.kernels.ops.attention.fla.layernorm_gated import RMSNorm
        from sglang.srt.model_executor.forward_mode import ForwardMode
        from sglang.srt.models import qwen3_5

        n_k, head_dim = 16, 128
        # value heads a rank, hidden, the norm's gate, the first token count where the norm's launch rounds as
        # the unfused one does, the first token count where the norm is fused
        configs = ((48, 2560, "sigmoid", 22, 22), (32, 4096, "swish", 17, 1))
        no_sp = types.SimpleNamespace(sp_active=False)
        decode = types.SimpleNamespace(
            forward_mode=ForwardMode.DECODE, logical_forward_mode=ForwardMode.DECODE
        )
        for n_v, hidden, activation, exact_from, fused_from in configs:
            with (
                _single_rank(),
                patch.object(qwen3_5, "get_forward", return_value=no_sp),
            ):
                norm = RMSNorm(
                    head_dim,
                    eps=1e-6,
                    activation=activation,
                    device="cuda",
                    dtype=torch.bfloat16,
                )
                norm.weight.data.normal_(1.0, 0.3)
                gdn = types.SimpleNamespace(
                    num_k_heads=n_k,
                    num_v_heads=n_v,
                    attn_tp_size=1,
                    head_k_dim=head_dim,
                    head_v_dim=head_dim,
                    norm=norm,
                    out_proj=_quark_row_linear(n_v * head_dim, hidden),
                )
                gdn.forward = types.MethodType(
                    qwen3_5.Qwen3_5GatedDeltaNet.forward, gdn
                )
                for t in (1, 16, 21, 22, 48, 176):
                    qkvz = torch.randn(
                        t, 2 * (n_k + n_v) * head_dim, device="cuda"
                    ).bfloat16()
                    ba = torch.randn(t, 2 * n_v, device="cuda").bfloat16()
                    core = (
                        torch.randn(t, n_v, head_dim, device="cuda") * 0.5
                    ).bfloat16()
                    gdn._forward_input_proj = lambda _: (qkvz, ba)
                    gdn.attn = lambda forward_batch, mixed_qkv, a, b: core
                    run = lambda: gdn.forward(None, decode)
                    with patch.object(qwen3_5, "_fp8_tuple_input", return_value=False):
                        ref, ref_quants = self.quant_launches(run)
                    got, got_quants = self.quant_launches(run)
                    with self.subTest(heads=n_v, t=t):
                        self.assertEqual(
                            (ref_quants, got_quants), (1, int(t < fused_from))
                        )
                        if t >= exact_from:
                            self.assertTrue(torch.equal(got, ref))

    def test_gdn_prefill_shares_the_in_proj_quant_and_fuses_the_norm_quant(self):
        """SGLANG_AMD_GDN_PREFILL_FUSED, plain extend: in_proj_qkvz and in_proj_ba take one (fp8, scale) of the
        layer input (not at TP4, where in_proj_ba's 24 outputs are served dequantized), and the gated RMSNorm
        writes out_proj's from 4-row norm tiles (from 22 tokens at 48 heads, 86 at 12); the output is the
        unfused path's bit for bit. Qwen3.8-Flash-Next TP1, TP2, TP4."""
        from sglang.kernels.ops.attention.fla.layernorm_gated import RMSNorm
        from sglang.srt.model_executor.forward_mode import ForwardMode
        from sglang.srt.model_loader.utils import set_default_torch_dtype
        from sglang.srt.models import qwen3_5

        head_dim, hidden = 128, 2560
        extend = types.SimpleNamespace(
            forward_mode=ForwardMode.EXTEND, logical_forward_mode=ForwardMode.EXTEND
        )
        no_sp = types.SimpleNamespace(sp_active=False)
        norms = []
        real_norm = qwen3_5.rms_norm_gated_fp8
        for n_v, n_k in ((48, 16), (24, 8), (12, 4)):
            with (
                _single_rank(),
                patch.object(qwen3_5, "get_forward", return_value=no_sp),
                patch.object(
                    qwen3_5,
                    "rms_norm_gated_fp8",
                    lambda *a, **k: norms.append(1) or real_norm(*a, **k),
                ),
            ):
                norm = RMSNorm(
                    head_dim,
                    eps=1e-6,
                    activation="sigmoid",
                    device="cuda",
                    dtype=torch.bfloat16,
                )
                norm.weight.data.normal_(1.0, 0.3)
                gdn = types.SimpleNamespace(
                    num_k_heads=n_k,
                    num_v_heads=n_v,
                    attn_tp_size=1,
                    head_k_dim=head_dim,
                    head_v_dim=head_dim,
                    norm=norm,
                    alt_stream=None,
                    _fused_in_proj_weight=None,
                    _fused_input_proj_cpu_enabled=types.SimpleNamespace(value=False),
                )
                # as the model is built: a linear served dequantized keeps BF16 weights
                with set_default_torch_dtype(torch.bfloat16):
                    gdn.in_proj_qkvz = _quark_row_linear(
                        hidden, 2 * (n_k + n_v) * head_dim
                    )
                    gdn.in_proj_ba = _quark_row_linear(hidden, 2 * n_v)
                    gdn.out_proj = _quark_row_linear(n_v * head_dim, hidden)
                for name in ("forward", "_forward_input_proj"):
                    method = getattr(qwen3_5.Qwen3_5GatedDeltaNet, name)
                    setattr(gdn, name, types.MethodType(method, gdn))

                def attn(forward_batch, mixed_qkv, a, b):
                    # stands in for the GDN core: reads v, a and b; token 5 all zero
                    v = mixed_qkv[:, 2 * n_k * head_dim :].reshape(-1, n_v, head_dim)
                    out = v * torch.sigmoid(b)[..., None] + a[..., None]
                    out[5] = 0
                    return out

                gdn.attn = attn
                for t in (65, 300, 2050):
                    h = torch.randn(t, hidden, device="cuda").bfloat16()
                    run = lambda: gdn.forward(h, extend)
                    del norms[:]
                    with patch.object(qwen3_5, "_gdn_prefill_fp8_fused", False):
                        ref, ref_quants = self.quant_launches(run)
                    shared = []
                    with (
                        patch.object(qwen3_5, "_gdn_prefill_fp8_fused", True),
                        patch.object(
                            qwen3_5,
                            "per_token_group_quant_fp8",
                            lambda *a, **k: shared.append(1) or self.quant(*a, **k),
                        ),
                    ):
                        got, got_quants = self.quant_launches(run)
                    norm_fused = t * n_v > 1024  # 4-row tiles on 256 CUs
                    shares = gdn.in_proj_ba.weight_scale is not None
                    with self.subTest(heads=n_v, t=t):
                        self.assertEqual(len(norms), int(norm_fused))
                        self.assertEqual(
                            (ref_quants, len(shared), got_quants),
                            (2 + shares, int(shares), 1 - shares + 1 - norm_fused),
                        )
                        self.assertTrue(torch.equal(got, ref))

    def test_gdn_prefill_gates_blocked(self):
        """The blocked prefill gates equal fused_gdn_gating's bit for bit on the strided b / a views of the GDN
        in_proj output, the softplus threshold branch included, and the backend routes to them by the switch."""
        from sglang.kernels.ops.attention.linear.fused_gdn_gating import (
            fused_gdn_gating,
            fused_gdn_gating_blocked,
        )
        from sglang.srt.layers.attention.linear import gdn_backend

        n_v = 48
        layer = types.SimpleNamespace(
            A_log=torch.randn(n_v, device="cuda"),
            dt_bias=torch.randn(n_v, device="cuda").bfloat16(),
        )
        for t in (1, 33, 6169):
            ba = (torch.randn(t, 2 * n_v, device="cuda") * 8).bfloat16()
            ba[:4, n_v:] = 40.0
            b, a = ba[:, :n_v], ba[:, n_v:]
            ref = fused_gdn_gating(layer.A_log, a, b, layer.dt_bias)
            calls = []
            with (
                patch.object(gdn_backend, "_prefill_gates_blocked", True),
                patch.object(
                    gdn_backend,
                    "fused_gdn_gating_blocked",
                    lambda *x: calls.append(1) or fused_gdn_gating_blocked(*x),
                ),
            ):
                got = gdn_backend.GDNAttnBackend._prefill_gates(layer, a, b)
            with self.subTest(t=t):
                self.assertEqual(len(calls), 1)
                self.assertTrue(all(torch.equal(g, r) for g, r in zip(got, ref)))

    def test_power_of_two_heads_keep_the_capped_launch(self):
        """_layer_norm_fwd(quant_heads=n) for a power-of-two n runs one program of n rows and
        min(n // rows_per_block, 16) warps per token at every token count: 32 heads below 17 tokens (Qwen3.5-397B
        TP2 decode) take 16 warps for 32 rows, as they always did."""
        from sglang.kernels.ops.attention.fla import layernorm_gated

        launches, kernel = [], layernorm_gated._layer_norm_fwd_1pass_kernel

        class Recorder:
            def __getitem__(self, grid):
                def launch(*args, **kwargs):
                    launches.append(
                        (grid, kwargs["num_warps"], kwargs["ROWS_PER_BLOCK"])
                    )
                    return kernel[grid](*args, **kwargs)

                return launch

        device = torch.empty(0, device="cuda").device
        with patch.object(layernorm_gated, "_layer_norm_fwd_1pass_kernel", Recorder()):
            for nv, t in ((16, 1), (16, 36), (32, 1), (32, 16), (32, 40), (64, 8)):
                x = torch.randn(t * nv, 128, device="cuda", dtype=torch.bfloat16)
                w = torch.ones(128, device="cuda", dtype=torch.bfloat16)
                rows_per_block = layernorm_gated.calc_rows_per_block(t * nv, device)
                with self.subTest(heads=nv, t=t):
                    first = len(launches)
                    (q, s), _, _ = layernorm_gated._layer_norm_fwd(
                        x,
                        w,
                        None,
                        1e-6,
                        z=torch.randn_like(x),
                        is_rms_norm=True,
                        quant_heads=nv,
                    )
                    self.assertEqual(
                        launches[first:],
                        [((t, 1), min(nv // rows_per_block, 16), nv)],
                    )
                    self.assertEqual((q.shape, s.shape), ((t, nv * 128), (t, 1)))

    def test_graph_replay_matches_eager(self):
        (w_in, s_in, _), _, (w_out, s_out, _) = self.shapes[:3]
        x = torch.randn(4, 4096, device="cuda", dtype=torch.bfloat16)
        (norm_q, sig_q), *_ = self.producers(4)

        def run():
            return (
                self.linear(x, w_in, s_in),
                self.linear(norm_q(), w_out, s_out),
                self.linear(sig_q(), w_out, s_out),
            )

        eager = [o.clone() for o in run()]
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            outs = run()
        g.replay()
        torch.cuda.synchronize()
        for e, o in zip(eager, outs):
            self.assertTrue(torch.equal(e, o))


if __name__ == "__main__":
    unittest.main()
