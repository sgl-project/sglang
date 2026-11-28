"""K3 attention overlap parity under CUDA graph capture and changed-input replay."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.models.kimi_k3 import (
    KimiK3DeltaAttention,
    KimiK3LinearForCausalLM,
    KimiK3MLAAttention,
    KimiK3MoE,
    _get_k3_dense_weight,
)
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=12, stage="base-b", runner_config="4-gpu-b200")

_H = 7168
_QKVG = 6144  # q,k,v,g slices per rank at TP8
_N_FA = 128
_N_B = 12
_BFA_W_ROWS = 144  # [f_a | b] padded to 8 rows like _merge_bfa_weights


def _make_owner(with_stream: bool):
    gen = torch.Generator(device="cuda").manual_seed(0)

    def _randn(*shape):
        return (
            torch.randn(*shape, generator=gen, device="cuda", dtype=torch.float32)
            .mul(0.05)
            .to(torch.bfloat16)
        )

    qkvg_w = _randn(_QKVG, _H)

    def fused_qkvg_proj(x):
        return torch.nn.functional.linear(x, qkvg_w), None

    owner = SimpleNamespace(
        use_full_rank_gate=True,
        _qkvg_gate_float_shard=False,
        _bfa_w=_randn(_BFA_W_ROWS, _H).contiguous(),
        _bfa_f_b_w=_randn(1536, _N_FA).contiguous(),
        _bfa_fa_size=_N_FA,
        _bfa_b_size=_N_B,
        fused_qkvg_proj=fused_qkvg_proj,
        split_sizes=[3 * 1536, 1536],
        _bfa_alt_stream=torch.cuda.Stream() if with_stream else None,
        _bfa_bs_limit=128 if with_stream else 0,
        _bfa_side_tgv_m_ranges=None,
    )
    return owner


def _run(owner, x):
    out = KimiK3DeltaAttention.forward_qkvbfg_fused(owner, x)
    return [t.clone() for t in out]


class TestKimiK3BfaOverlap(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        if not torch.cuda.is_available():
            raise unittest.SkipTest("CUDA is not available")

    def test_capture_replay_matches_serial(self):
        torch.manual_seed(0)
        for T in (1, 4, 12):
            for defer_f_b in (False, True):
                with self.subTest(T=T, defer_f_b=defer_f_b):
                    x = torch.empty(T, _H, device="cuda", dtype=torch.bfloat16)
                    x.normal_(std=0.05)
                    serial_owner = _make_owner(with_stream=False)
                    owner = _make_owner(with_stream=True)
                    forward = KimiK3DeltaAttention.forward_qkvbfg_fused
                    with patch(
                        "sglang.srt.models.kimi_k3.get_is_capture_mode",
                        return_value=True,
                    ):
                        # Warm up allocations/JIT outside capture.
                        forward(owner, x, defer_f_b=defer_f_b)
                        graph = torch.cuda.CUDAGraph()
                        with torch.cuda.graph(graph):
                            captured = forward(owner, x, defer_f_b=defer_f_b)
                        for _ in range(3):
                            # Changed inputs expose stale reads or missing dependencies.
                            x.normal_(std=0.05)
                            serial = forward(serial_owner, x, defer_f_b=defer_f_b)
                            graph.replay()
                            torch.cuda.synchronize()
                            for got, ref, name in zip(
                                captured, serial, ("qkv", "beta", "forget_gate", "g")
                            ):
                                self.assertTrue(torch.equal(got, ref), name)

    def test_side_bfa_tgv_capture_matches_eager_tgv(self):
        """Graph replay of the side-stream TGV [f_a | b] matches the eager kernel."""
        if torch.cuda.get_device_capability()[0] != 10:
            self.skipTest("TGV needs SM10x")
        from sglang.kernels.ops.gemm import cutedsl_bf16_gemm as cg
        from sglang.kernels.ops.gemm import kimi_k3_tiny_gemm
        from sglang.srt.models import kimi_k3

        forward = KimiK3DeltaAttention.forward_qkvbfg_fused
        ranges = cg.TGVMRanges()
        for m_lo, m_hi, tactic in kimi_k3._K3_GB300_BFA_SIDE_TGV_M_RANGES[
            (_BFA_W_ROWS, _H)
        ]:
            ranges.append(m_lo=m_lo, m_hi=m_hi, tactic=tactic)
        for T, tactic in ((72, 21), (144, 23), (264, 24)):
            with (
                self.subTest(T=T),
                patch(
                    "sglang.srt.models.kimi_k3.get_is_capture_mode",
                    return_value=True,
                ),
            ):
                x = torch.empty(T, _H, device="cuda", dtype=torch.bfloat16)
                x.normal_(std=0.05)
                owner = _make_owner(with_stream=True)
                owner._bfa_bs_limit = 288
                owner._bfa_side_tgv_m_ranges = ranges
                forward(owner, x)
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    captured = forward(owner, x)
                for _ in range(2):
                    x.normal_(std=0.05)
                    bfa = cg.cutedsl_bf16_gemm(x, owner._bfa_w, tactic=tactic)
                    beta = bfa[:, _N_FA : _N_FA + _N_B]
                    forget = kimi_k3_tiny_gemm(bfa[:, :_N_FA], owner._bfa_f_b_w)
                    graph.replay()
                    torch.cuda.synchronize()
                    self.assertTrue(torch.equal(captured[1], beta))
                    self.assertTrue(torch.equal(captured[2], forget))
                    ref = torch.nn.functional.linear(x, owner._bfa_w)
                    torch.testing.assert_close(
                        captured[1],
                        ref[:, _N_FA : _N_FA + _N_B],
                        rtol=2e-2,
                        atol=2e-3,
                    )

    def test_eager_stream_branch_not_taken(self):
        x = torch.randn(3, _H, device="cuda", dtype=torch.bfloat16)
        serial = _run(_make_owner(with_stream=False), x)
        overlap = _run(_make_owner(with_stream=True), x)  # capture mode False
        for got, ref in zip(overlap, serial):
            self.assertTrue(torch.equal(got, ref))

    def test_mla_gate_capture_matches_serial(self):
        x = torch.randn(4, 64, device="cuda", dtype=torch.bfloat16)
        qkv_weight = torch.randn(128, 64, device="cuda", dtype=torch.bfloat16)
        gate_weight = torch.randn_like(qkv_weight)
        project = torch.nn.functional.linear
        owner = SimpleNamespace(
            _gate_alt_stream=torch.cuda.Stream(),
            _gate_bs_limit=128,
            g_proj=lambda value: (project(value, gate_weight), None),
        )

        def forward():
            # Both branches must wait for this in-graph input producer.
            hidden = x * 0.5
            KimiK3MLAAttention._fork_output_gate(owner, hidden)
            qkv = project(hidden, qkv_weight)
            gate = KimiK3MLAAttention._compute_output_gate(owner, hidden)
            return qkv * torch.sigmoid(gate)

        for capture_mode, breakable in ((True, False), (True, True), (False, False)):
            with (
                self.subTest(capture_mode=capture_mode, breakable=breakable),
                patch(
                    "sglang.srt.models.kimi_k3.get_is_capture_mode",
                    return_value=capture_mode,
                ),
                patch(
                    "sglang.srt.models.kimi_k3.is_in_breakable_cuda_graph",
                    return_value=breakable,
                ),
            ):
                forward()
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    captured = forward()
                for _ in range(3):
                    x.normal_()
                    hidden = x * 0.5
                    expected = project(hidden, qkv_weight) * torch.sigmoid(
                        project(hidden, gate_weight)
                    )
                    graph.replay()
                    torch.cuda.synchronize()
                    self.assertTrue(torch.equal(captured, expected))

    def _check_reload(self, prepare, buffers, update):
        prepare()
        original = tuple(buffers())
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            [weight.clone() for weight in buffers()]
        torch.cuda.current_stream().wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            captured = tuple(weight.clone() for weight in buffers())
        initial = tuple(weight.clone() for weight in buffers())
        for reload in range(2):
            update()
            prepare()
            graph.replay()
            torch.cuda.synchronize()
            for index, current in enumerate(buffers()):
                with self.subTest(reload=reload, buffer=index):
                    self.assertFalse(torch.equal(current, initial[index]))
                    torch.testing.assert_close(captured[index], current, rtol=0, atol=0)
                    self.assertEqual(current.data_ptr(), original[index].data_ptr())

    def test_front_weight_reload_preserves_views(self):
        """Reload must refresh captured front weights and restore parameter views."""

        class UnshardedFront(SimpleNamespace):
            @property
            def _front_shard(self):
                return None

        def module(rows):
            return SimpleNamespace(
                weight=torch.ones(rows, 16, device="cuda", dtype=torch.bfloat16)
            )

        owner = UnshardedFront(
            use_latent_moe=True,
            shared_experts=None,
            gate=module(8),
            routed_expert_down_proj=module(16),
            _front_w=None,
        )

        def update():
            # A loader can either copy into the views or replace their storage.
            owner.gate.weight = owner.gate.weight + 1
            owner.routed_expert_down_proj.weight.add_(2)

        with patch(
            "sglang.srt.models.kimi_k3.envs.SGLANG_K3_FUSED_FRONT.get",
            return_value=True,
        ):
            self._check_reload(
                lambda: KimiK3MoE._merge_front_weights(owner),
                lambda: (owner._front_w,),
                update,
            )
        self.assertEqual(owner.gate.weight.data_ptr(), owner._front_w.data_ptr())
        torch.testing.assert_close(owner._front_w[:8], owner.gate.weight)
        torch.testing.assert_close(
            owner._front_w[8:], owner.routed_expert_down_proj.weight
        )

    def test_bfa_weight_reload_preserves_buffers(self):
        """Rebuilding dense or dequantized BFA weights must not strand a graph."""
        for block_fp8, mxfp8 in ((False, False), (True, False), (False, True)):
            with self.subTest(block_fp8=block_fp8, mxfp8=mxfp8):

                def module(rows):
                    return SimpleNamespace(
                        weight=torch.ones(rows, 16, device="cuda", dtype=torch.bfloat16)
                    )

                owner = SimpleNamespace(
                    use_full_rank_gate=True,
                    _bfa_uses_block_fp8=block_fp8,
                    _bfa_uses_mxfp8=mxfp8,
                    f_a_proj=module(4),
                    b_proj=module(2),
                    f_b_proj=module(8),
                    _bfa_w=None,
                    _bfa_f_b_w=None,
                )

                def update():
                    owner.f_a_proj.weight.add_(1)
                    owner.b_proj.weight.add_(2)
                    owner.f_b_proj.weight.add_(3)

                # Emulate a freshly dequantized allocation on every reload.
                with patch(
                    "sglang.srt.models.kimi_k3._get_k3_dense_weight",
                    side_effect=lambda mod: mod.weight.clone(),
                ):
                    self._check_reload(
                        lambda: KimiK3DeltaAttention._merge_bfa_weights(owner),
                        lambda: (owner._bfa_w, owner._bfa_f_b_w),
                        update,
                    )
                torch.testing.assert_close(owner._bfa_w[:4], owner.f_a_proj.weight)
                torch.testing.assert_close(owner._bfa_w[4:6], owner.b_proj.weight)
                self.assertEqual(torch.count_nonzero(owner._bfa_w[6:]).item(), 0)

    def test_hip_inproj_reload_preserves_whole_and_tail_views(self):
        """Reload must update captured whole/tail projections and parameter views."""

        def module(rows):
            return SimpleNamespace(
                weight=torch.ones(rows, 16, device="cuda", dtype=torch.bfloat16)
            )

        owner = SimpleNamespace(
            _may_fuse_kda_inproj=lambda: True,
            _kda_dwdp=None,
            _qkvgbfa_layer=None,
            fused_qkvg_proj=module(8),
            f_a_proj=module(4),
            b_proj=module(2),
            split_sizes=[2, 2, 2, 2],
        )

        def update():
            owner.fused_qkvg_proj.weight.add_(1)
            owner.f_a_proj.weight.add_(2)
            owner.b_proj.weight.add_(3)

        self._check_reload(
            lambda: KimiK3DeltaAttention._merge_kda_inproj_weights_hip(owner),
            lambda: (
                owner._qkvgbfa_layer.weight,
                owner._bfa_w,
                owner.fused_qkvg_proj.weight,
                owner.f_a_proj.weight,
                owner.b_proj.weight,
            ),
            update,
        )
        merged = owner._qkvgbfa_layer.weight
        for view, offset in (
            (owner.fused_qkvg_proj.weight, 0),
            (owner.f_a_proj.weight, 8),
            (owner.b_proj.weight, 12),
            (owner._bfa_w, 8),
        ):
            self.assertEqual(view.data_ptr(), merged[offset:].data_ptr())
        self.assertEqual(torch.count_nonzero(merged[14:]).item(), 0)

    def test_fused_decode_reload_preserves_buffers(self):
        """Captured KDA static tensors must observe subsequent weight loads."""

        def ones(*shape):
            return torch.ones(*shape, device="cuda", dtype=torch.float32)

        layer = SimpleNamespace(
            conv_weights=ones(3 * 1536, 4),
            bias=ones(3 * 1536),
            A_log=ones(12),
            dt_bias=ones(1536),
            _k3_fused_decode_args=None,
        )
        owner = SimpleNamespace(
            attn=layer,
            local_num_heads=12,
            o_norm=SimpleNamespace(weight=ones(128), eps=1e-6),
        )

        def update():
            layer.conv_weights.add_(1)
            layer.bias.add_(2)
            layer.A_log.add_(3)
            owner.o_norm.weight.add_(4)

        self._check_reload(
            lambda: KimiK3DeltaAttention._prepare_fused_decode(owner),
            lambda: layer._k3_fused_decode_args[:-1],
            update,
        )
        torch.testing.assert_close(
            layer._k3_fused_decode_args[0], layer.conv_weights[:1536].t()
        )

    def test_attn_res_reload_refreshes_both_cached_dtypes(self):
        """Both eager and captured AttnRes caches must track updated weights."""
        from sglang.srt.layers.attn_residual import get_cw

        proj = SimpleNamespace(
            weight=torch.ones(1, 16, device="cuda", dtype=torch.bfloat16)
        )
        norm = SimpleNamespace(
            weight=torch.ones(16, device="cuda", dtype=torch.bfloat16)
        )
        owner = SimpleNamespace(
            config=SimpleNamespace(full_attention_layer_ids=[]),
            model=SimpleNamespace(
                layers=[],
                output_attn_res_proj=proj,
                output_attn_res_norm=norm,
                kda_dwdp=None,
            ),
        )

        def update():
            proj.weight.add_(1)
            norm.weight.add_(2)

        self._check_reload(
            lambda: KimiK3LinearForCausalLM.post_load_weights(owner),
            lambda: (get_cw(proj, norm), get_cw(proj, norm, dtype=torch.bfloat16)),
            update,
        )
        expected = norm.weight.float() * proj.weight.squeeze().float()
        torch.testing.assert_close(get_cw(proj, norm), expected, rtol=0, atol=0)

    def test_mla_weight_reload_preserves_captured_projections(self):
        def values(shape):
            return (
                torch.arange(torch.Size(shape).numel(), device="cuda")
                .reshape(shape)
                .to(torch.bfloat16)
                / 512
            )

        attn = SimpleNamespace(
            kv_b_proj=SimpleNamespace(weight=values((24, 16))),
            qk_nope_head_dim=4,
            v_head_dim=8,
            w_kc=None,
            w_vc=None,
            _dcp_v_weights=None,
        )
        owner = SimpleNamespace(
            config=SimpleNamespace(full_attention_layer_ids=[0]),
            model=SimpleNamespace(
                layers=[
                    SimpleNamespace(self_attn=attn, use_attn_residuals=False, mlp=None)
                ],
                kda_dwdp=None,
            ),
        )
        post_load = KimiK3LinearForCausalLM.post_load_weights
        post_load(owner)
        # Keep captured storage alive even if a reload incorrectly replaces it.
        captured_weights = (attn.w_kc, attn.w_vc)
        query, latent = values((2, 3, 4)), values((2, 3, 16))

        def project():
            return torch.bmm(query, attn.w_kc), torch.bmm(latent, attn.w_vc)

        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            project()
        torch.cuda.current_stream().wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            captured = project()
        graph.replay()
        initial = [output.clone() for output in captured]

        for reload in range(2):
            attn.kv_b_proj.weight.add_(0.125)
            post_load(owner)
            expected = project()
            graph.replay()
            torch.cuda.synchronize()
            for index, name in enumerate(("w_kc", "w_vc")):
                with self.subTest(reload=reload, weight=name):
                    self.assertFalse(torch.equal(expected[index], initial[index]))
                    torch.testing.assert_close(
                        captured[index], expected[index], rtol=0, atol=0
                    )
                    self.assertEqual(
                        (attn.w_kc, attn.w_vc)[index].data_ptr(),
                        captured_weights[index].data_ptr(),
                    )

    def test_block_fp8_weight_is_dequantized_for_tiny_gemm(self):
        module = SimpleNamespace(
            weight=torch.nn.Parameter(
                torch.ones((130, 129), device="cuda", dtype=torch.float8_e4m3fn),
                requires_grad=False,
            ),
            weight_scale_inv=torch.nn.Parameter(
                torch.tensor([[1.0, 2.0], [3.0, 4.0]], device="cuda"),
                requires_grad=False,
            ),
            quant_method=SimpleNamespace(weight_block_size=[128, 128]),
            params_dtype=torch.bfloat16,
        )

        weight = _get_k3_dense_weight(module)

        self.assertEqual(weight.dtype, torch.bfloat16)
        torch.testing.assert_close(
            weight[[0, 0, 128, 128], [0, 128, 0, 128]].float(),
            torch.tensor([1.0, 2.0, 3.0, 4.0], device="cuda"),
        )

    def test_mxfp8_weight_is_dequantized_for_tiny_gemm(self):
        """MXFP8 scale bytes are UE8M0 exponents, not multipliers; the dense
        tiny-GEMM weight must apply 2**(byte - 127) per row and 32-wide group."""
        # Two rows x two 32-wide groups; UE8M0 exponents 127 / 128 / 129 / 126
        # are scales 1 / 2 / 4 / 0.5.
        module = SimpleNamespace(
            weight=torch.nn.Parameter(
                torch.ones((2, 64), device="cuda", dtype=torch.float8_e4m3fn),
                requires_grad=False,
            ),
            weight_scale_inv=torch.nn.Parameter(
                torch.tensor(
                    [[127, 128], [129, 126]], device="cuda", dtype=torch.uint8
                ),
                requires_grad=False,
            ),
            quant_method=SimpleNamespace(weight_block_size=[1, 32]),
            params_dtype=torch.bfloat16,
        )

        weight = _get_k3_dense_weight(module)

        self.assertEqual(weight.dtype, torch.bfloat16)
        torch.testing.assert_close(
            weight[[0, 0, 1, 1], [0, 32, 31, 63]].float(),
            torch.tensor([1.0, 2.0, 4.0, 0.5], device="cuda"),
        )

    def test_per_tensor_fp8_weight_is_not_block_dequantized(self):
        weight = torch.nn.Parameter(
            torch.ones((2, 2), device="cuda", dtype=torch.float8_e4m3fn),
            requires_grad=False,
        )
        module = SimpleNamespace(
            weight=weight, weight_scale=torch.ones(1, device="cuda")
        )

        self.assertEqual(_get_k3_dense_weight(module).data_ptr(), weight.data_ptr())

    def test_cuda_kda_inproj_merge_and_dispatch(self):
        """CUDA merges [qkvg | f_a | b] into one view-backed buffer; side-stream
        overlap stays active for small graphed M and falls over to the single
        merged GEMM when side-stream overlap is inactive (M > limit or eager)."""
        from sglang.srt.layers.quantization.unquant import UnquantizedLinearMethod

        qm = UnquantizedLinearMethod()
        gen = torch.Generator(device="cuda").manual_seed(7)

        def _randn(*shape):
            return (
                torch.randn(*shape, generator=gen, device="cuda", dtype=torch.float32)
                .mul(0.05)
                .to(torch.bfloat16)
            )

        class _CallableLinear:
            def __init__(self, rows, cols):
                self.weight = _randn(rows, cols)
                self.bias = None
                self.quant_method = qm

            def __call__(self, x):
                return qm.apply(self, x, None), None

        owner = SimpleNamespace(
            use_full_rank_gate=True,
            _qkvg_gate_float_shard=False,
            _bfa_uses_block_fp8=False,
            _bfa_uses_mxfp8=False,
            _kda_dwdp=False,
            fused_qkvg_proj=_CallableLinear(_QKVG, _H),
            f_a_proj=_CallableLinear(_N_FA, _H),
            b_proj=_CallableLinear(_N_B, _H),
            f_b_proj=_CallableLinear(1536, _N_FA),
            split_sizes=[3 * 1536, 1536],
            _bfa_w=None,
            _bfa_f_b_w=None,
            _qkvgbfa_layer=None,
            _qkvgbfa_sizes=None,
            _bfa_alt_stream=torch.cuda.Stream(),
            _bfa_bs_limit=8,
            _bfa_side_tgv_m_ranges=None,
        )
        ref_qkvg = owner.fused_qkvg_proj.weight.clone()
        ref_fa = owner.f_a_proj.weight.clone()
        ref_b = owner.b_proj.weight.clone()
        ref_fb = owner.f_b_proj.weight.clone()

        owner._may_fuse_kda_inproj = lambda: KimiK3DeltaAttention._may_fuse_kda_inproj(
            owner
        )
        owner._merge_kda_inproj_weights = (
            lambda: KimiK3DeltaAttention._merge_kda_inproj_weights(owner)
        )
        KimiK3DeltaAttention._merge_bfa_weights(owner)

        self.assertIsNotNone(owner._qkvgbfa_layer)
        self.assertEqual(owner._qkvgbfa_layer.weight.shape, (6288, _H))
        self.assertEqual(owner._bfa_w.shape, (_BFA_W_ROWS, _H))
        self.assertEqual(
            owner.fused_qkvg_proj.weight.data_ptr(),
            owner._qkvgbfa_layer.weight.data_ptr(),
        )

        # M > _bfa_bs_limit (and eager) must take the single merged GEMM and match
        # the separate reference projections.
        for T, capture_mode in ((4, False), (16, True)):
            with (
                self.subTest(T=T, capture_mode=capture_mode),
                patch(
                    "sglang.srt.models.kimi_k3.get_is_capture_mode",
                    return_value=capture_mode,
                ),
            ):
                x = _randn(T, _H)
                qkv, beta, fg, g = KimiK3DeltaAttention.forward_qkvbfg_fused(owner, x)
                exp_qkvg = torch.nn.functional.linear(x, ref_qkvg)
                exp_qkv, exp_g = exp_qkvg.split([3 * 1536, 1536], dim=-1)
                exp_fa = torch.nn.functional.linear(x, ref_fa)
                exp_b = torch.nn.functional.linear(x, ref_b)
                exp_fg = torch.nn.functional.linear(exp_fa, ref_fb)
                torch.testing.assert_close(qkv, exp_qkv, rtol=1e-2, atol=2e-3)
                torch.testing.assert_close(g, exp_g, rtol=1e-2, atol=2e-3)
                torch.testing.assert_close(beta, exp_b, rtol=1e-2, atol=2e-3)
                torch.testing.assert_close(fg, exp_fg, rtol=2e-2, atol=4e-3)

    def test_cuda_mla_qkv_a_gate_merge_and_strided_gate(self):
        """CUDA merges [fused_qkv_a_proj_with_mqa | g_proj] into one view-backed
        buffer for breakable prefill graphs / eager / M > limit, and strided
        mla_output_gate matches torch.sigmoid + mul bit-for-bit."""
        from sglang.kernels.ops.elementwise import mla_output_gate
        from sglang.srt.layers.quantization.unquant import UnquantizedLinearMethod

        qm = UnquantizedLinearMethod()
        gen = torch.Generator(device="cuda").manual_seed(11)

        def _randn(*shape):
            return (
                torch.randn(*shape, generator=gen, device="cuda", dtype=torch.float32)
                .mul(0.05)
                .to(torch.bfloat16)
            )

        qkv_a_w = _randn(2112, _H)
        gate_w = _randn(1536, _H)
        ref_qkv_a_w = qkv_a_w.clone()
        ref_gate_w = gate_w.clone()

        owner = SimpleNamespace(
            use_output_gate=True,
            has_fused_proj=True,
            _kimi_split_gguf_kv_b=False,
            fused_qkv_a_proj_with_mqa=SimpleNamespace(
                weight=qkv_a_w, bias=None, quant_method=qm
            ),
            g_proj=SimpleNamespace(weight=gate_w, bias=None, quant_method=qm),
            _qkv_a_gate_layer=None,
            _qkv_a_gate_sizes=None,
            _gate_alt_stream=torch.cuda.Stream(),
            _gate_bs_limit=8,
            _gate_pending_stream=None,
            _gate_hidden_states=None,
            _use_merged_qkv_a_gate=False,
            _precomputed_output_gate=None,
        )
        self.assertTrue(KimiK3MLAAttention._merge_qkv_a_gate_weights(owner))
        self.assertEqual(owner._qkv_a_gate_layer.weight.shape, (3648, _H))
        self.assertEqual(
            owner.fused_qkv_a_proj_with_mqa.weight.data_ptr(),
            owner._qkv_a_gate_layer.weight.data_ptr(),
        )
        self.assertEqual(
            owner.g_proj.weight.data_ptr(),
            owner._qkv_a_gate_layer.weight[2112:].data_ptr(),
        )

        # In breakable prefill graph or M > _gate_bs_limit, _fork_output_gate
        # stays inactive and prepare_qkv_latent computes both in one GEMM.
        x = _randn(16, _H)
        attn_out = _randn(16, 1536)
        owner._gate_hidden_states = x
        with (
            patch(
                "sglang.srt.models.kimi_k3.get_is_capture_mode",
                return_value=True,
            ),
            patch(
                "sglang.srt.models.kimi_k3.is_in_breakable_cuda_graph",
                return_value=True,
            ),
            patch(
                "sglang.srt.layers.layer_boundary.get_attn_tp_context",
                return_value=SimpleNamespace(input_scattered=False),
            ),
        ):
            KimiK3MLAAttention._fork_output_gate(owner, x)
            self.assertIsNone(owner._gate_pending_stream)
            owner._use_merged_qkv_a_gate = True
            qkv_a = KimiK3MLAAttention.prepare_qkv_latent(owner, x, None)
            gate = KimiK3MLAAttention._compute_output_gate(owner, x)

        self.assertFalse(gate.is_contiguous())
        self.assertTrue(mla_output_gate.covered(attn_out, gate))
        gated = mla_output_gate.kimi_k3_mla_output_gate(attn_out, gate)
        expected_gated = attn_out * torch.sigmoid(gate.contiguous())
        self.assertTrue(torch.equal(gated, expected_gated))
        torch.testing.assert_close(
            qkv_a, torch.nn.functional.linear(x, ref_qkv_a_w), rtol=1e-2, atol=2e-3
        )
        torch.testing.assert_close(
            gate, torch.nn.functional.linear(x, ref_gate_w), rtol=1e-2, atol=2e-3
        )

    def test_cuda_merged_projections_fallback_for_lora_wrappers(self):
        """Runtime LoRA replacement after weight load (and M > cap) must skip
        the merged quant_method.apply path and dispatch adapter-aware wrappers."""
        from sglang.srt.layers.quantization.unquant import UnquantizedLinearMethod
        from sglang.srt.models.deepseek_v2 import DeepseekV2AttentionMLA

        qm = UnquantizedLinearMethod()
        gen = torch.Generator(device="cuda").manual_seed(13)

        def _randn(*shape):
            return (
                torch.randn(*shape, generator=gen, device="cuda", dtype=torch.float32)
                .mul(0.05)
                .to(torch.bfloat16)
            )

        class _LoRAWrapper:
            """Mimics BaseLayerWithLoRA: has base_layer & weight, no quant_method."""

            def __init__(self, base_layer, delta: float):
                self.base_layer = base_layer
                self.weight = base_layer.weight
                self.bias = None
                self.delta = delta
                self.calls = 0

            def __call__(self, x):
                self.calls += 1
                return torch.nn.functional.linear(x, self.weight) + self.delta, None

        class _StubMLA(KimiK3MLAAttention):
            def __init__(self, **attrs):
                torch.nn.Module.__init__(self)
                for key, value in attrs.items():
                    setattr(self, key, value)

        base_qkv_a = SimpleNamespace(
            weight=_randn(2112, _H), bias=None, quant_method=qm
        )
        base_g = SimpleNamespace(weight=_randn(1536, _H), bias=None, quant_method=qm)
        mla = _StubMLA(
            use_output_gate=True,
            has_fused_proj=True,
            _kimi_split_gguf_kv_b=False,
            fused_qkv_a_proj_with_mqa=base_qkv_a,
            g_proj=base_g,
            _qkv_a_gate_layer=None,
            _qkv_a_gate_sizes=None,
            _gate_alt_stream=torch.cuda.Stream(),
            _gate_bs_limit=8,
            _gate_pending_stream=None,
            _gate_hidden_states=None,
            _use_merged_qkv_a_gate=False,
            _precomputed_output_gate=None,
        )
        self.assertTrue(mla._merge_qkv_a_gate_weights())
        x = _randn(16, _H)
        mla._gate_hidden_states = x
        mla._use_merged_qkv_a_gate = True

        # 1) LoRA wraps fused_qkv_a_proj_with_mqa after weight loading.
        lora_qkv_a = _LoRAWrapper(base_qkv_a, delta=0.25)
        mla.fused_qkv_a_proj_with_mqa = lora_qkv_a
        with (
            patch(
                "sglang.srt.layers.layer_boundary.get_attn_tp_context",
                return_value=SimpleNamespace(input_scattered=False),
            ),
            patch.object(
                DeepseekV2AttentionMLA,
                "prepare_qkv_latent",
                lambda self_, hs, fb: self_.fused_qkv_a_proj_with_mqa(hs)[0],
            ),
        ):
            out_qkv_a = mla.prepare_qkv_latent(x, None)
        self.assertEqual(lora_qkv_a.calls, 1)
        self.assertIsNone(mla._precomputed_output_gate)
        torch.testing.assert_close(
            out_qkv_a,
            torch.nn.functional.linear(x, base_qkv_a.weight) + 0.25,
            rtol=1e-2,
            atol=2e-3,
        )

        # 2) LoRA wraps only g_proj; prepare_qkv_latent must still fall back so
        #    _compute_output_gate executes the g_proj LoRA adapter.
        mla.fused_qkv_a_proj_with_mqa = base_qkv_a
        lora_g = _LoRAWrapper(base_g, delta=0.5)
        mla.g_proj = lora_g
        with (
            patch(
                "sglang.srt.layers.layer_boundary.get_attn_tp_context",
                return_value=SimpleNamespace(input_scattered=False),
            ),
            patch.object(
                DeepseekV2AttentionMLA,
                "prepare_qkv_latent",
                lambda self_, hs, fb: torch.nn.functional.linear(
                    hs, self_.fused_qkv_a_proj_with_mqa.weight
                ),
            ),
        ):
            mla.prepare_qkv_latent(x, None)
            out_gate = mla._compute_output_gate(x)
        self.assertEqual(lora_g.calls, 1)
        torch.testing.assert_close(
            out_gate,
            torch.nn.functional.linear(x, base_g.weight) + 0.5,
            rtol=1e-2,
            atol=2e-3,
        )

        # 3) Token count above _MERGED_QKV_A_GATE_MAX_TOKENS falls back too.
        mla.g_proj = base_g
        mla._MERGED_QKV_A_GATE_MAX_TOKENS = 8
        fallback_calls = 0

        def _super_prepare(self_, hs, fb):
            nonlocal fallback_calls
            fallback_calls += 1
            return torch.nn.functional.linear(
                hs, self_.fused_qkv_a_proj_with_mqa.weight
            )

        with (
            patch(
                "sglang.srt.layers.layer_boundary.get_attn_tp_context",
                return_value=SimpleNamespace(input_scattered=False),
            ),
            patch.object(
                DeepseekV2AttentionMLA,
                "prepare_qkv_latent",
                _super_prepare,
            ),
        ):
            mla.prepare_qkv_latent(x, None)
        self.assertEqual(fallback_calls, 1)
        self.assertIsNone(mla._precomputed_output_gate)

        # 4) KDA: LoRA wraps fused_qkvg_proj after _merge_bfa_weights().
        base_qkvg = SimpleNamespace(
            weight=_randn(_QKVG, _H), bias=None, quant_method=qm
        )
        kda = SimpleNamespace(
            use_full_rank_gate=True,
            _qkvg_gate_float_shard=False,
            _bfa_uses_block_fp8=False,
            _bfa_uses_mxfp8=False,
            _kda_dwdp=False,
            fused_qkvg_proj=base_qkvg,
            f_a_proj=SimpleNamespace(
                weight=_randn(_N_FA, _H), bias=None, quant_method=qm
            ),
            b_proj=SimpleNamespace(weight=_randn(_N_B, _H), bias=None, quant_method=qm),
            f_b_proj=SimpleNamespace(
                weight=_randn(1536, _N_FA), bias=None, quant_method=qm
            ),
            split_sizes=[3 * 1536, 1536],
            _bfa_w=None,
            _bfa_f_b_w=None,
            _qkvgbfa_layer=None,
            _qkvgbfa_sizes=None,
            _bfa_alt_stream=torch.cuda.Stream(),
            _bfa_bs_limit=8,
            _bfa_side_tgv_m_ranges=None,
        )
        KimiK3DeltaAttention._merge_bfa_weights(kda)
        self.assertIsNotNone(kda._qkvgbfa_layer)
        lora_qkvg = _LoRAWrapper(base_qkvg, delta=0.125)
        kda.fused_qkvg_proj = lora_qkvg
        qkv, _beta, _fg, g = KimiK3DeltaAttention.forward_qkvbfg_fused(kda, x)
        self.assertEqual(lora_qkvg.calls, 1)
        exp_qkv, exp_g = (
            torch.nn.functional.linear(x, base_qkvg.weight) + 0.125
        ).split([3 * 1536, 1536], dim=-1)
        torch.testing.assert_close(qkv, exp_qkv, rtol=1e-2, atol=2e-3)
        torch.testing.assert_close(g, exp_g, rtol=1e-2, atol=2e-3)

    def test_cuda_fp16_preserves_existing_layout_at_tp16(self):
        """TP16 + float16 must keep the existing 136-row [f_a|b] tail layout so
        1-token capture never hits BF16-only tiny_gemm_bf16 at 144 rows."""
        from sglang.srt.layers.quantization.unquant import UnquantizedLinearMethod

        qm = UnquantizedLinearMethod()
        gen = torch.Generator(device="cuda").manual_seed(17)

        def _randn_fp16(*shape):
            return (
                torch.randn(*shape, generator=gen, device="cuda", dtype=torch.float32)
                .mul(0.05)
                .to(torch.float16)
            )

        class _CallableLinear:
            def __init__(self, rows, cols):
                self.weight = _randn_fp16(rows, cols)
                self.bias = None
                self.quant_method = qm

            def __call__(self, x):
                return qm.apply(self, x, None), None

        # TP16 KDA geometry: local_heads=6 -> qkvg=3072, f_a=128, b=6 (tail=134 -> pad8=136)
        owner = SimpleNamespace(
            use_full_rank_gate=True,
            _qkvg_gate_float_shard=False,
            _bfa_uses_block_fp8=False,
            _bfa_uses_mxfp8=False,
            _kda_dwdp=False,
            fused_qkvg_proj=_CallableLinear(3072, _H),
            f_a_proj=_CallableLinear(128, _H),
            b_proj=_CallableLinear(6, _H),
            f_b_proj=_CallableLinear(768, 128),
            split_sizes=[3 * 768, 768],
            _bfa_w=None,
            _bfa_f_b_w=None,
            _qkvgbfa_layer=None,
            _qkvgbfa_sizes=None,
            _bfa_alt_stream=torch.cuda.Stream(),
            _bfa_bs_limit=8,
            _bfa_side_tgv_m_ranges=None,
        )
        KimiK3DeltaAttention._merge_bfa_weights(owner)
        self.assertIsNone(owner._qkvgbfa_layer)
        self.assertEqual(tuple(owner._bfa_w.shape), (136, _H))

        x = _randn_fp16(1, _H)
        with patch(
            "sglang.srt.models.kimi_k3.get_is_capture_mode",
            return_value=True,
        ):
            qkv, beta, fg, g = KimiK3DeltaAttention.forward_qkvbfg_fused(owner, x)
        self.assertEqual(qkv.dtype, torch.float16)
        self.assertEqual(beta.shape, (1, 6))
        self.assertEqual(fg.shape, (1, 768))
        self.assertEqual(g.shape, (1, 768))

        mla = SimpleNamespace(
            use_output_gate=True,
            has_fused_proj=True,
            _kimi_split_gguf_kv_b=False,
            fused_qkv_a_proj_with_mqa=SimpleNamespace(
                weight=_randn_fp16(2112, _H), bias=None, quant_method=qm
            ),
            g_proj=SimpleNamespace(
                weight=_randn_fp16(768, _H), bias=None, quant_method=qm
            ),
            _qkv_a_gate_layer=None,
            _qkv_a_gate_sizes=None,
        )
        self.assertFalse(KimiK3MLAAttention._merge_qkv_a_gate_weights(mla))
        self.assertIsNone(mla._qkv_a_gate_layer)

    def test_cuda_merge_skips_cpu_offloaded_weights(self):
        """Mixed CPU/CUDA and all-CPU offloaded weights must skip CUDA weight merges."""
        from sglang.srt.layers.quantization.unquant import UnquantizedLinearMethod

        qm = UnquantizedLinearMethod()

        def _mod(rows, device):
            return SimpleNamespace(
                weight=torch.zeros((rows, 64), device=device, dtype=torch.bfloat16),
                bias=None,
                quant_method=qm,
            )

        for devs in (("cpu", "cuda"), ("cuda", "cpu"), ("cpu", "cpu")):
            with self.subTest(mla_devices=devs):
                mla = SimpleNamespace(
                    use_output_gate=True,
                    has_fused_proj=True,
                    _kimi_split_gguf_kv_b=False,
                    fused_qkv_a_proj_with_mqa=_mod(32, devs[0]),
                    g_proj=_mod(16, devs[1]),
                    _qkv_a_gate_layer=None,
                    _qkv_a_gate_sizes=None,
                )
                self.assertFalse(KimiK3MLAAttention._merge_qkv_a_gate_weights(mla))
                self.assertIsNone(mla._qkv_a_gate_layer)

        for devs in (
            ("cpu", "cuda", "cuda"),
            ("cuda", "cpu", "cuda"),
            ("cpu", "cpu", "cpu"),
        ):
            with self.subTest(kda_devices=devs):
                kda = SimpleNamespace(
                    use_full_rank_gate=True,
                    _qkvg_gate_float_shard=False,
                    _bfa_uses_block_fp8=False,
                    _bfa_uses_mxfp8=False,
                    _kda_dwdp=False,
                    fused_qkvg_proj=_mod(64, devs[0]),
                    f_a_proj=_mod(16, devs[1]),
                    b_proj=_mod(4, devs[2]),
                    split_sizes=[48, 16],
                    _qkvgbfa_layer=None,
                    _qkvgbfa_sizes=None,
                )
                self.assertFalse(KimiK3DeltaAttention._merge_kda_inproj_weights(kda))
                self.assertIsNone(kda._qkvgbfa_layer)

    def test_k31_verify_binding_shares_merged_projection_storage(self):
        """K3.1 verify binding must reuse stock merged KDA/MLA buffers in place
        at both TP8 and TP16 (zero duplicate allocation) and retarget on mismatch."""
        from sglang.srt.kernel_plugins.nvidia.kimi_k31_verify_binding import (
            _pack_or_reuse_projection_weights,
        )
        from sglang.srt.layers.quantization.unquant import UnquantizedLinearMethod

        qm = UnquantizedLinearMethod()
        cols = 64

        def _param_mod(rows):
            return SimpleNamespace(
                weight=torch.nn.Parameter(
                    torch.randn((rows, cols), device="cuda", dtype=torch.bfloat16)
                    * 0.05,
                    requires_grad=False,
                ),
                bias=None,
                quant_method=qm,
            )

        # Both TP8 (qkvg=6144, fa=128, b=12 -> 6288; qkv_a=2112, g=1536 -> 3648)
        # and TP16 (qkvg=3072, fa=128, b=6 -> 3208; qkv_a=2112, g=768 -> 2880).
        for tp, qkvg_r, b_r, g_r in ((8, 6144, 12, 1536), (16, 3072, 6, 768)):
            with self.subTest(tp=tp):
                kda = SimpleNamespace(
                    use_full_rank_gate=True,
                    _qkvg_gate_float_shard=False,
                    _bfa_uses_block_fp8=False,
                    _bfa_uses_mxfp8=False,
                    _kda_dwdp=False,
                    fused_qkvg_proj=_param_mod(qkvg_r),
                    f_a_proj=_param_mod(128),
                    b_proj=_param_mod(b_r),
                    f_b_proj=_param_mod(b_r * 128),
                    split_sizes=[3 * b_r * 128, b_r * 128],
                    _bfa_w=None,
                    _bfa_f_b_w=None,
                    _qkvgbfa_layer=None,
                    _qkvgbfa_sizes=None,
                )
                self.assertTrue(KimiK3DeltaAttention._merge_kda_inproj_weights(kda))
                orig_kda_w = kda._qkvgbfa_layer.weight
                parts = (
                    kda.fused_qkvg_proj.weight,
                    kda.f_a_proj.weight,
                    kda.b_proj.weight,
                )
                padding = -sum(p.shape[0] for p in parts) % 8
                packed = _pack_or_reuse_projection_weights(
                    kda, parts, padding, "_qkvgbfa_layer", "_qkvgbfa_sizes"
                )
                self.assertIs(packed, orig_kda_w)
                self.assertEqual(packed.data_ptr(), orig_kda_w.data_ptr())
                self.assertIs(kda._k31_verify_projection_storage, orig_kda_w)
                self.assertEqual(
                    kda.fused_qkvg_proj.weight.data_ptr(), packed.data_ptr()
                )
                self.assertEqual(
                    kda.f_a_proj.weight.data_ptr(), packed[qkvg_r:].data_ptr()
                )

                mla = SimpleNamespace(
                    use_output_gate=True,
                    has_fused_proj=True,
                    _kimi_split_gguf_kv_b=False,
                    fused_qkv_a_proj_with_mqa=_param_mod(2112),
                    g_proj=_param_mod(g_r),
                    _qkv_a_gate_layer=None,
                    _qkv_a_gate_sizes=None,
                )
                self.assertTrue(KimiK3MLAAttention._merge_qkv_a_gate_weights(mla))
                orig_mla_w = mla._qkv_a_gate_layer.weight
                mla_parts = (mla.fused_qkv_a_proj_with_mqa.weight, mla.g_proj.weight)
                mla_packed = _pack_or_reuse_projection_weights(
                    mla, mla_parts, 0, "_qkv_a_gate_layer", "_qkv_a_gate_sizes"
                )
                self.assertIs(mla_packed, orig_mla_w)
                self.assertEqual(mla_packed.data_ptr(), orig_mla_w.data_ptr())
                self.assertIs(mla._k31_verify_projection_storage, orig_mla_w)

        # Mismatched existing buffer must be retargeted so only one allocation lives.
        stale = torch.zeros((48, cols), device="cuda", dtype=torch.bfloat16)
        mla_mismatch = SimpleNamespace(
            fused_qkv_a_proj_with_mqa=_param_mod(24),
            g_proj=_param_mod(16),
            _qkv_a_gate_layer=SimpleNamespace(weight=stale),
            _qkv_a_gate_sizes=[24, 16, 8],
        )
        m_parts = (
            mla_mismatch.fused_qkv_a_proj_with_mqa.weight,
            mla_mismatch.g_proj.weight,
        )
        repacked = _pack_or_reuse_projection_weights(
            mla_mismatch, m_parts, 0, "_qkv_a_gate_layer", "_qkv_a_gate_sizes"
        )
        self.assertIsNot(repacked, stale)
        self.assertIs(mla_mismatch._qkv_a_gate_layer.weight, repacked)
        self.assertEqual(mla_mismatch._qkv_a_gate_sizes[-1], 0)
        self.assertEqual(
            mla_mismatch.fused_qkv_a_proj_with_mqa.weight.data_ptr(),
            repacked.data_ptr(),
        )


if __name__ == "__main__":
    unittest.main()
