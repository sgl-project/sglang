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

register_cuda_ci(est_time=15, stage="base-b", runner_config="1-gpu-large")

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
        _bfa_w=_randn(_BFA_W_ROWS, _H).contiguous(),
        _bfa_f_b_w=_randn(1536, _N_FA).contiguous(),
        _bfa_fa_size=_N_FA,
        _bfa_b_size=_N_B,
        fused_qkvg_proj=fused_qkvg_proj,
        split_sizes=[3 * 1536, 1536],
        _bfa_alt_stream=torch.cuda.Stream() if with_stream else None,
        _bfa_bs_limit=128 if with_stream else 0,
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

        def module(rows):
            return SimpleNamespace(
                weight=torch.ones(rows, 16, device="cuda", dtype=torch.bfloat16)
            )

        owner = SimpleNamespace(
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
        for block_fp8 in (False, True):
            with self.subTest(block_fp8=block_fp8):

                def module(rows):
                    return SimpleNamespace(
                        weight=torch.ones(rows, 16, device="cuda", dtype=torch.bfloat16)
                    )

                owner = SimpleNamespace(
                    use_full_rank_gate=True,
                    _bfa_uses_block_fp8=block_fp8,
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
            attn=layer, o_norm=SimpleNamespace(weight=ones(128), eps=1e-6)
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
                layers=[], output_attn_res_proj=proj, output_attn_res_norm=norm
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
        )
        owner = SimpleNamespace(
            config=SimpleNamespace(full_attention_layer_ids=[0]),
            model=SimpleNamespace(
                layers=[
                    SimpleNamespace(self_attn=attn, use_attn_residuals=False, mlp=None)
                ]
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

    def test_per_tensor_fp8_weight_is_not_block_dequantized(self):
        weight = torch.nn.Parameter(
            torch.ones((2, 2), device="cuda", dtype=torch.float8_e4m3fn),
            requires_grad=False,
        )
        module = SimpleNamespace(
            weight=weight, weight_scale=torch.ones(1, device="cuda")
        )

        self.assertEqual(_get_k3_dense_weight(module).data_ptr(), weight.data_ptr())


if __name__ == "__main__":
    unittest.main()
