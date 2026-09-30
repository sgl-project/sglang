"""Single-GPU numerical checks for native Beam on the deterministic fixture.

Each case guards a precision decision of the serving policy: the BF16 head
projection ahead of FP32 output processing, FP32 rotary and fused-activation
intermediates under both eager and full-graph Inductor execution, the fused
Q/K norm and rotation against the unfused rounding up to 1M positions, the
single-rounding CUDA SSMax, the FP32 routing coefficients and runner-side
routed scale for a one-token compiled MoE, and the batch-one compiled decode
path against the eager oracle.
"""

import copy
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch
import torch.nn.functional as F
from torch._inductor import config as inductor_config

import sglang as sgl
from sglang.kernels.fused_op import BaseFusedOp
from sglang.kernels.ops.attention.fused_qknorm_rope import fused_qk_norm_rope
from sglang.srt.layers.logits_processor import LogitsMetadata, LogitsProcessor
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.model_loader.utils import set_default_torch_dtype
from sglang.srt.models.beam import (
    BeamAttention,
    BeamMLP,
    BeamMoE,
    apply_ssmax,
    compute_ssmax_scale,
)
from sglang.srt.runtime_context import get_context, get_parallel
from sglang.srt.utils import is_flashinfer_available
from sglang.srt.utils.common import (
    is_sm90_supported,
    is_sm100_supported,
    is_sm120_supported,
)
from sglang.test.beam_fixture import (
    assert_logprobs_close,
    eager_beam_logits,
    engine_token_logprobs,
    make_beam_fixture,
)
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=150, stage="base-b", runner_config="1-gpu-large")


class TestBeamNumerics(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        if not torch.cuda.is_available():
            raise unittest.SkipTest("Beam numerics tests require a CUDA GPU")
        cls.checkpoint = tempfile.TemporaryDirectory(prefix="beam-fixture-")
        cls.config, state = make_beam_fixture(cls.checkpoint.name)
        cls.state = {name: value.cuda() for name, value in state.items()}

    @classmethod
    def tearDownClass(cls):
        if hasattr(cls, "state"):
            del cls.state
        if hasattr(cls, "checkpoint"):
            cls.checkpoint.cleanup()

    def test_bf16_head_projection_precedes_fp32_output_processing(self):
        hidden = self.state["model.embed_tokens.weight"][:3]
        weight = self.state["lm_head.weight"]
        # The projection rounds to BF16 before the processor promotes its output
        # for FP32 softcap, probabilities and logprobs.
        with (
            get_context().override_server_args(
                enable_fp32_lm_head=False,
                enable_dp_lm_head=False,
            ),
            get_parallel().override(
                tp_size=1,
                attn_tp_size=1,
                attn_dp_size=1,
            ),
        ):
            processor = LogitsProcessor(self.config)
            head = SimpleNamespace(weight=weight)
            actual = processor._compute_lm_head(hidden, head)
            capped = processor._get_logits(
                hidden,
                head,
                LogitsMetadata(forward_mode=ForwardMode.DECODE),
                use_logits_buffer=False,
            )
        expected = F.linear(hidden, weight)
        self.assertEqual(weight.dtype, torch.bfloat16)
        self.assertEqual(hidden.dtype, torch.bfloat16)
        self.assertEqual(actual.dtype, torch.bfloat16)
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)
        self.assertFalse(
            torch.equal(actual.float(), F.linear(hidden.float(), weight.float()))
        )

        cap = self.config.final_logit_softcapping
        expected_capped = cap * torch.tanh(expected.float() / cap)
        self.assertEqual(capped.dtype, torch.float32)
        torch.testing.assert_close(capped, expected_capped, atol=1e-5, rtol=1e-5)
        bf16_capped = (cap * torch.tanh(expected / cap)).float()
        self.assertGreater((expected_capped - bf16_capped).abs().max().item(), 1e-3)
        logprobs, probabilities = capped.log_softmax(-1), capped.softmax(-1)
        self.assertEqual(logprobs.dtype, torch.float32)
        self.assertEqual(probabilities.dtype, torch.float32)
        torch.testing.assert_close(
            logprobs, expected_capped.log_softmax(-1), atol=1e-5, rtol=1e-5
        )
        torch.testing.assert_close(
            probabilities, expected_capped.softmax(-1), atol=1e-7, rtol=1e-5
        )

    @torch.inference_mode()
    def test_native_and_compiled_rotation_and_dense_activation_precision(self):
        with (
            get_context().override_server_args(),
            get_parallel().override(
                tp_size=1, tp_rank=0, attn_tp_size=1, attn_tp_rank=0, attn_dp_size=1
            ),
            set_default_torch_dtype(torch.bfloat16),
            torch.device("cuda"),
        ):
            # Obtain the operators from real Beam attention/dense construction.
            attention = BeamAttention(self.config, layer_id=0)
            dense = BeamMLP(self.config)
        rotary, activation = attention.rotary_emb, dense.act_fn
        self.assertEqual(rotary.cos_sin_cache.dtype, torch.float32)
        positions = torch.tensor([511], device="cuda")
        # Powers of two avoid product-rounding ambiguity in the FP32 oracle.
        q = torch.ones(1, attention.q_size, device="cuda", dtype=torch.bfloat16)
        k = torch.full((1, attention.kv_size), 0.5, device="cuda", dtype=torch.bfloat16)
        gate_up = torch.cat(
            (
                torch.full((1, self.config.intermediate_size), 0.75, device="cuda"),
                torch.full((1, self.config.intermediate_size), 1.5, device="cuda"),
            ),
            dim=-1,
        ).bfloat16()
        cos, sin = rotary.cos_sin_cache[positions].chunk(2, dim=-1)

        def rotation_expected(value, *, round_coefficients):
            half = self.config.head_dim // 2
            shaped = value.reshape(1, -1, self.config.head_dim)
            first, second = shaped[..., :half], shaped[..., half:]
            c, s = cos[:, None], sin[:, None]
            if round_coefficients:
                c, s = c.bfloat16(), s.bfloat16()
            else:
                first, second = first.float(), second.float()
            return (
                torch.cat((first * c - second * s, first * s + second * c), dim=-1)
                .reshape_as(value)
                .bfloat16()
            )

        expected_q = rotation_expected(q, round_coefficients=False)
        expected_k = rotation_expected(k, round_coefficients=False)
        expected_activation = torch.full(
            (1, self.config.intermediate_size),
            0.765625,
            device="cuda",
            dtype=torch.bfloat16,
        )
        # These inputs distinguish the obsolete eager BF16 intermediates.
        self.assertFalse(
            torch.equal(expected_q, rotation_expected(q, round_coefficients=True))
        )
        gate, up = gate_up.chunk(2, dim=-1)
        self.assertFalse(torch.equal(expected_activation, F.silu(gate) * up))

        def run(positions, q, k, gate_up):
            # Native RoPE mutates Q/K, so repeated calls get independent copies.
            rotated_q, rotated_k = rotary(positions, q.clone(), k.clone())
            return rotated_q, rotated_k, activation(gate_up)

        eager = run(positions, q, k, gate_up)
        for actual, expected in zip(
            eager, (expected_q, expected_k, expected_activation)
        ):
            torch.testing.assert_close(actual, expected, atol=0, rtol=0)
        try:
            rotary.enter_torch_compile(1)
            activation.enter_torch_compile(1)

            def compiled_run(positions, q, k, gate_up):
                assert torch.compiler.is_compiling()
                return run(positions, q, k, gate_up)

            # Compile synchronously so the check does not depend on a worker pool.
            with inductor_config.patch(compile_threads=1):
                compiled = torch.compile(
                    compiled_run, backend="inductor", fullgraph=True
                )
                for _ in range(2):
                    actual = compiled(positions, q, k, gate_up)
                    for result, expected in zip(actual, eager):
                        torch.testing.assert_close(result, expected, atol=0, rtol=0)
        finally:
            activation.leave_torch_compile()
            rotary.leave_torch_compile()

    @torch.inference_mode()
    def test_fused_sliding_prologue_keeps_unfused_rounding_at_long_positions(self):
        config = copy.deepcopy(self.config)
        config.max_position_embeddings = 1 << 20
        with (
            get_context().override_server_args(),
            get_parallel().override(
                tp_size=1, tp_rank=0, attn_tp_size=1, attn_tp_rank=0, attn_dp_size=1
            ),
            set_default_torch_dtype(torch.bfloat16),
            torch.device("cuda"),
        ):
            attention = BeamAttention(config, layer_id=0)
        self.assertTrue(attention.use_fused_qk_norm_rope)
        generator = torch.Generator(device="cuda").manual_seed(47)
        for norm in (attention.q_norm, attention.k_norm):
            norm.weight.copy_(
                0.5 + torch.rand(norm.weight.shape, device="cuda", generator=generator)
            )
        head_dim = config.head_dim
        sizes = [attention.q_size, attention.kv_size, attention.kv_size]
        positions = torch.cat(
            [
                torch.arange(start, start + 256, device="cuda")
                for start in (0, 131072, 524288, (1 << 20) - 256)
            ]
        )
        shape = (positions.numel(), sum(sizes))
        qkv = (3 * torch.randn(shape, device="cuda", generator=generator)).bfloat16()

        q, k, _ = qkv.clone().split(sizes, dim=-1)
        q = attention.q_norm(q.reshape(-1, head_dim)).view(q.shape)
        k = attention.k_norm(k.reshape(-1, head_dim)).view(k.shape)
        unfused = torch.cat(attention.rotary_emb(positions, q, k), dim=-1)

        fused_qkv = qkv.clone()
        q, k, v = fused_qkv.split(sizes, dim=-1)
        attention._fused_qk_norm_rope(positions, q, k)
        torch.testing.assert_close(v, qkv[:, -attention.kv_size :], atol=0, rtol=0)
        fused = fused_qkv[:, : -attention.kv_size]

        # Norm reductions may still differ by one BF16 ULP in rare elements.
        self.assertLess((fused != unfused).float().mean().item(), 1e-3)
        torch.testing.assert_close(fused, unfused, atol=0.05, rtol=2**-7)

        # The fast-math kernel recomputes the angles and fails the same bound.
        approx_qkv = qkv.clone()
        fused_qk_norm_rope(
            qkv=approx_qkv,
            num_heads_q=attention.num_heads,
            num_heads_k=attention.num_kv_heads,
            num_heads_v=attention.num_kv_heads,
            head_dim=head_dim,
            eps=config.rms_norm_eps,
            q_weight=attention.q_norm.weight,
            k_weight=attention.k_norm.weight,
            base=config.rope_parameters["sliding_attention"]["rope_theta"],
            is_neox=True,
            position_ids=positions.int(),
            factor=1.0,
            low=0.0,
            high=0.0,
            attention_factor=1.0,
        )
        approx = approx_qkv[:, : -attention.kv_size]
        self.assertGreater((approx != unfused).float().mean().item(), 1e-3)

    @torch.inference_mode()
    def test_cuda_ssmax_rounds_once_on_packed_query_rows(self):
        positions = torch.tensor([0, 511, 2047, 8191, 262143, 1048575], device="cuda")
        generator = torch.Generator(device="cuda").manual_seed(48)
        q_size, kv_size = 6 * 128, 128
        qkv = torch.randn(
            positions.numel(),
            q_size + 2 * kv_size,
            device="cuda",
            generator=generator,
        ).bfloat16()
        q = qkv[:, :q_size]
        scale = compute_ssmax_scale(positions)
        expected = (q.float() * scale.unsqueeze(-1)).bfloat16()

        actual = apply_ssmax(q, scale)

        torch.testing.assert_close(actual, expected, atol=0, rtol=0)
        self.assertFalse(torch.equal(expected, q * scale.unsqueeze(-1).bfloat16()))

    @torch.inference_mode()
    def test_single_token_compiled_moe_preserves_runner_scale(self):
        # No process group is needed at TP1; these lookups only feed the disabled
        # symmetric-memory context. All routing and expert kernels remain real.
        group = SimpleNamespace(world_size=1)
        with (
            get_context().override_server_args(moe_runner_backend="triton"),
            get_parallel().override(
                tp_size=1,
                tp_rank=0,
                attn_tp_size=1,
                attn_dp_size=1,
                moe_tp_size=1,
                moe_tp_rank=0,
                moe_ep_size=1,
                moe_ep_rank=0,
            ),
            patch("sglang.srt.layers.moe.topk.get_tp_group", new=lambda: group),
            patch(
                "sglang.srt.layers.moe.fused_moe_triton.layer.get_tp_group",
                new=lambda: group,
            ),
            patch(
                "sglang.srt.layers.moe.moe_runner.triton_utils.fused_moe.get_tp_group",
                new=lambda: group,
            ),
            set_default_torch_dtype(torch.bfloat16),
            torch.device("cuda"),
        ):
            moe = BeamMoE(self.config, layer_id=1)
            hidden = torch.ones(1, self.config.hidden_size, dtype=torch.bfloat16)
            # Every expert maps ones to gate=.75, up=1.5 and then uses identity
            # down weights. Its unweighted fused output is exactly BF16 .765625.
            width = self.config.moe_intermediate_size
            self.assertEqual(width, self.config.hidden_size)
            moe.experts.w13_weight[:, :width].fill_(0.75 / self.config.hidden_size)
            moe.experts.w13_weight[:, width:].fill_(1.5 / self.config.hidden_size)
            moe.experts.w2_weight.copy_(
                torch.eye(width).expand(self.config.num_experts, -1, -1)
            )
            moe.gate.weight.zero_()
            moe.gate.weight[:, 0].copy_(torch.linspace(-1, 1, self.config.num_experts))
            moe.gate.e_score_correction_bias.copy_(
                self.state["model.layers.1.mlp.gate.e_score_correction_bias"]
            )
            moe.experts.quant_method.process_weights_after_loading(moe.experts)
            logits, _ = moe.gate(hidden.float())
            routing = moe.topk(hidden, logits)
            self.assertEqual(routing.topk_weights.dtype, torch.float32)
            self.assertFalse(
                torch.equal(
                    routing.topk_weights, routing.topk_weights.bfloat16().float()
                )
            )
            contributions = (0.765625 * routing.topk_weights).bfloat16().float()
            expected = (
                (
                    contributions.sum(-1, keepdim=True)
                    * self.config.routed_scaling_factor
                )
                .bfloat16()
                .expand_as(hidden)
            )
            unscaled = contributions.sum(-1, keepdim=True).bfloat16().expand_as(hidden)
            self.assertGreater((expected - unscaled).abs().min().item(), 0.5)
            eager = moe(hidden)
            torch.testing.assert_close(eager, expected, atol=0, rtol=0)

            operators = [
                module for module in moe.modules() if isinstance(module, BaseFusedOp)
            ]
            try:
                for operator in operators:
                    operator.enter_torch_compile(1)

                def compiled_forward(hidden):
                    assert torch.compiler.is_compiling()
                    output = moe(hidden)
                    # Expose the same-input native selector's coefficients from
                    # the compiled graph, without observing through Python hooks.
                    logits, _ = moe.gate(hidden.float())
                    weights = moe.topk(hidden, logits).topk_weights
                    return output, weights

                with inductor_config.patch(compile_threads=1):
                    compiled = torch.compile(
                        compiled_forward, backend="inductor", fullgraph=True
                    )
                    for _ in range(2):
                        actual, weights = compiled(hidden)
                        self.assertEqual(weights.dtype, torch.float32)
                        torch.testing.assert_close(
                            weights, routing.topk_weights, atol=0, rtol=0
                        )
                        torch.testing.assert_close(actual, expected, atol=0, rtol=0)
            finally:
                for operator in reversed(operators):
                    operator.leave_torch_compile()

    def test_single_token_compiled_decode(self):
        if not is_flashinfer_available() or not (
            is_sm90_supported() or is_sm100_supported() or is_sm120_supported()
        ):
            raise unittest.SkipTest(
                "Beam TRT-LLM MHA decode requires FlashInfer and SM90, SM100, or SM120"
            )
        engine = sgl.Engine(
            model_path=self.checkpoint.name,
            skip_tokenizer_init=True,
            dtype="bfloat16",
            tp_size=1,
            attention_backend="trtllm_mha",
            page_size=64,
            moe_runner_backend="triton",
            enable_torch_compile=True,
            torch_compile_max_bs=1,
            cuda_graph_backend_decode="full",
            cuda_graph_bs_decode=[1],
            disable_prefill_cuda_graph=True,
            disable_radix_cache=True,
            chunked_prefill_size=-1,
            max_running_requests=1,
            max_total_tokens=4096,
            mem_fraction_static=0.2,
        )
        try:
            # Exercise Beam's batch-one compile dispatch and CUDA graph replay.
            prompt = [1 + (i * 17) % 127 for i in range(6)]
            generated, actual = engine_token_logprobs(
                engine, prompt, 4, self.config.vocab_size
            )
            self.assertEqual(len(generated), 4)
            logits = eager_beam_logits(self.config, self.state, prompt + generated[:-1])
            expected = logits[len(prompt) - 1 :].log_softmax(-1)
            with self.subTest(comparison="reference"):
                assert_logprobs_close(
                    self, actual, expected, tag="compiled tp=1 prompt=6 reference"
                )
            for step in range(1, 4):
                with self.subTest(comparison="cached_vs_fresh", step=step):
                    _, fresh = engine_token_logprobs(
                        engine, prompt + generated[:step], 1, self.config.vocab_size
                    )
                    assert_logprobs_close(
                        self,
                        actual[step],
                        fresh[0],
                        tag=f"compiled tp=1 prompt=6 cached step={step}",
                    )
            with self.subTest(comparison="graph_replay"):
                repeated_ids, repeated = engine_token_logprobs(
                    engine, prompt, 4, self.config.vocab_size
                )
                self.assertEqual(repeated_ids, generated)
                assert_logprobs_close(
                    self, repeated, actual, tag="compiled tp=1 prompt=6 graph replay"
                )
        finally:
            engine.shutdown()


if __name__ == "__main__":
    unittest.main()
