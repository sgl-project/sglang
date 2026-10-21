"""Unit tests for FlashInfer CUTLASS MoE runner helpers."""

import inspect
import unittest
from unittest import mock

import torch

from sglang.srt.layers.moe.moe_runner.base import MoeRunnerConfig
from sglang.srt.layers.moe.moe_runner.flashinfer_cutlass import (
    FlashInferCutlassMoeQuantInfo,
    FlashInferCutlassMxfp4MoeQuantInfo,
    _fused_experts_flashinfer_mxfp4_cutlass,
    _run_flashinfer_cutlass,
    materialize_swiglu_params_for_cutlass,
)
from sglang.srt.layers.moe.token_dispatcher.standard import StandardDispatchOutput
from sglang.srt.layers.moe.topk import StandardTopKOutput
from sglang.srt.runtime_context import get_context, get_parallel
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")

GPT_OSS_ALPHA = 1.702
GPT_OSS_LIMIT = 7.0
NUM_LOCAL_EXPERTS = 4
HIDDEN = 128
INTERMEDIATE = 128
TOP_K = 2
STARTUP_AUTOTUNE_TOKENS = 256
CUTLASS_FUSED_MOE_IMPORT = (
    "sglang.srt.layers.moe.moe_runner.flashinfer_cutlass._flashinfer_cutlass_fused_moe"
)


class _SingleRankMoeLayer(torch.nn.Module):
    moe_tp_size = 1
    moe_tp_rank = 0
    moe_ep_size = 1
    moe_ep_rank = 0


class TestMaterializeSwigluParamsForCutlass(CustomTestCase):
    def _materialize(self, activation="silu", **config_kwargs):
        config = MoeRunnerConfig(activation=activation, is_gated=True, **config_kwargs)
        return materialize_swiglu_params_for_cutlass(
            config, num_local_experts=NUM_LOCAL_EXPERTS, device=torch.device("cpu")
        )

    def test_gpt_oss_triple(self):
        # GPT-OSS ships gemm1_alpha=1.702 with a 7.0 clamp and never sets
        # gemm1_beta; alpha implies the +1 up term.
        alpha, beta, limit = self._materialize(
            gemm1_alpha=GPT_OSS_ALPHA, gemm1_clamp_limit=GPT_OSS_LIMIT
        )
        # 1.702 is not exactly representable in fp32; compare with tolerance.
        self.assertTrue(torch.allclose(alpha, torch.full_like(alpha, GPT_OSS_ALPHA)))
        self.assertTrue(torch.allclose(beta, torch.ones_like(beta)))
        self.assertTrue(torch.allclose(limit, torch.full_like(limit, GPT_OSS_LIMIT)))

    def test_gpt_oss_triple_matches_reference_epilogue(self):
        # The reference epilogue is silu(gate) * clamp(alpha*up + 1, ±limit).
        # With the old beta=0.0 default the up term lost its +1.
        torch.manual_seed(0)
        gate = torch.randn(64, dtype=torch.float32)
        up = torch.randn(64, dtype=torch.float32)

        def reference(alpha_v, beta_v, limit_v):
            return torch.nn.functional.silu(gate) * torch.clamp(
                alpha_v * up + beta_v, -limit_v, limit_v
            )

        alpha, beta, limit = self._materialize(
            gemm1_alpha=GPT_OSS_ALPHA, gemm1_clamp_limit=GPT_OSS_LIMIT
        )
        got = reference(alpha[0], beta[0], limit[0])
        want = reference(GPT_OSS_ALPHA, 1.0, GPT_OSS_LIMIT)
        self.assertTrue(torch.allclose(got, want))
        wrong = reference(GPT_OSS_ALPHA, 0.0, GPT_OSS_LIMIT)
        self.assertFalse(torch.allclose(got, wrong))

    def test_explicit_beta_wins(self):
        _, beta, _ = self._materialize(
            gemm1_alpha=GPT_OSS_ALPHA,
            gemm1_beta=0.5,
            gemm1_clamp_limit=GPT_OSS_LIMIT,
        )
        self.assertEqual(beta.tolist(), [0.5] * NUM_LOCAL_EXPERTS)

    def test_no_alpha_stays_neutral(self):
        alpha, beta, _ = self._materialize(gemm1_clamp_limit=GPT_OSS_LIMIT)
        self.assertEqual(alpha.tolist(), [1.0] * NUM_LOCAL_EXPERTS)
        self.assertEqual(beta.tolist(), [0.0] * NUM_LOCAL_EXPERTS)

    def test_swiglu_limit_alias(self):
        _, _, limit = self._materialize(swiglu_limit=GPT_OSS_LIMIT)
        self.assertEqual(limit.tolist(), [GPT_OSS_LIMIT] * NUM_LOCAL_EXPERTS)

    def test_no_clamp_disables(self):
        self.assertEqual(
            self._materialize(gemm1_alpha=GPT_OSS_ALPHA), (None, None, None)
        )

    def test_situ_disables(self):
        # SiTU carries its clamp inside the activation itself.
        self.assertEqual(
            self._materialize(activation="situ", gemm1_clamp_limit=GPT_OSS_LIMIT),
            (None, None, None),
        )

    def test_dtype_and_shape(self):
        for tensor in self._materialize(
            gemm1_alpha=GPT_OSS_ALPHA, gemm1_clamp_limit=GPT_OSS_LIMIT
        ):
            self.assertEqual(tensor.dtype, torch.float32)
            self.assertEqual(tuple(tensor.shape), (NUM_LOCAL_EXPERTS,))


def _dispatch_output(num_tokens):
    return StandardDispatchOutput(
        hidden_states=torch.empty(num_tokens, HIDDEN, dtype=torch.bfloat16),
        hidden_states_scale=None,
        topk_output=StandardTopKOutput(
            topk_weights=torch.full(size=(num_tokens, TOP_K), fill_value=1.0 / TOP_K),
            topk_ids=torch.zeros(num_tokens, TOP_K, dtype=torch.int32),
            router_logits=torch.empty(num_tokens, NUM_LOCAL_EXPERTS),
        ),
    )


def _run_bf16_cutlass(num_tokens):
    _run_flashinfer_cutlass(
        dispatch_output=_dispatch_output(num_tokens),
        quant_info=FlashInferCutlassMoeQuantInfo(
            quant_type="bf16",
            w13_weight=torch.empty(
                NUM_LOCAL_EXPERTS, 2 * INTERMEDIATE, HIDDEN, dtype=torch.bfloat16
            ),
            w2_weight=torch.empty(
                NUM_LOCAL_EXPERTS, HIDDEN, INTERMEDIATE, dtype=torch.bfloat16
            ),
        ),
        runner_config=MoeRunnerConfig(layer=_SingleRankMoeLayer()),
    )


def _run_mxfp4_w4a16_cutlass(num_tokens):
    _fused_experts_flashinfer_mxfp4_cutlass(
        dispatch_output=_dispatch_output(num_tokens),
        quant_info=FlashInferCutlassMxfp4MoeQuantInfo(
            w13_weight=torch.empty(
                NUM_LOCAL_EXPERTS, 2 * INTERMEDIATE, HIDDEN // 2, dtype=torch.uint8
            ),
            w2_weight=torch.empty(
                NUM_LOCAL_EXPERTS, HIDDEN, INTERMEDIATE // 2, dtype=torch.uint8
            ),
            w13_weight_scale=torch.empty(
                NUM_LOCAL_EXPERTS, 2 * INTERMEDIATE, HIDDEN // 32, dtype=torch.uint8
            ),
            w2_weight_scale=torch.empty(
                NUM_LOCAL_EXPERTS, HIDDEN, INTERMEDIATE // 32, dtype=torch.uint8
            ),
        ),
        runner_config=MoeRunnerConfig(layer=_SingleRankMoeLayer()),
    )


def _captured_tune_max_num_tokens(run_site, num_tokens):
    from flashinfer.fused_moe.core import ActivationType

    kwargs_seen = {}

    def fused_moe_stub(**kwargs):
        kwargs_seen.update(kwargs)
        return [kwargs["output"]]

    with mock.patch(
        target=CUTLASS_FUSED_MOE_IMPORT,
        return_value=(fused_moe_stub, ActivationType),
    ):
        run_site(num_tokens)
    return kwargs_seen["tune_max_num_tokens"]


class TestCutlassMoeTuneMaxNumTokens(CustomTestCase):
    def test_cap_is_flashinfer_default(self):
        """The cap must stay FlashInfer's own default tune range for this call."""
        from flashinfer.fused_moe import cutlass_fused_moe

        self.assertEqual(
            inspect.signature(cutlass_fused_moe)
            .parameters["tune_max_num_tokens"]
            .default,
            8192,
        )

    def test_every_token_count_hits_a_startup_tuned_bucket(self):
        """A prefill-sized MoE call must look up a bucket the decode-shaped startup
        autotune tuned, up to the DP-gathered chunk capped at FlashInfer's default."""
        from flashinfer.fused_moe.utils import (
            get_hybrid_num_tokens_buckets,
            map_to_hybrid_bucket,
        )

        dp_attention = dict(tp_size=4, attn_dp_size=4)
        for fields, tuned_ceiling in (
            (dict(chunked_prefill_size=8192), 8192),
            (dict(chunked_prefill_size=-1), 8192),
            (dict(chunked_prefill_size=1024, **dp_attention), 4096),
            (dict(chunked_prefill_size=4096, **dp_attention), 8192),
            (dict(chunked_prefill_size=8192, disaggregation_mode="prefill"), 8192),
        ):
            with (
                get_context().override_server_args(**fields),
                get_parallel().override(tp_group=None),
            ):
                for run_site in (_run_bf16_cutlass, _run_mxfp4_w4a16_cutlass):
                    tuned = get_hybrid_num_tokens_buckets(
                        _captured_tune_max_num_tokens(
                            run_site=run_site, num_tokens=STARTUP_AUTOTUNE_TOKENS
                        )
                    )
                    for num_tokens in (1, 257, 777, 2048, 4097, 8192, 16384):
                        with self.subTest(
                            site=run_site.__name__, num_tokens=num_tokens, **fields
                        ):
                            bucket = map_to_hybrid_bucket(
                                x=num_tokens,
                                max_num_tokens=_captured_tune_max_num_tokens(
                                    run_site=run_site, num_tokens=num_tokens
                                ),
                            )
                            self.assertIn(bucket, tuned)
                            self.assertGreaterEqual(
                                bucket, min(num_tokens, tuned_ceiling)
                            )
                            self.assertLessEqual(bucket, tuned_ceiling)

    def test_pd_decode_server_sizes_each_call(self):
        """A PD decode server runs no prefill, so it must not pay for prefill-sized
        tuning: each call passes its own token count rounded up to a power of 2."""
        with (
            get_context().override_server_args(
                chunked_prefill_size=8192, disaggregation_mode="decode"
            ),
            get_parallel().override(tp_group=None),
        ):
            for run_site in (_run_bf16_cutlass, _run_mxfp4_w4a16_cutlass):
                for num_tokens, per_call in (
                    (1, 1),
                    (STARTUP_AUTOTUNE_TOKENS, 256),
                    (257, 512),
                    (777, 1024),
                    (16384, 16384),
                ):
                    with self.subTest(site=run_site.__name__, num_tokens=num_tokens):
                        self.assertEqual(
                            _captured_tune_max_num_tokens(
                                run_site=run_site, num_tokens=num_tokens
                            ),
                            per_call,
                        )


if __name__ == "__main__":
    unittest.main()
