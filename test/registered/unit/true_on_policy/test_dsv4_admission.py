"""validate_dsv4 refuses DeepSeek-V4 serving paths outside the qualified TOP program."""

import os
import unittest
from contextlib import ExitStack
from types import SimpleNamespace
from unittest.mock import patch

from sglang.kernels.ops.attention.dsv4.kv_layout import KVLayout
from sglang.srt.environ import envs
from sglang.srt.layers.moe.utils import DeepEPMode
from sglang.srt.true_on_policy import dsv4
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


_QUALIFIED_ENV = {
    "SGLANG_OPT_USE_TILELANG_MHC_PRE": True,
    "SGLANG_OPT_USE_TILELANG_MHC_POST": True,
    "SGLANG_OPT_DEEPGEMM_HC_PRENORM": True,
    "SGLANG_DSV4_FP4_EXPERTS": False,
    "SGLANG_OPT_USE_ONLINE_COMPRESS": False,
    "SGLANG_DSV4_FP4_DEQUANT": False,
    "SGLANG_OPT_USE_FLASHINFER_MHC": False,
    "SGLANG_DSV4_USE_BF16_KV_QUANT_SOURCE": False,
    "SGLANG_OPT_DSV4_NONPAGED_INDEXER": False,
    "SGLANG_OPT_USE_TILELANG_INDEXER": False,
    "SGLANG_OPT_USE_AITER_INDEXER": False,
    "SGLANG_FP8_PAGED_MQA_LOGITS_TORCH": False,
    "SGLANG_RAGGED_VERIFY_MODE": "static",
}


def _config():
    return SimpleNamespace(
        model_type="deepseek_v4",
        n_routed_experts=256,
        hidden_act="silu",
        scoring_func="sqrtsoftplus",
        norm_topk_prob=True,
        hc_pre_from_prev_sublayer=False,
        num_hash_layers=3,
    )


class TestDsv4Admission(unittest.TestCase):
    def _validate(
        self, *, args=None, kernel=None, layout=(KVLayout.V4, None), sbo=False, env=None
    ):
        server_args = SimpleNamespace(
            enable_hisparse=False,
            enforce_shared_experts_fusion=False,
            disaggregation_mode="null",
            enable_hierarchical_cache=False,
        )
        server_args.__dict__.update(args or {})
        kernel_args = SimpleNamespace(
            attention_backend="dsv4",
            dsv4_attn_backend="flashmla",
            dsv4_prefill_backend="flashmla_sparse",
            enable_deepseek_v4_fp4_indexer=False,
        )
        kernel_args.__dict__.update(kernel or {})
        rc = "sglang.srt.runtime_context"
        with ExitStack() as stack:
            for target, value in (
                ("sglang.srt.true_on_policy.fp8.enabled", lambda: True),
                ("sglang.srt.true_on_policy.fp8.validate_quant_config", lambda _: None),
                ("sglang.srt.true_on_policy.fp8.validate_device", lambda: None),
                ("sglang.srt.true_on_policy.moe_ep.validate_ep_runtime", lambda: None),
                ("sglang.srt.true_on_policy.mtp.validate_mtp", lambda *a, **k: None),
                (f"{rc}.get_server_args", lambda: server_args),
                (f"{rc}.get_exec", lambda: SimpleNamespace(kernel=kernel_args)),
                (f"{rc}.get_model", lambda: SimpleNamespace(kv_cache_dtype="fp8_e4m3")),
                (
                    f"{rc}.get_parallel",
                    lambda: SimpleNamespace(attn_tp_size=1, pp_size=1, moe_ep_size=8),
                ),
                (
                    "sglang.srt.layers.moe.utils.get_deepep_mode",
                    lambda: DeepEPMode.AUTO,
                ),
                ("sglang.srt.layers.moe.utils.is_tbo_enabled", lambda: False),
                ("sglang.srt.layers.moe.utils.is_sbo_enabled", lambda: sbo),
                (
                    "sglang.srt.mem_cache.deepseek_v4_memory_pool.select_dsv4_kv_layout",
                    lambda: layout,
                ),
                ("sglang.srt.true_on_policy.dsv4._unified_kv", lambda: False),
                (
                    "sglang.srt.layers.deep_gemm_wrapper.configurer.ENABLE_JIT_DEEPGEMM",
                    True,
                ),
            ):
                stack.enter_context(patch(target, value))
            # envs' override() does not restore on an exception; patch.dict does.
            stack.enter_context(patch.dict(os.environ))
            for name, value in {**_QUALIFIED_ENV, **(env or {})}.items():
                getattr(envs, name).set(value)
            dsv4.validate_dsv4(_config(), quant_config=None)

    def test_the_qualified_program_is_admitted(self):
        self._validate()

    def test_refusals(self):
        cases = {
            "V4.1 KV": (dict(layout=(KVLayout.V41, None)), "V4 KV layout"),
            "V4.1 FP4 compressed KV": (
                dict(layout=(KVLayout.V4, "fp4")),
                "V4 KV layout",
            ),
            "FP4 indexer": (
                dict(kernel={"enable_deepseek_v4_fp4_indexer": True}),
                "FP4 indexer",
            ),
            "PD disaggregation": (
                dict(args={"disaggregation_mode": "prefill"}),
                "PD disaggregation",
            ),
            "HiCache": (dict(args={"enable_hierarchical_cache": True}), "HiCache"),
            "compact ragged verify": (
                dict(env={"SGLANG_RAGGED_VERIFY_MODE": "compact"}),
                "ragged verify mode",
            ),
            "SBO": (dict(sbo=True), "SBO"),
        }
        for name, (kwargs, refusal) in cases.items():
            with (
                self.subTest(name),
                self.assertRaisesRegex(NotImplementedError, rf"\[top\].*{refusal}"),
            ):
                self._validate(**kwargs)


if __name__ == "__main__":
    unittest.main()
