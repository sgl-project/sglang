"""K3 routed experts on DeepGEMM's one-rank MegaMoE: layer rules, the DeepGEMM capability check and the
call keywords (no GPU, no deep_gemm call)."""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

import types
import unittest
from unittest import mock

from sglang.srt.environ import envs
from sglang.srt.layers.moe import k3_deepgemm_routed as routed


def _layer(**overrides):
    cfg = dict(
        activation="situ",
        gemm1_alpha=4.0,
        gemm1_clamp_limit=25.0,
        gate_up_interleaved=False,
    )
    cfg.update(overrides)
    return types.SimpleNamespace(moe_runner_config=types.SimpleNamespace(**cfg))


def _fake_deep_gemm(call_keywords, buffer_keywords, version="0.0.0"):
    """A module with the signatures of a deep_gemm build, nothing else."""
    scope = {}
    exec(
        f"def fp8_fp4_mega_moe(y, l1_weights, l2_weights, sym_buffer, {', '.join(k + '=None' for k in call_keywords)}): pass\n"
        f"class SymmBuffer:\n"
        f"    def __init__(self, group, num_experts, num_max_tokens_per_rank, num_topk, hidden, intermediate_hidden,"
        f" {', '.join(k + '=None' for k in buffer_keywords)}): pass\n",
        scope,
    )
    return types.SimpleNamespace(
        fp8_fp4_mega_moe=scope["fp8_fp4_mega_moe"],
        mega=types.SimpleNamespace(SymmBuffer=scope["SymmBuffer"]),
        get_num_locality_domains=lambda: 2,
        __version__=version,
    )


# sgl-deep-gemm 0.2.9: has weight_layout (NVFP4 only) but none of the one-rank keywords
RELEASED_CALL = ("weight_layout", "global_num_tokens", "activation", "recipe")
RELEASED_BUFFER = ("mma_type", "activation", "base")


class _Capability(unittest.TestCase):
    def setUp(self):
        routed.missing_capability.cache_clear()
        self.addCleanup(routed.missing_capability.cache_clear)

    def _with(self, module):
        routed.missing_capability.cache_clear()
        return mock.patch.object(routed, "_deep_gemm", return_value=module)


class TestLayerRules(_Capability):
    def test_off_by_default(self):
        self.assertFalse(envs.SGLANG_K3_ROUTED_MOE_DEEPGEMM.get())

    def test_mismatch_reasons(self):
        self.assertIn(
            "activation", routed.mismatch(_layer(activation="silu"), False, "default")
        )
        self.assertIn(
            "SiTU", routed.mismatch(_layer(gemm1_alpha=1.0), False, "default")
        )
        self.assertIn(
            "interleaved",
            routed.mismatch(_layer(gate_up_interleaved=True), False, "default"),
        )
        self.assertEqual(routed.mismatch(_layer(), True, "default"), "expert bias")
        self.assertIn("bf16", routed.mismatch(_layer(), False, "bf16"))

    def test_ar_fusion_is_not_a_mismatch(self):
        with envs.SGLANG_K3_AR_FUSION.override(True), self._with(None):
            reason = routed.mismatch(_layer(), False, "default")
        self.assertIn("deep_gemm", reason)  # the next rule in line, not AR fusion

    def test_select_routing(self):
        ids, words, std = object(), object(), (object(), object())
        metadata = types.SimpleNamespace(expert_weights=object())
        # kimi_k3_fused_routing: plain ids + metadata weights, never the ids alone
        self.assertEqual(
            routed.select_routing(ids, (metadata, object()), lambda: std),
            (ids, metadata.expert_weights),
        )
        # route_quant_fused: packed words
        self.assertIs(routed.select_routing(words, None, lambda: std), words)
        # nothing prepared: the standard top-k
        self.assertIs(routed.select_routing(None, None, lambda: std), std)

    def test_call_keywords(self):
        with (
            mock.patch.object(routed._group, "first_expert", 336),
            mock.patch.object(routed._group, "stats", "stats"),
        ):
            small, large = routed._call_keywords(84), routed._call_keywords(85)
        # trtllm-gen: four accumulators up to 84 tokens, one chain above
        self.assertEqual(small["split4_combine_max_tokens"], 84)
        self.assertEqual(large["split4_combine_max_tokens"], 0)
        for keywords in (small, large):
            self.assertEqual(
                {
                    k: keywords[k]
                    for k in (
                        "weight_layout",
                        "numerics",
                        "topk_format",
                        "topk_first_expert",
                        "activation",
                    )
                },
                dict(
                    weight_layout="trtllm",
                    numerics="trtllm",
                    topk_format="packed",
                    topk_first_expert=336,
                    activation="situ",
                ),
            )
            # every keyword the capability check asks for is one the call passes
            self.assertLessEqual(set(routed._CALL_KEYWORDS), set(keywords))


class TestCapability(_Capability):
    def test_not_installed(self):
        with self._with(None):
            self.assertEqual(
                routed.missing_capability(), "deep_gemm cannot be imported"
            )

    def test_released_wheel_lacks_the_api(self):
        with self._with(
            _fake_deep_gemm(RELEASED_CALL, RELEASED_BUFFER, version="0.2.9")
        ):
            reason = routed.missing_capability()
            self.assertIn("deep_gemm 0.2.9 lacks", reason)
            for name in (
                "numerics",
                "split4_combine_max_tokens",
                "topk_format",
                "topk_first_expert",
                "exact_num_max_tokens",
                "max_block_m",
            ):
                self.assertIn(name, reason)
            self.assertNotIn(
                "weight_layout", reason
            )  # present in the release, for NVFP4 only
            self.assertEqual(routed.mismatch(_layer(), False, "default"), reason)

    def test_call_keywords_without_the_buffer_keywords(self):
        # FP8 x FP4 `weight_layout='trtllm'` arrived with the buffer keywords: a build without them does not qualify
        with self._with(_fake_deep_gemm(routed._CALL_KEYWORDS, RELEASED_BUFFER)):
            reason = routed.missing_capability()
            self.assertIn("exact_num_max_tokens", reason)
            self.assertNotIn("numerics", reason)

    def test_module_without_mega_moe(self):
        with self._with(types.SimpleNamespace()):
            self.assertIn(
                "no inspectable fp8_fp4_mega_moe", routed.missing_capability()
            )

    def test_never_raises_on_an_unusual_build(self):
        # A build may expose its entry points through a proxy whose attribute lookup fails with something other
        # than AttributeError, or as native ops without a Python signature: a reason, not an exception at model load.
        class Proxy:
            def __getattr__(self, name):
                raise RuntimeError(f"no op {name}")

        native_op = types.SimpleNamespace(
            fp8_fp4_mega_moe=len.__call__, mega=types.SimpleNamespace(SymmBuffer=int)
        )
        no_version = _fake_deep_gemm(RELEASED_CALL, RELEASED_BUFFER)
        del no_version.__version__
        for module, text in (
            (Proxy(), "no inspectable"),
            (native_op, None),
            (no_version, "deep_gemm unknown lacks"),
        ):
            with self._with(module):
                reason = routed.missing_capability()
                self.assertIsInstance(reason, str)
                if text:
                    self.assertIn(text, reason)
        capable = _fake_deep_gemm(routed._CALL_KEYWORDS, routed._BUFFER_KEYWORDS)
        capable.get_num_locality_domains = Proxy().__getattr__
        with (
            self._with(capable),
            mock.patch("torch.distributed.is_initialized", return_value=True),
        ):
            self.assertIn(
                "cannot report locality domains",
                routed.mismatch(_layer(), False, "default"),
            )

    def test_capable_build(self):
        module = _fake_deep_gemm(
            routed._CALL_KEYWORDS, routed._BUFFER_KEYWORDS + RELEASED_BUFFER
        )
        with self._with(module):
            self.assertIsNone(routed.missing_capability())
            with mock.patch("torch.distributed.is_initialized", return_value=True):
                self.assertIsNone(routed.mismatch(_layer(), False, "default"))
                module.get_num_locality_domains = lambda: 1
                self.assertIn(
                    "locality domain", routed.mismatch(_layer(), False, "default")
                )


if __name__ == "__main__":
    unittest.main()
