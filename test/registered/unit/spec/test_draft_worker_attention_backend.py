"""Which attention backend a draft worker gets from the published spec."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

from sglang.srt.arg_groups.kimi_k3_hook import apply_kimi_k3_spec_backend_defaults
from sglang.srt.arg_groups.overrides import resolution_result
from sglang.srt.environ import envs
from sglang.srt.runtime_context import override_platform
from sglang.srt.server_args import ServerArgs
from sglang.srt.speculative import draft_worker_common
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class TestDraftWorkerAttentionBackend(CustomTestCase):
    def _resolve(
        self,
        *,
        algorithm="DSPARK",
        named_backend="aiter",
        aiter_non_causal=True,
        draft_is_mla=False,
        draft_has_sliding_window=False,
        layout="nhd",
        unified_verify=True,
        unified_memory=False,
        deterministic=False,
    ):
        spec = SimpleNamespace(
            speculative_algorithm=algorithm,
            speculative_draft_attention_backend=named_backend,
        )
        memory = SimpleNamespace(enable_unified_memory=unified_memory)
        exec_bag = SimpleNamespace(
            deterministic=SimpleNamespace(enable_deterministic_inference=deterministic)
        )
        with (
            patch.object(draft_worker_common, "get_spec", return_value=spec),
            patch.object(draft_worker_common, "get_memory", return_value=memory),
            patch.object(draft_worker_common, "get_exec", return_value=exec_bag),
            patch.object(
                draft_worker_common,
                "attention_backends",
                return_value=("aiter", "aiter"),
            ),
            patch(
                "sglang.srt.utils.aiter.aiter_unified_attention_supports_non_causal",
                return_value=aiter_non_causal,
            ),
            patch(
                "sglang.srt.speculative.dspark_components.dspark_config.draft_is_mla",
                return_value=draft_is_mla,
            ),
            patch(
                "sglang.srt.speculative.dspark_components.dspark_config."
                "draft_has_sliding_window",
                return_value=draft_has_sliding_window,
            ),
            envs.SGLANG_AITER_KV_CACHE_LAYOUT.override(layout),
            envs.SGLANG_AITER_UNIFIED_VERIFY.override(unified_verify),
        ):
            return draft_worker_common.resolve_draft_worker_attention_backend()

    def test_named_aiter_serves_a_dspark_mha_draft_or_raises(self):
        """A DSpark draft verifies its block bidirectionally. A named aiter that
        cannot run it (an AITER without causal=False, an MLA or sliding-window
        draft, a SHUFFLE 5D pool, unified verify off, unified memory,
        deterministic inference) is an unmet explicit request and raises."""
        self.assertEqual(self._resolve(), "aiter")
        for case in (
            dict(aiter_non_causal=False),
            dict(draft_is_mla=True),
            dict(draft_has_sliding_window=True),
            dict(layout="vectorized_5d"),
            dict(unified_verify=False),
            dict(unified_memory=True),
            dict(deterministic=True),
        ):
            with self.subTest(**case):
                with self.assertRaisesRegex(ValueError, "cannot run this DSpark draft"):
                    self._resolve(**case)

    def test_aiter_is_taken_only_when_named_for_a_dspark_draft(self):
        """An --attention-backend aiter the draft merely inherits, or aiter named
        for another algorithm's draft, keeps the platform fallback as before."""
        self.assertEqual(self._resolve(named_backend=None), "triton")
        self.assertEqual(self._resolve(algorithm="DFLASH"), "triton")


class TestKimiK3DSparkDraftBackendDefault(CustomTestCase):
    def _declared_draft_backend(
        self, *, is_gfx950=True, aiter_non_causal=True, **server_args
    ):
        server_args = (
            dict(
                speculative_algorithm="DSPARK",
                prefill_attention_backend="aiter",
                decode_attention_backend="aiter",
                linear_attn_verify_backend="triton",
            )
            | server_args
        )
        args = ServerArgs(model_path="dummy", **server_args)
        with (
            override_platform(is_gfx950=is_gfx950, is_sm100=False),
            patch(
                "sglang.srt.utils.aiter.aiter_unified_attention_supports_non_causal",
                return_value=aiter_non_causal,
            ),
        ):
            apply_kimi_k3_spec_backend_defaults(args)
        return resolution_result(args, "speculative_draft_attention_backend")

    def test_kimi_k3_names_aiter_for_its_dspark_draft_on_gfx950(self):
        """The Kimi-K3 hook is the allowlist entry for the aiter DSpark draft, which
        was validated on gfx950 only. Another HIP arch (gfx942), an AITER without
        causal=False, deterministic inference, an operator-named draft backend, a
        non-aiter prefill backend or another algorithm must not get it."""
        self.assertEqual(self._declared_draft_backend(), "aiter")
        for case in (
            dict(is_gfx950=False),
            dict(aiter_non_causal=False),
            dict(enable_deterministic_inference=True),
            dict(speculative_draft_attention_backend="triton"),
            dict(prefill_attention_backend="triton"),
            dict(speculative_algorithm="EAGLE"),
        ):
            with self.subTest(**case):
                self.assertNotEqual(self._declared_draft_backend(**case), "aiter")


class TestIsGfx950Supported(CustomTestCase):
    def _probe(self, hip):
        from sglang.srt.utils import device

        device.is_gfx950_supported.cache_clear()
        self.addCleanup(device.is_gfx950_supported.cache_clear)
        with patch(
            "sglang.kernels.arch.current_arch",
            return_value=SimpleNamespace(hip=hip),
        ):
            return device.is_gfx950_supported()

    def test_exact_gfx950_only(self):
        """The Kimi-K3 aiter draft default is gfx950-only: another gfx95x, gfx942 or
        a non-HIP device must not pass the gate a "gfx95" substring test admits."""
        self.assertTrue(self._probe(SimpleNamespace(gcn_arch="gfx950")))
        for hip in (
            SimpleNamespace(gcn_arch="gfx951"),
            SimpleNamespace(gcn_arch="gfx942"),
            None,
        ):
            with self.subTest(hip=hip):
                self.assertFalse(self._probe(hip))


if __name__ == "__main__":
    unittest.main()
