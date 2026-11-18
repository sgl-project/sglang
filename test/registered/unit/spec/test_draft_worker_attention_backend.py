"""Which attention backend a draft worker gets from the published spec."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

from sglang.srt.arg_groups.kimi_k3_hook import apply_kimi_k3_spec_backend_defaults
from sglang.srt.arg_groups.overrides import resolution_result
from sglang.srt.environ import envs
from sglang.srt.layers.attention import attention_registry
from sglang.srt.layers.attention.attention_registry import NonCausalDraftAttention
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
        self, *, is_gfx950=True, aiter_non_causal=True, declared=None, **server_args
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
            override_platform(is_sm100=False),
            patch(
                "sglang.kernels.arch.current_arch",
                return_value=SimpleNamespace(
                    hip=SimpleNamespace(gcn_arch="gfx950" if is_gfx950 else "gfx942")
                ),
            ),
            patch(
                "sglang.srt.utils.aiter.aiter_unified_attention_supports_non_causal",
                return_value=aiter_non_causal,
            ),
            patch.dict(
                attention_registry.NONCAUSAL_DRAFT_ATTENTION_BACKENDS, declared or {}
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

    def test_hook_asks_the_declared_capability_not_the_backend_name(self):
        """The hook defaults the draft to whichever target backend declared
        NonCausalDraftAttention and passes its checks; one that declared nothing,
        is not the default here, or rejects the draft keeps the draft unset."""
        reasons = []

        def declared(*, default_here=True, reject_reason=None):
            def reject(**needs):
                reasons.append(needs)
                return reject_reason

            return {
                "fake_noncausal": NonCausalDraftAttention(
                    reject_reason=reject,
                    default_on_this_device=lambda: default_here,
                )
            }

        fake_target = dict(
            prefill_attention_backend="fake_noncausal",
            decode_attention_backend="fake_noncausal",
        )
        self.assertEqual(
            self._declared_draft_backend(declared=declared(), **fake_target),
            "fake_noncausal",
        )
        self.assertFalse(reasons[-1]["draft_is_mla"])
        for case in (
            dict(default_here=False),
            dict(reject_reason="cannot"),
        ):
            with self.subTest(**case):
                self.assertIsNone(
                    self._declared_draft_backend(
                        declared=declared(**case), **fake_target
                    )
                )
        self.assertIsNone(
            self._declared_draft_backend(
                prefill_attention_backend="triton", decode_attention_backend="triton"
            )
        )


class TestAiterDSparkDraftDefault(CustomTestCase):
    def _default_here(self, hip):
        from sglang.srt.utils.aiter import aiter_dspark_draft_default_on_this_device

        with patch(
            "sglang.kernels.arch.current_arch",
            return_value=SimpleNamespace(hip=hip),
        ):
            return aiter_dspark_draft_default_on_this_device()

    def test_exact_gfx950_only(self):
        """aiter's DSpark draft default is gfx950-only: another gfx95x, gfx942 or
        a non-HIP device must not pass the gate a "gfx95" substring test admits."""
        self.assertTrue(self._default_here(SimpleNamespace(gcn_arch="gfx950")))
        for hip in (
            SimpleNamespace(gcn_arch="gfx951"),
            SimpleNamespace(gcn_arch="gfx942"),
            None,
        ):
            with self.subTest(hip=hip):
                self.assertFalse(self._default_here(hip))

    def test_aiter_declares_the_noncausal_draft_capability(self):
        """The resolver and the Kimi-K3 hook reach aiter's checks through its
        registration, not by naming aiter."""
        from sglang.srt.utils import aiter as aiter_utils

        declared = attention_registry.NONCAUSAL_DRAFT_ATTENTION_BACKENDS["aiter"]
        self.assertIs(
            declared.reject_reason, aiter_utils.aiter_dspark_draft_reject_reason
        )
        self.assertIs(
            declared.default_on_this_device,
            aiter_utils.aiter_dspark_draft_default_on_this_device,
        )


if __name__ == "__main__":
    unittest.main()
