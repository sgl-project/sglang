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
        unified_attn=None,
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
            envs.SGLANG_USE_AITER_UNIFIED_ATTN.override(unified_attn),
            envs.SGLANG_AITER_UNIFIED_VERIFY.override(unified_verify),
        ):
            return draft_worker_common.resolve_draft_worker_attention_backend()

    def test_aiter_serves_only_a_dspark_mha_draft_on_a_non_causal_aiter(self):
        """A DSpark draft verifies its block bidirectionally. An AITER without
        causal=False, the causal-only aiter MLA verify, a sliding-window draft, a
        SHUFFLE 5D pool, unified attention or verify switched off, unified memory,
        deterministic inference, and DFLASH must all leave the draft on the fallback."""
        self.assertEqual(self._resolve(), "aiter")
        for case in (
            dict(aiter_non_causal=False),
            dict(draft_is_mla=True),
            dict(draft_has_sliding_window=True),
            dict(layout="vectorized_5d"),
            dict(unified_attn=False),
            dict(unified_verify=False),
            dict(unified_memory=True),
            dict(deterministic=True),
            dict(algorithm="DFLASH"),
        ):
            with self.subTest(**case):
                self.assertEqual(self._resolve(**case), "triton")

    def test_inherited_aiter_needs_an_explicit_opt_in(self):
        """aiter serves a DSpark draft only where it was validated: named for the
        draft (a model hook or the operator), or forced by
        SGLANG_USE_AITER_UNIFIED_ATTN=1. An --attention-backend aiter the draft
        merely inherits keeps the triton draft, so an AITER upgrade alone does not
        move every DSpark draft."""
        self.assertEqual(self._resolve(named_backend=None), "triton")
        self.assertEqual(self._resolve(named_backend=None, unified_attn=True), "aiter")
        self.assertEqual(self._resolve(unified_attn=True), "aiter")
        self.assertEqual(self._resolve(unified_attn=False), "triton")


class TestKimiK3DSparkDraftBackendDefault(CustomTestCase):
    def _declared_draft_backend(self, *, is_hip=True, **server_args):
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
        with override_platform(is_hip=is_hip, is_sm100=False):
            apply_kimi_k3_spec_backend_defaults(args)
        return resolution_result(args, "speculative_draft_attention_backend")

    def test_kimi_k3_names_aiter_for_its_dspark_draft_on_rocm(self):
        """The Kimi-K3 hook is the allowlist entry for the aiter DSpark draft; an
        operator-named draft backend, a non-aiter prefill backend, another
        algorithm, or a non-ROCm platform must not get it."""
        self.assertEqual(self._declared_draft_backend(), "aiter")
        for case in (
            dict(is_hip=False),
            dict(speculative_draft_attention_backend="triton"),
            dict(prefill_attention_backend="triton"),
            dict(speculative_algorithm="EAGLE"),
        ):
            with self.subTest(**case):
                self.assertNotEqual(self._declared_draft_backend(**case), "aiter")


if __name__ == "__main__":
    unittest.main()
