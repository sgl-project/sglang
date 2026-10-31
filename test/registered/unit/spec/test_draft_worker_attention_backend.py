"""Which attention backend a draft worker gets from the published spec."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

from sglang.srt.environ import envs
from sglang.srt.speculative import draft_worker_common
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class TestDraftWorkerAttentionBackend(CustomTestCase):
    def _resolve(
        self, *, algorithm, aiter_non_causal=True, draft_is_mla=False, layout="nhd"
    ):
        spec = SimpleNamespace(
            speculative_algorithm=algorithm, speculative_draft_attention_backend=None
        )
        memory = SimpleNamespace(enable_unified_memory=False)
        with (
            patch.object(draft_worker_common, "get_spec", return_value=spec),
            patch.object(draft_worker_common, "get_memory", return_value=memory),
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
            envs.SGLANG_AITER_KV_CACHE_LAYOUT.override(layout),
        ):
            return draft_worker_common.resolve_draft_worker_attention_backend()

    def test_aiter_serves_only_a_dspark_mha_draft_on_a_non_causal_aiter(self):
        """A DSpark draft verifies its block bidirectionally. An AITER without
        causal=False, the causal-only aiter MLA verify, a SHUFFLE 5D pool and
        DFLASH must all leave the draft on the fallback."""
        self.assertEqual(self._resolve(algorithm="DSPARK"), "aiter")
        for case in (
            dict(algorithm="DSPARK", aiter_non_causal=False),
            dict(algorithm="DSPARK", draft_is_mla=True),
            dict(algorithm="DSPARK", layout="shuffle"),
            dict(algorithm="DFLASH"),
        ):
            with self.subTest(**case):
                self.assertNotEqual(self._resolve(**case), "aiter")


if __name__ == "__main__":
    unittest.main()
