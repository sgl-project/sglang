"""Deprecated attention-backend aliases are normalized before anything reads them."""

import unittest

from sglang.srt.arg_groups.model_override_base import resolved_view
from sglang.srt.arg_groups.serving_hook import handle_deprecated_args
from sglang.srt.layers.cp.base import CPAttentionBackendKind
from sglang.srt.server_args import ServerArgs
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class TestDeprecatedAttentionBackendAliases(CustomTestCase):
    def _resolve(self, **kwargs):
        server_args = ServerArgs(model_path="dummy", **kwargs)
        handle_deprecated_args(server_args)
        return resolved_view(server_args)

    def test_nsa_normalizes_to_dsa_on_every_backend_field(self):
        cfg = self._resolve(
            attention_backend="nsa",
            prefill_attention_backend="nsa",
            decode_attention_backend="nsa",
        )
        self.assertEqual(cfg.attention_backend, "dsa")
        self.assertEqual(cfg.prefill_attention_backend, "dsa")
        self.assertEqual(cfg.decode_attention_backend, "dsa")

    def test_compressed_still_normalizes_to_dsv4(self):
        cfg = self._resolve(attention_backend="compressed")
        self.assertEqual(cfg.attention_backend, "dsv4")

    def test_normalized_nsa_selects_the_dsa_cp_backend(self):
        cfg = self._resolve(attention_backend="nsa")
        self.assertIs(
            CPAttentionBackendKind.from_string(cfg.attention_backend),
            CPAttentionBackendKind.DSA,
        )

    def test_cp_backend_match_is_exact_not_substring(self):
        # `value in ("dsa")` was a substring test on the string "dsa".
        for value in ("a", "s", "ds", "sa", ""):
            with self.assertRaises(ValueError):
                CPAttentionBackendKind.from_string(value)


if __name__ == "__main__":
    unittest.main()
