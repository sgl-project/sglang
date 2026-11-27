import unittest
from types import SimpleNamespace

from sglang.srt.environ import envs
from sglang.srt.true_on_policy.gdn_chunked import (
    row_cache_bytes_per_layer,
    select_chunked_programs,
    top_gdn_chunked,
    top_gdn_mode,
    validate_gdn_chunked_contract,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def _config(**overrides):
    fields = dict(
        linear_num_key_heads=16,
        linear_num_value_heads=32,
        linear_key_head_dim=128,
        linear_value_head_dim=128,
    )
    fields.update(overrides)
    return SimpleNamespace(**fields)


def _args(**overrides):
    fields = dict(
        true_on_policy_contract="true_on_policy_v1",
        speculative_algorithm=None,
        speculative_num_draft_tokens=None,
        pp_size=1,
        enable_two_batch_overlap=False,
        enable_linear_replayssm_spec=False,
    )
    fields.update(overrides)
    return SimpleNamespace(**fields)


class TestSelectChunkedPrograms(unittest.TestCase):
    def _selected(self, ops):
        with (
            envs.SGLANG_TOP_GDN_MODE.override(""),
            envs.SGLANG_TOP_KDA_MODE.override(""),
        ):
            select_chunked_programs(ops)
            return envs.SGLANG_TOP_GDN_MODE.get(), envs.SGLANG_TOP_KDA_MODE.get()

    def test_each_family_gets_only_its_own_program(self):
        """GDN and KDA share the FLA state kernel, whose chunk close the GDN program pins, so a KDA family must not
        select the GDN program, and a family without either selects neither."""
        self.assertEqual(
            self._selected(("bi_linear", "gdn", "decode_kv_split")), ("chunked", "")
        )
        self.assertEqual(
            self._selected(("mhc", "kda", "glm5_next_dsa")), ("", "chunked")
        )
        self.assertEqual(self._selected(("bi_linear", "dsa")), ("", ""))


class TestGdnChunkedContract(unittest.TestCase):
    def test_mode(self):
        with envs.SGLANG_TOP_GDN_MODE.override(""):
            self.assertEqual(top_gdn_mode(), "")
            self.assertFalse(top_gdn_chunked(_args()))
        with envs.SGLANG_TOP_GDN_MODE.override("chunked"):
            self.assertTrue(top_gdn_chunked(_args()))
            self.assertFalse(top_gdn_chunked(_args(true_on_policy_contract=None)))
        with envs.SGLANG_TOP_GDN_MODE.override("fast"):
            with self.assertRaisesRegex(ValueError, "SGLANG_TOP_GDN_MODE"):
                top_gdn_mode()

    def test_row_cache_bytes(self):
        # k, v, v_new rows (BF16), gate and beta rows (FP32), the solved 64 x 64 inverse, e_chunk.
        self.assertEqual(
            row_cache_bytes_per_layer(
                num_k_heads=16, num_v_heads=32, head_k_dim=128, head_v_dim=128
            ),
            1_851_520,
        )

    def test_admits_qwen35_and_chain_spec(self):
        validate_gdn_chunked_contract(_args(), _config())
        validate_gdn_chunked_contract(
            _args(speculative_algorithm="EAGLE", speculative_num_draft_tokens=4),
            _config(),
        )

    def test_refusals(self):
        cases = (
            (_args(), _config(linear_num_key_heads=8), "16 key heads"),
            (
                _args(speculative_algorithm="EAGLE", speculative_num_draft_tokens=17),
                _config(),
                "at most 16",
            ),
            (_args(pp_size=2), _config(), "pipeline"),
            (_args(enable_two_batch_overlap=True), _config(), "two-batch"),
            (_args(enable_linear_replayssm_spec=True), _config(), "ReplaySSM"),
        )
        for args, config, message in cases:
            with self.assertRaisesRegex(ValueError, message):
                validate_gdn_chunked_contract(args, config)


if __name__ == "__main__":
    unittest.main()
