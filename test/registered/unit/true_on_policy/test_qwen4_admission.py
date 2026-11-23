"""TOP Qwen3.8-Flash-Next (qwen4_exp) admission: the layouts its ops admit and what the model check refuses."""

import unittest
from types import SimpleNamespace

from sglang.srt.true_on_policy.attention import decode_kv_split
from sglang.srt.true_on_policy.gdn_contract import (
    GDN_TOP_ARCHITECTURES,
    validate_qwen4_exp_top,
)
from sglang.srt.true_on_policy.registry import (
    OPS_BY_NAME,
    ServingLayout,
    admit,
    find_family,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


def _admit(**layout):
    fields = dict(ep=4, dp_attention=4, pp=1, spec=None)
    fields.update(layout)
    return admit(
        "qwen4_exp",
        ServingLayout(**fields),
        fp8=False,
        side="serving",
        envelopes_of=lambda name: OPS_BY_NAME[name].serving,
    )


def _config(**overrides):
    fields = dict(hc_count=4, hc_lowrank=320, hidden_size=2560, ple_embed_dim=2560)
    fields.update(overrides)
    return SimpleNamespace(**fields)


class TestQwen4Admission(unittest.TestCase):
    def test_attention_dp_with_ep_is_admitted(self):
        self.assertEqual(_admit().name, "Qwen3.8-Flash-Next")
        self.assertIs(find_family("qwen4_exp_text"), find_family("qwen4_exp"))

    def test_only_the_checkpoint_mtp_chain_speculates(self):
        """QSA's pending key ring holds one 4-token group, so a wider verify window would collide in one forward."""
        self.assertEqual(_admit(spec=(3, 1, 4)).name, "Qwen3.8-Flash-Next")
        with self.assertRaises(NotImplementedError) as refusal:
            _admit(spec=(5, 1, 6))
        self.assertIn("op qsa: spec=(5, 1, 6) refused", str(refusal.exception))
        with self.assertRaises(NotImplementedError) as refusal:
            _admit(spec=(3, 2, 4))
        message = str(refusal.exception)
        self.assertIn("op ple: spec=(3, 2, 4) refused", message)
        self.assertIn("op gdn: spec=(3, 2, 4) refused", message)
        with self.assertRaisesRegex(
            NotImplementedError, "op qwen4_mtp_draft: spec=\\(1, 1, 2\\)"
        ):
            _admit(spec=(1, 1, 2))

    def test_local_experts_and_serving_stages_are_refused(self):
        with self.assertRaisesRegex(NotImplementedError, "op qwen2_moe_experts: ep=1"):
            _admit(ep=1, dp_attention=1)
        with self.assertRaisesRegex(NotImplementedError, "op hyper_connection: pp=2"):
            _admit(pp=2)

    def test_a_block_fp8_checkpoint_is_refused(self):
        with self.assertRaisesRegex(NotImplementedError, "no block-FP8 program"):
            find_family("qwen4_exp").ops_for(fp8=True)

    def test_qsa_layers_bind_no_fa_decode_split(self):
        self.assertIn("qwen4_exp_text", GDN_TOP_ARCHITECTURES)
        self.assertEqual(decode_kv_split("qwen4_exp_text"), 0)
        self.assertGreater(decode_kv_split("qwen3_5_moe_text"), 0)

    def test_the_model_check_refuses_a_quantized_checkpoint(self):
        validate_qwen4_exp_top(_config(), quant_config=None)
        with self.assertRaisesRegex(NotImplementedError, "BF16 checkpoint only"):
            validate_qwen4_exp_top(_config(), quant_config=object())

    def test_the_model_check_pins_the_branch_count_and_widths(self):
        for overrides in (
            dict(hc_count=2),
            dict(hc_lowrank=100),
            dict(ple_embed_dim=1000),
        ):
            with (
                self.subTest(**overrides),
                self.assertRaisesRegex(NotImplementedError, "4 branches"),
            ):
                validate_qwen4_exp_top(_config(**overrides), quant_config=None)


if __name__ == "__main__":
    unittest.main()
