import unittest
from types import SimpleNamespace

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.layers.logits_processor import (  # noqa: E402
    should_enable_multimem_logits_all_gather,
)

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class TestLogitsMultimemGate(unittest.TestCase):
    def test_gate_matrix(self):
        kimi = SimpleNamespace(
            model_type="kimi_k3",
            architectures=["KimiK3ForConditionalGeneration"],
        )
        generic = SimpleNamespace(
            model_type="llama", architectures=["LlamaForCausalLM"]
        )
        cases = [
            ("generic", generic, True, False, False, "null", True),
            ("disabled gather", generic, False, False, False, "null", False),
            ("attention TP", generic, True, True, False, "null", False),
            ("kimi DCP", kimi, True, False, True, "decode", False),
            ("kimi prefill", kimi, True, False, False, "prefill", False),
            ("kimi ordinary decode", kimi, True, False, False, "decode", True),
            ("generic prefill", generic, True, False, False, "prefill", True),
        ]

        for name, config, gather, attn_tp, dcp, mode, expected in cases:
            with self.subTest(name=name):
                self.assertEqual(
                    should_enable_multimem_logits_all_gather(
                        config,
                        do_tensor_parallel_all_gather=gather,
                        use_attn_tp_group=attn_tp,
                        dcp_enabled=dcp,
                        disaggregation_mode=mode,
                    ),
                    expected,
                )


if __name__ == "__main__":
    unittest.main()
