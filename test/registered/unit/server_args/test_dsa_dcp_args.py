import unittest
from types import SimpleNamespace

from sglang.srt.arg_groups.overrides import (
    collect_model_override_declarations,
    declare_resolution,
    resolution_result,
    validate_declarations,
)
from sglang.srt.arg_groups.parallel_hook import handle_deprecated_dp_attention
from sglang.srt.server_args import ServerArgs
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestDsaDcpArgs(CustomTestCase):
    @staticmethod
    def _resolve(index_kpool=None, **kwargs):
        args = ServerArgs(
            **{"model_path": "dummy", "tp_size": 4, "dcp_size": 2, **kwargs}
        )
        arch = "GlmMoeDsaForCausalLM"
        hf_config = SimpleNamespace(architectures=[arch], index_topk=2048)
        if index_kpool is not None:
            hf_config.index_kpool = index_kpool
        args._model_config = SimpleNamespace(hf_config=hf_config)
        handle_deprecated_dp_attention(args)
        declarations = collect_model_override_declarations(arch, args, hf_config)
        validate_declarations(args, declarations)
        for source, fields in declarations:
            declare_resolution(args, source, **fields)
        return args

    def test_dcp_selects_the_trtllm_sparse_kernels(self):
        args = self._resolve()
        self.assertEqual(resolution_result(args, "dsa_prefill_backend"), "trtllm")
        self.assertEqual(resolution_result(args, "dsa_decode_backend"), "trtllm")

    def test_rejected_combinations(self):
        for kwargs in (
            {"index_kpool": 4},
            {"enable_prefill_cp": True},
            {"speculative_algorithm": "EAGLE", "speculative_eagle_topk": 2},
            {"dsa_decode_backend": "flashmla_sparse"},
        ):
            with self.subTest(**kwargs), self.assertRaisesRegex(ValueError, "DSA DCP"):
                self._resolve(**kwargs)

    def test_replicated_q_default_follows_the_comm_backend(self):
        for backend, expected in (("fi_a2a", True), ("ag_rs", None)):
            with self.subTest(backend=backend):
                args = self._resolve(dcp_comm_backend=backend)
                self.assertEqual(
                    resolution_result(args, "dcp_replicate_q_proj"), expected
                )

    def test_dcp_off_leaves_the_backends_alone(self):
        args = self._resolve(dcp_size=1)
        self.assertIsNone(resolution_result(args, "dsa_prefill_backend"))


if __name__ == "__main__":
    unittest.main()
