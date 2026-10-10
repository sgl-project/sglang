import unittest
from types import SimpleNamespace

from sglang.srt.arg_groups.overrides import (
    collect_model_override_declarations,
    declare_resolution,
    resolution_result,
    validate_declarations,
)
from sglang.srt.arg_groups.parallel_hook import handle_deprecated_dp_attention
from sglang.srt.runtime_context import override_platform
from sglang.srt.server_args import ServerArgs
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

EAGLE_CHAIN = dict(
    speculative_algorithm="EAGLE",
    speculative_num_steps=5,
    speculative_eagle_topk=1,
    speculative_num_draft_tokens=6,
)


class TestDsaDcpArgs(CustomTestCase):
    def setUp(self):
        platform = override_platform(
            is_cuda=True,
            is_hip=False,
            is_npu=False,
            is_xpu=False,
            is_sm100=True,
            device_sm=103,
        )
        platform.__enter__()
        self.addCleanup(platform.__exit__, None, None, None)

    @staticmethod
    def _resolve(arch="GlmMoeDsaForCausalLM", **kwargs):
        args = ServerArgs(
            **{"model_path": "dummy", "tp_size": 4, "dcp_size": 2, **kwargs}
        )
        hf_config = SimpleNamespace(architectures=[arch], index_topk=2048)
        args._model_config = SimpleNamespace(hf_config=hf_config)
        handle_deprecated_dp_attention(args)
        declarations = collect_model_override_declarations(arch, args, hf_config)
        validate_declarations(args, declarations)
        for source, fields in declarations:
            declare_resolution(args, source, **fields)
        return args

    def test_defaults(self):
        args = self._resolve()
        self.assertEqual(resolution_result(args, "attention_backend"), "dsa")
        self.assertEqual(resolution_result(args, "dsa_prefill_backend"), "trtllm")
        self.assertEqual(resolution_result(args, "dsa_decode_backend"), "trtllm")

    def test_supported_combinations(self):
        for kwargs in (
            {"dcp_size": 4},
            {"kv_cache_dtype": "bfloat16"},
            {"attn_dp_size": 2},
            {"ep_size": 4},
            {"dcp_comm_backend": "ag_rs", "enable_symm_mem": True},
            EAGLE_CHAIN,
        ):
            with self.subTest(**kwargs):
                self._resolve(**kwargs)

    def test_unsupported_combinations(self):
        for kwargs in (
            {"enable_hisparse": True},
            {"enable_prefill_cp": True},
            {"enable_hierarchical_cache": True},
            {"enable_unified_memory": True},
            {"disaggregation_mode": "decode"},
            {"dcp_replicate_q_proj": True, "enable_lora": True},
            {"kv_cache_dtype": "fp8_e5m2"},
            {"dsa_decode_backend": "flashmla_sparse"},
            {"attention_backend": "flashinfer"},
            {"attn_dp_size": 4},
            {"speculative_algorithm": "EAGLE3"},
            EAGLE_CHAIN | {"speculative_eagle_topk": 2},
            EAGLE_CHAIN | {"speculative_num_draft_tokens": 7},
        ):
            with (
                self.subTest(**kwargs),
                self.assertRaisesRegex(ValueError, "RoPE DSA DCP"),
            ):
                self._resolve(**kwargs)

    def test_unsupported_gpu(self):
        with (
            override_platform(is_cuda=True, is_sm100=False, device_sm=90),
            self.assertRaisesRegex(ValueError, "SM100 or SM103"),
        ):
            self._resolve()

    def test_replicated_q_default(self):
        for kwargs, expected in (
            ({"dcp_comm_backend": "fi_a2a"}, True),
            ({"dcp_comm_backend": "ag_rs"}, None),
            ({"dcp_comm_backend": "fi_a2a", "enable_lora": True}, None),
            ({"dcp_comm_backend": "fi_a2a", "dcp_replicate_q_proj": False}, False),
        ):
            with self.subTest(**kwargs):
                args = self._resolve(**kwargs)
                self.assertEqual(
                    resolution_result(args, "dcp_replicate_q_proj"), expected
                )

    def test_other_paths_unchanged(self):
        for kwargs in ({"dcp_size": 1}, {"arch": "Glm5NextForConditionalGeneration"}):
            with self.subTest(**kwargs):
                args = self._resolve(enable_hisparse=True, **kwargs)
                self.assertIsNone(resolution_result(args, "dsa_prefill_backend"))


if __name__ == "__main__":
    unittest.main()
