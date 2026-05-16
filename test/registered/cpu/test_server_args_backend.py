import unittest
from types import SimpleNamespace
from unittest.mock import patch

from sglang.srt.server_args import ServerArgs
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="base-b-test-cpu")


class TestServerArgsCPUBackend(unittest.TestCase):
    def _make_server_args(self, attention_backend=None):
        server_args = ServerArgs.__new__(ServerArgs)
        server_args.device = "cpu"
        server_args.attention_backend = attention_backend
        server_args.sampling_backend = None
        return server_args

    @patch("sglang.srt.server_args.is_host_cpu_arm64", return_value=True)
    def test_arm_cpu_defaults_to_torch_native(self, _mock_is_arm64):
        server_args = self._make_server_args()

        ServerArgs._handle_cpu_backends(server_args)

        self.assertEqual(server_args.attention_backend, "torch_native")
        self.assertEqual(server_args.sampling_backend, "pytorch")

    @patch("sglang.srt.server_args.is_host_cpu_arm64", return_value=False)
    def test_x86_cpu_defaults_to_intel_amx(self, _mock_is_arm64):
        server_args = self._make_server_args()

        ServerArgs._handle_cpu_backends(server_args)

        self.assertEqual(server_args.attention_backend, "intel_amx")
        self.assertEqual(server_args.sampling_backend, "pytorch")


class TestServerArgsDeterministicBackend(unittest.TestCase):
    def _make_server_args(self, attention_backend=None, disable_radix_cache=False):
        server_args = ServerArgs.__new__(ServerArgs)
        server_args.rl_on_policy_target = None
        server_args.enable_deterministic_inference = True
        server_args.enable_aiter_allreduce_fusion = False
        server_args.enable_flashinfer_allreduce_fusion = False
        server_args.sampling_backend = None
        server_args.model_path = "dummy"
        server_args.attention_backend = attention_backend
        server_args.disable_radix_cache = disable_radix_cache
        server_args.tp_size = 1
        server_args.disable_custom_all_reduce = False
        return server_args

    @patch("sglang.srt.server_args.is_sm120_supported", return_value=False)
    @patch("sglang.srt.server_args.is_sm100_supported", return_value=False)
    @patch.object(
        ServerArgs,
        "get_model_config",
        return_value=SimpleNamespace(
            hf_config=SimpleNamespace(architectures=["Qwen2ForCausalLM"])
        ),
    )
    def test_default_deterministic_hopper_backend_is_fa3(self, *_):
        server_args = self._make_server_args()

        ServerArgs._handle_deterministic_inference(server_args)

        self.assertEqual(server_args.attention_backend, "fa3")
        self.assertFalse(server_args.disable_radix_cache)

    @patch.object(
        ServerArgs,
        "get_model_config",
        return_value=SimpleNamespace(
            hf_config=SimpleNamespace(architectures=["Qwen2ForCausalLM"])
        ),
    )
    def test_explicit_fa3_deterministic_keeps_radix(self, _mock_model_config):
        server_args = self._make_server_args(attention_backend="fa3")

        ServerArgs._handle_deterministic_inference(server_args)

        self.assertEqual(server_args.attention_backend, "fa3")
        self.assertFalse(server_args.disable_radix_cache)


if __name__ == "__main__":
    unittest.main()
