import importlib.util
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import patch

REPO_ROOT = Path(__file__).resolve().parents[3]
CI_REGISTER_PATH = REPO_ROOT / "python" / "sglang" / "test" / "ci" / "ci_register.py"
DSA_MTP_FIXTURE_PATH = (
    REPO_ROOT / "python" / "sglang" / "test" / "server_fixtures" / "dsa_mtp_fixture.py"
)


def _load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _fixture_stubs():
    sglang_module = types.ModuleType("sglang")
    sglang_module.__path__ = []
    srt_module = types.ModuleType("sglang.srt")
    srt_module.__path__ = []
    utils_module = types.ModuleType("sglang.srt.utils")
    utils_module.kill_process_tree = lambda pid: None
    test_module = types.ModuleType("sglang.test")
    test_module.__path__ = []
    test_utils_module = types.ModuleType("sglang.test.test_utils")
    test_utils_module.CustomTestCase = unittest.TestCase
    test_utils_module.DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH = 1
    test_utils_module.DEFAULT_URL_FOR_TEST = "http://127.0.0.1:30000"
    test_utils_module.popen_launch_server = lambda *args, **kwargs: None

    return {
        "sglang": sglang_module,
        "sglang.srt": srt_module,
        "sglang.srt.utils": utils_module,
        "sglang.test": test_module,
        "sglang.test.test_utils": test_utils_module,
    }


def _load_dsa_mtp_fixture():
    with patch.dict(sys.modules, _fixture_stubs()):
        return _load_module("dsa_mtp_fixture_under_test", DSA_MTP_FIXTURE_PATH)


register_cpu_ci = _load_module("ci_register", CI_REGISTER_PATH).register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

dsa_mtp_fixture = _load_dsa_mtp_fixture()
DsaMtpEvalConfigDefaults = dsa_mtp_fixture.DsaMtpEvalConfigDefaults
DsaMtpServerBase = dsa_mtp_fixture.DsaMtpServerBase
CustomTestCase = dsa_mtp_fixture.CustomTestCase


def _arg_value(args, flag):
    return args[args.index(flag) + 1]


class TestDsaMtpFixture(CustomTestCase):
    def test_eval_defaults_match_current_mtp_thresholds(self):
        self.assertEqual(DsaMtpEvalConfigDefaults.gsm8k_accuracy_thres, 0.935)
        self.assertEqual(DsaMtpEvalConfigDefaults.gsm8k_accept_length_thres, 4.5)
        self.assertEqual(DsaMtpEvalConfigDefaults.gsm8k_num_questions, 500)
        self.assertEqual(DsaMtpEvalConfigDefaults.gsm8k_num_threads, 500)
        self.assertEqual(DsaMtpEvalConfigDefaults.gsm8k_num_shots, 20)
        self.assertEqual(DsaMtpEvalConfigDefaults.accept_length_thres, 4.5)

    def test_default_server_args_use_shared_mtp_defaults(self):
        class Server(DsaMtpServerBase):
            model = "test/model"

        args = Server.get_server_args()

        self.assertEqual(_arg_value(args, "--tp"), "8")
        self.assertEqual(_arg_value(args, "--speculative-num-steps"), "5")
        self.assertEqual(_arg_value(args, "--speculative-eagle-topk"), "1")
        self.assertEqual(_arg_value(args, "--speculative-num-draft-tokens"), "6")

    def test_variant_can_override_parallel_sizes_and_append_args(self):
        class Server(DsaMtpServerBase):
            model = "test/model"
            tp_size = 4
            dp_size = 4
            enable_dp_attention = True
            extra_server_args = [
                "--moe-runner-backend",
                "flashinfer_trtllm",
                "--quantization",
                "modelopt_fp4",
            ]

        args = Server.get_server_args()

        self.assertEqual(_arg_value(args, "--tp"), "4")
        self.assertEqual(_arg_value(args, "--dp"), "4")
        self.assertIn("--enable-dp-attention", args)
        self.assertEqual(_arg_value(args, "--moe-runner-backend"), "flashinfer_trtllm")
        self.assertEqual(_arg_value(args, "--quantization"), "modelopt_fp4")


if __name__ == "__main__":
    unittest.main()
