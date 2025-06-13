import unittest
from types import SimpleNamespace

from sglang.srt.utils import kill_process_tree
from sglang.test.run_eval import run_eval
from sglang.test.test_utils import (
    DEFAULT_MLA_MODEL_NAME_FOR_TEST,
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)


def support_custom_all_to_all():
    try:
        from sgl_kernel import all_to_all

        return True
    except Exception:
        return False


is_all_to_all_supported = support_custom_all_to_all()


class TestDPMla(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = DEFAULT_MLA_MODEL_NAME_FOR_TEST
        cls.base_url = DEFAULT_URL_FOR_TEST
        if not is_all_to_all_supported:
            print("custom_all_to_all is not supported, skip test_dp_mla")
            return
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=[
                "--trust-remote-code",
                "--tp-size",
                "2",
                "--dp-size",
                "2",
                "--attention-backend",
                "flashinfer",
                "--enable-dp-mla",
                "--enable-torch-compile",
                "--torch-compile-max-bs",
                "2",
            ],
        )

    @classmethod
    def tearDownClass(cls):
        if not is_all_to_all_supported:
            return
        kill_process_tree(cls.process.pid)

    def test_mmlu(self):
        if not is_all_to_all_supported:
            print("custom_all_to_all is not supported, skip test_dp_mla::test_mmlu")
            return
        args = SimpleNamespace(
            base_url=self.base_url,
            model=self.model,
            eval_name="mmlu",
            num_examples=64,
            num_threads=32,
        )

        metrics = run_eval(args)
        print(f"{metrics=}")
        self.assertGreater(metrics["score"], 0.5)

    def test_mgsm_en(self):
        if not is_all_to_all_supported:
            print("custom_all_to_all is not supported, skip test_dp_mla::test_mgsm_en")
            return
        args = SimpleNamespace(
            base_url=self.base_url,
            model=self.model,
            eval_name="mgsm_en",
            num_examples=None,
            num_threads=1024,
        )

        metrics = run_eval(args)
        print(f"{metrics=}")
        self.assertGreater(metrics["score"], 0.8)


if __name__ == "__main__":
    unittest.main()
