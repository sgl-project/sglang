import unittest
from types import SimpleNamespace

from sglang.srt.utils import kill_child_process
from sglang.test.run_eval import run_eval
from sglang.test.test_utils import (
    DEFAULT_MODEL_NAME_FOR_TEST,
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    popen_launch_server,
)


class TestTorchCompile(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = DEFAULT_MODEL_NAME_FOR_TEST
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=["--enable-torch-compile", "--disable-radix-cache"],
        )

    @classmethod
    def tearDownClass(cls):
        kill_child_process(cls.process.pid)

    def test_mmlu(self):
        args = SimpleNamespace(
            base_url=self.base_url,
            model=self.model,
            eval_name="mmlu",
            num_examples=32,
            num_threads=32,
        )

        metrics = run_eval(args)
        assert metrics["score"] >= 0.6

    def test_throughput(self):
        import time

        import torch

        import sglang as sgl

        @sgl.function
        def test_gen(s):
            s += "Hello, my name is"
            s += sgl.gen("res", temperature=0, ignore_eos=True, max_tokens=256)

        sgl.set_default_backend(sgl.RuntimeEndpoint(self.base_url))
        torch.cuda.synchronize()
        tic = time.time()
        res = test_gen.run()["res"]
        torch.cuda.synchronize()
        tok = time.time()
        print(res)
        print(f"Throughput: {256 / (tok - tic)} tokens/s")


if __name__ == "__main__":
    unittest.main()
