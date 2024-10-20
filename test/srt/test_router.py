import socket
import unittest
from types import SimpleNamespace

from sglang.srt.utils import kill_child_process
from sglang.test.few_shot_gsm8k import run_eval
from sglang.test.test_utils import (
    DEFAULT_MODEL_NAME_FOR_TEST,
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    popen_launch_router,
    popen_launch_server,
)


def find_available_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        # specify 0 tells the OS to assign a random port
        s.bind(("127.0.0.1", 0))
        # call listen to make it a server socket
        s.listen(1)
        port = s.getsockname()[1]
    return port


# # Example usage
# available_port = find_available_port()
# print(f"Found available port: {available_port}")
DEFAULT_MODEL_NAME_FOR_TEST = "/shared/public/elr-models/meta-llama/Meta-Llama-3.1-8B-Instruct/07eb05b21d191a58c577b4a45982fe0c049d0693/"


class TestRouter(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = DEFAULT_MODEL_NAME_FOR_TEST
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.proc_list = []
        host = "127.0.0.1"
        worker_urls = []
        for i in range(2):
            port = find_available_port()
            url = f"http://{host}:{port}"

            cls.proc_list.append(
                popen_launch_server(
                    cls.model,
                    url,
                    device_id=str(i),
                    timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
                    # return_stdout_stderr=True
                )
            )
            worker_urls.append(url)

        router_port = find_available_port()

        cls.base_url = f"http://{host}:{router_port}"

        cls.proc_list.append(
            popen_launch_router(
                router_url=f"http://{host}:{router_port}",
                worker_urls=worker_urls,
                policy="round_robin",
                timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            )
        )

    @classmethod
    def tearDownClass(cls):
        for p in cls.proc_list:
            kill_child_process(p.pid)

    def test_nothing(self):

        host, port = self.base_url.split("http://")[1].split(":")

        args = SimpleNamespace(
            num_shots=5,
            data_path="/home/jobuser/resources/data/test.jsonl",
            num_questions=1024,
            max_new_tokens=512,
            parallel=128,
            host=f"http://{host}",
            port=port,
        )

        metrics = run_eval(args)

        assert metrics["accuracy"] >= 0.65

    # def test_mmlu(self):
    #     args = SimpleNamespace(
    #         base_url=self.base_url,
    #         model=self.model,
    #         eval_name="mmlu",
    #         num_examples=64,
    #         num_threads=32,
    #     )

    #     metrics = run_eval(args)
    #     assert metrics["score"] >= 0.65


if __name__ == "__main__":
    unittest.main()
