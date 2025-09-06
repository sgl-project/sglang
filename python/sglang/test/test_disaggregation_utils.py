import time

import requests

from sglang.srt.utils import kill_process_tree
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    CustomTestCase,
    popen_with_error_check,
)


class TestDisaggregationBase(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        cls.process_lb, cls.process_decode, cls.process_prefill = None, None, None
        pass

    @classmethod
    def launch_lb(cls):
        lb_command = [
            "python3",
            "-m",
            "sglang_router.launch_router",
            "--pd-disaggregation",
            "--mini-lb",  # FIXME: remove this
            "--prefill",
            cls.prefill_url,
            "--decode",
            cls.decode_url,
            "--host",
            cls.base_host,
            "--port",
            cls.lb_port,
        ]
        print("Starting load balancer:", " ".join(lb_command))
        cls.process_lb = popen_with_error_check(lb_command)
        cls.wait_server_ready(cls.lb_url + "/health")

    @classmethod
    def wait_server_ready(cls, url, timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH):
        start_time = time.perf_counter()
        while True:
            try:
                response = requests.get(url)
                if response.status_code == 200:
                    print(f"Server {url} is ready")
                    return
            except Exception:
                pass

            if time.perf_counter() - start_time > timeout:
                raise RuntimeError(f"Server {url} failed to start in {timeout}s")
            time.sleep(1)

    @classmethod
    def tearDownClass(cls):
        for process in [cls.process_lb, cls.process_decode, cls.process_prefill]:
            if process:
                try:
                    kill_process_tree(process.pid)
                except Exception as e:
                    print(f"Error killing process {process.pid}: {e}")

        # wait for 5 seconds
        time.sleep(5)
