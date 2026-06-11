import requests


def configure_nixl_pd_backend(test_cls):
    test_cls.transfer_backend = ["--disaggregation-transfer-backend", "nixl"]
    # NIXL backend/network selection is driven by NIXL environment variables
    # such as SGLANG_DISAGGREGATION_NIXL_BACKEND and backend params, not by the
    # Mooncake-specific --disaggregation-ib-device argument.
    test_cls.rdma_devices = []


def assert_process_healthy(test_case, name, process, url, health_path="/health"):
    test_case.assertIsNotNone(process, f"{name} process was not started")
    test_case.assertIsNone(
        process.poll(),
        f"{name} exited unexpectedly with code {process.returncode}",
    )
    response = requests.get(f"{url}{health_path}", timeout=10)
    test_case.assertEqual(response.status_code, 200, response.text)
