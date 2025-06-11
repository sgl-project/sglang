import multiprocessing
import time
import unittest
from types import SimpleNamespace


def terminate_process(process: multiprocessing.Process, timeout: float = 1.0) -> None:
    """Terminate a process gracefully, with forced kill as fallback.

    Args:
        process: The process to terminate
        timeout: Seconds to wait for graceful termination before forcing kill
    """
    if not process.is_alive():
        return

    process.terminate()
    process.join(timeout=timeout)
    if process.is_alive():
        process.kill()  # Force kill if terminate didn't work
        process.join()


class TestLaunchRouter(unittest.TestCase):
    def setUp(self):
        """Set up default arguments for router tests."""
        self.default_args = SimpleNamespace(
            host="127.0.0.1",
            port=30000,
            policy="cache_aware",
            worker_startup_timeout_secs=600,
            worker_startup_check_interval=10,
            cache_threshold=0.5,
            balance_abs_threshold=32,
            balance_rel_threshold=1.0001,
            eviction_interval=60,
            max_tree_size=2**24,
            max_payload_size=4 * 1024 * 1024,  # 4MB
            verbose=False,
            log_dir=None,
            service_discovery=False,
            selector=None,
            service_discovery_port=80,
            service_discovery_namespace=None,
            # PDLB specific args
            prefill_urls=[],
            decode_urls=[],
            pd_selection_policy="random",
        )

    def create_router_args(self, **kwargs):
        """Create router arguments by updating default args with provided kwargs."""
        args_dict = vars(self.default_args).copy()
        args_dict.update(kwargs)
        return SimpleNamespace(**args_dict)

    def run_router_process(self, args):
        """Run router in a separate process and verify it starts successfully."""

        def run_router():
            try:
                from sglang_router.launch_router import launch_router

                router = launch_router(args)
                if router is None:
                    return 1
                return 0
            except Exception as e:
                print(e)
                return 1

        process = multiprocessing.Process(target=run_router)
        try:
            process.start()
            # Wait for router to start or fail
            process.join(timeout=5)  # Increased timeout slightly for PD startup
            # If process exited, check exit code
            if process.exitcode is not None:
                self.assertEqual(
                    process.exitcode,
                    0,
                    f"Router process failed with exit code {process.exitcode}",
                )
            else:  # Still alive, means it started
                self.assertTrue(
                    process.is_alive(),
                    "Router process did not start or exited prematurely",
                )
        finally:
            if process.is_alive():
                terminate_process(process)

    def test_launch_router_common(self):
        args = self.create_router_args(worker_urls=["http://localhost:8000"])
        self.run_router_process(args)

    def test_launch_router_non_pd_missing_worker_urls(self):
        # Test that non-PD router fails if worker_urls is not provided
        from sglang_router.launch_router import launch_router

        args = self.create_router_args(policy="random", worker_urls=None)
        with self.assertRaises(ValueError) as context:
            launch_router(args)
        self.assertIn("Missing worker_urls for selected policy", str(context.exception))

    def test_launch_router_with_service_discovery(self):
        # Test router startup with service discovery enabled but no selectors
        args = self.create_router_args(
            worker_urls=[], service_discovery=True, selector=["app=test-worker"]
        )
        self.run_router_process(args)

    def test_launch_router_with_service_discovery_namespace(self):
        # Test router startup with service discovery enabled and namespace specified
        args = self.create_router_args(
            worker_urls=[],
            service_discovery=True,
            selector=["app=test-worker"],
            service_discovery_namespace="test-namespace",
        )
        self.run_router_process(args)

    # --- PrefillDecode Router Tests ---

    def test_launch_router_pd_successful_random(self):
        args = self.create_router_args(
            policy="prefill_decode",
            prefill_urls=["http://localhost:8001"],
            decode_urls=["http://localhost:8002"],
            pd_selection_policy="random",
            port=30001,  # Use a different port to avoid conflict
        )
        self.run_router_process(args)

    def test_launch_router_pd_successful_cache_aware(self):
        args = self.create_router_args(
            policy="prefill_decode",
            prefill_urls=["http://localhost:8003"],
            decode_urls=["http://localhost:8004"],
            pd_selection_policy="cache_aware",
            cache_threshold=0.6,  # Ensure these are passed
            balance_abs_threshold=10,
            balance_rel_threshold=1.5,
            port=30002,  # Use a different port
        )
        self.run_router_process(args)

    def test_launch_router_pd_missing_prefill_urls(self):
        from sglang_router.launch_router import launch_router

        args = self.create_router_args(
            policy="prefill_decode",
            # prefill_urls missing
            decode_urls=["http://localhost:8002"],
            pd_selection_policy="random",
            port=30003,
        )
        with self.assertRaises(ValueError) as context:
            launch_router(args)
        self.assertIn(
            "Missing prefill_urls or decode_urls for prefill_decode policy",
            str(context.exception),
        )

    def test_launch_router_pd_missing_decode_urls(self):
        from sglang_router.launch_router import launch_router

        args = self.create_router_args(
            policy="prefill_decode",
            prefill_urls=["http://localhost:8001"],
            # decode_urls missing
            pd_selection_policy="random",
            port=30004,
        )
        with self.assertRaises(ValueError) as context:
            launch_router(args)
        self.assertIn(
            "Missing prefill_urls or decode_urls for prefill_decode policy",
            str(context.exception),
        )


if __name__ == "__main__":
    unittest.main()
