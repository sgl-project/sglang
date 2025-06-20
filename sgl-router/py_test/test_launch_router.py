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
            max_payload_size=256 * 1024 * 1024,  # 256MB
            verbose=False,
            log_dir=None,
            service_discovery=False,
            selector=None,
            service_discovery_port=80,
            service_discovery_namespace=None,
            prometheus_port=None,
            prometheus_host=None,
            # PD-specific attributes
            pd_disaggregated=False,
            prefill=None,
            decode=None,
            # Keep worker_urls for regular mode
            worker_urls=[],
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
            # Wait 3 seconds
            time.sleep(3)
            # Process is still running means router started successfully
            self.assertTrue(process.is_alive())
        finally:
            terminate_process(process)

    def test_launch_router_common(self):
        args = self.create_router_args(worker_urls=["http://localhost:8000"])
        self.run_router_process(args)

    def test_launch_router_with_empty_worker_urls(self):
        args = self.create_router_args(worker_urls=[])
        self.run_router_process(args)  # Expected error

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

    def test_launch_router_pd_mode_basic(self):
        """Test basic PD router functionality without actually starting servers."""
        # This test just verifies the PD router can be created and configured
        # without actually starting it (which would require real prefill/decode servers)
        from sglang_router import Router
        from sglang_router.launch_router import RouterArgs
        from sglang_router_rs import PolicyType

        # Test RouterArgs parsing for PD mode
        # Simulate the parsed args structure from argparse with action="append"
        args = self.create_router_args(
            pd_disaggregated=True,
            policy="power_of_two",  # PowerOfTwo is only valid in PD mode
            prefill=[
                ["http://prefill1:8080", "9000"],
                ["http://prefill2:8080", "none"],
            ],
            decode=[
                ["http://decode1:8081"],
                ["http://decode2:8081"],
            ],
            worker_urls=[],  # Empty for PD mode
        )

        router_args = RouterArgs.from_cli_args(args)
        self.assertTrue(router_args.pd_disaggregated)
        self.assertEqual(router_args.policy, "power_of_two")
        self.assertEqual(len(router_args.prefill_urls), 2)
        self.assertEqual(len(router_args.decode_urls), 2)

        # Verify the parsed URLs and bootstrap ports
        self.assertEqual(router_args.prefill_urls[0], ("http://prefill1:8080", 9000))
        self.assertEqual(router_args.prefill_urls[1], ("http://prefill2:8080", None))
        self.assertEqual(router_args.decode_urls[0], "http://decode1:8081")
        self.assertEqual(router_args.decode_urls[1], "http://decode2:8081")

        # Test Router creation in PD mode
        router = Router(
            worker_urls=[],  # Empty for PD mode
            pd_disaggregated=True,
            prefill_urls=[
                ("http://prefill1:8080", 9000),
                ("http://prefill2:8080", None),
            ],
            decode_urls=["http://decode1:8081", "http://decode2:8081"],
            policy=PolicyType.CacheAware,
            host="127.0.0.1",
            port=3001,
        )
        self.assertIsNotNone(router)

    def test_policy_validation(self):
        """Test that policy validation works correctly for PD and regular modes."""
        from sglang_router.launch_router import RouterArgs, launch_router

        # Test 1: PowerOfTwo is only valid in PD mode
        args = self.create_router_args(
            pd_disaggregated=False,
            policy="power_of_two",
            worker_urls=["http://localhost:8000"],
        )

        # Should raise error
        with self.assertRaises(ValueError) as cm:
            launch_router(args)
        self.assertIn(
            "PowerOfTwo policy is only supported in PD disaggregated mode",
            str(cm.exception),
        )

        # Test 2: RoundRobin is not valid in PD mode
        args = self.create_router_args(
            pd_disaggregated=True,
            policy="round_robin",
            prefill=[["http://prefill1:8080", "9000"]],
            decode=[["http://decode1:8081"]],
            worker_urls=[],
        )

        # Should raise error
        with self.assertRaises(ValueError) as cm:
            launch_router(args)
        self.assertIn(
            "RoundRobin policy is not supported in PD disaggregated mode",
            str(cm.exception),
        )

        # Test 3: Valid combinations should not raise errors
        # Regular mode with RoundRobin
        args = self.create_router_args(
            pd_disaggregated=False,
            policy="round_robin",
            worker_urls=["http://localhost:8000"],
        )
        # This should not raise (though it may fail to connect)

        # PD mode with PowerOfTwo
        args = self.create_router_args(
            pd_disaggregated=True,
            policy="power_of_two",
            prefill=[["http://prefill1:8080", "9000"]],
            decode=[["http://decode1:8081"]],
            worker_urls=[],
        )
        # This should not raise (though it may fail to connect)


if __name__ == "__main__":
    unittest.main()
