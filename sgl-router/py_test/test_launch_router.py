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
        self.run_router_process(args)


if __name__ == "__main__":
    unittest.main()
