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
    def test_launch_router_no_exception(self):

        # Create SimpleNamespace with default arguments
        args = SimpleNamespace(
            worker_urls=["http://localhost:8000"],
            host="127.0.0.1",
            port=30000,
            policy="cache_aware",
            cache_threshold=0.5,
            balance_abs_threshold=32,
            balance_rel_threshold=1.0001,
            eviction_interval=60,
            max_tree_size=2**24,
            verbose=False,
        )

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

        # Start router in separate process
        process = multiprocessing.Process(target=run_router)
        try:
            process.start()
            # Wait 3 seconds
            time.sleep(3)
            # Process is still running means router started successfully
            self.assertTrue(process.is_alive())
        finally:
            terminate_process(process)


if __name__ == "__main__":
    unittest.main()
