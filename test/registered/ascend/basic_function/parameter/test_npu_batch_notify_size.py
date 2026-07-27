import unittest
from concurrent.futures import ThreadPoolExecutor

import requests

from sglang.srt.utils import kill_process_tree
from sglang.test.ascend.test_ascend_utils import LLAMA_3_2_1B_INSTRUCT_WEIGHTS_PATH
from sglang.test.ci.ci_register import register_npu_ci
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)

register_npu_ci(est_time=600, suite="full-1-npu-a3", nightly=True)


class TestBatchNotifySize(CustomTestCase):
    """Test --batch-notify-size parameter: controls streaming response
    coroutine notification batching in tokenizer_manager.

    [Test Category] Parameter
    [Test Target] --batch-notify-size
    """

    model = LLAMA_3_2_1B_INSTRUCT_WEIGHTS_PATH
    base_url = DEFAULT_URL_FOR_TEST
    NPU_FIXTURE_ARGS = ["--attention-backend", "ascend"]

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @classmethod
    def _launch_server(cls, other_args):
        return popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=cls.NPU_FIXTURE_ARGS + other_args,
        )

    # ------------------------------------------------------------------
    # TC-BN: batch_notify_size with concurrent streaming
    #
    #   Covers both the threshold path (len >= size) and the loop-exit
    # ------------------------------------------------------------------

    def test_batch_notify_size(self):
        """Verify --batch-notify-size is accepted and concurrent
        streaming requests complete successfully.

        Tests both immediate-notify (size=1) and batch-flush (size=32)
        semantics with 16 concurrent streams.
        """
        for size in ("1", "32"):
            print(
                f"\n[TC-BN] batch_notify_size={size} — " f"16 concurrent streams",
                flush=True,
            )

            process = self._launch_server(
                [
                    "--batch-notify-size",
                    size,
                    "--mem-fraction-static",
                    "0.8",
                ]
            )
            try:
                health = requests.get(f"{self.base_url}/health_generate")
                self.assertEqual(health.status_code, 200)

                def _one_stream(idx):
                    session = requests.Session()
                    try:
                        resp = session.post(
                            f"{self.base_url}/v1/chat/completions",
                            json={
                                "messages": [{"role": "user", "content": "Hello"}],
                                "stream": True,
                                "max_tokens": 16,
                            },
                        )
                        resp.raise_for_status()
                        count = 0
                        for line in resp.iter_lines():
                            if line:
                                count += 1
                        return {"idx": idx, "ok": True, "chunks": count}
                    except Exception as e:
                        return {"idx": idx, "ok": False, "error": str(e)}
                    finally:
                        session.close()

                with ThreadPoolExecutor(max_workers=16) as pool:
                    results = list(pool.map(_one_stream, range(16)))

                succeeded = [r for r in results if r["ok"]]
                failed = [r for r in results if not r["ok"]]
                print(
                    f"[TC-BN-{size}] {len(succeeded)}/16 OK, " f"{len(failed)} failed",
                    flush=True,
                )
                for r in failed:
                    print(f"[TC-BN-{size}] Stream {r['idx']}: {r['error']}", flush=True)

                self.assertEqual(len(results), 16)
                self.assertTrue(
                    all(r["ok"] for r in results),
                    f"{len(failed)} streams failed " f"(batch_notify_size={size})",
                )
            finally:
                kill_process_tree(process.pid)

        print(f"\n[TC-BN] PASSED — both batch_notify_size=1 and 32", flush=True)


if __name__ == "__main__":
    unittest.main()
