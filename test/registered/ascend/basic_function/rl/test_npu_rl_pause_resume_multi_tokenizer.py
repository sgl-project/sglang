"""
E2E tests for NPU pause/resume generation — multi-tokenizer worker PauseContinueBroadcast.
"""

import logging
import time
import unittest
from concurrent.futures import ThreadPoolExecutor, as_completed

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

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)

register_npu_ci(est_time=90, suite="full-2-npu-a3", nightly=True)


class TestNpuMultiTokenizerPauseResume(CustomTestCase):
    """E2E: pause/resume in multi-tokenizer worker mode.

    [Test Category] RL Pause/Resume + Multi-Worker
    [Test Target] POST /pause_generation, POST /continue_generation
                  with PauseContinueBroadcast (multi_tokenizer_mixin.py)
    """

    PAUSE_NUM_REQUESTS = 16
    PAUSE_MAX_NEW_TOKENS = 512
    PAUSE_DURATION = 5
    REQUEST_TIMEOUT = 180

    @classmethod
    def setUpClass(cls):
        cls.model = LLAMA_3_2_1B_INSTRUCT_WEIGHTS_PATH
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=[
                "--attention-backend",
                "ascend",
                "--tp-size",
                2,
                "--tokenizer-worker-num",
                2,
                "--cuda-graph-bs",
                4,
                16,
            ],
        )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)

    def _generate(
        self, prompt_id: int, max_tokens: int = None, ignore_eos: bool = False
    ):
        sampling_params: dict = {
            "temperature": 0.8,
            "max_new_tokens": max_tokens or self.PAUSE_MAX_NEW_TOKENS,
        }
        if ignore_eos:
            sampling_params["ignore_eos"] = True
        return requests.post(
            self.base_url + "/generate",
            json={
                "text": (
                    f"Question {prompt_id}: "
                    f"Write a short essay about the number {prompt_id}."
                ),
                "sampling_params": sampling_params,
            },
            timeout=self.REQUEST_TIMEOUT,
        )

    def _pause(self, mode: str = "in_place"):
        return requests.post(
            self.base_url + "/pause_generation",
            json={"mode": mode},
            timeout=30,
        )

    def _continue(self, torch_empty_cache: bool = True):
        return requests.post(
            self.base_url + "/continue_generation",
            json={"torch_empty_cache": torch_empty_cache},
            timeout=30,
        )

    def test_in_place_pause_multi_worker(self):
        """
        in_place pause with multiple tokenizer workers.

        Verification strategy:
          1. Submit N concurrent requests → distributed across 2 workers
          2. Brief sleep, then pause — race pause against generation
          3. Assert no request completes during pause window (core invariant)
          4. Resume → assert all N complete

        If any worker missed the broadcast, it would continue generating
        during pause (violating step 3) or hang permanently after resume
        (violating step 4).

        Router broadcasts PauseContinueBroadcast(is_pause=True) to
        all workers; resume broadcasts is_pause=False.
        """
        with ThreadPoolExecutor(max_workers=self.PAUSE_NUM_REQUESTS) as executor:
            futures = {
                executor.submit(self._generate, i, ignore_eos=True): i
                for i in range(self.PAUSE_NUM_REQUESTS)
            }

            time.sleep(0.5)

            self._pause("in_place").raise_for_status()

            try:
                time.sleep(0.5)
                done_before = sum(1 for f in futures if f.done())

                time.sleep(self.PAUSE_DURATION)
                done_after = sum(1 for f in futures if f.done())

                logger.info(
                    "Pause check: done_before=%d done_after=%d total=%d",
                    done_before,
                    done_after,
                    self.PAUSE_NUM_REQUESTS,
                )

                self.assertLess(
                    done_before,
                    self.PAUSE_NUM_REQUESTS,
                    "All %d requests completed before pause took effect "
                    "— pause window never exercised" % self.PAUSE_NUM_REQUESTS,
                )
                self.assertEqual(
                    done_after - done_before,
                    0,
                    f"{done_after - done_before} requests completed during pause "
                    f"— PauseContinueBroadcast not respected by a worker",
                )
            finally:
                self._continue().raise_for_status()

            # All requests should complete after resume on both workers
            completed = 0
            errors = []
            for future in as_completed(futures, timeout=self.REQUEST_TIMEOUT):
                prompt_id = futures[future]
                try:
                    resp = future.result()
                    if resp.status_code == 200:
                        body = resp.json()
                        if body.get("text") and len(body["text"]) > 0:
                            # ignore_eos=True → finish_reason must be "length"
                            self.assertEqual(
                                body["meta_info"]["finish_reason"]["type"],
                                "length",
                                f"Request {prompt_id}: expected finish_reason='length', "
                                f"got {body['meta_info'].get('finish_reason')}",
                            )
                            # 512 tokens should produce substantial output
                            self.assertGreater(
                                len(body["text"]),
                                50,
                                f"Request {prompt_id}: output too short "
                                f"({len(body['text'])} chars)",
                            )
                            completed += 1
                        else:
                            errors.append(f"Request {prompt_id}: empty text")
                    else:
                        errors.append(f"Request {prompt_id}: status={resp.status_code}")
                except Exception as e:
                    errors.append(f"Request {prompt_id}: exception={e}")

        self.assertEqual(completed + len(errors), self.PAUSE_NUM_REQUESTS)
        self.assertEqual(
            completed,
            self.PAUSE_NUM_REQUESTS,
            f"Failed: {completed}/{self.PAUSE_NUM_REQUESTS}. Errors: {errors}",
        )

    def test_retract_pause_multi_worker(self):
        """
        retract pause with multiple tokenizer workers.
        Each worker independently retracts its running requests
        into its own waiting_queue.
        """
        num_requests = 4

        with ThreadPoolExecutor(max_workers=num_requests) as executor:
            futures = {
                executor.submit(
                    self._generate,
                    i,
                    ignore_eos=True,
                ): i
                for i in range(num_requests)
            }

            time.sleep(0.5)

            self._pause("retract").raise_for_status()

            time.sleep(0.5)
            done_before = sum(1 for f in futures if f.done())

            time.sleep(self.PAUSE_DURATION)
            done_after = sum(1 for f in futures if f.done())

            logger.info(
                "Retract pause check: done_before=%d done_after=%d total=%d",
                done_before,
                done_after,
                num_requests,
            )
            self.assertLess(
                done_before,
                num_requests,
                "All %d requests completed before retract took effect "
                "— retract window never exercised" % num_requests,
            )
            self.assertEqual(
                done_after - done_before,
                0,
                f"{done_after - done_before} requests completed during retract pause "
                f"— retract did not pause generation",
            )

            self._continue().raise_for_status()

            completed = 0
            errors = []
            for future in as_completed(futures, timeout=self.REQUEST_TIMEOUT):
                prompt_id = futures[future]
                try:
                    resp = future.result()
                    if resp.status_code == 200:
                        body = resp.json()
                        if body.get("text") and len(body["text"]) > 0:
                            # ignore_eos=True → finish_reason must be "length"
                            self.assertEqual(
                                body["meta_info"]["finish_reason"]["type"],
                                "length",
                                f"Request {prompt_id}: expected finish_reason='length', "
                                f"got {body['meta_info'].get('finish_reason')}",
                            )
                            self.assertGreater(
                                len(body["text"]),
                                50,
                                f"Request {prompt_id}: output too short "
                                f"({len(body['text'])} chars)",
                            )
                            completed += 1
                        else:
                            errors.append(f"Request {prompt_id}: empty text")
                    else:
                        errors.append(f"Request {prompt_id}: status={resp.status_code}")
                except Exception as e:
                    errors.append(f"Request {prompt_id}: exception={e}")

        self.assertEqual(completed + len(errors), num_requests)
        self.assertEqual(
            completed,
            num_requests,
            f"Failed: {completed}/{num_requests}. Errors: {errors}",
        )


if __name__ == "__main__":
    unittest.main()
