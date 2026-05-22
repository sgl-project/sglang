import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests

_REQUEST_TIMEOUT = 180


class PauseResumeInPlaceMixin:
    """Test pause/resume with in_place mode.

    Sends concurrent requests, pauses mid-generation, verifies no progress
    during the pause window, then resumes and verifies all requests complete.

    Subclass must set:
      - pause_generate_url: URL to send /generate requests (or falls back to self.base_url)
      - pause_target_urls: list of URLs to send /pause_generation and /continue_generation
    """

    pause_num_requests: int = 32
    pause_max_new_tokens: int = 512
    pause_duration: float = 5
    pause_generate_url: str = ""
    pause_target_urls: list = []

    def test_pause_resume_in_place(self):
        generate_url = self.pause_generate_url or self.base_url
        target_urls = self.pause_target_urls or [self.base_url]
        num_requests = self.pause_num_requests

        def _generate(prompt_id):
            return requests.post(
                generate_url + "/generate",
                json={
                    "text": f"Question {prompt_id}: Write a short essay about the number {prompt_id}.",
                    "sampling_params": {
                        "temperature": 0.8,
                        "max_new_tokens": self.pause_max_new_tokens,
                    },
                },
                timeout=_REQUEST_TIMEOUT,
            )

        with ThreadPoolExecutor(max_workers=num_requests) as executor:
            futures = {executor.submit(_generate, i): i for i in range(num_requests)}

            time.sleep(1)

            # Pause all targets
            for url in target_urls:
                requests.post(
                    url + "/pause_generation",
                    json={"mode": "in_place"},
                    timeout=30,
                ).raise_for_status()

            time.sleep(0.5)
            done_before = sum(1 for f in futures if f.done())

            time.sleep(self.pause_duration)
            done_after = sum(1 for f in futures if f.done())

            self.assertLess(
                done_before,
                num_requests,
                "All requests completed before pause took effect -- "
                "increase pause_max_new_tokens to make the test meaningful.",
            )

            self.assertEqual(
                done_after - done_before,
                0,
                f"{done_after - done_before} requests completed during pause "
                f"({done_before} before, {done_after} after) -- "
                f"pause_generation was not respected by the scheduler.",
            )

            # Resume all targets (reverse order to unblock downstream first)
            for url in reversed(target_urls):
                requests.post(
                    url + "/continue_generation",
                    json={},
                    timeout=30,
                ).raise_for_status()

            completed = 0
            errors = []
            for future in as_completed(futures, timeout=_REQUEST_TIMEOUT):
                prompt_id = futures[future]
                try:
                    resp = future.result()
                    if resp.status_code == 200:
                        body = resp.json()
                        self.assertIn("text", body)
                        self.assertGreater(len(body["text"]), 0)
                        completed += 1
                    else:
                        errors.append(f"Request {prompt_id}: status={resp.status_code}")
                except Exception as e:
                    errors.append(f"Request {prompt_id}: exception={e}")

        self.assertEqual(
            completed + len(errors),
            num_requests,
            "Some requests did not resolve within the timeout -- likely hung during pause.",
        )
        self.assertEqual(
            completed,
            num_requests,
            f"Some requests failed: {completed}/{num_requests} succeeded. Errors: {errors}",
        )
