from urllib.parse import urlparse

import requests

from sglang.test.send_one import BenchArgs, send_one_prompt
from sglang.test.test_utils import is_in_ci, write_github_step_summary


class SpecDecodingMixin:
    bs_1_speed_thres: float
    accept_length_thres: float
    bs_1_speed_attempts: int = 3

    def test_bs_1_speed(self):
        args = BenchArgs(port=int(self.base_url.split(":")[-1]), max_new_tokens=2048)
        acc_length, speed = 0.0, 0.0
        for attempt in range(1, self.bs_1_speed_attempts + 1):
            requests.get(self.base_url + "/flush_cache")
            acc_length, speed = send_one_prompt(
                args, label=f"attempt {attempt}", print_output=False
            )
            if acc_length > self.accept_length_thres and speed > self.bs_1_speed_thres:
                break
        requests.get(self.base_url + "/flush_cache")

        if is_in_ci():
            write_github_step_summary(
                f"### test_bs_1_speed ({self.model})\n"
                f"{acc_length=:.2f}\n"
                f"{speed=:.2f} token/s\n"
            )

        self.assertGreater(acc_length, self.accept_length_thres)
        self.assertGreater(speed, self.bs_1_speed_thres)


class MTPAcceptanceLengthMixin:
    mtp_accept_length_thres: float
    mtp_bs_1_speed_thres: float = 1.0
    mtp_max_new_tokens: int = 2048
    mtp_summary_name: str | None = None

    def test_z_mtp_accept_length(self):
        parsed_url = urlparse(self.base_url)
        args = BenchArgs(
            host=parsed_url.hostname or BenchArgs.host,
            port=parsed_url.port or BenchArgs.port,
            max_new_tokens=self.mtp_max_new_tokens,
        )
        acc_length, speed = send_one_prompt(args)
        print(f"{acc_length=:.2f} {speed=:.2f}")

        if is_in_ci():
            name = self.mtp_summary_name or self.model
            write_github_step_summary(
                f"### test_z_mtp_accept_length ({name})\n"
                f"{acc_length=:.2f}\n"
                f"{speed=:.2f} token/s\n"
            )

        self.assertGreater(acc_length, self.mtp_accept_length_thres)
        self.assertGreater(speed, self.mtp_bs_1_speed_thres)
