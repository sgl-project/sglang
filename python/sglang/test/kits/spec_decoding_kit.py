from sglang.test.send_one import BenchArgs, send_one_prompt
from sglang.test.test_utils import is_in_ci, write_github_step_summary


class SpecDecodingMixin:
    bs_1_speed_thres: float
    accept_length_thres: float

    def test_bs_1_speed(self):
        args = BenchArgs(port=int(self.base_url.split(":")[-1]), max_new_tokens=2048)
        acc_length, speed = send_one_prompt(args)

        print(f"{acc_length=:.2f} {speed=:.2f}")

        if is_in_ci():
            write_github_step_summary(
                f"### test_bs_1_speed ({self.model})\n"
                f"{acc_length=:.2f}\n"
                f"{speed=:.2f} token/s\n"
            )

        self.assertGreater(acc_length, self.accept_length_thres)
        self.assertGreater(speed, self.bs_1_speed_thres)
