import pickle
import runpy
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from sglang.srt.managers.io_struct import GenerateReqInput
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class TestReplayRequestDump(CustomTestCase):
    def test_request_selection(self):
        """Index zero and count limits must restrict which requests get sent."""
        script = (
            Path(__file__).resolve().parents[4]
            / "scripts/playground/replay_request_dump.py"
        )
        records = [
            (GenerateReqInput(text=text), {}, time, time)
            for text, time in [("third", 0.003), ("first", 0.001), ("second", 0.002)]
        ]
        with tempfile.TemporaryDirectory() as folder:
            dump = Path(folder) / "requests.pkl"
            dump.write_bytes(pickle.dumps({"requests": records}))
            argv = [str(script), "--input-file", str(dump), "--parallel", "1"]
            for flags, expected in [
                (["--idx", "0"], ["first"]),
                (["--req-start", "1", "--req-number", "1"], ["second"]),
                (["--req-number", "0"], []),
                (["--req-start", "1", "--req-number", "2", "--idx", "1"], ["third"]),
            ]:
                with (
                    self.subTest(flags=flags),
                    patch.object(sys, "argv", argv + flags),
                    patch("sglang.benchmark.utils.set_ulimit"),
                    patch("requests.post") as post,
                ):
                    post.return_value.json.return_value = {
                        "meta_info": {"prompt_tokens": 1, "completion_tokens": 1}
                    }
                    runpy.run_path(str(script), run_name="__main__")
                    sent = [call.kwargs["json"]["text"] for call in post.call_args_list]
                    self.assertEqual(sent, expected)


if __name__ == "__main__":
    unittest.main()
