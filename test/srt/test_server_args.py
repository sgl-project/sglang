import unittest

from sglang.srt.server_args import prepare_server_args


class TestPrepareServerArgs(unittest.TestCase):
    def test_prepare_server_args(self):
        server_args = prepare_server_args(
            [
                "--model-path",
                "model_path",
                "--model-override-args",
                '{"rope_scaling": {"factor": 2.0, "type": "linear"}}',
            ]
        )
        self.assertEqual(server_args.model_path, "model_path")
        self.assertEqual(
            server_args.model_override_args,
            {"rope_scaling": {"factor": 2.0, "type": "linear"}},
        )


if __name__ == "__main__":
    unittest.main()
