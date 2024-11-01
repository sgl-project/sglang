import json
import unittest

from sglang.srt.server_args import prepare_server_args


class TestPrepareServerArgs(unittest.TestCase):
    def test_prepare_server_args(self):
        server_args = prepare_server_args(
            [
                "--model-path",
                "model_path",
                "--json-model-override-args",
                '{"rope_scaling": {"factor": 2.0, "rope_type": "linear"}}',
            ]
        )
        self.assertEqual(server_args.model_path, "model_path")
        self.assertEqual(
            json.loads(server_args.json_model_override_args),
            {"rope_scaling": {"factor": 2.0, "rope_type": "linear"}},
        )


if __name__ == "__main__":
    unittest.main()
