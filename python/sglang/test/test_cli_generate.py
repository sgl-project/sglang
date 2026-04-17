import json
import tempfile
import unittest

from sglang.cli.generate import _get_model_path_for_generate


class TestCliGenerate(unittest.TestCase):
    def test_config_file_can_provide_model_path(self):
        with tempfile.NamedTemporaryFile("w", suffix=".json") as config_file:
            json.dump({"model_path": "baidu/ERNIE-Image-Turbo"}, config_file)
            config_file.flush()

            self.assertEqual(
                _get_model_path_for_generate(["--config", config_file.name]),
                "baidu/ERNIE-Image-Turbo",
            )

    def test_inline_config_file_can_provide_model_path(self):
        with tempfile.NamedTemporaryFile("w", suffix=".json") as config_file:
            json.dump({"model_path": "baidu/ERNIE-Image-Turbo"}, config_file)
            config_file.flush()

            self.assertEqual(
                _get_model_path_for_generate([f"--config={config_file.name}"]),
                "baidu/ERNIE-Image-Turbo",
            )


if __name__ == "__main__":
    unittest.main()
