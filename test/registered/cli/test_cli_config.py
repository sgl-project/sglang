import unittest

from pydantic import ValidationError

from sglang.cli.config import ServerConfig
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="stage-a-cpu-only")


class TestServerConfig(unittest.TestCase):
    def test_valid_config(self):
        """Valid config should construct without error."""
        config = ServerConfig(
            model_path="/path/to/model",
            port=8000,
            host="127.0.0.1",
        )
        self.assertEqual(config.model_path, "/path/to/model")
        self.assertEqual(config.port, 8000)
        self.assertEqual(config.host, "127.0.0.1")

    def test_invalid_port(self):
        """Port below 1024 should raise ValidationError immediately."""
        with self.assertRaises(ValidationError):
            ServerConfig(
                model_path="/path/to/model",
                port=80,
                host="127.0.0.1",
            )

    def test_empty_model_path(self):
        """Empty model_path should raise ValidationError immediately."""
        with self.assertRaises(ValidationError):
            ServerConfig(
                model_path="",
                port=8000,
                host="127.0.0.1",
            )


if __name__ == "__main__":
    unittest.main(verbosity=2)
