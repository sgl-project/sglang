import json
import unittest

from sglang.cli.render import extract_engine_url, write_renderer_config


class TestStandaloneRendererCli(unittest.TestCase):
    def test_engine_url_is_required_and_removed_from_server_arguments(self):
        engine_url, remaining = extract_engine_url(
            ["model", "--engine-url", "http://engine:30000", "--port", "8000"]
        )
        self.assertEqual(engine_url, "http://engine:30000")
        self.assertEqual(remaining, ["model", "--port", "8000"])

        with self.assertRaisesRegex(ValueError, "requires --engine-url"):
            extract_engine_url(["model"])

    def test_engine_url_equals_form_is_supported(self):
        engine_url, remaining = extract_engine_url(
            ["model", "--engine-url=http://engine:30000"]
        )
        self.assertEqual(engine_url, "http://engine:30000")
        self.assertEqual(remaining, ["model"])

    def test_config_file_contains_the_exact_launch_payload(self):
        payload = {"engine_url": "http://engine:30000", "renderer": {"model": "m"}}
        path = write_renderer_config(payload)
        try:
            self.assertEqual(json.loads(path.read_text()), payload)
        finally:
            path.unlink(missing_ok=True)


if __name__ == "__main__":
    unittest.main()
