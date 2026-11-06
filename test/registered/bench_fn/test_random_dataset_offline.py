import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from huggingface_hub.errors import OfflineModeIsEnabled

from sglang.benchmark.datasets.random import sample_random_requests
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestRandomDatasetOffline(unittest.TestCase):
    def setUp(self):
        self.tokenizer = Mock(vocab_size=32)
        self.tokenizer.num_special_tokens_to_add.return_value = 0
        self.tokenizer.encode.return_value = [1, 2, 3]
        self.tmpdir = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmpdir.cleanup)

    def _write_sharegpt_json(self):
        path = Path(self.tmpdir.name) / "sharegpt.json"
        path.write_text(
            json.dumps(
                [
                    {"conversations": [{"value": "hello"}, {"value": "world"}]},
                    {"conversations": [{"value": "question"}, {"value": "answer"}]},
                ]
            )
        )
        return str(path)

    def test_random_sampler_download_failure_has_offline_guidance(self):
        error = OfflineModeIsEnabled("Cannot reach Hugging Face in offline mode")
        with patch(
            "sglang.benchmark.datasets.random.download_and_cache_hf_file",
            side_effect=error,
        ) as download:
            with self.assertRaises(OfflineModeIsEnabled) as caught:
                sample_random_requests(
                    input_len=8,
                    output_len=4,
                    num_prompts=2,
                    range_ratio=1.0,
                    tokenizer=self.tokenizer,
                    dataset_path="",
                )
        download.assert_called_once()
        self.assertIs(caught.exception, error)
        notes = " ".join(getattr(caught.exception, "__notes__", []))
        self.assertIn("--dataset-name random-ids", notes)
        self.assertIn("--dataset-path", notes)
        self.assertIn("ShareGPT", notes)

    def test_random_sampler_offline_sources_do_not_download(self):
        dataset_path = self._write_sharegpt_json()
        for random_sample, path in ((True, dataset_path), (False, "")):
            with self.subTest(random_sample=random_sample):
                with patch(
                    "sglang.benchmark.datasets.random.download_and_cache_hf_file",
                    side_effect=AssertionError("Unexpected dataset download"),
                ) as download:
                    rows = sample_random_requests(
                        input_len=8,
                        output_len=4,
                        num_prompts=2,
                        range_ratio=1.0,
                        tokenizer=self.tokenizer,
                        dataset_path=path,
                        random_sample=random_sample,
                        return_text=False,
                    )
                download.assert_not_called()
                self.assertEqual(len(rows), 2)
                self.assertTrue(all(len(row.prompt) == 8 for row in rows))
                self.assertTrue(all(row.output_len == 4 for row in rows))


if __name__ == "__main__":
    unittest.main()
