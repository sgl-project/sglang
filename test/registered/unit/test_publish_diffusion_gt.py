import importlib.util
import io
import json
import sys
import tempfile
import unittest
import wave
from pathlib import Path
from unittest.mock import patch

import numpy as np

from sglang.multimodal_gen.test.test_utils import encode_audio_gt_wav
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

SCRIPT = (
    Path(__file__).resolve().parents[3]
    / "scripts/ci/utils/diffusion/publish_diffusion_gt.py"
)
spec = importlib.util.spec_from_file_location("publish_diffusion_gt", SCRIPT)
publisher = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = publisher
spec.loader.exec_module(publisher)


class TestPublishDiffusionAudioGT(CustomTestCase):
    def test_generated_audio_is_collected_validated_and_deduplicated(self):
        content = encode_audio_gt_wav(np.linspace(-0.5, 0.5, 16000))
        with tempfile.TemporaryDirectory() as directory:
            Path(directory, "case_2gpu_audio.wav").write_bytes(content)
            Path(directory, "case.mp4").write_bytes(b"not a GT")
            files = publisher.collect_gt_files(directory, "gt/h100")
        self.assertEqual(files, [("gt/h100/case_2gpu_audio.wav", content)])
        publisher.validate_gt_files(files, files, {}, None)
        entries = [
            {
                "type": "file",
                "path": files[0][0],
                "sha": publisher.git_blob_sha(content),
            }
        ]
        with patch.object(
            publisher, "make_github_request", return_value=json.dumps(entries)
        ):
            shas = publisher.get_remote_blob_shas("owner", "repo", "gt/h100", None)
        self.assertEqual(publisher.filter_changed_files(files, shas), [])

    def test_invalid_audio_is_rejected(self):
        valid = encode_audio_gt_wav(np.ones(16000) * 0.1)
        output = io.BytesIO()
        with wave.open(output, "wb") as wav:
            wav.setnchannels(2)
            wav.setsampwidth(2)
            wav.setframerate(16000)
            wav.writeframes(b"\0" * 32)
        for content in [
            b"not a WAV",
            valid[:-2],
            encode_audio_gt_wav(np.zeros(0)),
            output.getvalue(),
        ]:
            with self.subTest(length=len(content)), self.assertRaises(SystemExit):
                publisher.validate_gt_files([("gt/audio.wav", content)], [], {}, None)


if __name__ == "__main__":
    unittest.main()
