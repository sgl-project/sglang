"""
python3 -m unittest test_deepseek_ocr.py
"""

import json
import os
import unittest
from pathlib import Path
from unittest.mock import patch

import requests
import torch

from sglang.srt.managers.schedule_batch import Modality, MultimodalDataItem
from sglang.srt.models.deepseek_ocr import DeepseekOCRForCausalLM
from sglang.srt.utils import kill_process_tree
from sglang.srt.utils.hf_transformers import get_tokenizer
from sglang.test.ci.ci_register import register_xpu_ci
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)

register_xpu_ci(est_time=360, suite="stage-b-test-1-gpu-xpu")


class TestDeepSeekOCR(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = "deepseek-ai/DeepSeek-OCR"
        cls.tokenizer = get_tokenizer(cls.model)
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.image_path = str(
            (Path(__file__).resolve().parents[3] / "examples/assets/example_image.png")
        )
        if not os.path.exists(cls.image_path):
            raise FileNotFoundError(f"Image not found: {cls.image_path}")
        cls.common_args = [
            "--device",
            "xpu",
            "--attention-backend",
            "intel_xpu",
        ]
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=[
                *cls.common_args,
            ],
        )

    @classmethod
    def tearDownClass(cls):
        """Fixture that is run once after all tests in the class."""
        if hasattr(cls, "process") and cls.process:
            cls.process.terminate()
            try:
                cls.process.wait(timeout=30)
            except Exception:
                # Force kill if it didn't exit cleanly in time
                kill_process_tree(cls.process.pid)

    def get_request_json(self, max_new_tokens=32, n=1):
        response = requests.post(
            self.base_url + "/generate",
            json={
                "text": "<image>\n<|grounding|>Convert the document to pure text.",
                "image_data": self.image_path,
                "sampling_params": {
                    "temperature": 0 if n == 1 else 0.5,
                    "max_new_tokens": max_new_tokens,
                },
            },
        )
        return response.json()

    def run_decode(
        self,
        max_new_tokens=128,
        n=1,
    ):

        ret = self.get_request_json(max_new_tokens=max_new_tokens, n=n)
        print(json.dumps(ret, indent=2))

        def assert_one_item(item):
            if item["meta_info"]["finish_reason"]["type"] == "stop":
                self.assertEqual(
                    item["meta_info"]["finish_reason"]["matched"],
                    self.tokenizer.eos_token_id,
                )
            elif item["meta_info"]["finish_reason"]["type"] == "length":
                self.assertEqual(
                    len(item["output_ids"]), item["meta_info"]["completion_tokens"]
                )
                self.assertEqual(len(item["output_ids"]), max_new_tokens)

        # Determine whether to assert a single item or multiple items based on n
        if n == 1:
            assert_one_item(ret)
        else:
            self.assertEqual(len(ret), n)
            for i in range(n):
                assert_one_item(ret[i])

        print("=" * 100)

    def test_moe(self):
        self.run_decode()


def _device_available(device: str) -> bool:
    if device == "cpu":
        return True
    if device == "cuda":
        return torch.cuda.is_available()
    if device == "xpu":
        return hasattr(torch, "xpu") and torch.xpu.is_available()
    return False


class TestDeepSeekOCRProcessImageInputBatchedUnequalCrops(CustomTestCase):
    """Regression: `_process_image_input` used to `torch.stack` `images_crop`
    across `mm_items`, which crashed when two OCR requests in the same batch
    had different `num_patches` (e.g. 6 vs 4). Guards against reintroducing
    the stack. Runs on each device the host provides (CPU / XPU / CUDA)."""

    def _make_item(self, device, num_patches, tiles_w, tiles_h):
        item = MultimodalDataItem(modality=Modality.IMAGE)
        item.feature = torch.zeros(3, 640, 640, dtype=torch.float32, device=device)
        item.images_crop = torch.zeros(
            1, num_patches, 3, 640, 640, dtype=torch.float32, device=device
        )
        item.images_spatial_crop = torch.tensor(
            [[[tiles_w, tiles_h]]], dtype=torch.long, device=device
        )
        item.has_local_crops = True
        return item

    def _run_batched_unequal_num_patches(self, device: str):
        if not _device_available(device):
            self.skipTest(f"device {device!r} not available on this host")
        torch_device = torch.device(device)

        items = [
            self._make_item(torch_device, num_patches=6, tiles_w=3, tiles_h=2),
            self._make_item(torch_device, num_patches=4, tiles_w=2, tiles_h=2),
        ]

        def fake_pixel_values_to_embedding(
            pixel_values, images_crop, images_spatial_crop, has_local_crops
        ):
            self.assertEqual(pixel_values.shape[0], 1)
            self.assertEqual(images_crop.dim(), 6)
            self.assertEqual(images_spatial_crop.dim(), 3)
            self.assertEqual(pixel_values.device.type, torch_device.type)
            self.assertEqual(images_crop.device.type, torch_device.type)
            self.assertEqual(images_spatial_crop.device.type, torch_device.type)
            n_patches = images_crop.shape[2]
            return [torch.zeros(n_patches, 8, dtype=torch.float32, device=torch_device)]

        instance = DeepseekOCRForCausalLM.__new__(DeepseekOCRForCausalLM)
        instance.is_ocr2 = False

        stub_param = torch.zeros(1, dtype=torch.float32, device=torch_device)

        class _Stub:
            dtype = torch.float32

            @staticmethod
            def parameters():
                return iter([stub_param])

        instance.sam_model = _Stub()
        instance.vision_model = _Stub()

        with patch.object(
            DeepseekOCRForCausalLM,
            "_pixel_values_to_embedding",
            side_effect=fake_pixel_values_to_embedding,
            autospec=False,
        ):
            out = DeepseekOCRForCausalLM._process_image_input(instance, items)

        # Feature sequences from both items must be concatenated in order.
        # 6 rows for item A + 4 rows for item B = 10 rows.
        self.assertEqual(out.shape, (10, 8))
        self.assertEqual(out.device.type, torch_device.type)

    def test_batched_unequal_num_patches_no_crash_cpu(self):
        self._run_batched_unequal_num_patches("cpu")

    def test_batched_unequal_num_patches_no_crash_xpu(self):
        self._run_batched_unequal_num_patches("xpu")

    def test_batched_unequal_num_patches_no_crash_cuda(self):
        self._run_batched_unequal_num_patches("cuda")


if __name__ == "__main__":
    unittest.main()
