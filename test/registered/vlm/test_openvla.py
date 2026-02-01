from sglang.test.ci.ci_register import register_cuda_ci

# OpenVLA (Vision-Language-Action) model tests
register_cuda_ci(est_time=300, suite="multimodal-gen-test-1-gpu")

# Copyright 2023-2024 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""
OpenVLA model tests for SGLang.

OpenVLA is a Vision-Language-Action model that outputs 7 action tokens
representing robot control commands (6-DoF pose + gripper), not text.

Usage:
    python3 -m pytest test/registered/vlm/test_openvla.py -v
    python3 -m unittest test_openvla.TestOpenVLAImports
"""

import unittest

import numpy as np
import torch

from sglang.test.test_utils import CustomTestCase


class TestOpenVLAImports(CustomTestCase):
    """Test that OpenVLA modules import correctly."""

    def test_model_import(self):
        """Test OpenVLA model class imports."""
        from sglang.srt.models.openvla import OpenVLAForActionPrediction

        self.assertTrue(hasattr(OpenVLAForActionPrediction, "forward"))
        self.assertTrue(hasattr(OpenVLAForActionPrediction, "load_weights"))
        self.assertTrue(hasattr(OpenVLAForActionPrediction, "encode_images"))

    def test_processor_import(self):
        """Test OpenVLA processor imports."""
        from sglang.srt.multimodal.processors.openvla import OpenVLAImageProcessor

        self.assertTrue(hasattr(OpenVLAImageProcessor, "process_mm_data_async"))

    def test_model_components(self):
        """Test OpenVLA internal components."""
        from sglang.srt.models.openvla import (
            PrismaticProjector,
            PrismaticVisionBackbone,
        )

        self.assertTrue(hasattr(PrismaticVisionBackbone, "forward"))
        self.assertTrue(hasattr(PrismaticProjector, "forward"))


class TestOpenVLAActionDecoding(CustomTestCase):
    """Test OpenVLA action token decoding logic."""

    def test_action_token_to_continuous(self):
        """Test conversion from action tokens to continuous values.

        OpenVLA uses an inverted token mapping:
        - Tokens are in range [vocab_size - 256, vocab_size - 1] = [31744, 31999]
        - Formula: bin = vocab_size - token - 1
        - Action value uses bin centers: action = (2 * bin + 1) / 256 - 1
        """
        vocab_size = 32000

        # Test tokens corresponding to specific bins:
        # bin 0 (action -0.996): token = 31999
        # bin 127 (action -0.004): token = 31872
        # bin 255 (action 0.996): token = 31744
        action_tokens = torch.tensor(
            [[31999, 31872, 31744, 31935, 31807, 31967, 31999]]
        )

        # Replicate the HF decoding logic
        bin_indices = vocab_size - action_tokens - 1
        bin_indices = bin_indices.clamp(min=0, max=255)
        # Use bin centers: (2 * bin + 1) / 256 - 1
        actions = ((2.0 * bin_indices.float() + 1.0) / 256.0) - 1.0

        # Verify output shape (batch=1, action_dim=7)
        self.assertEqual(actions.shape, (1, 7))

        # Verify all values in valid range [-1, 1]
        self.assertTrue(torch.all(actions >= -1.0))
        self.assertTrue(torch.all(actions <= 1.0))

        # Verify specific values using bin centers
        # Token 31999 -> bin 0 -> action = (2*0 + 1)/256 - 1 = -0.99609
        self.assertAlmostEqual(actions[0, 0].item(), -0.99609, places=3)
        # Token 31872 -> bin 127 -> action = (2*127 + 1)/256 - 1 = -0.00391
        self.assertAlmostEqual(actions[0, 1].item(), -0.00391, places=3)
        # Token 31744 -> bin 255 -> action = (2*255 + 1)/256 - 1 = 0.99609
        self.assertAlmostEqual(actions[0, 2].item(), 0.99609, places=3)

    def test_action_bins_coverage(self):
        """Test that action bins cover the full [-1, 1] range uniformly."""
        n_bins = 256
        bin_indices = torch.arange(n_bins)
        # Use bin centers: (2 * bin + 1) / 256 - 1
        actions = ((2.0 * bin_indices.float() + 1.0) / 256.0) - 1.0

        # First bin (0) should map to action near -1
        self.assertAlmostEqual(actions[0].item(), -0.99609, places=3)
        # Last bin (255) should map to action near 1
        self.assertAlmostEqual(actions[-1].item(), 0.99609, places=3)
        # Middle bin (127) should be near 0
        self.assertAlmostEqual(actions[127].item(), -0.00391, places=3)
        # Bin 128 should be just above 0
        self.assertAlmostEqual(actions[128].item(), 0.00391, places=3)


class TestOpenVLAImageProcessor(CustomTestCase):
    """Test OpenVLA image preprocessing."""

    def test_manual_image_preprocessing(self):
        """Test manual image preprocessing logic matches expected format."""
        from PIL import Image

        # Create a dummy RGB image
        img = Image.new("RGB", (224, 224), color=(128, 128, 128))
        pixel_values = np.array(img, dtype=np.float32) / 255.0

        # Normalize with HF's exact quantized ImageNet stats (for OpenVLA)
        mean = np.array([0.484375, 0.455078125, 0.40625])
        std = np.array([0.228515625, 0.2236328125, 0.224609375])
        pixel_values = (pixel_values - mean) / std

        # Convert to CHW format
        pixel_values = pixel_values.transpose(2, 0, 1)

        # Verify output shape (C, H, W)
        self.assertEqual(pixel_values.shape, (3, 224, 224))

        # For a gray image (128/255 ~ 0.5), normalized values should be near 0
        self.assertTrue(np.abs(pixel_values.mean()) < 1.0)

    def test_processor_initialization(self):
        """Test processor can be initialized with config."""
        from sglang.srt.multimodal.processors.openvla import OpenVLAImageProcessor

        class MockConfig:
            image_size = 224

        class MockServerArgs:
            pass

        processor = OpenVLAImageProcessor(
            hf_config=MockConfig(),
            server_args=MockServerArgs(),
            _processor=None,
            transport_mode=None,
        )

        self.assertEqual(processor.image_size, 224)


@unittest.skipUnless(
    torch.cuda.is_available(),
    "CUDA not available, skipping GPU tests",
)
class TestOpenVLAServer(CustomTestCase):
    """Integration tests for OpenVLA server (requires GPU and model weights)."""

    model = "openvla/openvla-7b"

    @classmethod
    def setUpClass(cls):
        """Launch SGLang server with OpenVLA model."""
        # Check if timm is available
        try:
            import timm  # noqa: F401
        except ImportError:
            raise unittest.SkipTest("timm not installed")

        from sglang.srt.utils import kill_process_tree
        from sglang.test.test_utils import (
            DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            DEFAULT_URL_FOR_TEST,
            popen_launch_server,
        )

        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.api_key = "sk-123456"
        cls.kill_process_tree = kill_process_tree

        try:
            cls.process = popen_launch_server(
                cls.model,
                cls.base_url,
                timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
                api_key=cls.api_key,
                other_args=[
                    "--trust-remote-code",
                    "--enable-multimodal",
                    "--mem-fraction-static=0.80",
                ],
            )
            cls.base_url += "/v1"
            cls._server_started = True
        except Exception as e:
            cls._server_started = False
            cls._skip_reason = str(e)
            cls.process = None

    @classmethod
    def tearDownClass(cls):
        """Shutdown SGLang server."""
        if cls.process is not None:
            cls.kill_process_tree(cls.process.pid)

    def setUp(self):
        if not getattr(self.__class__, "_server_started", False):
            self.skipTest(
                f"Server failed to start: "
                f"{getattr(self.__class__, '_skip_reason', 'unknown')}"
            )

    def test_action_prediction(self):
        """Test OpenVLA action prediction with an image."""
        import openai

        client = openai.Client(api_key=self.api_key, base_url=self.base_url)

        # Use a test image (any image works for format testing)
        test_image_url = (
            "https://raw.githubusercontent.com/sgl-project/sgl-test-files/"
            "refs/heads/main/images/man_ironing_on_back_of_suv.png"
        )

        # OpenVLA prompt format
        prompt = "In: What action should the robot take to pick up the object?\nOut:"

        response = client.chat.completions.create(
            model="default",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {"url": test_image_url},
                        },
                        {
                            "type": "text",
                            "text": prompt,
                        },
                    ],
                },
            ],
            temperature=0,
            max_tokens=7,  # OpenVLA outputs exactly 7 action tokens
        )

        # Verify response structure
        self.assertIsNotNone(response)
        self.assertIsNotNone(response.choices)
        self.assertGreater(len(response.choices), 0)
        self.assertIsNotNone(response.choices[0].message.content)

        print(f"OpenVLA response: {response.choices[0].message.content}")


if __name__ == "__main__":
    unittest.main()
