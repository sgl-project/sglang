import os
import unittest

from PIL import Image

from sglang.multimodal_gen.configs.sample.sampling_params import DataType
from sglang.multimodal_gen.test.cli.test_generate_common import CLIBase, run_command
from sglang.multimodal_gen.test.test_utils import (
    DEFAULT_QWEN_IMAGE_EDIT_2511_MODEL_NAME_FOR_TEST,
    check_image_size,
)


class TestQwenImageEditI2I(CLIBase):
    model_path: str = DEFAULT_QWEN_IMAGE_EDIT_2511_MODEL_NAME_FOR_TEST
    data_type: DataType = DataType.IMAGE
    width: int = 512
    height: int = 512

    test_image_urls = [
        "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-Image/edit2509/edit2509_1.jpg",
        "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-Image/edit2509/edit2509_2.jpg",
    ]

    def get_base_command(self):
        return [
            "sglang",
            "generate",
            "--save-output",
            "--log-level=info",
            f"--width={self.width}",
            f"--height={self.height}",
            f"--output-path={self.output_path}",
        ]

    def verify_multi_output(self, name: str, num_outputs: int):
        output_files = []
        try:
            all_files = os.listdir(self.output_path)
            ext = self.data_type.get_default_extension()
            for f in all_files:
                if f.endswith(f".{ext}"):
                    output_files.append(f)

            self.assertEqual(
                len(output_files),
                num_outputs,
                f"Expected {num_outputs} output files, found {len(output_files)}: {output_files}",
            )

            for f in output_files:
                path = os.path.join(self.output_path, f)
                with Image.open(path) as image:
                    check_image_size(self, image, self.width, self.height)
        finally:
            for f in output_files:
                path = os.path.join(self.output_path, f)
                if os.path.exists(path):
                    os.remove(path)

    def test_single_prompt_single_image(self):
        """Case 1: Single prompt + single image."""
        name = "single_prompt_single_image"

        command = self.get_base_command() + [
            f"--model-path={self.model_path}",
            "--prompt",
            "Add a red hat",
            "--image-path",
            self.test_image_urls[0],
        ]

        succeed = run_command(command)
        self.assertTrue(succeed, f"{name} command failed")
        self.verify_multi_output(name, 1)

    def test_single_prompt_multi_image(self):
        """Case 2: Single prompt + multiple images (image composition)."""
        name = "single_prompt_multi_image"

        command = self.get_base_command() + [
            f"--model-path={self.model_path}",
            "--prompt",
            "Combine both images",
            "--image-path",
            *self.test_image_urls,
        ]

        succeed = run_command(command)
        self.assertTrue(succeed, f"{name} command failed")
        self.verify_multi_output(name, 1)

    def test_multi_prompt_multi_image(self):
        """Case 3: Multiple prompts + multiple images (image editing)."""
        name = "multi_prompt_multi_image"

        command = self.get_base_command() + [
            f"--model-path={self.model_path}",
            "--prompt",
            "Convert to oil painting style",
            "Convert to watercolor style",
            "--image-path",
            *self.test_image_urls,
        ]

        succeed = run_command(command)
        self.assertTrue(succeed, f"{name} command failed")
        self.verify_multi_output(name, 2)

    def test_multi_prompt_single_image(self):
        """Case 4: Multiple prompts + single image (image editing)."""
        name = "multi_prompt_single_image"

        command = self.get_base_command() + [
            f"--model-path={self.model_path}",
            "--prompt",
            "Add a red hat",
            "Change to blue background",
            "--image-path",
            self.test_image_urls[0],
        ]

        succeed = run_command(command)
        self.assertTrue(succeed, f"{name} command failed")
        self.verify_multi_output(name, 2)


del CLIBase


if __name__ == "__main__":
    unittest.main()
