# Copied and adapted from: https://github.com/hao-ai-lab/FastVideo

import os
import shlex
import unittest

from PIL import Image

from sglang.multimodal_gen.configs.sample.sampling_params import DataType
from sglang.multimodal_gen.test.single_test_file.cli_generate_common import (
    CLIBase,
    check_image_size,
)
from sglang.multimodal_gen.test.test_utils import DEFAULT_SMALL_MODEL_NAME_FOR_TEST


class TestZImageTurboCLI(CLIBase):
    model_path = DEFAULT_SMALL_MODEL_NAME_FOR_TEST
    extra_args = ("--num-inference-steps=4",)
    data_type: DataType = DataType.IMAGE
    width = 512
    height = 512

    def test_output_file_path_alias(self):
        output_file_path = os.path.join(self.output_path, "zimage_turbo_alias.png")
        _, status = self._run_command(
            "ignored_output_file_name",
            self.model_path,
            args=f"--output-file-path {shlex.quote(output_file_path)}",
        )

        self.assertEqual(status, "Success", "output-file-path command failed")
        self.assertTrue(
            os.path.exists(output_file_path),
            f"Output file not exist for {output_file_path}",
        )
        with Image.open(output_file_path) as image:
            check_image_size(self, image, self.width, self.height)


del CLIBase


if __name__ == "__main__":
    unittest.main()
