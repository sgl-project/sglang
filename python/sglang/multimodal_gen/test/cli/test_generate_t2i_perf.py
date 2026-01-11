# Copied and adapted from: https://github.com/hao-ai-lab/FastVideo

import unittest

from sglang.multimodal_gen.configs.sample.sampling_params import DataType
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger
from sglang.multimodal_gen.test.cli.test_generate_common import CLIBase

logger = init_logger(__name__)


class TestFlux_T2V(CLIBase):
    model_path = "black-forest-labs/FLUX.1-dev"
    extra_args = []
    data_type: DataType = DataType.IMAGE


del CLIBase


if __name__ == "__main__":
    unittest.main()
