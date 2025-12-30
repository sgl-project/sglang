# Copied and adapted from: https://github.com/hao-ai-lab/FastVideo

import unittest

from sglang.multimodal_gen.configs.sample.sampling_params import DataType
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger
from sglang.multimodal_gen.test.cli.test_generate_common import TestGenerateBase

logger = init_logger(__name__)


class TestFlux_T2V(TestGenerateBase):
    model_path = "black-forest-labs/FLUX.1-dev"
    extra_args = []
    data_type: DataType = DataType.IMAGE
    thresholds = {
        "test_single_gpu": 6.5 * 1.05,
        "test_usp": 8.3 * 1.05,
    }

    def test_cfg_parallel(self):
        pass

    def test_mixed(self):
        pass


if __name__ == "__main__":
    del TestGenerateBase
    unittest.main()
