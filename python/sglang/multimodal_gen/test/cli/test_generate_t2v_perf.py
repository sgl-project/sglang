# Copied and adapted from: https://github.com/hao-ai-lab/FastVideo

import unittest

from sglang.multimodal_gen.configs.sample.base import DataType
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger
from sglang.multimodal_gen.test.test_utils import TestGenerateBase

logger = init_logger(__name__)


class TestFastWan2_1_T2V(TestGenerateBase):
    model_path = "FastVideo/FastWan2.1-T2V-1.3B-Diffusers"
    extra_args = ["--attention-backend=video_sparse_attn"]
    data_type: DataType = DataType.VIDEO
    thresholds = {
        "test_single_gpu": 13.0,
        "test_cfg_parallel": 15.0,
        "test_usp": 15.0,
        "test_mixed": 15.0 * 1.05,
    }

    # disabled for vsa
    def test_usp(self):
        pass


class TestFastWan2_2_T2V(TestGenerateBase):
    model_path = "FastVideo/FastWan2.2-TI2V-5B-FullAttn-Diffusers"
    extra_args = []
    data_type: DataType = DataType.VIDEO
    thresholds = {
        "test_single_gpu": 25.0,
        "test_cfg_parallel": 30.0,
        "test_usp": 30.0,
        "test_mixed": 30.0,
    }


class TestWan2_1_T2V(TestGenerateBase):
    model_path = "Wan-AI/Wan2.1-T2V-1.3B-Diffusers"
    extra_args = []
    data_type: DataType = DataType.VIDEO
    thresholds = {
        "test_single_gpu": 76.0 * 1.05,
        "test_cfg_parallel": 46.5 * 1.05,
        "test_usp": 39.8 * 1.05,
        "test_mixed": 37.3 * 1.05,
    }

    def test_mixed(self):
        pass

    def test_cfg_parallel(self):
        pass


class TestWan2_2_T2V(TestGenerateBase):
    model_path = "Wan-AI/Wan2.2-T2V-A14B-Diffusers"
    extra_args = []
    data_type: DataType = DataType.VIDEO
    thresholds = {
        "test_single_gpu": 904.3 * 1.05,
        "test_cfg_parallel": 446,
        "test_usp": 316 * 1.05,
        "test_mixed": 159,
    }

    def test_single_gpu(self):
        pass

    def test_mixed(self):
        pass

    def test_cfg_parallel(self):
        pass


if __name__ == "__main__":
    del TestGenerateBase
    unittest.main()
