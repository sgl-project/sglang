import os.path
import unittest
from pathlib import Path

from PIL import Image

from sglang.multimodal_gen.api.configs.sample.base import DataType
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger
from sglang.multimodal_gen.test.test_utils import TestCLIBase, check_image_size

logger = init_logger(__name__)


class TestGenerateBase(TestCLIBase):
    model_path: str = None
    extra_args = []
    data_type: DataType = None
    # tested on h100
    thresholds = {}

    width: int = 720
    height: int = 720
    output_path: str = "outputs"
    image_path: str | None = None
    prompt: str | None = "A curious raccoon"

    base_command = [
        "sglang",
        "generate",
        "--text-encoder-cpu-offload",
        "--pin-cpu-memory",
        f"--prompt='{prompt}'",
        "--save-output",
        "--log-level=debug",
        f"--width={width}",
        f"--height={height}",
        f"--output-path={output_path}",
    ]

    results = []

    @classmethod
    def setUpClass(cls):
        cls.results = []

    @classmethod
    def tearDownClass(cls):
        # Print markdown table
        print("\n## Test Results\n")
        print("| Test Case                      | Duration | Status  |")
        print("|--------------------------------|----------|---------|")
        test_keys = ["test_single_gpu", "test_cfg_parallel", "test_usp", "test_mixed"]
        test_key_to_order = {
            test_key: order for order, test_key in enumerate(test_keys)
        }

        ordered_results: list[dict] = [{}] * len(test_keys)

        for result in cls.results:
            order = test_key_to_order[result["key"]]
            ordered_results[order] = result

        for result in ordered_results:
            if not result:
                continue
            print(
                f"| {result['name']:<30} | {result['duration']:<8} | {result['status']:<7} |"
            )
        print()
        durations = [result["duration"] for result in cls.results]
        print(" | ".join([""] + durations + [""]))

    def _run_test(self, name, args, model_path: str, test_key: str):
        time_threshold = self.thresholds[test_key]
        name, duration, status = self._run_command(
            name, args=args, model_path=model_path, test_key=test_key
        )
        self.verify(status, name, duration, time_threshold)

    def verify(self, status, name, duration, time_threshold):
        print("-" * 80)
        print("\n" * 3)

        # test task status
        self.assertEqual(status, "Success", f"{name} command failed")
        self.assertIsNotNone(duration, f"Could not parse duration for {name}")
        self.assertLessEqual(
            duration,
            time_threshold,
            f"{name} failed with {duration:.4f}s > {time_threshold}s",
        )

        # test output file
        path = os.path.join(
            self.output_path, f"{name}.{self.data_type.get_default_extension()}"
        )
        self.assertTrue(os.path.exists(path), f"Output file not exist for {path}")
        if self.data_type == DataType.IMAGE:
            with Image.open(path) as image:
                check_image_size(self, image, self.width, self.height)
        logger.info(f"{name} passed in {duration:.4f}s (threshold: {time_threshold}s)")

    def model_name(self):
        return self.model_path.split("/")[-1]

    def test_single_gpu(self):
        """single gpu"""
        self._run_test(
            name=f"{self.model_name()}, single gpu",
            args=None,
            model_path=self.model_path,
            test_key="test_single_gpu",
        )

    def test_cfg_parallel(self):
        """cfg parallel"""
        if self.data_type == DataType.IMAGE:
            return
        self._run_test(
            name=f"{self.model_name()}, cfg parallel",
            args="--num-gpus 2 --enable-cfg-parallel",
            model_path=self.model_path,
            test_key="test_cfg_parallel",
        )

    def test_usp(self):
        """usp"""
        if self.data_type == DataType.IMAGE:
            return
        self._run_test(
            name=f"{self.model_name()}, usp",
            args="--num-gpus 4 --ulysses-degree=2 --ring-degree=2",
            model_path=self.model_path,
            test_key="test_usp",
        )

    def test_mixed(self):
        """mixed"""
        if self.data_type == DataType.IMAGE:
            return
        self._run_test(
            name=f"{self.model_name()}, mixed",
            args="--num-gpus 4 --ulysses-degree=2 --ring-degree=1 --enable-cfg-parallel",
            model_path=self.model_path,
            test_key="test_mixed",
        )


class TestFastWan2_1_T2V(TestGenerateBase):
    model_path = "FastVideo/FastWan2.1-T2V-1.3B-Diffusers"
    extra_args = ["--attention-backend=video_sparse_attn"]
    data_type: DataType = DataType.VIDEO
    thresholds = {
        "test_single_gpu": 13.0,
        "test_cfg_parallel": 15.0,
        "test_usp": 15.0,
        "test_mixed": 15.0,
    }


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
        "test_single_gpu": 76.0,
        "test_cfg_parallel": 46.5 * 1.05,
        "test_usp": 22.5,
        "test_mixed": 26.5,
    }


class TestWan2_2_T2V(TestGenerateBase):
    model_path = "Wan-AI/Wan2.2-T2V-A14B-Diffusers"
    extra_args = []
    data_type: DataType = DataType.VIDEO
    thresholds = {
        "test_single_gpu": 865,
        "test_cfg_parallel": 446,
        "test_usp": 124,
        "test_mixed": 159,
    }

    def test_mixed(self):
        pass

    def test_cfg_parallel(self):
        pass


class TestFlux_T2V(TestGenerateBase):
    model_path = "black-forest-labs/FLUX.1-dev"
    extra_args = []
    data_type: DataType = DataType.IMAGE
    thresholds = {
        "test_single_gpu": 6.16 * 1.05,
    }


class TestQwenImage(TestGenerateBase):
    model_path = "Qwen/Qwen-Image"
    extra_args = []
    data_type: DataType = DataType.IMAGE
    thresholds = {
        "test_single_gpu": 10.0 * 1.05,
    }


class TestQwenImageEdit(TestGenerateBase):
    model_path = "Qwen/Qwen-Image-Edit"
    extra_args = []
    data_type: DataType = DataType.IMAGE
    thresholds = {
        "test_single_gpu": 40.5 * 1.05,
    }

    prompt: str | None = (
        "Change the rabbit's color to purple, with a flash light background."
    )

    def test_cfg_parallel(self):
        pass

    def test_mixed(self):
        pass

    def test_usp(self):
        pass

    def test_single_gpu(self):
        test_dir = Path(__file__).parent
        img_path = (test_dir / ".." / "test_files" / "rabbit.jpg").resolve().as_posix()
        self.base_command = [
            "sglang",
            "generate",
            "--text-encoder-cpu-offload",
            "--pin-cpu-memory",
            f"--prompt='{self.prompt}'",
            "--save-output",
            "--log-level=debug",
            f"--width={self.width}",
            f"--height={self.height}",
            f"--output-path={self.output_path}",
        ] + [f"--image-path={img_path}"]

        self._run_test(
            name=f"{self.model_name()}, single gpu",
            args=None,
            model_path=self.model_path,
            test_key="test_single_gpu",
        )


if __name__ == "__main__":
    del TestGenerateBase
    unittest.main()
