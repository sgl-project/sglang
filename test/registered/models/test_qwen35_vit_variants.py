"""
Qwen3.5-9B MMMU 精度测试（对应 launch_server_command.sh 中的 basic/opt1~opt4 五种启动配置）。

用法（直接用文件路径跑，不要用 dotted 模块路径，因为 test/ 不是 python 包）：
    # 单独跑一个
    python3 sglang/test/registered/models/test_qwen35_vit_variants.py -v TestQwen35Basic
    # 跑全部
    python3 sglang/test/registered/models/test_qwen35_vit_variants.py -v

每个测试类会：
  1) 按对应的环境变量 + server flag 拉起 sglang server
  2) 通过 lmms-eval 在 MMMU-val 上跑精度
  3) 打印精度，并断言 >= mmmu_accuracy 阈值
"""

import os
import tempfile
import unittest
from types import SimpleNamespace

from sglang.test.kits.mmmu_vlm_kit import MMMUMultiModelTestBase
from sglang.test.test_utils import DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH

# ---------------------------------------------------------------------------
# 全局配置：所有变体共享的 server 参数（对应 sh 脚本里完全一致的那段）
# ---------------------------------------------------------------------------
# MODEL_PATH = "/data01/models/Qwen3.5-9B/"
MODEL_PATH = "/data00/models/Qwen3-VL-8B-Instruct/"

# MMMU 精度阈值下限，实际可按基线精度微调
MMMU_ACCURACY_THRESHOLD = 0.30

# 所有变体共用的 sglang.launch_server flag（不含 --model-path / --host / --port）
COMMON_SERVER_ARGS = [
    "--mem-fraction-static",
    "0.8",
    "--cuda-graph-max-bs",
    "128",
    "--tensor-parallel-size",
    "1",
    "--mm-attention-backend",
    "fa3",
    "--cuda-graph-bs",
    "128",
    "120",
    "112",
    "104",
    "96",
    "88",
    "80",
    "72",
    "64",
    "56",
    "48",
    "40",
    "32",
    "24",
    "16",
    "8",
    "4",
    "2",
    "1",
    "--disable-radix-cache",
    "--trust-remote-code",
    # "--context-length", "262144",
    # "--reasoning-parser", "qwen3",
]


class _Qwen35VariantBase(MMMUMultiModelTestBase):
    """抽象基类：子类通过 variant_* 字段描述一个具体的启动配置。"""

    # 子类必须覆盖
    variant_name: str = ""
    variant_port: int = 0
    variant_cuda_device: str = ""
    variant_env: dict = (
        {}
    )  # 额外环境变量（除 CUDA_VISIBLE_DEVICES/SGLANG_VLM_CACHE_SIZE_MB 外）
    variant_extra_args: list = []  # 额外的 server flag

    # 所有变体都要设的环境变量
    _COMMON_ENV = {"SGLANG_VLM_CACHE_SIZE_MB": "0"}

    @classmethod
    def setUpClass(cls):
        assert cls.variant_name, "variant_name must be set"
        assert cls.variant_port, "variant_port must be set"

        # 覆盖 base_url，使 server 和 lmms-eval 指向各变体专属端口
        cls.base_url = f"http://127.0.0.1:{cls.variant_port}"

        # 拼装 server 启动 flag
        cls.other_args = list(COMMON_SERVER_ARGS) + list(cls.variant_extra_args)

        # 显存占比（MMMUMultiModelTestBase 会读 parsed_args.mem_fraction_static）
        # 我们把 --mem-fraction-static 放到 other_args 里了，这里给个保守值避免 kit 额外再加一份
        cls.parsed_args = SimpleNamespace(mem_fraction_static=0.7)

        super().setUpClass()

    def _make_env(self) -> dict:
        """合并公共环境变量 + CUDA_VISIBLE_DEVICES + 变体独有的 env。"""
        env = os.environ.copy()
        env.update(self._COMMON_ENV)
        env["CUDA_VISIBLE_DEVICES"] = self.variant_cuda_device
        env.update(self.variant_env)
        return env

    def _run(self):
        model = SimpleNamespace(
            model=MODEL_PATH,
            mmmu_accuracy=MMMU_ACCURACY_THRESHOLD,
        )
        prefix = f"qwen35_mmmu_{self.variant_name}_"
        with tempfile.TemporaryDirectory(prefix=prefix) as out_dir:
            self._run_vlm_mmmu_test(
                model,
                out_dir,
                test_name=f" [{self.variant_name}]",
                custom_env=self._make_env(),
            )


# ---------------------------------------------------------------------------
# basic：无额外特性
# ---------------------------------------------------------------------------
class TestBasic(_Qwen35VariantBase):
    variant_name = "basic"
    variant_port = 8080
    variant_cuda_device = "0"
    variant_env: dict = {}
    variant_extra_args: list = []

    def test_mmmu(self):
        self._run()


# ---------------------------------------------------------------------------
# opt1：--enforce-piecewise-cuda-graph
# ---------------------------------------------------------------------------
class TestQwen35Opt1PiecewiseCudaGraph(_Qwen35VariantBase):
    variant_name = "opt1_piecewise_cuda_graph"
    variant_port = 8070
    variant_cuda_device = "1"
    variant_env: dict = {}
    variant_extra_args = ["--enforce-piecewise-cuda-graph"]

    def test_mmmu(self):
        self._run()


# ---------------------------------------------------------------------------
# opt2：SGLANG_VISION_ATTN_FP8=1
# ---------------------------------------------------------------------------
class TestQwen35Opt2VisionAttnFp8(_Qwen35VariantBase):
    variant_name = "opt2_vision_attn_fp8"
    variant_port = 8060
    variant_cuda_device = "2"
    variant_env = {"SGLANG_VISION_ATTN_FP8": "1"}
    variant_extra_args: list = []

    def test_mmmu(self):
        self._run()


# ---------------------------------------------------------------------------
# opt3：SGLANG_VIT_PACK=1
# ---------------------------------------------------------------------------
class TestQwen35Opt3VitPack(_Qwen35VariantBase):
    variant_name = "opt3_vit_pack"
    variant_port = 8050
    variant_cuda_device = "3"
    variant_env = {"SGLANG_VIT_INFER_MAX_SHAPE": "4096"}
    variant_extra_args: list = []

    def test_mmmu(self):
        self._run()


# ---------------------------------------------------------------------------
# opt4：SGLANG_VIT_ENABLE_CUDA_GRAPH=1
# ---------------------------------------------------------------------------
# class TestQwen35Opt4VitCudaGraph(_Qwen35VariantBase):
#     variant_name = "opt4_vit_enable_cuda_graph"
#     variant_port = 8040
#     variant_cuda_device = "4"
#     variant_env = {"SGLANG_VIT_ENABLE_CUDA_GRAPH": "1"}
#     variant_extra_args: list = []

#     def test_mmmu(self):
#         self._run()


if __name__ == "__main__":
    # 尊重 server 启动耗时
    os.environ.setdefault(
        "SGLANG_TEST_SERVER_LAUNCH_TIMEOUT",
        str(DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH),
    )
    unittest.main()
