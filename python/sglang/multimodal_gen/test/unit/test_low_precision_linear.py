import unittest
from contextlib import contextmanager
from unittest.mock import PropertyMock, patch

import torch
import torch.nn as nn

from sglang.multimodal_gen.runtime.layers import low_precision_linear as lpl
from sglang.multimodal_gen.runtime.layers.low_precision_linear import (
    TE_NVFP4_LINEAR_TARGETS_ENV,
    TeNvfp4LinearRunner,
    maybe_get_te_nvfp4_linear_runner,
    te_nvfp4_linear_target_enabled,
)
from sglang.multimodal_gen.runtime.models.dits import ltx_2


class UnquantizedLinearMethod:
    pass


class FakeTeLinear:
    instances = []

    def __init__(
        self,
        input_size: int,
        output_size: int,
        *,
        bias: bool,
        params_dtype: torch.dtype,
        device: torch.device,
    ) -> None:
        del input_size, bias, params_dtype, device
        self.output_size = output_size
        self.weight = None
        self.bias = None
        self.calls = []
        FakeTeLinear.instances.append(self)

    def train(self, training: bool):
        self.training = training
        return self

    def __call__(
        self,
        x: torch.Tensor,
        *,
        is_first_microbatch: bool | None = None,
    ) -> torch.Tensor:
        self.calls.append(is_first_microbatch)
        return torch.zeros(
            (*x.shape[:-1], self.output_size),
            dtype=x.dtype,
            device=x.device,
        )


@contextmanager
def fake_fp8_autocast(*, enabled, fp8_recipe):
    del enabled, fp8_recipe
    yield


class TestTeNvfp4LinearTargetPolicy(unittest.TestCase):
    def setUp(self):
        FakeTeLinear.instances.clear()

    def test_targets_default_disabled(self):
        with patch.dict("os.environ", {}, clear=True):
            self.assertFalse(te_nvfp4_linear_target_enabled("ltx2.video_ffn"))
            self.assertIsNone(maybe_get_te_nvfp4_linear_runner("ltx2.video_ffn"))

    def test_specific_target_enabled(self):
        with patch.dict(
            "os.environ",
            {TE_NVFP4_LINEAR_TARGETS_ENV: "qwen_image.ffn, ltx2.video_ffn"},
            clear=True,
        ):
            self.assertTrue(te_nvfp4_linear_target_enabled("ltx2.video_ffn"))
            self.assertFalse(te_nvfp4_linear_target_enabled("wan.video_ffn"))

            runner = maybe_get_te_nvfp4_linear_runner("ltx2.video_ffn")
            self.assertIsInstance(runner, TeNvfp4LinearRunner)
            self.assertEqual(runner.target, "ltx2.video_ffn")

    def test_all_target_enabled(self):
        with patch.dict(
            "os.environ",
            {TE_NVFP4_LINEAR_TARGETS_ENV: "all"},
            clear=True,
        ):
            self.assertTrue(te_nvfp4_linear_target_enabled("ltx2.video_ffn"))
            self.assertTrue(te_nvfp4_linear_target_enabled("wan.video_ffn"))

    def test_ltx2_video_ffn_target_default_disabled(self):
        with patch.dict("os.environ", {}, clear=True):
            self.assertIsNone(ltx_2._ltx2_te_nvfp4_video_ffn_target())

    def test_ltx2_video_ffn_target_requires_env(self):
        with patch.dict(
            "os.environ",
            {TE_NVFP4_LINEAR_TARGETS_ENV: "ltx2.video_ffn"},
            clear=True,
        ):
            self.assertEqual(
                ltx_2._ltx2_te_nvfp4_video_ffn_target(),
                "ltx2.video_ffn",
            )

    def test_cpu_input_short_circuits_before_te_or_distributed_setup(self):
        runner = TeNvfp4LinearRunner(target="unit.test")
        layer = nn.Linear(4, 4)
        x = torch.ones(2, 4, dtype=torch.float32)

        self.assertIsNone(runner.try_apply("linear", layer, x, training=False))

    def test_inference_weight_workspace_cache_hint_reuses_after_first_call(self):
        runner = TeNvfp4LinearRunner(target="unit.test")
        layer = nn.Linear(4, 4, dtype=torch.float16)
        layer.quant_method = UnquantizedLinearMethod()
        x = torch.ones(3, 4, dtype=torch.float16)

        with (
            patch.object(torch.Tensor, "is_cuda", new_callable=PropertyMock) as is_cuda,
            patch.object(lpl, "get_tp_world_size", return_value=1),
            patch.object(
                lpl,
                "_get_te_nvfp4_context",
                return_value=(FakeTeLinear, fake_fp8_autocast, object()),
            ),
        ):
            is_cuda.return_value = True
            self.assertIsNotNone(runner.try_apply("linear", layer, x, training=False))
            self.assertIsNotNone(runner.try_apply("linear", layer, x, training=False))

        self.assertEqual(len(FakeTeLinear.instances), 1)
        self.assertEqual(FakeTeLinear.instances[0].calls, [True, False])

    def test_weight_workspace_cache_hint_resets_when_weight_object_changes(self):
        runner = TeNvfp4LinearRunner(target="unit.test")
        layer = nn.Linear(4, 4, dtype=torch.float16)
        layer.quant_method = UnquantizedLinearMethod()
        x = torch.ones(3, 4, dtype=torch.float16)

        with (
            patch.object(torch.Tensor, "is_cuda", new_callable=PropertyMock) as is_cuda,
            patch.object(lpl, "get_tp_world_size", return_value=1),
            patch.object(
                lpl,
                "_get_te_nvfp4_context",
                return_value=(FakeTeLinear, fake_fp8_autocast, object()),
            ),
        ):
            is_cuda.return_value = True
            self.assertIsNotNone(runner.try_apply("linear", layer, x, training=False))
            layer.weight = nn.Parameter(torch.ones_like(layer.weight))
            self.assertIsNotNone(runner.try_apply("linear", layer, x, training=False))

        self.assertEqual(len(FakeTeLinear.instances), 2)
        self.assertEqual(FakeTeLinear.instances[0].calls, [True])
        self.assertEqual(FakeTeLinear.instances[1].calls, [True])

    def test_training_does_not_enable_weight_workspace_cache_hint(self):
        runner = TeNvfp4LinearRunner(target="unit.test")
        layer = nn.Linear(4, 4, dtype=torch.float16)
        layer.quant_method = UnquantizedLinearMethod()
        x = torch.ones(3, 4, dtype=torch.float16)

        with (
            patch.object(torch.Tensor, "is_cuda", new_callable=PropertyMock) as is_cuda,
            patch.object(lpl, "get_tp_world_size", return_value=1),
            patch.object(
                lpl,
                "_get_te_nvfp4_context",
                return_value=(FakeTeLinear, fake_fp8_autocast, object()),
            ),
        ):
            is_cuda.return_value = True
            self.assertIsNotNone(runner.try_apply("linear", layer, x, training=True))
            self.assertIsNotNone(runner.try_apply("linear", layer, x, training=True))

        self.assertEqual(len(FakeTeLinear.instances), 1)
        self.assertEqual(FakeTeLinear.instances[0].calls, [None, None])


if __name__ == "__main__":
    unittest.main()
