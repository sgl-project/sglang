import unittest
from contextlib import contextmanager
from unittest.mock import PropertyMock, patch

import torch
import torch.nn as nn

from sglang.multimodal_gen.runtime.layers import low_precision_linear as lpl
from sglang.multimodal_gen.runtime.layers import mlp as mlp_module
from sglang.multimodal_gen.runtime.layers.low_precision_linear import (
    TE_NVFP4_LINEAR_ENABLED_ENV,
    TeNvfp4LinearRunner,
    maybe_get_te_nvfp4_linear_runner,
    te_nvfp4_linear_enabled,
)
from sglang.multimodal_gen.runtime.layers.mlp import MLP
from sglang.multimodal_gen.runtime.models.dits import ltx_2, mova_video_dit, wanvideo


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


class FakeMlpTeRunner:
    def __init__(self):
        self.calls = []

    def try_apply(
        self,
        cache_key: str,
        layer: nn.Module,
        x: torch.Tensor,
        *,
        training: bool,
    ) -> torch.Tensor:
        self.calls.append((cache_key, layer, tuple(x.shape), training))
        output_dim = 8 if cache_key == "fc_in" else 4
        return torch.ones((*x.shape[:-1], output_dim), dtype=x.dtype, device=x.device)


class FakeParallelLinear(nn.Module):
    def __init__(
        self,
        input_size: int,
        output_size: int,
        *args,
        **kwargs,
    ) -> None:
        del args, kwargs
        super().__init__()
        self.output_size = output_size
        self.weight = nn.Parameter(torch.empty(output_size, input_size))
        self.bias = nn.Parameter(torch.empty(output_size))

    def forward(self, x: torch.Tensor):
        return torch.zeros(
            (*x.shape[:-1], self.output_size),
            dtype=x.dtype,
            device=x.device,
        ), None


@contextmanager
def fake_mlp_parallel_linears():
    with (
        patch.object(mlp_module, "ColumnParallelLinear", FakeParallelLinear),
        patch.object(mlp_module, "RowParallelLinear", FakeParallelLinear),
    ):
        yield


@contextmanager
def fake_fp8_autocast(*, enabled, fp8_recipe):
    del enabled, fp8_recipe
    yield


class TestTeNvfp4LinearEnablePolicy(unittest.TestCase):
    def setUp(self):
        FakeTeLinear.instances.clear()

    def test_default_disabled(self):
        with patch.dict("os.environ", {}, clear=True):
            self.assertFalse(te_nvfp4_linear_enabled())
            self.assertIsNone(maybe_get_te_nvfp4_linear_runner("ltx2.video_ffn"))

    def test_bool_env_enabled(self):
        with patch.dict(
            "os.environ",
            {TE_NVFP4_LINEAR_ENABLED_ENV: "1"},
            clear=True,
        ):
            self.assertTrue(te_nvfp4_linear_enabled())

            runner = maybe_get_te_nvfp4_linear_runner("ltx2.video_ffn")
            self.assertIsInstance(runner, TeNvfp4LinearRunner)
            self.assertEqual(runner.target, "ltx2.video_ffn")

    def test_false_bool_env_disabled(self):
        with patch.dict(
            "os.environ",
            {TE_NVFP4_LINEAR_ENABLED_ENV: "0"},
            clear=True,
        ):
            self.assertFalse(te_nvfp4_linear_enabled())
            self.assertIsNone(maybe_get_te_nvfp4_linear_runner("ltx2.video_ffn"))

    def test_ltx2_video_ffn_target_default_disabled(self):
        with patch.dict("os.environ", {}, clear=True):
            self.assertIsNone(ltx_2._ltx2_te_nvfp4_video_ffn_target())

    def test_ltx2_video_ffn_target_requires_env(self):
        with patch.dict(
            "os.environ",
            {TE_NVFP4_LINEAR_ENABLED_ENV: "true"},
            clear=True,
        ):
            self.assertEqual(
                ltx_2._ltx2_te_nvfp4_video_ffn_target(),
                "ltx2.video_ffn",
            )

    def test_wan_and_mova_video_ffn_targets_require_env_and_shape(self):
        with patch.dict("os.environ", {}, clear=True):
            self.assertIsNone(
                wanvideo._wan_te_nvfp4_video_ffn_target(5120, 13824, 5120)
            )
            self.assertIsNone(
                mova_video_dit._mova_te_nvfp4_video_ffn_target(5120, 13824, 5120)
            )

        with patch.dict(
            "os.environ",
            {TE_NVFP4_LINEAR_ENABLED_ENV: "1"},
            clear=True,
        ):
            self.assertEqual(
                wanvideo._wan_te_nvfp4_video_ffn_target(5120, 13824, 5120),
                "wan.video_ffn",
            )
            self.assertEqual(
                mova_video_dit._mova_te_nvfp4_video_ffn_target(5120, 13824, 5120),
                "mova.video_ffn",
            )
            self.assertIsNone(
                wanvideo._wan_te_nvfp4_video_ffn_target(4096, 16384, 4096)
            )
            self.assertIsNone(
                mova_video_dit._mova_te_nvfp4_video_ffn_target(5120, 13824, 16)
            )

    def test_mlp_only_creates_te_runner_when_target_is_present(self):
        runner = object()
        with (
            fake_mlp_parallel_linears(),
            patch.object(
                mlp_module,
                "maybe_get_te_nvfp4_linear_runner",
                return_value=runner,
            ) as maybe_get_runner,
        ):
            mlp = MLP(4, 8, output_dim=4)
            self.assertIsNone(mlp._te_nvfp4_linear)
            maybe_get_runner.assert_not_called()

            mlp = MLP(4, 8, output_dim=4, te_nvfp4_target="wan.video_ffn")
            self.assertIs(mlp._te_nvfp4_linear, runner)
            maybe_get_runner.assert_called_once_with("wan.video_ffn")

    def test_mlp_forward_tries_te_runner_for_both_linear_layers(self):
        with fake_mlp_parallel_linears():
            mlp = MLP(4, 8, output_dim=4)
            runner = FakeMlpTeRunner()
            mlp._te_nvfp4_linear = runner
            x = torch.zeros(2, 3, 4)

            out = mlp(x)

        self.assertEqual(tuple(out.shape), (2, 3, 4))
        self.assertEqual([call[0] for call in runner.calls], ["fc_in", "fc_out"])
        self.assertIs(runner.calls[0][1], mlp.fc_in)
        self.assertIs(runner.calls[1][1], mlp.fc_out)
        self.assertEqual(runner.calls[0][2], (2, 3, 4))
        self.assertEqual(runner.calls[1][2], (2, 3, 8))
        self.assertEqual(runner.calls[0][3], mlp.training)
        self.assertEqual(runner.calls[1][3], mlp.training)

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
