# SPDX-License-Identifier: Apache-2.0

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from safetensors.torch import safe_open, save_file
from torch import nn

from sglang.multimodal_gen.runtime.layers.linear import (
    ColumnParallelLinear,
    ReplicatedLinear,
    UnquantizedLinearMethod,
)
from sglang.multimodal_gen.runtime.layers.quantization.bitsandbytes import (
    BitsAndBytesConfig,
)
from sglang.multimodal_gen.runtime.loader import fsdp_load, rank_local_checkpoint
from sglang.multimodal_gen.runtime.loader.weight_load_plan import WeightLoadPlan
from sglang.srt.layers.quantization.fp8 import Fp8Config


class _UniformDtypeModel(nn.Module):
    param_names_mapping = {}
    # Every model the FSDP loader accepts declares its custom forward entry
    # points, as BaseDiT and TextEncoder do; none here drive one.
    _fsdp_forward_methods: tuple[str, ...] = ()

    def __init__(self) -> None:
        super().__init__()

    def post_load_weights(self) -> None:
        pass


class _MixedDtypeModel(_UniformDtypeModel):
    _fsdp_mixed_dtype_params = True


class _ReplicatedLinearModel(_UniformDtypeModel):
    def __init__(self) -> None:
        super().__init__()
        self.proj = ReplicatedLinear(4, 4, bias=False)


class _CustomEntrypointModel(_UniformDtypeModel):
    _fsdp_forward_methods = ("refine_prompt_embeds",)

    def refine_prompt_embeds(self) -> None:
        pass


class _MetaBufferModel(_UniformDtypeModel):
    """Mirrors cosmos3: a non-checkpoint buffer stays on meta until
    post_load_weights() rebuilds it."""

    def __init__(self) -> None:
        super().__init__()
        self.proj = ReplicatedLinear(4, 4, bias=False)
        self.register_buffer("inv_freq", torch.empty(2), persistent=False)

    def post_load_weights(self) -> None:
        if self.inv_freq.is_meta:
            device = next(self.parameters()).device
            self.register_buffer(
                "inv_freq", torch.ones(2, device=device), persistent=False
            )


class TestFSDPMixedPrecisionPolicy(unittest.TestCase):
    def test_quant_config_detection_uses_the_runtime_instance(self):
        self.assertTrue(fsdp_load._is_bitsandbytes_quant_config(BitsAndBytesConfig()))
        self.assertFalse(fsdp_load._is_bitsandbytes_quant_config(Fp8Config()))

    def _load_and_capture_policy(
        self,
        model_cls: type[nn.Module],
        *,
        fsdp_inference: bool,
    ):
        with (
            patch.object(fsdp_load.current_platform, "is_mps", return_value=False),
            patch.object(fsdp_load, "init_device_mesh", return_value=object()),
            patch.object(fsdp_load, "shard_model") as shard_model,
            patch.object(
                fsdp_load,
                "safetensors_weights_iterator",
                return_value=iter(()),
            ),
            patch.object(fsdp_load, "load_model_from_full_model_state_dict"),
            patch.object(fsdp_load, "set_mixed_precision_policy") as set_policy,
        ):
            fsdp_load.maybe_load_fsdp_model(
                model_cls=model_cls,
                init_params={},
                weight_dir_list=[],
                device=torch.device("cpu"),
                hsdp_replicate_dim=1,
                hsdp_shard_dim=1,
                param_dtype=torch.bfloat16,
                reduce_dtype=torch.float32,
                fsdp_inference=fsdp_inference,
            )

        state_kwargs = set_policy.call_args.kwargs
        return state_kwargs["mp_policy"], state_kwargs, shard_model

    def test_uniform_dtype_model_uses_requested_fsdp_param_dtype(self):
        policy, state_kwargs, shard_model = self._load_and_capture_policy(
            _UniformDtypeModel,
            fsdp_inference=True,
        )

        self.assertEqual(policy.param_dtype, torch.bfloat16)
        self.assertEqual(policy.reduce_dtype, torch.float32)
        self.assertIsNone(policy.output_dtype)
        self.assertFalse(policy.cast_forward_inputs)
        self.assertEqual(state_kwargs["param_dtype"], torch.bfloat16)
        self.assertIs(shard_model.call_args.kwargs["mp_policy"], policy)

    def test_mixed_dtype_model_preserves_original_fsdp_param_dtypes(self):
        policy, state_kwargs, shard_model = self._load_and_capture_policy(
            _MixedDtypeModel,
            fsdp_inference=True,
        )

        self.assertIsNone(policy.param_dtype)
        self.assertEqual(policy.reduce_dtype, torch.float32)
        self.assertIsNone(policy.output_dtype)
        self.assertFalse(policy.cast_forward_inputs)
        self.assertEqual(state_kwargs["param_dtype"], torch.bfloat16)
        self.assertIs(shard_model.call_args.kwargs["mp_policy"], policy)

    def test_mixed_dtype_opt_in_does_not_change_non_fsdp_policy(self):
        policy, state_kwargs, shard_model = self._load_and_capture_policy(
            _MixedDtypeModel,
            fsdp_inference=False,
        )

        self.assertEqual(policy.param_dtype, torch.bfloat16)
        self.assertEqual(state_kwargs["param_dtype"], torch.bfloat16)
        shard_model.assert_not_called()


class TestFSDPEntrypointRegistration(unittest.TestCase):
    def _load_and_capture_registrations(self, model_cls: type[nn.Module]):
        with (
            patch.object(fsdp_load.current_platform, "is_mps", return_value=False),
            patch.object(fsdp_load, "init_device_mesh", return_value=object()),
            patch.object(fsdp_load, "shard_model"),
            patch.object(
                fsdp_load,
                "safetensors_weights_iterator",
                return_value=iter(()),
            ),
            patch.object(fsdp_load, "load_model_from_full_model_state_dict"),
            patch.object(fsdp_load, "register_fsdp_forward_method") as register,
        ):
            model = fsdp_load.maybe_load_fsdp_model(
                model_cls=model_cls,
                init_params={},
                weight_dir_list=[],
                device=torch.device("cpu"),
                hsdp_replicate_dim=1,
                hsdp_shard_dim=1,
                param_dtype=torch.bfloat16,
                reduce_dtype=torch.float32,
                fsdp_inference=True,
            )

        return model, register

    def test_declared_entry_points_are_registered(self):
        model, register = self._load_and_capture_registrations(_CustomEntrypointModel)

        register.assert_called_once_with(model, "refine_prompt_embeds")

    def test_model_without_entry_points_registers_nothing(self):
        _, register = self._load_and_capture_registrations(_UniformDtypeModel)

        register.assert_not_called()


class TestOrdinaryWeightLoading(unittest.TestCase):
    def _load_replicated_weight(
        self, device: torch.device, *, allow_device_tensor_assignment: bool = False
    ) -> tuple[torch.nn.Parameter, torch.Tensor]:
        with torch.device("meta"):
            model = _ReplicatedLinearModel()
        checkpoint_weight = torch.arange(
            16, dtype=torch.float32, device=device
        ).reshape(4, 4)

        fsdp_load.load_model_from_full_model_state_dict(
            model,
            iter((("proj.weight", checkpoint_weight),)),
            checkpoint_load_device=device,
            param_dtype=torch.float32,
            strict=True,
            param_names_mapping=fsdp_load.get_param_names_mapping({}),
            allow_device_tensor_assignment=allow_device_tensor_assignment,
        )
        return model.proj.weight, checkpoint_weight

    def test_direct_device_loading_skips_rank_local_cpu_checkpoint(self):
        load_plan = WeightLoadPlan(
            checkpoint_load_device=torch.device("cuda:0"),
            load_full_state_dict_on_device=True,
        )
        with (
            patch.object(fsdp_load.current_platform, "is_mps", return_value=False),
            patch.object(
                rank_local_checkpoint,
                "try_load_rank_local_tp_state_dict",
            ) as rank_local_load,
            patch.object(
                fsdp_load,
                "safetensors_weights_iterator",
                return_value=iter(()),
            ) as weight_iterator,
            patch.object(fsdp_load, "load_model_from_full_model_state_dict"),
        ):
            fsdp_load.maybe_load_fsdp_model(
                model_cls=_UniformDtypeModel,
                init_params={},
                weight_dir_list=["model.safetensors"],
                device=torch.device("cuda:0"),
                hsdp_replicate_dim=1,
                hsdp_shard_dim=1,
                param_dtype=torch.bfloat16,
                reduce_dtype=torch.float32,
                weight_load_plan=load_plan,
            )

        rank_local_load.assert_not_called()
        weight_iterator.assert_called_once_with(
            ["model.safetensors"],
            key_filter=None,
            weight_load_plan=load_plan,
        )

    def test_tp1_unquantized_linear_adopts_cpu_checkpoint_storage(self):
        model_weight, checkpoint_weight = self._load_replicated_weight(
            torch.device("cpu")
        )

        self.assertEqual(model_weight.data_ptr(), checkpoint_weight.data_ptr())
        torch.testing.assert_close(model_weight, checkpoint_weight)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
    def test_ordinary_cuda_loading_preserves_materialization_path(self):
        model_weight, checkpoint_weight = self._load_replicated_weight(
            torch.device("cuda:0")
        )

        self.assertNotEqual(model_weight.data_ptr(), checkpoint_weight.data_ptr())
        torch.testing.assert_close(model_weight, checkpoint_weight)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
    def test_direct_cuda_loading_adopts_checkpoint_storage(self):
        model_weight, checkpoint_weight = self._load_replicated_weight(
            torch.device("cuda:0"), allow_device_tensor_assignment=True
        )

        self.assertEqual(model_weight.data_ptr(), checkpoint_weight.data_ptr())
        torch.testing.assert_close(model_weight, checkpoint_weight)

    def test_zero_copy_assignment_rejects_incompatible_layout_or_tp_weights(self):
        with torch.device("meta"):
            model = _ReplicatedLinearModel()
        param = model.proj.weight
        tensor = torch.empty(4, 4)

        self.assertFalse(
            fsdp_load._can_assign_tensor_without_copy(
                param, tensor.as_strided((4, 4), (1, 4)), param
            )
        )

        tp_owner = ColumnParallelLinear.__new__(ColumnParallelLinear)
        nn.Module.__init__(tp_owner)
        tp_owner.quant_method = UnquantizedLinearMethod()
        tp_owner.tp_size = 1
        tp_param = nn.Parameter(tensor)
        tp_param.weight_loader = tp_owner.weight_loader
        tp_owner.tp_size = 2
        self.assertFalse(
            fsdp_load._can_assign_tensor_without_copy(tp_param, tensor, tp_param)
        )


class TestDevicePostprocessMove(unittest.TestCase):
    def test_postprocess_move_preserves_meta_buffers_for_post_load_weights(self):
        # The pre-postprocess device move must not copy buffers that are
        # still on meta awaiting post_load_weights() (cosmos3's RoPE inv_freq).
        load_plan = WeightLoadPlan(
            checkpoint_load_device=torch.device("cpu"),
            weight_postprocess_device=torch.device("cpu"),
        )
        checkpoint_weight = torch.arange(16, dtype=torch.float32).reshape(4, 4)

        with patch.object(fsdp_load.current_platform, "is_mps", return_value=False):
            model = fsdp_load.maybe_load_fsdp_model(
                model_cls=_MetaBufferModel,
                init_params={},
                weight_dir_list=[],
                device=torch.device("cpu"),
                hsdp_replicate_dim=1,
                hsdp_shard_dim=1,
                param_dtype=torch.float32,
                reduce_dtype=torch.float32,
                weight_load_plan=load_plan,
                weights_iterator=iter((("proj.weight", checkpoint_weight),)),
            )

        self.assertFalse(model.inv_freq.is_meta)
        torch.testing.assert_close(model.inv_freq, torch.ones(2))
        torch.testing.assert_close(model.proj.weight, checkpoint_weight)


class TestRankLocalSafetensorsRead(unittest.TestCase):
    def _source(
        self,
        file_path: str,
        param_name: str,
        shape: tuple[int, ...],
        merge_index: int | None = None,
        num_params_to_merge: int | None = None,
    ) -> rank_local_checkpoint.SafetensorsSource:
        return rank_local_checkpoint.SafetensorsSource(
            file_path=file_path,
            param_name=param_name,
            shape=shape,
            dtype="BF16",
            merge_index=merge_index,
            num_params_to_merge=num_params_to_merge,
        )

    def test_reads_rank_local_slice(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            file_path = str(Path(temp_dir) / "model.safetensors")
            weight = torch.arange(24, dtype=torch.bfloat16).reshape(6, 4)
            save_file({"weight": weight}, file_path)

            with safe_open(file_path, framework="pt", device="cpu") as handle:
                tensor = rank_local_checkpoint.read_rank_local_tensor(
                    [self._source(file_path, "weight", (6, 4))],
                    {file_path: handle},
                    local_shape=(2, 4),
                    global_offset=(2, 0),
                )

            torch.testing.assert_close(tensor, weight[2:4])

    def test_reads_rank_local_slice_across_merged_sources(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            file_path = str(Path(temp_dir) / "model.safetensors")
            first = torch.arange(12, dtype=torch.bfloat16).reshape(3, 4)
            second = torch.arange(12, 24, dtype=torch.bfloat16).reshape(3, 4)
            save_file({"first": first, "second": second}, file_path)
            sources = [
                self._source(file_path, "first", (3, 4), 0, 2),
                self._source(file_path, "second", (3, 4), 1, 2),
            ]

            with safe_open(file_path, framework="pt", device="cpu") as handle:
                tensor = rank_local_checkpoint.read_rank_local_tensor(
                    sources,
                    {file_path: handle},
                    local_shape=(4, 4),
                    global_offset=(1, 0),
                )

            torch.testing.assert_close(tensor, torch.cat((first, second))[1:5])

    def test_applies_layout_transform_before_rank_local_slice(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            file_path = str(Path(temp_dir) / "model.safetensors")
            grouped_qkv = torch.arange(12, dtype=torch.bfloat16).reshape(6, 2)
            save_file({"weight": grouped_qkv}, file_path)

            def reorder_qkv(weight: torch.Tensor) -> torch.Tensor:
                grouped = weight.view(2, 3, 2)
                return grouped.permute(1, 0, 2).reshape(6, 2)

            with safe_open(file_path, framework="pt", device="cpu") as handle:
                tensor = rank_local_checkpoint.read_fsdp_rank_local_tensor(
                    [self._source(file_path, "weight", (6, 2))],
                    {file_path: handle},
                    global_shape=(6, 2),
                    local_shape=(2, 2),
                    global_offset=(2, 0),
                    transform=reorder_qkv,
                )

            expected = reorder_qkv(grouped_qkv)[2:4]
            torch.testing.assert_close(tensor, expected)
            self.assertEqual(tensor.untyped_storage().nbytes(), expected.nbytes)

    def test_reads_zero_sized_rank_local_shard(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            file_path = str(Path(temp_dir) / "model.safetensors")
            weight = torch.ones((1, 4), dtype=torch.bfloat16)
            save_file({"weight": weight}, file_path)

            with safe_open(file_path, framework="pt", device="cpu") as handle:
                tensor = rank_local_checkpoint.read_rank_local_tensor(
                    [self._source(file_path, "weight", (1, 4))],
                    {file_path: handle},
                    local_shape=(0, 4),
                    global_offset=(1, 0),
                )

            self.assertEqual(tensor.shape, (0, 4))
            self.assertEqual(tensor.dtype, torch.bfloat16)
            self.assertEqual(tensor.numel(), 0)

    def test_reads_tp_local_merged_column_slice_per_source(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            file_path = str(Path(temp_dir) / "model.safetensors")
            first = torch.arange(16, dtype=torch.bfloat16).reshape(4, 4)
            second = torch.arange(16, 32, dtype=torch.bfloat16).reshape(4, 4)
            save_file({"first": first, "second": second}, file_path)
            sources = [
                self._source(file_path, "first", (4, 4), 0, 2),
                self._source(file_path, "second", (4, 4), 1, 2),
            ]

            with safe_open(file_path, framework="pt", device="cpu") as handle:
                tensor = rank_local_checkpoint.read_tp_local_tensor(
                    sources,
                    {file_path: handle},
                    shard_dim=0,
                    tp_rank=1,
                    tp_size=2,
                )

            expected = torch.cat((first[2:4], second[2:4]))
            torch.testing.assert_close(tensor, expected)
            self.assertEqual(
                rank_local_checkpoint.tp_local_shape(sources, 0, 2), (4, 4)
            )

    def test_reads_tp_local_merged_row_slice(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            file_path = str(Path(temp_dir) / "model.safetensors")
            first = torch.arange(16, dtype=torch.bfloat16).reshape(4, 4)
            second = torch.arange(16, 32, dtype=torch.bfloat16).reshape(4, 4)
            save_file({"first": first, "second": second}, file_path)
            sources = [
                self._source(file_path, "first", (4, 4), 0, 2),
                self._source(file_path, "second", (4, 4), 1, 2),
            ]

            with safe_open(file_path, framework="pt", device="cpu") as handle:
                tensor = rank_local_checkpoint.read_tp_local_tensor(
                    sources,
                    {file_path: handle},
                    shard_dim=1,
                    tp_rank=1,
                    tp_size=2,
                )

            expected = torch.cat((first[:, 2:4], second[:, 2:4]))
            torch.testing.assert_close(tensor, expected)
            self.assertEqual(
                rank_local_checkpoint.tp_local_shape(sources, 1, 2), (8, 2)
            )
