# SPDX-License-Identifier: Apache-2.0

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from safetensors.torch import safe_open, save_file
from torch import nn

from sglang.multimodal_gen.runtime.loader import fsdp_load, rank_local_checkpoint


class _UniformDtypeModel(nn.Module):
    param_names_mapping = {}

    def __init__(self) -> None:
        super().__init__()

    def post_load_weights(self) -> None:
        pass


class _MixedDtypeModel(_UniformDtypeModel):
    _fsdp_mixed_dtype_params = True


class TestFSDPMixedPrecisionPolicy(unittest.TestCase):
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
