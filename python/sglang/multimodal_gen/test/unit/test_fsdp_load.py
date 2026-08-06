# SPDX-License-Identifier: Apache-2.0

import unittest
from unittest.mock import patch

import torch
from torch import nn

from sglang.multimodal_gen.runtime.loader import fsdp_load


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
