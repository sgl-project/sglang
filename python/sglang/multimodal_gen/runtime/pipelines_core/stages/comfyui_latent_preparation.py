# SPDX-License-Identifier: Apache-2.0
"""
ComfyUI latent preparation stage with device mismatch fix.
This stage extends LatentPreparationStage to handle device mismatch issues
that occur when tensors are pickled and unpickled via broadcast_pyobj in
multi-GPU scenarios.
"""
import dataclasses

import torch

from sglang.multimodal_gen.runtime.distributed import (
    get_local_torch_device,
    get_sp_group,
)
from sglang.multimodal_gen.runtime.distributed.parallel_state import get_sp_world_size
from sglang.multimodal_gen.runtime.pipelines_core.stages.latent_preparation import (
    LatentPreparationStage,
)
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger

logger = init_logger(__name__)


class ComfyUILatentPreparationStage(LatentPreparationStage):
    """
    ComfyUI-specific latent preparation stage with device mismatch fix.

    This stage extends LatentPreparationStage to automatically fix device
    mismatches for tensor fields on non-source ranks in multi-GPU scenarios.
    """

    @staticmethod
    def _fix_tensor_device(value, target_device):
        """Recursively fix tensor device, handling single tensors, lists, and tuples."""
        if isinstance(value, torch.Tensor):
            if value.device != target_device:
                return value.detach().clone().to(target_device)
            return value
        elif isinstance(value, list):
            return [
                ComfyUILatentPreparationStage._fix_tensor_device(v, target_device)
                for v in value
            ]
        elif isinstance(value, tuple):
            return tuple(
                ComfyUILatentPreparationStage._fix_tensor_device(v, target_device)
                for v in value
            )
        return value

    @staticmethod
    def _has_tensor(value):
        """Check if value contains any tensor."""
        if isinstance(value, torch.Tensor):
            return True
        elif isinstance(value, (list, tuple)):
            return any(ComfyUILatentPreparationStage._has_tensor(v) for v in value)
        return False

    def forward(self, batch, server_args):
        """
        Prepare latents with device mismatch fix for ComfyUI pipelines.

        This method first fixes device mismatches for all tensor fields,
        then calls the parent class's forward method, and ensures raw_latent_shape
        is set correctly (before packing, for proper unpadding later).
        """
        # Fix device mismatch for tensor fields on non-source ranks
        if get_sp_world_size() > 1:
            sp_group = get_sp_group()
            target_device = get_local_torch_device()

            if sp_group.rank != 0:
                logger.debug(
                    f"[ComfyUILatentPreparationStage] Fixing tensor device on rank={sp_group.rank} "
                    f"target_device={target_device}"
                )

                if dataclasses.is_dataclass(batch):
                    for field in dataclasses.fields(batch):
                        value = getattr(batch, field.name, None)
                        if value is not None and self._has_tensor(value):
                            fixed_value = self._fix_tensor_device(value, target_device)
                            setattr(batch, field.name, fixed_value)
                else:
                    for attr_name in dir(batch):
                        if not attr_name.startswith("_") and not callable(
                            getattr(batch, attr_name, None)
                        ):
                            try:
                                value = getattr(batch, attr_name, None)
                                if value is not None and self._has_tensor(value):
                                    fixed_value = self._fix_tensor_device(
                                        value, target_device
                                    )
                                    setattr(batch, attr_name, fixed_value)
                            except (AttributeError, TypeError):
                                continue

        original_latents_shape = None
        if batch.latents is not None:
            original_latents_shape = batch.latents.shape

        # Call parent class's forward method
        result = super().forward(batch, server_args)

        if original_latents_shape is not None:
            # Preserve the original shape before any potential packing/conversion
            # (e.g., 4D spatial -> 3D sequence) to ensure proper unpadding later.
            result.raw_latent_shape = original_latents_shape

        return result
