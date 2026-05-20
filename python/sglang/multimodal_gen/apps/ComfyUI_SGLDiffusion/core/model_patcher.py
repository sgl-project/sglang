"""
Model patcher for SGLang Diffusion ComfyUI integration.
"""

import copy

from comfy.model_patcher import ModelPatcher


class SGLDModelPatcher(ModelPatcher):
    """Model patcher for SGLang Diffusion models in ComfyUI."""

    def __init__(
        self,
        model,
        load_device,
        offload_device,
        size=0,
        weight_inplace_update=False,
        model_type=None,
    ):
        super().__init__(
            model, load_device, offload_device, size, weight_inplace_update
        )
        self.lora_cache = {}
        self.model_type = model_type
        self.model_size_dict = {
            "flux": 27 * 1024 * 1024 * 1024,
            "lumina2": 8 * 1024 * 1024 * 1024,
        }

    def clone(self):
        """Clone the model patcher."""
        n = SGLDModelPatcher(
            self.model,
            self.load_device,
            self.offload_device,
            self.size,
            weight_inplace_update=self.weight_inplace_update,
        )
        n.patches = {}
        for k in self.patches:
            n.patches[k] = self.patches[k][:]
        n.patches_uuid = self.patches_uuid

        n.object_patches = self.object_patches.copy()
        n.model_options = copy.deepcopy(self.model_options)
        n.backup = self.backup
        n.object_patches_backup = self.object_patches_backup
        n.lora_cache = copy.copy(self.lora_cache)
        return n

    def model_size(self):
        """Get the model size in bytes."""
        if self.model_type in self.model_size_dict:
            return self.model_size_dict[self.model_type]
        else:
            return 0

    def load(
        self,
        device_to=None,
        lowvram_model_memory=0,
        force_patch_weights=False,
        full_load=False,
    ):
        """Load model (no-op for SGLang Diffusion)."""
        pass

    def patch_model(
        self,
        device_to=None,
        lowvram_model_memory=0,
        load_weights=True,
        force_patch_weights=False,
    ):
        """Patch model (no-op for SGLang Diffusion)."""
        pass

    def unpatch_model(self, device_to=None, unpatch_weights=True):
        """Unpatch model (no-op for SGLang Diffusion)."""
        pass
