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
        worker=None,
        worker_token=None,
    ):
        super().__init__(
            model, load_device, offload_device, size, weight_inplace_update
        )
        self.lora_cache = {}
        self.model_type = model_type
        # Owner of the worker process (SGLDiffusionGenerator). The weights
        # live there, so ComfyUI freeing this patcher has to reach it.
        self.worker = worker
        # Identifies which worker process this patcher belongs to; clones
        # share it, a patcher for a replaced worker does not match.
        self.worker_token = worker_token
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
            worker=self.worker,
            worker_token=self.worker_token,
        )
        n.patches = {}
        for k in self.patches:
            n.patches[k] = self.patches[k][:]
        n.patches_uuid = self.patches_uuid

        n.object_patches = self.object_patches.copy()
        n.model_options = copy.deepcopy(self.model_options)
        n.parent = self
        n.backup = self.backup
        n.object_patches_backup = self.object_patches_backup
        n.lora_cache = copy.copy(self.lora_cache)
        n.model_type = self.model_type
        n.wrappers = {}
        for wrapper_type, keyed in self.wrappers.items():
            n.wrappers[wrapper_type] = {
                key: wrappers.copy() for key, wrappers in keyed.items()
            }
        n.callbacks = {}
        for call_type, keyed in self.callbacks.items():
            n.callbacks[call_type] = {
                key: callbacks.copy() for key, callbacks in keyed.items()
            }
        return n

    def model_size(self):
        """Bytes the worker needs on ComfyUI's GPU; 0 when it runs on another GPU."""
        if self.worker is not None and not getattr(
            self.worker, "shares_comfy_gpu", True
        ):
            return 0
        if self.size:
            return self.size
        return self.model_size_dict.get(self.model_type, 0)

    def _is_stale(self):
        """True when the worker was rebuilt after this patcher was made."""
        if self.worker is None or self.worker_token is None:
            return False
        return self.worker_token is not getattr(
            self.worker, "generator", self.worker_token
        )

    def loaded_size(self):
        """VRAM ComfyUI can reclaim: the worker's weights while it is awake."""
        if (
            self.worker is None
            or self._is_stale()
            or getattr(self.worker, "worker_asleep", False)
            or not getattr(self.worker, "shares_comfy_gpu", True)
        ):
            return 0
        # measured after the first run; the checkpoint size until then
        return getattr(self.worker, "resident_bytes", 0) or self.model_size()

    def partially_unload(self, device_to, memory_to_free=0, force_patch_weights=False):
        """Nothing is partially offloadable; ``detach`` does the real unload."""
        return 0

    def detach(self, unpatch_all=True):
        """Every ComfyUI unload ends here; sleep the worker (woken by ensure_executor)."""
        if self.worker is not None and not self._is_stale():
            self.worker.sleep_worker(self.worker_token)
        return super().detach(unpatch_all)

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
