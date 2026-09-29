"""Metadata-only eligibility checks for B200/B300 BF16 query packing."""

import torch


class PackedPrefillPolicy:
    def __init__(self, device, *, mixed_min_rows=4096, swa_min_rows=512):
        if mixed_min_rows < 1 or swa_min_rows < 1:
            raise ValueError("Packed prefill thresholds must be positive")
        self.device = torch.device(device)
        self.supported_gpu = False
        if self.device.type == "cuda":
            if self.device.index is None:
                self.device = torch.device("cuda", torch.cuda.current_device())
            properties = torch.cuda.get_device_properties(self.device)
            self.supported_gpu = (
                properties.major == 10
                and properties.minor in (0, 3)
                and properties.multi_processor_count == 148
            )
        self.mixed_min_rows = mixed_min_rows
        self.swa_min_rows = swa_min_rows

    def can_use(self, q, indices, *, real_heads):
        """Choose an algorithm for tensors built by the sparse-prefill backend.

        The native entry point validates devices, layouts, alignment and index
        bounds on tensor dimensions. Keep that validation out of Python's
        per-layer dispatch; the backend owns workspace and index construction.
        """
        if not self.supported_gpu or real_heads != 16:
            return False
        if q.ndim != 3 or indices.ndim != 2:
            return False
        width = indices.shape[1]
        threshold = self.swa_min_rows if width == 128 else self.mixed_min_rows
        return (
            threshold <= q.shape[0] <= 2**31 - 4
            and q.shape[1:] in ((16, 512), (64, 512))
            and q.dtype == torch.bfloat16
            and width in (128, 640)
        )
