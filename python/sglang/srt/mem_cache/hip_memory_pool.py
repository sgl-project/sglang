import torch
from hip import HiPAttentionOutputMetadata


class HiPMetadataCachePool:

    def __init__(
        self,
        size: int,
        dtype: torch.dtype,
        head_num: int,
        head_dim: int,
        layer_num: int,
        device: str,
    ):
        self.metadata = None

    def get_hip_metadata_cache(self):
        return self.metadata

    def set_hip_metadata_cache(self, metadata: HiPAttentionOutputMetadata):
        self.metadata = metadata
