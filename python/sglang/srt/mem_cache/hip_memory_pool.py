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
        self.metadata = HiPAttentionOutputMetadata(
            indices=torch.zeros([], dtype=dtype, device=device),  # FIXME: correct shape
            ks=torch.zeros([], dtype=dtype, device=device),
            ks_count=torch.zeros([], dtype=dtype, device=device),
            ks_start_end=torch.zeros([], dtype=dtype, device=device),
            key_access_log=None,
            key_access_count=None,
            block_access_log=None,
            block_access_score=None,
            block_access_count=None,
        )

    def get_hip_metadata_cache(self):
        return self.metadata

    def set_hip_metadata_cache(self, metadata: HiPAttentionOutputMetadata):
        self.metadata = metadata
