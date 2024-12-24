import torch
from hip import HiPAttentionOutputMetadata


class HiPMetadataCachePool:

    def __init__(
        self,
        size: int,  # max_total_num_tokens = bsz * seq_len
        dtype: torch.dtype,
        head_num: int,
        head_dim: int,
        layer_num: int,
        device: str,
    ):
        self.indices_pool = []
        self.ks_pool = []
        self.ks_count_pool = []
        self.ks_start_end_pool = []

        for layer_idx in range(layer_num):
            self.indices_pool.append(
                torch.zeros(size, dtype=dtype, device=device)
            )

    def get_hip_metadata_cache(self, layer_id: int):
        return None
        # return HiPAttentionOutputMetadata(
        #     indices=indices,
        #     ks=ks,
        #     ks_count=ks_count,
        #     ks_start_end=ks_start_end,
        #     key_access_log=None,
        #     key_access_count=None,
        #     block_access_log=None,
        #     block_access_score=None,
        #     block_access_count=None,
        # )

    def set_hip_metadata_cache(
        self,
        layer_id: int,
        cache_loc: torch.Tensor,
        metadata: HiPAttentionOutputMetadata
    ):
        pass
