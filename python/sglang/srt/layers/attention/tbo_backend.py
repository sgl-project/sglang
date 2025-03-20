from typing import Optional, Union, List

import torch
from sglang.srt.layers.attention.base_attn_backend import AttentionBackend
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, ForwardMode
from sglang.srt.speculative.eagle_utils import EagleDraftInput, EagleVerifyInput


class TboAttnBackend(AttentionBackend):
    def __init__(self, primary: AttentionBackend, children: List[AttentionBackend]):
        super().__init__()
        self.primary = primary
        self.children = children

    def init_forward_metadata(self, forward_batch: ForwardBatch):
        TODO

    def init_cuda_graph_state(self, max_bs: int):
        TODO

    def init_forward_metadata_capture_cuda_graph(
            self,
            bs: int,
            num_tokens: int,
            req_pool_indices: torch.Tensor,
            seq_lens: torch.Tensor,
            encoder_lens: Optional[torch.Tensor],
            forward_mode: ForwardMode,
            spec_info: Optional[Union[EagleDraftInput, EagleVerifyInput]],
    ):
        TODO

    def init_forward_metadata_replay_cuda_graph(
            self,
            bs: int,
            num_kv_heads: int,
            req_pool_indices: torch.Tensor,
            seq_lens: torch.Tensor,
            seq_lens_sum: int,
            encoder_lens: Optional[torch.Tensor],
            forward_mode: ForwardMode,
            spec_info: Optional[Union[EagleDraftInput, EagleVerifyInput]],
            seq_lens_cpu: Optional[torch.Tensor],
    ):
        TODO

    def get_cuda_graph_seq_len_fill_value(self):
        TODO

    def forward_extend(self, *args, **kwargs):
        TODO

    def forward_decode(self, *args, **kwargs):
        TODO
