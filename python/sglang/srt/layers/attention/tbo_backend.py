from typing import Optional, Union, List

import torch
from sglang.srt import two_batch_overlap
from sglang.srt.layers.attention.base_attn_backend import AttentionBackend
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, ForwardMode
from sglang.srt.speculative.eagle_utils import EagleDraftInput, EagleVerifyInput


class TboAttnBackend(AttentionBackend):
    def __init__(self, primary: AttentionBackend, children: List[AttentionBackend]):
        super().__init__()
        self.primary = primary
        self.children = children

    def init_forward_metadata(self, forward_batch: ForwardBatch):
        for item in self._primary_and_children:
            item.init_forward_metadata(forward_batch=forward_batch)

    def init_cuda_graph_state(self, max_bs: int):
        for item in self._primary_and_children:
            # TODO for children, maybe can provide *smaller* max_bs to optimize
            item.init_cuda_graph_state(max_bs=max_bs)

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
        self.primary.init_forward_metadata_capture_cuda_graph(
            bs=bs,
            num_tokens=num_tokens,
            req_pool_indices=req_pool_indices,
            seq_lens=seq_lens,
            encoder_lens=encoder_lens,
            forward_mode=forward_mode,
            spec_info=spec_info,
        )

        args_left, args_right = self._compute_cuda_graph_children_args(
            bs=bs,
            num_tokens=num_tokens,
            req_pool_indices=req_pool_indices,
            seq_lens=seq_lens,
            encoder_lens=encoder_lens,
            forward_mode=forward_mode,
            spec_info=spec_info,
        )
        child_left, child_right = self.children
        child_left.init_forward_metadata_capture_cuda_graph(**args_left)
        child_right.init_forward_metadata_capture_cuda_graph(**args_right)

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
        self.primary.init_forward_metadata_replay_cuda_graph(
            bs=bs,
            num_kv_heads=num_kv_heads,
            req_pool_indices=req_pool_indices,
            seq_lens=seq_lens,
            seq_lens_sum=seq_lens_sum,
            encoder_lens=encoder_lens,
            forward_mode=forward_mode,
            spec_info=spec_info,
            seq_lens_cpu=seq_lens_cpu,
        )

        args_left, args_right = self._compute_cuda_graph_children_args(
            bs=bs,
            num_tokens=num_tokens,
            req_pool_indices=req_pool_indices,
            seq_lens=seq_lens,
            encoder_lens=encoder_lens,
            forward_mode=forward_mode,
            spec_info=spec_info,
            # TODO
            seq_lens_sum=seq_lens_sum,
            seq_lens_cpu=seq_lens_cpu,
        )
        common_args = dict(
            num_kv_heads=num_kv_heads,
        )

        child_left, child_right = self.children
        child_left.init_forward_metadata_capture_cuda_graph(**args_left, **common_args)
        child_right.init_forward_metadata_capture_cuda_graph(**args_right, **common_args)

    @staticmethod
    def _compute_cuda_graph_children_args(
        bs: int,
        num_tokens: int,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        encoder_lens: Optional[torch.Tensor],
        forward_mode: ForwardMode,
        spec_info: Optional[Union[EagleDraftInput, EagleVerifyInput]],
    ):
        tbo_split_seq_index = two_batch_overlap.compute_split_seq_index(
            forward_mode=forward_mode,
            num_tokens=num_tokens,
            extend_lens=None,
        )
        tbo_split_token_index = two_batch_overlap.compute_split_token_index(
            split_seq_index=tbo_split_seq_index,
            forward_mode=forward_mode,
            extend_lens=None,
        )

        num_tokens_child_left = tbo_split_token_index
        num_tokens_child_right = num_tokens - tbo_split_token_index

        # not yet support `num_tokens_per_bs>1`
        assert num_tokens == bs
        bs_child_left = num_tokens_child_left
        bs_child_right = num_tokens_child_right

        assert encoder_lens is None, 'encoder_lens is not supported yet'
        assert spec_info is None, 'spec_info is not supported yet'

        args_left = dict(
            bs=bs_child_left,
            num_tokens=num_tokens_child_left,
            req_pool_indices=req_pool_indices[:tbo_split_seq_index],
            seq_lens=seq_lens[:tbo_split_seq_index],
            encoder_lens=None,
            forward_mode=forward_mode,
            spec_info=None,
        )
        args_right = dict(
            bs=bs_child_right,
            num_tokens=num_tokens_child_right,
            req_pool_indices=req_pool_indices[tbo_split_seq_index:],
            seq_lens=seq_lens[tbo_split_seq_index:],
            encoder_lens=None,
            forward_mode=forward_mode,
            spec_info=None,
        )
        return args_left, args_right

    def get_cuda_graph_seq_len_fill_value(self):
        ans = self.primary.get_cuda_graph_seq_len_fill_value()
        for child in self.children:
            assert ans == child.get_cuda_graph_seq_len_fill_value()
        return ans

    def forward_extend(self, *args, **kwargs):
        return self.forward_extend(*args, **kwargs)

    def forward_decode(self, *args, **kwargs):
        return self.forward_decode(*args, **kwargs)

    @property
    def _primary_and_children(self):
        yield self.primary
        yield from self.children
