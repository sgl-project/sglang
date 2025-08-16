from typing import TYPE_CHECKING, Callable, List, Optional, Union

import torch

from sglang.srt import two_batch_overlap
from sglang.srt.layers.attention.base_attn_backend import AttentionBackend
from sglang.srt.speculative.eagle_utils import EagleDraftInput, EagleVerifyInput

if TYPE_CHECKING:
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch, ForwardMode


class TboAttnBackend(AttentionBackend):
    def __init__(self, primary: AttentionBackend, children: List[AttentionBackend]):
        super().__init__()
        self.primary = primary
        self.children = children

    @classmethod
    def init_new(cls, creator: Callable[[], AttentionBackend]):
        return cls(
            primary=creator(),
            children=[creator() for _ in range(2)],
        )

    def init_forward_metadata(self, forward_batch: "ForwardBatch"):
        self.primary.init_forward_metadata(forward_batch=forward_batch)
        if forward_batch.tbo_children is not None:
            for child, forward_batch_child in zip(
                self.children, forward_batch.tbo_children, strict=True
            ):
                if forward_batch_child.batch_size > 0:
                    child.init_forward_metadata(forward_batch=forward_batch_child)

    def init_cuda_graph_state(self, max_bs: int, max_num_tokens: int):
        self.primary.init_cuda_graph_state(max_bs=max_bs, max_num_tokens=max_num_tokens)
        for item in self.children:
            # TODO for children, maybe can provide *smaller* max_bs to optimize
            item.init_cuda_graph_state(max_bs=max_bs, max_num_tokens=max_num_tokens)

    def init_forward_metadata_capture_cuda_graph(
        self,
        bs: int,
        num_tokens: int,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        encoder_lens: Optional[torch.Tensor],
        forward_mode: "ForwardMode",
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

        self._init_forward_metadata_cuda_graph_children(
            fn_name="init_forward_metadata_capture_cuda_graph",
            bs=bs,
            req_pool_indices=req_pool_indices,
            seq_lens=seq_lens,
            encoder_lens=encoder_lens,
            forward_mode=forward_mode,
            spec_info=spec_info,
            capture_num_tokens=num_tokens,
        )

    def init_forward_metadata_replay_cuda_graph(
        self,
        bs: int,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        seq_lens_sum: int,
        encoder_lens: Optional[torch.Tensor],
        forward_mode: "ForwardMode",
        spec_info: Optional[Union[EagleDraftInput, EagleVerifyInput]],
        seq_lens_cpu: Optional[torch.Tensor],
    ):
        self.primary.init_forward_metadata_replay_cuda_graph(
            bs=bs,
            req_pool_indices=req_pool_indices,
            seq_lens=seq_lens,
            seq_lens_sum=seq_lens_sum,
            encoder_lens=encoder_lens,
            forward_mode=forward_mode,
            spec_info=spec_info,
            seq_lens_cpu=seq_lens_cpu,
        )

        self._init_forward_metadata_cuda_graph_children(
            fn_name="init_forward_metadata_replay_cuda_graph",
            bs=bs,
            req_pool_indices=req_pool_indices,
            seq_lens=seq_lens,
            encoder_lens=encoder_lens,
            forward_mode=forward_mode,
            spec_info=spec_info,
            replay_seq_lens_sum=seq_lens_sum,
            replay_seq_lens_cpu=seq_lens_cpu,
        )

    def _init_forward_metadata_cuda_graph_children(
        self,
        fn_name: str,
        # common args
        bs: int,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        encoder_lens: Optional[torch.Tensor],
        forward_mode: "ForwardMode",
        spec_info: Optional[Union[EagleDraftInput, EagleVerifyInput]],
        # capture args
        capture_num_tokens: int = None,
        # replay args
        replay_seq_lens_sum: int = None,
        replay_seq_lens_cpu: Optional[torch.Tensor] = None,
    ):
        token_num_per_seq = two_batch_overlap.get_token_num_per_seq(
            forward_mode=forward_mode, spec_info=spec_info
        )
        if fn_name == "init_forward_metadata_capture_cuda_graph":
            assert (
                capture_num_tokens == bs * token_num_per_seq
            ), "For target-verify or decode mode, num_tokens should be equal to token_num_per_seq * bs"
        num_tokens = bs * token_num_per_seq

        tbo_split_seq_index, tbo_split_token_index = (
            two_batch_overlap.compute_split_indices_for_cuda_graph_replay(
                forward_mode=forward_mode,
                cuda_graph_num_tokens=num_tokens,
                spec_info=spec_info,
            )
        )

        num_tokens_child_left = tbo_split_token_index
        num_tokens_child_right = num_tokens - tbo_split_token_index
        bs_child_left = tbo_split_seq_index
        bs_child_right = bs - bs_child_left

        assert (
            num_tokens_child_left > 0 and num_tokens_child_right > 0
        ), f"{num_tokens_child_left=} {num_tokens_child_right=} {forward_mode=} {num_tokens=}"

        common_pre_split_args = dict(
            fn_name=fn_name,
            bs=bs,
            req_pool_indices=req_pool_indices,
            seq_lens=seq_lens,
            encoder_lens=encoder_lens,
            forward_mode=forward_mode,
            spec_info=spec_info,
            capture_num_tokens=capture_num_tokens,
            replay_seq_lens_sum=replay_seq_lens_sum,
            replay_seq_lens_cpu=replay_seq_lens_cpu,
        )

        args_left = _init_forward_metadata_cuda_graph_split(
            output_bs=bs_child_left,
            seq_slice=slice(None, tbo_split_seq_index),
            **common_pre_split_args,
        )
        args_right = _init_forward_metadata_cuda_graph_split(
            output_bs=bs_child_right,
            seq_slice=slice(tbo_split_seq_index, None),
            **common_pre_split_args,
        )

        child_left, child_right = self.children
        getattr(child_left, fn_name)(**args_left)
        getattr(child_right, fn_name)(**args_right)

    def get_cuda_graph_seq_len_fill_value(self):
        ans = self.primary.get_cuda_graph_seq_len_fill_value()
        for child in self.children:
            assert ans == child.get_cuda_graph_seq_len_fill_value()
        return ans

    def forward_extend(self, *args, **kwargs):
        return self.primary.forward_extend(*args, **kwargs)

    def forward_decode(self, *args, **kwargs):
        return self.primary.forward_decode(*args, **kwargs)


def _init_forward_metadata_cuda_graph_split(
    fn_name: str,
    seq_slice: slice,
    output_bs: int,
    # common args
    bs: int,
    req_pool_indices: torch.Tensor,
    seq_lens: torch.Tensor,
    encoder_lens: Optional[torch.Tensor],
    forward_mode: "ForwardMode",
    spec_info: Optional[EagleVerifyInput],
    # capture args
    capture_num_tokens: int = None,
    # replay args
    replay_seq_lens_sum: int = None,
    replay_seq_lens_cpu: Optional[torch.Tensor] = None,
):
    token_num_per_seq = two_batch_overlap.get_token_num_per_seq(
        forward_mode=forward_mode, spec_info=spec_info
    )
    assert encoder_lens is None, "encoder_lens is not supported yet"
    if spec_info is not None:
        output_spec_info = two_batch_overlap.split_spec_info(
            spec_info=spec_info,
            start_seq_index=seq_slice.start if seq_slice.start is not None else 0,
            end_seq_index=seq_slice.stop if seq_slice.stop is not None else bs,
            start_token_index=(
                seq_slice.start * token_num_per_seq
                if seq_slice.start is not None
                else 0
            ),
            end_token_index=(
                seq_slice.stop * token_num_per_seq
                if seq_slice.stop is not None
                else bs * token_num_per_seq
            ),
        )

    else:
        output_spec_info = None
    ans = dict(
        bs=output_bs,
        req_pool_indices=req_pool_indices[seq_slice],
        seq_lens=seq_lens[seq_slice],
        # directly forward
        forward_mode=forward_mode,
        # ignore
        encoder_lens=None,
        spec_info=output_spec_info,
    )

    if fn_name == "init_forward_metadata_capture_cuda_graph":
        assert (
            capture_num_tokens == bs * token_num_per_seq
        ), "Only support num_tokens==bs * token_num_per_seq for target-verify or decode mode"
        ans.update(
            dict(
                num_tokens=output_bs * token_num_per_seq,
            )
        )
    elif fn_name == "init_forward_metadata_replay_cuda_graph":
        output_seq_lens_cpu = replay_seq_lens_cpu[seq_slice]
        ans.update(
            dict(
                seq_lens_sum=output_seq_lens_cpu.sum().item(),
                seq_lens_cpu=output_seq_lens_cpu,
            )
        )
    else:
        raise NotImplementedError

    return ans
