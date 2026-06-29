from types import SimpleNamespace
from typing import TYPE_CHECKING, Callable, List

from sglang.srt.batch_overlap import two_batch_overlap
from sglang.srt.layers.attention.base_attn_backend import AttentionBackend
from sglang.srt.layers.moe import is_decode_tbo_enabled

if TYPE_CHECKING:
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch


class TboAttnBackend(AttentionBackend):
    def __init__(self, primary: AttentionBackend, children: List[AttentionBackend]):
        super().__init__()
        self.primary = primary
        self.children = children
        # Dispatcher aliases the primary's pool refs so get_attn_backend()
        # reads through TboAttnBackend resolve to the underlying pool.
        self.token_to_kv_pool = primary.token_to_kv_pool
        self.req_to_token_pool = primary.req_to_token_pool

    @classmethod
    def init_new(cls, creator: Callable[[], AttentionBackend]):
        return cls(
            primary=creator(),
            children=[creator() for _ in range(2)],
        )

    def init_forward_metadata_out_graph(
        self,
        forward_batch: "ForwardBatch",
        in_capture: bool = False,
    ):
        self.primary.init_forward_metadata_out_graph(
            forward_batch=forward_batch, in_capture=in_capture
        )
        tbo_children = getattr(forward_batch, "tbo_children", None)
        if tbo_children is not None:
            for child, forward_batch_child in zip(
                self.children, tbo_children, strict=True
            ):
                if forward_batch_child.batch_size > 0:
                    child.init_forward_metadata_out_graph(
                        forward_batch=forward_batch_child, in_capture=in_capture
                    )
            return
        if in_capture:
            return
        # Replay path: build_replay_fb_view returns a SimpleNamespace and
        # tbo_plugin.replay_prepare does not call prepare_raw, so split the
        # padded buffers here using the same indices the eager path would.
        self._dispatch_children_from_replay_view(forward_batch)

    def _dispatch_children_from_replay_view(self, fb_view) -> None:
        bs = fb_view.batch_size
        forward_mode = fb_view.forward_mode
        spec_info = fb_view.spec_info
        if not is_decode_tbo_enabled() and (
            forward_mode.is_decode() or forward_mode.is_target_verify()
        ):
            # Decode-only TBO disabled: replay_prepare did not split this batch,
            # so there are no children to dispatch.
            return
        token_num_per_seq = two_batch_overlap.get_token_num_per_seq(
            forward_mode=forward_mode, spec_info=spec_info
        )
        num_tokens = bs * token_num_per_seq
        (
            tbo_split_seq_index,
            tbo_split_token_index,
        ) = two_batch_overlap.compute_split_indices_for_cuda_graph_replay(
            forward_mode=forward_mode,
            cuda_graph_num_tokens=num_tokens,
            spec_info=spec_info,
        )
        bs_left = tbo_split_seq_index
        bs_right = bs - bs_left
        for child, child_bs, seq_slice, tok_slice in (
            (
                self.children[0],
                bs_left,
                slice(None, tbo_split_seq_index),
                slice(None, tbo_split_token_index),
            ),
            (
                self.children[1],
                bs_right,
                slice(tbo_split_seq_index, None),
                slice(tbo_split_token_index, None),
            ),
        ):
            if child_bs == 0:
                continue
            child_fb_view = _build_tbo_child_replay_fb_view(
                fb_view,
                child_bs=child_bs,
                seq_slice=seq_slice,
                tok_slice=tok_slice,
                token_num_per_seq=token_num_per_seq,
            )
            child.init_forward_metadata_out_graph(
                forward_batch=child_fb_view, in_capture=False
            )

    def init_forward_metadata_in_graph(self, forward_batch: "ForwardBatch"):
        self.primary.init_forward_metadata_in_graph(forward_batch=forward_batch)
        tbo_children = getattr(forward_batch, "tbo_children", None)
        if tbo_children is not None:
            for child, forward_batch_child in zip(
                self.children, tbo_children, strict=True
            ):
                if forward_batch_child.batch_size > 0:
                    child.init_forward_metadata_in_graph(
                        forward_batch=forward_batch_child
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

    def on_after_cuda_graph_warmup(self):
        self.primary.on_after_cuda_graph_warmup()
        for child in self.children:
            child.on_after_cuda_graph_warmup()

    def get_cuda_graph_seq_len_fill_value(self):
        ans = self.primary.get_cuda_graph_seq_len_fill_value()
        for child in self.children:
            assert ans == child.get_cuda_graph_seq_len_fill_value()
        return ans

    def forward(self, *args, **kwargs):
        return self.primary.forward(*args, **kwargs)

    def forward_extend(self, *args, **kwargs):
        return self.primary.forward_extend(*args, **kwargs)

    def forward_decode(self, *args, **kwargs):
        return self.primary.forward_decode(*args, **kwargs)

    def get_indexer_metadata(self, layer_id: int, forward_batch: "ForwardBatch"):
        return self.primary.get_indexer_metadata(layer_id, forward_batch)


def _build_tbo_child_replay_fb_view(
    fb_view,
    *,
    child_bs: int,
    seq_slice: slice,
    tok_slice: slice,
    token_num_per_seq: int,
) -> SimpleNamespace:
    """Slice a parent replay fb_view into a per-child view.

    Mirrors the legacy ``_init_forward_metadata_cuda_graph_split`` (deleted
    along with the cuda_graph variants) for the new
    ``init_forward_metadata_out_graph(fb_view)`` contract: padded
    capture-time buffers are sliced per child, spec_info is split, and
    seq_lens_sum is recomputed from the sliced ``seq_lens_cpu``.
    """
    assert (
        getattr(fb_view, "encoder_lens", None) is None
    ), "TBO replay split does not support encoder_lens yet"
    spec_info = getattr(fb_view, "spec_info", None)
    if spec_info is not None:
        start_seq = seq_slice.start or 0
        end_seq = seq_slice.stop if seq_slice.stop is not None else start_seq + child_bs
        child_spec_info = two_batch_overlap.split_spec_info(
            spec_info=spec_info,
            start_seq_index=start_seq,
            end_seq_index=end_seq,
            start_token_index=start_seq * token_num_per_seq,
            end_token_index=end_seq * token_num_per_seq,
        )
    else:
        child_spec_info = None
    child_seq_lens_cpu = fb_view.seq_lens_cpu[seq_slice]
    parent_input_ids = getattr(fb_view, "input_ids", None)
    parent_out_cache_loc = getattr(fb_view, "out_cache_loc", None)
    return SimpleNamespace(
        batch_size=child_bs,
        forward_mode=fb_view.forward_mode,
        actual_forward_mode=getattr(
            fb_view, "actual_forward_mode", fb_view.forward_mode
        ),
        input_ids=(
            parent_input_ids[tok_slice] if parent_input_ids is not None else None
        ),
        req_pool_indices=fb_view.req_pool_indices[seq_slice],
        seq_lens=fb_view.seq_lens[seq_slice],
        seq_lens_sum=int(child_seq_lens_cpu.sum()),
        seq_lens_cpu=child_seq_lens_cpu,
        encoder_lens=None,
        out_cache_loc=(
            parent_out_cache_loc[tok_slice]
            if parent_out_cache_loc is not None
            else None
        ),
        spec_info=child_spec_info,
    )
