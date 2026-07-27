from types import SimpleNamespace
from typing import TYPE_CHECKING, Callable, List

import torch

from sglang.srt.batch_overlap import two_batch_overlap
from sglang.srt.layers.attention.base_attn_backend import AttentionBackend

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
        # These are ABC class attributes, so __getattr__ cannot delegate them.
        # Mirror the primary explicitly: otherwise the wrapper silently rejects
        # captured ragged verify and forces an unnecessary decode seq-lens D2H.
        self.supports_ragged_verify_graph = primary.supports_ragged_verify_graph
        self.needs_cpu_seq_lens = primary.needs_cpu_seq_lens

    @classmethod
    def init_new(cls, creator: Callable[[], AttentionBackend]):
        return cls(
            primary=creator(),
            children=[creator() for _ in range(2)],
        )

    def _children_use_cuda_graph(self, forward_batch=None) -> bool:
        """Whether the TBO child backends participate in CUDA-graph capture/replay.

        Some models only run TBO in eager prefill and keep their graph-captured
        modes (decode / target-verify) NON-TBO on the primary backend. For those,
        the children must NOT be driven through the cuda-graph paths: doing so
        rebuilds their per-step metadata on every replay even though the captured
        graph never uses them. For DeepSeek-V4 that metadata build (compressor /
        indexer) leaks ROCm HSA resources across the 2 children -> eventual
        HSA_STATUS_ERROR_OUT_OF_RESOURCES. Eager prefill TBO (init_forward_metadata)
        is unaffected; only the *_graph paths are gated.
        """
        if forward_batch is not None:
            supports_mode = getattr(self.primary, "tbo_supports_cuda_graph_for", None)
            if supports_mode is not None:
                return supports_mode(forward_batch.forward_mode)
        return getattr(self.primary, "tbo_supports_cuda_graph", True)

    def init_forward_metadata_out_graph(
        self,
        forward_batch: "ForwardBatch",
        in_capture: bool = False,
    ):
        children_supported = self._children_use_cuda_graph(forward_batch)
        tbo_children = getattr(forward_batch, "tbo_children", None)
        use_children = children_supported and (
            tbo_children is not None
            or (
                not in_capture
                and forward_batch.forward_mode.is_extend_without_speculative()
            )
        )
        if not use_children:
            self.primary.init_forward_metadata_out_graph(
                forward_batch=forward_batch, in_capture=in_capture
            )
            return
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
        if fb_view.forward_mode.is_extend_without_speculative():
            self._dispatch_prefill_children_from_replay_view(fb_view)
            return

        bs = fb_view.batch_size
        forward_mode = fb_view.forward_mode
        spec_info = fb_view.spec_info
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

    def _dispatch_prefill_children_from_replay_view(self, fb_view) -> None:
        """Refresh fixed-shape child metadata for a full prefill graph replay."""
        num_tokens = fb_view.positions.numel()
        split_token_index = num_tokens // 2
        child_layouts = _split_prefill_replay_layout(
            extend_seq_lens=fb_view.extend_seq_lens_cpu,
            extend_prefix_lens=fb_view.extend_prefix_lens_cpu,
            split_token_index=split_token_index,
        )
        for child_backend, layout, tok_slice in zip(
            self.children,
            child_layouts,
            (slice(None, split_token_index), slice(split_token_index, None)),
            strict=True,
        ):
            child_fb_view = _build_tbo_prefill_child_replay_fb_view(
                fb_view,
                layout=layout,
                tok_slice=tok_slice,
            )
            child_backend.init_forward_metadata_out_graph(
                forward_batch=child_fb_view,
                in_capture=False,
            )

    def init_forward_metadata_in_graph(self, forward_batch: "ForwardBatch"):
        tbo_children = getattr(forward_batch, "tbo_children", None)
        use_children = (
            self._children_use_cuda_graph(forward_batch) and tbo_children is not None
        )
        if not use_children:
            self.primary.init_forward_metadata_in_graph(forward_batch=forward_batch)
            return
        for child, forward_batch_child in zip(self.children, tbo_children, strict=True):
            if forward_batch_child.batch_size > 0:
                child.init_forward_metadata_in_graph(forward_batch=forward_batch_child)

    def init_forward_metadata(self, forward_batch: "ForwardBatch"):
        if forward_batch.tbo_children is None:
            self.primary.init_forward_metadata(forward_batch=forward_batch)
            return
        for child, forward_batch_child in zip(
            self.children, forward_batch.tbo_children, strict=True
        ):
            if forward_batch_child.batch_size > 0:
                child.init_forward_metadata(forward_batch=forward_batch_child)

    def init_cuda_graph_state(self, max_bs: int, max_num_tokens: int):
        self.primary.init_cuda_graph_state(max_bs=max_bs, max_num_tokens=max_num_tokens)
        if not getattr(
            self.primary,
            "tbo_requires_decode_cuda_graph_state",
            self._children_use_cuda_graph(),
        ):
            return
        for item in self.children:
            # TODO for children, maybe can provide *smaller* max_bs to optimize
            item.init_cuda_graph_state(max_bs=max_bs, max_num_tokens=max_num_tokens)

    def on_after_cuda_graph_warmup(self):
        self.primary.on_after_cuda_graph_warmup()
        if not self._children_use_cuda_graph():
            return
        for child in self.children:
            child.on_after_cuda_graph_warmup()

    def get_cuda_graph_seq_len_fill_value(self):
        ans = self.primary.get_cuda_graph_seq_len_fill_value()
        if not self._children_use_cuda_graph():
            return ans
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

    def __getattr__(self, name):
        # Delegate backend-specific attributes/methods not explicitly wrapped
        # above (e.g. DSV4's get_unified_swa_loc / get_swa_out_cache_loc, which
        # the model calls directly via get_attn_backend()) to the primary
        # full-batch backend. Inside TBO the per-child backend is resolved
        # directly from the forward context, so this path only serves the
        # non-overlapped forward (warmup / decode / TBO-ineligible batches).
        # NOTE: __getattr__ runs only when normal lookup fails; guard `primary`
        # to avoid infinite recursion before __init__ sets it.
        if name == "primary":
            raise AttributeError(name)
        return getattr(self.primary, name)


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


def _split_prefill_replay_layout(
    *,
    extend_seq_lens,
    extend_prefix_lens,
    split_token_index: int,
):
    """Split live request metadata at a fixed flat-token graph boundary.

    Both returned layouts keep the parent's fixed request-slot count.  A
    request that has no query tokens in a child becomes a true zero sentinel;
    a request straddling the boundary keeps the original prefix in child A and
    advances child B's prefix by A's query-token contribution.
    """
    extend_seq_lens = [int(value) for value in extend_seq_lens]
    extend_prefix_lens = [int(value) for value in extend_prefix_lens]
    if len(extend_seq_lens) != len(extend_prefix_lens):
        raise RuntimeError(
            "TBO prefill replay metadata has mismatched request axes: "
            f"extend={len(extend_seq_lens)}, prefix={len(extend_prefix_lens)}"
        )
    if split_token_index < 0:
        raise RuntimeError(
            f"TBO prefill replay split must be non-negative: {split_token_index}"
        )

    child_extend = [[], []]
    child_prefix = [[], []]
    child_seq = [[], []]
    remaining_left = split_token_index
    for extend_len, prefix_len in zip(extend_seq_lens, extend_prefix_lens, strict=True):
        if extend_len < 0 or prefix_len < 0:
            raise RuntimeError(
                "TBO prefill replay lengths must be non-negative: "
                f"extend={extend_len}, prefix={prefix_len}"
            )
        left_len = min(extend_len, max(remaining_left, 0))
        right_len = extend_len - left_len
        remaining_left -= left_len

        child_extend[0].append(left_len)
        child_prefix[0].append(prefix_len if left_len else 0)
        child_seq[0].append(prefix_len + left_len if left_len else 0)

        child_extend[1].append(right_len)
        child_prefix[1].append(prefix_len + left_len if right_len else 0)
        child_seq[1].append(prefix_len + extend_len if right_len else 0)

    if remaining_left < 0:
        raise RuntimeError(
            "TBO prefill replay split consumed more tokens than its boundary"
        )

    layouts = []
    for extend_lens, prefix_lens, seq_lens in zip(
        child_extend, child_prefix, child_seq, strict=True
    ):
        start_locs = []
        cursor = 0
        for extend_len in extend_lens:
            start_locs.append(cursor)
            cursor += extend_len
        layouts.append(
            SimpleNamespace(
                extend_seq_lens=extend_lens,
                extend_prefix_lens=prefix_lens,
                seq_lens=seq_lens,
                extend_start_loc=start_locs,
                extend_num_tokens=cursor,
            )
        )
    return layouts


def _build_tbo_prefill_child_replay_fb_view(
    fb_view,
    *,
    layout,
    tok_slice: slice,
) -> SimpleNamespace:
    device = fb_view.seq_lens.device
    seq_dtype = fb_view.seq_lens.dtype
    req_pool_indices = fb_view.req_pool_indices.clone()
    active = torch.tensor(
        [value > 0 for value in layout.extend_seq_lens],
        dtype=torch.bool,
        device=device,
    )
    req_pool_indices.masked_fill_(~active, 0)
    seq_lens = torch.tensor(layout.seq_lens, dtype=seq_dtype, device=device)
    extend_seq_lens = torch.tensor(
        layout.extend_seq_lens, dtype=fb_view.extend_seq_lens.dtype, device=device
    )
    extend_prefix_lens = torch.tensor(
        layout.extend_prefix_lens,
        dtype=fb_view.extend_prefix_lens.dtype,
        device=device,
    )
    extend_start_loc = torch.tensor(
        layout.extend_start_loc,
        dtype=fb_view.extend_start_loc.dtype,
        device=device,
    )
    seq_lens_cpu = torch.tensor(layout.seq_lens, dtype=torch.int64, device="cpu")
    parent_input_ids = getattr(fb_view, "input_ids", None)
    parent_out_cache_loc = getattr(fb_view, "out_cache_loc", None)
    parent_positions = getattr(fb_view, "positions", None)
    return SimpleNamespace(
        batch_size=fb_view.batch_size,
        forward_mode=fb_view.forward_mode,
        actual_forward_mode=getattr(
            fb_view, "actual_forward_mode", fb_view.forward_mode
        ),
        input_ids=(
            parent_input_ids[tok_slice] if parent_input_ids is not None else None
        ),
        positions=(
            parent_positions[tok_slice] if parent_positions is not None else None
        ),
        req_pool_indices=req_pool_indices,
        seq_lens=seq_lens,
        seq_lens_sum=sum(layout.seq_lens),
        seq_lens_cpu=seq_lens_cpu,
        encoder_lens=None,
        out_cache_loc=(
            parent_out_cache_loc[tok_slice]
            if parent_out_cache_loc is not None
            else None
        ),
        extend_num_tokens=layout.extend_num_tokens,
        extend_seq_lens=extend_seq_lens,
        extend_seq_lens_cpu=layout.extend_seq_lens,
        extend_prefix_lens=extend_prefix_lens,
        extend_prefix_lens_cpu=layout.extend_prefix_lens,
        extend_start_loc=extend_start_loc,
        spec_info=None,
    )
