import contextlib
from typing import TYPE_CHECKING, Callable, List, Optional

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
        # FIXME: side channel propagated to primary and each child at replay.
        # Some inner backends (dsv4 / dsa) discriminate capture vs replay via
        # ``self._replay_forward_batch``; the dispatcher must mirror the value
        # set by ``cuda_graph_runner.replay_prepare`` onto each inner backend
        # before recursing, else inner backends silently take the capture
        # branch at replay and rebuild metadata that the captured graph holds
        # capture-time pointers for. Step 04 removes the side channel.
        self._replay_forward_batch: Optional["ForwardBatch"] = None

    @classmethod
    def init_new(cls, creator: Callable[[], AttentionBackend]):
        return cls(
            primary=creator(),
            children=[creator() for _ in range(2)],
        )

    @contextlib.contextmanager
    def _propagate_replay_forward_batch(self):
        """Mirror ``self._replay_forward_batch`` onto primary and each child
        for the duration of the dispatch, then clear on exit.

        ``cuda_graph_runner.replay_prepare`` sets the side channel on the
        outer TboAttnBackend only; inner backends (dsv4 / dsa) need it set
        on themselves to take their replay branch. Capture-path callers
        leave ``_replay_forward_batch=None`` and this becomes a no-op.
        """
        replay_fb = self._replay_forward_batch
        if replay_fb is None:
            yield
            return
        if hasattr(self.primary, "_replay_forward_batch"):
            self.primary._replay_forward_batch = replay_fb
        for child in self.children:
            if hasattr(child, "_replay_forward_batch"):
                child._replay_forward_batch = replay_fb
        try:
            yield
        finally:
            if hasattr(self.primary, "_replay_forward_batch"):
                self.primary._replay_forward_batch = None
            for child in self.children:
                if hasattr(child, "_replay_forward_batch"):
                    child._replay_forward_batch = None

    def init_forward_data(self, forward_batch: "ForwardBatch") -> None:
        """Dispatcher eager wrapper -- recurse to primary + (optional) children."""
        self.primary.init_forward_data(forward_batch)
        if forward_batch.tbo_children is not None:
            for child, forward_batch_child in zip(
                self.children, forward_batch.tbo_children, strict=True
            ):
                if forward_batch_child.batch_size > 0:
                    child.init_forward_data(forward_batch_child)

    def init_forward_data_out_graph(self, forward_batch: "ForwardBatch") -> None:
        """Dispatcher out-graph init -- recurse to primary + (optional) children.

        Capture and replay both flow through this method via the unified
        step 03 contract; child-fb split details that the old asymmetric
        capture/replay variants assembled are now derived per-backend
        directly from the (already-split-by-caller) ``forward_batch.tbo_children``.
        At replay, ``_replay_forward_batch`` is fanned out to primary/children
        so inner DSV4/DSA backends take their replay branch (not the capture
        branch, which would rebuild metadata into freshly-allocated tensors).
        """
        with self._propagate_replay_forward_batch():
            self.primary.init_forward_data_out_graph(forward_batch)
            if forward_batch.tbo_children is not None:
                for child, forward_batch_child in zip(
                    self.children, forward_batch.tbo_children, strict=True
                ):
                    if forward_batch_child.batch_size > 0:
                        child.init_forward_data_out_graph(forward_batch_child)

    def init_forward_data_in_graph(self, forward_batch: "ForwardBatch") -> None:
        """Dispatcher in-graph init -- recurse to primary + (optional) children.

        Override the ABC default no-op so a backend that overrides
        ``_in_graph`` (none in the initial stage; reserved for future
        per-backend perf PRs) is correctly reached through this dispatcher.
        """
        with self._propagate_replay_forward_batch():
            self.primary.init_forward_data_in_graph(forward_batch)
            if forward_batch.tbo_children is not None:
                for child, forward_batch_child in zip(
                    self.children, forward_batch.tbo_children, strict=True
                ):
                    if forward_batch_child.batch_size > 0:
                        child.init_forward_data_in_graph(forward_batch_child)

    def init_cuda_graph_state(self, max_bs: int, max_num_tokens: int):
        self.primary.init_cuda_graph_state(max_bs=max_bs, max_num_tokens=max_num_tokens)
        for item in self.children:
            # TODO for children, maybe can provide *smaller* max_bs to optimize
            item.init_cuda_graph_state(max_bs=max_bs, max_num_tokens=max_num_tokens)

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
