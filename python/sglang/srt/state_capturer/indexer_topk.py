import logging
from typing import Optional

import numpy as np
import pybase64
import torch

from sglang.srt.layers.dp_attention import (
    get_attention_dp_rank,
    get_attention_tp_size,
    get_dp_local_slice_cpu,
    is_dp_attention_enabled,
)
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.state_capturer.base import BaseTopkCapturer

logger = logging.getLogger(__name__)


class IndexerTopkCapturer(BaseTopkCapturer):
    def __init__(
        self,
        num_tokens: int,
        num_indexer_layers: int,
        index_topk: int,
        max_running_requests: int,
        device: str,
    ):
        from sglang.srt.server_args import get_global_server_args

        self.num_indexer_layers = num_indexer_layers
        self.index_topk = index_topk

        attn_tp_size = get_attention_tp_size()
        assert attn_tp_size == 1, "IndexerTopkCapturer now only supports DP attention"

        # device_cache holds the global DP-rank-padded batch; each rank's slice is
        # read back via _get_local_slice (see RoutedExpertsCapturer for the same pattern)
        server_args = get_global_server_args()
        max_batch_size = max(
            server_args.chunked_prefill_size * server_args.dp_size,
            max_running_requests,
        )

        super().__init__(
            num_tokens=num_tokens,
            max_batch_size=max_batch_size,
            num_layers=self.num_indexer_layers,
            topk_size=self.index_topk,
            device=device,
            name="indexer_topk",
        )

    def capture(self, layer_id: int, topk_indices: torch.Tensor, forward_batch=None):
        # Each DP rank only computes its own tokens' topk; write them at this
        # rank's offset in the shared rank-padded buffer (writing the head would
        # clobber across ranks). The offset matches _get_local_slice's read.
        if forward_batch is not None and is_dp_attention_enabled():
            gnt = forward_batch.global_num_tokens_cpu
            dp_rank = get_attention_dp_rank()
            batch = topk_indices.shape[0]
            # gnt is None during cuda-graph capture (uniform dummy batch); the
            # per-rank stride is then the captured batch size itself.
            start = (
                dp_rank * batch if gnt is None else sum(int(n) for n in gnt[:dp_rank])
            )
            self.device_cache.buffer[start : start + batch, layer_id, :] = topk_indices
        else:
            super().capture(layer_id, topk_indices)

    def _get_local_slice(
        self,
        forward_batch: ForwardBatch,
        can_run_graph: bool,
        cuda_graph_batch: Optional[int],
    ) -> torch.Tensor:
        # Under DP attention the device buffer is rank-padded; read this rank's
        # slice (handles eager, cuda-graph-replay, and cuda-graph-capture layouts)
        # instead of the head, which would only be correct for DP rank 0.
        if not is_dp_attention_enabled():
            num = forward_batch.out_cache_loc.shape[0]
            return self.device_cache.buffer[:num, :, : self.topk_size]
        gnt = forward_batch.global_num_tokens_cpu
        if gnt is None:
            # cuda-graph capture: gnt not populated, uniform dummy batch.
            num = forward_batch.out_cache_loc.shape[0]
            start = get_attention_dp_rank() * (
                cuda_graph_batch if cuda_graph_batch is not None else num
            )
        else:
            start, num = get_dp_local_slice_cpu(
                forward_batch, can_run_graph, cuda_graph_batch
            )
        return self.device_cache.buffer[start : start + num, :, : self.topk_size]


_global_indexer_capturer: Optional[IndexerTopkCapturer] = None


def get_global_indexer_capturer() -> Optional[IndexerTopkCapturer]:
    return _global_indexer_capturer


def set_global_indexer_capturer(capturer: Optional[IndexerTopkCapturer]):
    global _global_indexer_capturer
    _global_indexer_capturer = capturer


def maybe_capture_indexer_topk(
    layer_id: int, topk_indices: Optional[torch.Tensor], forward_batch=None
) -> Optional[torch.Tensor]:
    """Capture topk for layer_id if a capturer is set; pass through unchanged.

    Works in both expression context (`return maybe_capture_indexer_topk(...)`)
    and statement context (call for side-effect, ignore return).
    """
    if topk_indices is None:
        return None
    if (cap := get_global_indexer_capturer()) is not None:
        cap.capture(
            layer_id=layer_id, topk_indices=topk_indices, forward_batch=forward_batch
        )
    return topk_indices


def extract_indexer_topk_from_meta_info(data):
    # Mirrors extract_routed_experts_from_meta_info: indices are returned as
    # base64-encoded int32 bytes. Caller reshapes to (seqlen-1, num_indexer_layers,
    # index_topk).
    indexer_topk_base64 = data["meta_info"].get("indexer_topk", None)
    indexer_topk = np.frombuffer(
        pybase64.b64decode(indexer_topk_base64.encode("utf-8")), dtype=np.int32
    )
    return indexer_topk


def create_indexer_capturer(
    enable: bool,
    num_indexer_layers: int,
    index_topk: int,
    num_tokens: int,
    max_running_requests: int,
    device: str,
) -> Optional[IndexerTopkCapturer]:
    if not enable:
        return None
    if num_indexer_layers == 0:
        logger.warning("No indexer layers found, IndexerTopkCapturer disabled")
        return None
    return IndexerTopkCapturer(
        num_tokens=num_tokens,
        num_indexer_layers=num_indexer_layers,
        index_topk=index_topk,
        max_running_requests=max_running_requests,
        device=device,
    )
