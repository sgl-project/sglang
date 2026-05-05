import logging
from typing import Optional

import numpy as np
import pybase64
import torch

from sglang.srt.layers.dp_attention import get_attention_tp_size
from sglang.srt.layers.topk_capturer_base import BaseTopkCapturer

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

        # DP-attention capture is per-rank-local: each rank writes [:local_batch, ...]
        # to its own device_cache, so the buffer only needs to fit one rank's batch.
        server_args = get_global_server_args()
        max_batch_size = max(server_args.chunked_prefill_size, max_running_requests)

        super().__init__(
            num_tokens=num_tokens,
            max_batch_size=max_batch_size,
            num_layers=self.num_indexer_layers,
            topk_size=self.index_topk,
            device=device,
            name="indexer_topk",
        )


_global_indexer_capturer: Optional[IndexerTopkCapturer] = None


def get_global_indexer_capturer() -> Optional[IndexerTopkCapturer]:
    return _global_indexer_capturer


def set_global_indexer_capturer(capturer: Optional[IndexerTopkCapturer]):
    global _global_indexer_capturer
    _global_indexer_capturer = capturer


def maybe_capture_indexer_topk(
    layer_id: int, topk_indices: Optional[torch.Tensor]
) -> Optional[torch.Tensor]:
    """Capture topk for layer_id if a capturer is set; pass through unchanged.

    Works in both expression context (`return maybe_capture_indexer_topk(...)`)
    and statement context (call for side-effect, ignore return).
    """
    if topk_indices is None:
        return None
    if (cap := get_global_indexer_capturer()) is not None:
        cap.capture(layer_id=layer_id, topk_indices=topk_indices)
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
