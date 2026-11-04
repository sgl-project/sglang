import logging
from typing import Optional

import numpy as np
import pybase64
import torch

from sglang.srt.configs.kimi_linear import KimiLinearConfig
from sglang.srt.configs.model_config import ModelConfig, get_num_indexer_layers
from sglang.srt.layers.dp_attention import is_dp_attention_enabled
from sglang.srt.runtime_context import (
    get_exec,
    get_parallel,
    get_resources,
    get_schedule,
)
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
        layer_ids: Optional[list[int]] = None,
    ):
        self.num_indexer_layers = num_indexer_layers
        self.index_topk = index_topk
        # None: stream i is layer i; otherwise stream i is layer_ids[i].
        self._stream_of_layer = (
            None
            if layer_ids is None
            else {layer: i for i, layer in enumerate(layer_ids)}
        )

        # Plain TP attention runs the replicated indexer over the whole batch on every
        # rank, so each rank's capture is already complete; attention TP inside DP
        # attention would need a gather that is not implemented.
        assert get_parallel().attn_tp_size == 1 or not is_dp_attention_enabled(), (
            "IndexerTopkCapturer supports DP attention with attn_tp_size 1 or plain TP attention"
        )

        # Capture is per-rank-local: each rank writes [:local_batch, ...] to its own
        # device_cache, so the buffer only needs to fit one rank's batch.
        max_batch_size = max(get_schedule().chunked_prefill_size, max_running_requests)

        super().__init__(
            num_tokens=num_tokens,
            max_batch_size=max_batch_size,
            num_layers=self.num_indexer_layers,
            topk_size=self.index_topk,
            device=device,
            name="indexer_topk",
            # Every attention-TP rank attends rank 0's broadcast top-k; only rank 0's rows
            # reach the client.
            keeps_host_rows=get_parallel().attn_tp_rank == 0,
        )

    def capture(self, layer_id: int, topk_indices: torch.Tensor):
        stream = (
            layer_id
            if self._stream_of_layer is None
            else self._stream_of_layer[layer_id]
        )
        super().capture(layer_id=stream, topk_indices=topk_indices)


def get_global_indexer_capturer() -> Optional[IndexerTopkCapturer]:

    return get_resources().indexer_capturer


def set_global_indexer_capturer(capturer: Optional[IndexerTopkCapturer]):

    get_resources().indexer_capturer = capturer


def destroy_global_indexer_capturer():
    if (capturer := get_resources().indexer_capturer) is not None:
        capturer.destroy()
    get_resources().indexer_capturer = None


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
    *,
    model_config: ModelConfig,
    num_tokens: int,
    max_running_requests: int,
    device: str,
) -> Optional[IndexerTopkCapturer]:

    enable = get_exec().features.enable_return_indexer_topk
    # Producer wiring is CUDA/NPU-only; other backends
    # would create a capturer but never feed it.
    if enable and device != "cuda" and device != "npu":
        logger.warning(
            "indexer-topk capture is not wired for %s backend. Disabling capturer.",
            device,
        )
        return None

    hf_text_config = model_config.hf_text_config
    # Kimi-K3.1 has indexers only on its DSA layers, so only those get a stream.
    if (
        isinstance(hf_text_config, KimiLinearConfig)
        and hf_text_config.dsa_config is not None
    ):
        layer_ids = hf_text_config.dsa_layer_ids()
        num_indexer_layers = len(layer_ids)
    else:
        layer_ids = None
        num_indexer_layers = get_num_indexer_layers(hf_text_config)
    index_topk = getattr(hf_text_config, "index_topk", 0)
    return _create_indexer_capturer_raw(
        enable=enable,
        num_indexer_layers=num_indexer_layers,
        index_topk=index_topk,
        num_tokens=num_tokens,
        max_running_requests=max_running_requests,
        device=device,
        layer_ids=layer_ids,
    )


def _create_indexer_capturer_raw(
    enable: bool,
    num_indexer_layers: int,
    index_topk: int,
    num_tokens: int,
    max_running_requests: int,
    device: str,
    layer_ids: Optional[list[int]] = None,
) -> Optional[IndexerTopkCapturer]:
    if not enable:
        return None
    if num_indexer_layers <= 0:
        logger.warning("No indexer layers found, IndexerTopkCapturer disabled")
        return None
    if index_topk <= 0:
        logger.warning(
            "Invalid index_topk=%s, IndexerTopkCapturer disabled", index_topk
        )
        return None
    return IndexerTopkCapturer(
        num_tokens=num_tokens,
        num_indexer_layers=num_indexer_layers,
        index_topk=index_topk,
        max_running_requests=max_running_requests,
        device=device,
        layer_ids=layer_ids,
    )
