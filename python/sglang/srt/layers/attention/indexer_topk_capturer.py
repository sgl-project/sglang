import logging
from typing import TYPE_CHECKING, Optional

from sglang.srt.layers.dp_attention import get_attention_tp_size
from sglang.srt.layers.topk_capturer_base import (
    _GB,
    _MB,
    BaseTopkCapturer,
    BaseTopkCapturerNoop,
)

if TYPE_CHECKING:
    from sglang.srt.configs.model_config import ModelConfig

logger = logging.getLogger(__name__)

INDEX_TOPK = 512


def _count_indexer_layers(model_config: "ModelConfig") -> int:
    # TODO very hacky now
    compress_ratios = getattr(model_config.hf_text_config, "compress_ratios", None)
    if compress_ratios is None:
        return 0
    return sum(1 for r in compress_ratios if r == 4)


class IndexerTopkCapturer(BaseTopkCapturer):
    def __init__(
        self,
        model_config: "ModelConfig",
        num_tokens: int,
        max_running_requests: int,
        device: str,
    ):
        from sglang.srt.server_args import get_global_server_args

        self.num_indexer_layers = _count_indexer_layers(model_config)
        self.index_topk = getattr(model_config.hf_text_config, "index_topk", INDEX_TOPK)

        if self.num_indexer_layers == 0:
            logger.warning("No indexer layers found, IndexerTopkCapturer disabled")
            self._enabled = False
            return

        self._enabled = True

        server_args = get_global_server_args()
        max_batch_size = max(
            server_args.chunked_prefill_size * server_args.dp_size,
            max_running_requests,
        )

        attn_tp_size = get_attention_tp_size()
        assert attn_tp_size == 1, "IndexerTopkCapturer now only supports DP attention"

        super().__init__(
            num_tokens=num_tokens,
            max_batch_size=max_batch_size,
            num_layers=self.num_indexer_layers,
            topk_size=self.index_topk,
            device=device,
        )

        self._log_allocation()

    def _log_allocation(self):
        host_size_gb = self.host_cache.get_buffer_size_bytes() / _GB
        device_size_mb = self.device_cache.get_buffer_size_bytes() / _MB
        logger.info(
            f"IndexerTopkCapturer allocated: "
            f"num_indexer_layers={self.num_indexer_layers}, index_topk={self.index_topk}, "
            f"host_cache={host_size_gb:.2f}GB, device_cache={device_size_mb:.2f}MB"
        )

    def _sync_to_host(self, forward_batch, can_run_graph, cuda_graph_batch):
        # b/c DP attention, we will not use a global buffer and gather it (like MoE),
        # and each rank should directly write to host
        num_tokens = forward_batch.out_cache_loc.shape[0]
        out_cache_loc_cpu = forward_batch.out_cache_loc.cpu()
        self.host_cache.buffer[out_cache_loc_cpu] = self.device_cache.buffer[
            :num_tokens, :, : self.topk_size
        ].cpu()

    def is_enabled(self) -> bool:
        return self._enabled


class IndexerTopkCapturerNoop(BaseTopkCapturerNoop):
    pass


_global_indexer_capturer: Optional[IndexerTopkCapturer] = IndexerTopkCapturerNoop()


def get_global_indexer_capturer():
    return _global_indexer_capturer


def set_global_indexer_capturer(capturer):
    global _global_indexer_capturer
    _global_indexer_capturer = capturer


def create_indexer_capturer(
    enable: bool,
    model_config: "ModelConfig",
    num_tokens: int,
    max_running_requests: int,
    device: str,
):
    if enable:
        capturer = IndexerTopkCapturer(
            model_config=model_config,
            num_tokens=num_tokens,
            max_running_requests=max_running_requests,
            device=device,
        )
        if capturer.is_enabled():
            return capturer
    return IndexerTopkCapturerNoop()
