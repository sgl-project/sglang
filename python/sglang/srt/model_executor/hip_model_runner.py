import json
import logging
from typing import Optional

import torch

from sglang.srt.configs.model_config import ModelConfig
from sglang.srt.layers.attention.hip_attention import HiPRadixAttentionBackend
from sglang.srt.layers.attention.hip_attention.hip_config import HiPAttentionConfig
from sglang.srt.layers.logits_processor import LogitsProcessorOutput
from sglang.srt.mem_cache.hip_memory_pool import HiPMetadataCachePool
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.model_executor.model_runner import ModelRunner
from sglang.srt.server_args import ServerArgs
from sglang.srt.utils import get_available_gpu_memory

logger = logging.getLogger(__name__)


class HiPModelRunner(ModelRunner):
    hip_attention_config: HiPAttentionConfig

    def __init__(
        self,
        model_config: ModelConfig,
        mem_fraction_static: float,
        gpu_id: int,
        tp_rank: int,
        tp_size: int,
        nccl_port: int,
        server_args: ServerArgs,
        is_draft_worker: bool = False,
    ):
        if server_args.enable_hip_attention:
            logger.info("HIP attention is turned on.")
            server_args.attention_backend = "hip_attention"
            self.init_hip_attention_config(server_args.hip_attention_config)

        super().__init__(
            model_config=model_config,
            mem_fraction_static=mem_fraction_static,
            gpu_id=gpu_id,
            tp_rank=tp_rank,
            tp_size=tp_size,
            nccl_port=nccl_port,
            server_args=server_args,
            is_draft_worker=is_draft_worker,
        )

    def init_attention_backend(self):
        if self.server_args.enable_hip_attention:
            self.attn_backend = HiPRadixAttentionBackend(self)
        else:
            super().init_attention_backend()

    def init_hip_attention_config(self, hip_attention_config):
        if hip_attention_config is None:
            hip_attention_config = {}
        elif hip_attention_config.startswith("{"):
            hip_attention_config = json.loads(hip_attention_config)
        else:
            with open(hip_attention_config, "r") as f:
                hip_attention_config = json.load(f)
        self.hip_attention_config = HiPAttentionConfig(parsed_json=hip_attention_config)

    def init_memory_pool(
        self,
        total_gpu_memory: int,
        max_num_reqs: Optional[int] = None,
        max_total_tokens: Optional[int] = None,
    ):
        super().init_memory_pool(total_gpu_memory, max_num_reqs, max_total_tokens)

        if self.server_args.enable_hip_attention:
            self.hip_metadata_cache_pool = HiPMetadataCachePool(
                query_head_num=self.model_config.num_attention_heads
                // self.server_args.tp_size,
                layer_num=self.model_config.num_hidden_layers,
                context_length=self.model_config.context_len,
                device=self.device,
                hip_config=self.hip_attention_config,
            )
        logger.info(
            f"Memory + HiP pool end. "
            f"avail mem={get_available_gpu_memory(self.device, self.gpu_id):.2f} GB"
        )
