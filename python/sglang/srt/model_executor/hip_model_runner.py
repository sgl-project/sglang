import json
import logging
from typing import Optional

import torch

from sglang.srt.configs.model_config import ModelConfig
from sglang.srt.layers.attention.hip_attention import HiPRadixAttentionBackend
from sglang.srt.layers.attention.hip_attention.hip_config import HiPAttentionConfig
from sglang.srt.mem_cache.hip_memory_pool import HiPMetadataCachePool
from sglang.srt.model_executor.model_runner import ModelRunner
from sglang.srt.server_args import ServerArgs
from sglang.srt.utils import get_available_gpu_memory
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.layers.logits_processor import LogitsProcessorOutput

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
    ):
        if server_args.enable_hip_attention:
            logger.info("HIP attention is turned on.")
            server_args.attention_backend = "hip_attention"
            self.init_hip_attention_config(
                server_args.hip_attention_config
            )

        super().__init__(
            model_config=model_config,
            mem_fraction_static=mem_fraction_static,
            gpu_id=gpu_id,
            tp_rank=tp_rank,
            tp_size=tp_size,
            nccl_port=nccl_port,
            server_args=server_args,
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
                query_head_num=self.model_config.num_attention_heads // self.server_args.tp_size,
                layer_num=self.model_config.num_hidden_layers,
                device=self.device,
                hip_config=self.hip_attention_config,
            )
        logger.info(
            f"Memory + HiP pool end. "
            f"avail mem={get_available_gpu_memory(self.device, self.gpu_id):.2f} GB"
        )
    
    def forward(self, forward_batch: ForwardBatch) -> LogitsProcessorOutput:
        if forward_batch.forward_mode.is_decode():
            start_event = torch.cuda.Event(enable_timing=True)
            start_event.record()
            
        if forward_batch.forward_mode.is_decode():
            result = self.forward_decode(forward_batch)
        elif forward_batch.forward_mode.is_extend():
            result = self.forward_extend(forward_batch)
        elif forward_batch.forward_mode.is_idle():
            result = self.forward_idle(forward_batch)
        else:
            raise ValueError(f"Invaid forward mode: {forward_batch.forward_mode}")

        if forward_batch.forward_mode.is_decode():
            end_event = torch.cuda.Event(enable_timing=True)
            end_event.record()
            
            end_event.synchronize()
            elapsed = start_event.elapsed_time(end_event)
        
        if (forward_batch.hip_metadata_cache_pool is not None) and forward_batch.forward_mode.is_decode():
            cache = forward_batch.hip_metadata_cache_pool
            statistics = cache.compute_cache_statistics(forward_batch.batch_size)
            statistics = dict(map(lambda x: (x[0], x[1].item()), statistics.items()))
            logger.info(f'took {elapsed} ms, cache statistics {statistics}')
        elif forward_batch.forward_mode.is_decode():
            logger.info(f'took {elapsed} ms')

        return result