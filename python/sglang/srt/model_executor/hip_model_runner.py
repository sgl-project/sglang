import json
import logging
from typing import Optional

from sglang.srt.configs.model_config import ModelConfig
from sglang.srt.layers.attention.hip_attention import HiPRadixAttentionBackend
from sglang.srt.layers.attention.hip_attention.hip_config import HiPAttentionConfig
from sglang.srt.mem_cache.hip_memory_pool import HiPMetadataCachePool
from sglang.srt.model_executor.model_runner import ModelRunner
from sglang.srt.server_args import ServerArgs
from sglang.srt.utils import get_available_gpu_memory

logger = logging.getLogger(__name__)


class HiPModelRunner(ModelRunner):

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
        self.hip_attention_config: HiPAttentionConfig
        if server_args.enable_hip_attention:
            logger.info("HIP attention is turned on.")
            server_args.attention_backend = "hip_attention"
            self.init_hip_attention_config(
                server_args.hip_attention_config_path
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

    def init_hip_attention_config(self, hip_attention_config_path):
        if hip_attention_config_path is None:
            hip_attention_config = {}
        elif hip_attention_config_path.startswith("{"):
            hip_attention_config = json.loads(hip_attention_config_path)
        else:
            with open(hip_attention_config_path, "r") as f:
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
                self.max_total_num_tokens,
                head_num=self.model_config.num_attention_heads,
                layer_num=self.model_config.num_hidden_layers,
                device=self.device,
                hip_config=self.hip_attention_config,
            )
        logger.info(
            f"Memory + HiP pool end. "
            f"avail mem={get_available_gpu_memory(self.device, self.gpu_id):.2f} GB"
        )
