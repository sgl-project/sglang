import logging
from abc import ABC
from typing import Optional
import torch
import numpy as np
from sglang.srt.configs.model_config import ModelConfig
from sglang.srt.server_args import get_global_server_args
# from sglang.srt.utils.torch_memory_saver_adapter import TorchMemorySaverAdapter
# from sglang.srt.constants import GPU_MEMORY_TYPE_KV_CACHE

logger = logging.getLogger(__name__)

GB = 1024 * 1024 * 1024
MB = 1024 * 1024

def get_tensor_size_bytes(t: torch.Tensor):
    return np.prod(t.shape) * t.dtype.itemsize

class RoutedExpertsDeviceCache:
    def __init__(
        self, 
        model_config: ModelConfig, 
        max_running_requests: int, 
        device: str
    ) -> None:
        self.buffer = torch.zeros(
            (
                max(
                    get_global_server_args().chunked_prefill_size,
                    max_running_requests
                ),
                model_config.hf_text_config.num_hidden_layers,
                model_config.hf_text_config.num_experts_per_tok
            ),
            dtype=torch.int32,
            device=device,
        )
        self._finalize_allocation_log()

    def get_buffer_size_bytes(self):
        assert hasattr(self, "buffer")
        return get_tensor_size_bytes(self.buffer)
    
    def capture_fwd_routed_experts(
        self,
        layer_id: int,
        topk_ids: torch.Tensor
    ):
        assert layer_id is not None, "capturing routing experts but get layer_id None"
        batch, _ = topk_ids.shape
        self.buffer[:batch, layer_id, :] = topk_ids

    def get_experts_buffer(self, layer_id: int):
        return self.buffer

    def _finalize_allocation_log(self):
        """Common logging and memory usage computation for captured experts buffers.
        """
        buffer_size_MB = self.get_buffer_size_bytes() / MB
        logger.info(
            f"Routing experts device buffer allocated. #shape: {tuple(self.buffer.shape)}, size: {buffer_size_MB:.2f} MB"
        )

class RoutedExpertsHostCache:
    def __init__(
        self, 
        model_config: ModelConfig, 
        num_tokens: int, 
    ) -> None:
        self.num_tokens = num_tokens
        self.buffer = torch.zeros(
            (
                num_tokens,
                model_config.hf_text_config.num_hidden_layers,
                model_config.hf_text_config.num_experts_per_tok
            ),
            dtype=torch.int32,
            device="cpu",
        )
        self._finalize_allocation_log()

    def get_buffer_size_bytes(self):
        assert hasattr(self, "buffer")
        return get_tensor_size_bytes(self.buffer)
    
    def set_experts_buffer(
        self, 
        layer_id: int,
        loc: torch.Tensor,
        top_k: torch.Tensor
    ):
        self.buffer[layer_id, loc, :] = top_k.cpu()
    
    def get_experts_buffer(self, layer_id: int):
        return self.buffer

    def _finalize_allocation_log(self):
        """Common logging and memory usage computation for captured experts buffers.
        """
        buffer_size_GB = self.get_buffer_size_bytes() / GB
        logger.info(
            f"Routing experts host buffer allocated. #tokens: {self.num_tokens}, size: {buffer_size_GB:.2f} GB"
        )

class RoutedExpertsCapturer(ABC):
    @staticmethod
    def create(
        enable: bool, 
        model_config: ModelConfig,
        num_tokens: int,
        max_running_requests: int,
        device: str
    ):
        if enable:
            return _RoutedExpertsCapturerReal(
                model_config,
                num_tokens=num_tokens,
                max_running_requests=max_running_requests,
                device=device
            )
        else:
            return _RoutedExpertsCapturerNoop()

    def capture(self, layer_id: int, topk_ids: torch.Tensor):
        raise NotImplementedError
    
    def get_host_cache(self):
        raise NotImplementedError
    
    def get_device_cache(self):
        raise NotImplementedError

class _RoutedExpertsCapturerReal(RoutedExpertsCapturer):
    """Capturer for routed experts with host buffer"""
    def __init__(
        self,
        model_config: ModelConfig,
        num_tokens: int,
        max_running_requests: int,
        device: str
    ):

        self.host_cache = RoutedExpertsHostCache(
            model_config,
            num_tokens
        )

        self.device_cache = RoutedExpertsDeviceCache(
            model_config,
            max_running_requests,
            device
        )

    def capture(self, layer_id: int, topk_ids: torch.Tensor):
        self.device_cache.capture_fwd_routed_experts(
            layer_id,
            topk_ids
        )

    def get_host_cache(self):
        return self.host_cache

    def get_device_cache(self):
        return self.device_cache

class _RoutedExpertsCapturerNoop(RoutedExpertsCapturer):
    def __init__(self):
        pass

    def capture(self, layer_id: int, topk_ids: torch.Tensor):
        pass

    def clear_buffer(self):
        pass
    
    def get_captured_experts(self):
        pass

_global_expert_capturer: Optional[RoutedExpertsCapturer] = (
    _RoutedExpertsCapturerNoop()
)

def get_global_experts_capturer():
    return _global_expert_capturer

def set_global_experts_capturer(capturer: RoutedExpertsCapturer):
    global _global_expert_capturer
    _global_expert_capturer = capturer

def sync_fwd_experts_buffer_DtoH():
    capturer = get_global_experts_capturer()
    if isinstance(capturer, _RoutedExpertsCapturerReal):
        device_cache = capturer.get_device_cache()
        host_cache = capturer.get_host_cache()
        for layer_id in range(device_cache.buffer.shape[0]):
            host_cache.set_experts_buffer(
                layer_id,
                torch.arange(host_cache.num_tokens),
                device_cache.get_experts_buffer(layer_id).cpu()
            )