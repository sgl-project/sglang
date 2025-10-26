import logging
from abc import ABC
import torch
from sglang.srt.configs.model_config import ModelConfig
from sglang.srt.server_args import get_global_server_args
from sglang.srt.layers.moe.topk import StandardTopKOutput

logger = logging.getLogger(__name__)

_experts_capturer_host_buffer = None
_experts_capturer_device_buffer = None

class RoutedExpertsCapturer(ABC):
    @staticmethod
    def create(enable: bool):
        if enable:
            return _RoutedExpertsCapturerReal()
        else:
            return _RoutedExpertsCapturerNoop()

    def init_buffer(self, max_running_requests: int, model_config: ModelConfig):
        raise NotImplementedError

    def is_initialized(self):
        raise NotImplementedError

    def capture(self, layer_id: int, topk_output: StandardTopKOutput):
        raise NotImplementedError

    def clear_buffer(self):
        raise NotImplementedError
    
    def get_captured_experts(self):
        raise NotImplementedError


class _RoutedExpertsCapturerReal(RoutedExpertsCapturer):
    """Capturer for routed experts with host buffer"""
    def init_buffer(self, max_running_requests: int, model_config: ModelConfig):
        global _experts_capturer_host_buffer
        if (
            get_global_server_args().enable_return_routed_experts
            and _experts_capturer_host_buffer is None
        ):
            _experts_capturer_host_buffer = torch.zeros(
                (
                    max_running_requests, 
                    model_config.hf_text_config.num_hidden_layers, 
                    model_config.hf_text_config.num_experts_per_tok
                ),
                dtype=torch.int32,
                device="cpu",
            )
            logger.debug(
                f"Initialized routed experts capturer host buffer with shape {_experts_capturer_host_buffer.shape}."
            )
            print(f"{_experts_capturer_host_buffer.shape=}")

    def capture(self, layer_id: int, topk_output: StandardTopKOutput):
        batch_size, num_routed_experts = topk_output.topk_ids.shape
        _experts_capturer_host_buffer[:batch_size, layer_id, :] = topk_output.topk_ids.cpu()

    def clear_buffer(self):
        global _experts_capturer_host_buffer
        _experts_capturer_host_buffer.zero_()
    
    def get_captured_experts(self):
        global _experts_capturer_host_buffer
        return _experts_capturer_host_buffer

class _RoutedExpertsCapturerNoop(RoutedExpertsCapturer):
    def init_buffer(self, max_running_requests: int, model_config: ModelConfig):
        pass

    def is_initialized(self):
        pass

    def capture(self, layer_id: int, topk_output: StandardTopKOutput):
        pass

    def clear_buffer(self):
        pass
    
    def get_captured_experts(self):
        pass