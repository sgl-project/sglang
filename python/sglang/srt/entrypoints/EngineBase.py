from abc import ABC, abstractmethod
from typing import Dict, Iterator, List, Optional, Tuple, Union

import torch


class EngineBase(ABC):
    """
    Abstract base class for engine interfaces that support generation, weight updating, and memory control.
    This base class provides a unified API for both HTTP-based engines and engines.
    """

    @abstractmethod
    def generate(
        self,
        prompt: Optional[Union[List[str], str]] = None,
        sampling_params: Optional[Union[List[Dict], Dict]] = None,
        input_ids: Optional[Union[List[List[int]], List[int]]] = None,
        image_data: Optional[Union[List[str], str]] = None,
        return_logprob: Optional[Union[List[bool], bool]] = False,
        logprob_start_len: Optional[Union[List[int], int]] = None,
        top_logprobs_num: Optional[Union[List[int], int]] = None,
        token_ids_logprob: Optional[Union[List[List[int]], List[int]]] = None,
        lora_path: Optional[Union[List[Optional[str]], Optional[str]]] = None,
        custom_logit_processor: Optional[Union[List[str], str]] = None,
        return_hidden_states: Optional[bool] = None,
        stream: Optional[bool] = None,
        bootstrap_host: Optional[Union[List[str], str]] = None,
        bootstrap_port: Optional[Union[List[int], int]] = None,
        bootstrap_room: Optional[Union[List[int], int]] = None,
        data_parallel_rank: Optional[int] = None,
    ) -> Union[Dict, Iterator[Dict]]:
        """Generate outputs based on given inputs."""
        pass

    @abstractmethod
    def flush_cache(self):
        """Flush the cache of the engine."""
        pass

    @abstractmethod
    def update_weights_from_tensor(
        self,
        named_tensors: List[Tuple[str, torch.Tensor]],
        load_format: Optional[str] = None,
        flush_cache: bool = True,
    ):
        """Update model weights with in-memory tensor data."""
        pass

    @abstractmethod
    def release_memory_occupation(self):
        """Release GPU memory occupation temporarily."""
        pass

    @abstractmethod
    def resume_memory_occupation(self):
        """Resume GPU memory occupation which is previously released."""
        pass

    @abstractmethod
    def shutdown(self):
        """Shutdown the engine and clean up resources."""
        pass
