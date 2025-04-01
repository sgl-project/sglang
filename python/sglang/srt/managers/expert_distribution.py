import logging
import time
from abc import ABC
from collections import defaultdict
from typing import Dict, List, Tuple

import torch
from sglang.srt.utils import Withable

logger = logging.getLogger(__name__)


class _ExpertDistributionRecorder:
    """Global expert distribution recording"""

    def __init__(self):
        self._recording = False
        self._current_layer_idx = Withable()
        self._forward_gatherer: _ForwardGatherer = TODO

        # TODO
        # the length of the dictionary is the number of layers
        # the length of the list is the number of tokens
        # the length of the tuple is topk's k value
        self._expert_distribution_record: Dict[int, List[Tuple[int]]] = defaultdict(
            list
        )

    def with_current_layer(self, layer_idx):
        return self._current_layer_idx.with_value(layer_idx)

    def on_select_experts(self, topk_ids: torch.Tensor):
        if not self._recording:
            return
        self._forward_gatherer.on_select_experts(layer_idx=self._current_layer_idx.value, topk_ids=topk_ids)

    def on_deepep_dispatch_normal(self, num_recv_tokens_per_expert_list: List[int]):
        if not self._recording:
            return
        self._forward_gatherer.on_deepep_dispatch_normal(self._current_layer_idx.value, num_recv_tokens_per_expert_list)

    def reset(self):
        """Reset the expert distribution recorder."""
        logger.info("Resetting expert distribution record...")
        self._recording = False
        self._expert_distribution_record.clear()
        assert self._current_layer_idx.value is None

    def start_record(self):
        """Start recording the expert distribution. Reset the recorder and set the recording flag to True."""
        if self._recording:
            logger.warning(
                "SGLang server is already recording expert ids. Did you forget to dump the expert ids recorded so far by sending requests to the `/stop_expert_distribution_record` and `/dump_expert_distribution_record` endpoints?"
            )
        self.reset()
        self._recording = True

    def stop_record(self):
        """Stop recording the expert distribution. Set the recording flag to False."""
        if not self._recording:
            logger.warning(
                "SGLang server has not been recording expert ids. Did you forget to start recording by sending request to the `/start_expert_distribution_record` endpoint?"
            )
        self._recording = False

    def dump_record(self):
        """Dump the expert distribution record to a file. Reset the recorder after dumping."""
        results = {}
        for layer_idx, layer_record in self._expert_distribution_record.items():
            results[layer_idx] = defaultdict(int)
            for token_record in layer_record:
                for expert_idx in token_record:
                    results[layer_idx][expert_idx] += 1
        with open(
            f"expert_distribution_rank{torch.distributed.get_rank()}_timestamp{time.time()}.csv",
            "w",
        ) as fd:
            fd.write("layer_id,expert_id,count\n")
            for layer_idx, layer_results in results.items():
                for expert_idx, count in layer_results.items():
                    fd.write(f"{layer_idx},{expert_idx},{count}\n")
        self.reset()


class _ForwardGatherer(ABC):
    def on_select_experts(self, layer_idx: int, topk_ids: torch.Tensor):
        pass

    def on_deepep_dispatch_normal(self, layer_idx: int, num_recv_tokens_per_expert_list: List[int]):
        pass

    def collect(self):
        raise NotImplementedError


class _LayerBasedForwardGatherer(_ForwardGatherer):
    def __init__(self):
        self._num_recv_tokens_per_expert_list_of_layer = {}

    def _on_layer_data(self, layer_idx: int, num_recv_tokens_per_expert_list: List[int]):
        # TODO for TBO, we may need to relax this restriction
        assert layer_idx not in self._num_recv_tokens_per_expert_list_of_layer
        self._num_recv_tokens_per_expert_list_of_layer[layer_idx] = num_recv_tokens_per_expert_list

    def collect(self):
        return TODO


class _SelectExpertsForwardGatherer(_LayerBasedForwardGatherer):
    # pretty slow, but we will use the DeepEP Gatherer in production
    def on_select_experts(self, layer_idx: int, topk_ids: torch.Tensor):
        topk_ids_list = topk_ids.to("cpu", non_blocking=True).numpy().tolist()
        torch.cuda.synchronize()

        num_recv_tokens_per_expert_list = [0] * num_local_physical_experts
        for token_record in topk_ids_list:
            for expert_idx in token_record:
                num_recv_tokens_per_expert_list[expert_idx] += 1

        self._on_layer_data(layer_idx, num_recv_tokens_per_expert_list)


class _DeepepNormalForwardGatherer(_LayerBasedForwardGatherer):
    def on_deepep_dispatch_normal(self, layer_idx: int, num_recv_tokens_per_expert_list: List[int]):
        assert isinstance(num_recv_tokens_per_expert_list, list)
        self._on_layer_data(layer_idx, num_recv_tokens_per_expert_list)


# TODO Wait for LowLatency DeepEP merging
class _DeepepLowLatencyForwardGatherer(_ForwardGatherer):
    pass


expert_distribution_recorder = _ExpertDistributionRecorder()
