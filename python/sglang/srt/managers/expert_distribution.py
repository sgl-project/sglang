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
        # the length of the dictionary is the number of layers
        # the length of the list is the number of tokens
        # the length of the tuple is topk's k value
        self._expert_distribution_record: Dict[int, List[Tuple[int]]] = defaultdict(
            list
        )
        self._record = False
        self._current_layer_id = Withable()

    def with_current_layer(self, layer_idx):
        return self._current_layer_id.with_value(layer_idx)

    def on_select_experts(self, topk_ids):
        if not self._record:
            return
        TODO

    def reset(self):
        """Reset the expert distribution recorder."""
        logger.info("Resetting expert distribution record...")
        self._record = False
        self._expert_distribution_record.clear()
        assert self._current_layer_id.value is None

    def start_record(self):
        """Start recording the expert distribution. Reset the recorder and set the recording flag to True."""
        if self._record:
            logger.warning(
                "SGLang server is already recording expert ids. Did you forget to dump the expert ids recorded so far by sending requests to the `/stop_expert_distribution_record` and `/dump_expert_distribution_record` endpoints?"
            )
        self.reset()
        self._record = True

    def stop_record(self):
        """Stop recording the expert distribution. Set the recording flag to False."""
        if not self._record:
            logger.warning(
                "SGLang server has not been recording expert ids. Did you forget to start recording by sending request to the `/start_expert_distribution_record` endpoint?"
            )
        self._record = False

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
    pass


class _SelectExpertsGatherer(_ForwardGatherer):
    def on_select_experts(self, topk_ids):
        topk_ids_list = topk_ids.to("cpu", non_blocking=True).numpy().tolist()
        torch.cuda.synchronize()
        for i in topk_ids_list:
            self._expert_distribution_record[self._current_layer_id.value].append(tuple(i))


expert_distribution_recorder = _ExpertDistributionRecorder()
