import logging
from http import HTTPStatus
from typing import Optional, List, Tuple
from collections import defaultdict
import json
import time

from sglang.srt.managers.schedule_batch import FINISH_ABORT, Req

logger = logging.getLogger(__name__)


def validate_input_length(
    req: Req, max_req_input_len: int, allow_auto_truncate: bool
) -> Optional[str]:
    """Validate and potentially truncate input length.

    Args:
        req: The request containing input_ids to validate
        max_req_input_len: Maximum allowed input length
        allow_auto_truncate: Whether to truncate long inputs

    Returns:
        Error message if validation fails, None if successful
    """
    if len(req.origin_input_ids) >= max_req_input_len:
        if allow_auto_truncate:
            logger.warning(
                "Request length is longer than the KV cache pool size or "
                "the max context length. Truncated. "
                f"{len(req.origin_input_ids)=}, {max_req_input_len=}."
            )
            req.origin_input_ids = req.origin_input_ids[:max_req_input_len]
            return None
        else:
            error_msg = (
                f"Input length ({len(req.origin_input_ids)} tokens) exceeds "
                f"the maximum allowed length ({max_req_input_len} tokens). "
                f"Use a shorter input or enable --allow-auto-truncate."
            )
            logger.error(error_msg)
            req.finished_reason = FINISH_ABORT(
                error_msg, HTTPStatus.BAD_REQUEST, "BadRequestError"
            )
            return error_msg

    return None

# global expert distribution recording
#TODO make this a singleton and add endpoints to construct/destruct this
class ExpertDistributionRecorder:
    # This class is a singleton class
    def __new__(cls):
        if not hasattr(cls, 'instance'):
            cls.instance = super(ExpertDistributionRecorder, cls).__new__(cls)
        return cls.instance

    def __init__(self):
        # the length of the first list is the number of layers
        # the length of the second list is the number of tokens
        # the length of the tuple is topk's k value
        self._expert_distribution_record: Dict[int, List[Tuple[int]]] = defaultdict(list)
        self._current_layer_id = None

    def set_current_layer(self, layer_idx):
        self._current_layer_id = layer_idx

    def record_new_token(self, topk_ids):
        topk_ids_list = topk_ids.cpu().numpy().tolist()
        for i in topk_ids_list:
            self._expert_distribution_record[self._current_layer_id].append(tuple(i))

    def reset(self):
        self._expert_distribution_record.clear()

    def dump_record(self):
        results = {}
        for layer_idx, layer_record in self._expert_distribution_record.items():
            results[layer_idx] = defaultdict(lambda: defaultdict(int))
            for token_record in layer_record:
                for k_idx, expert_idx in enumerate(token_record):
                    results[layer_idx][k_idx][expert_idx] += 1
        with open(f"expert_distribution_{time.time()}.json", 'w') as fd:
            json.dump(results, fd)
