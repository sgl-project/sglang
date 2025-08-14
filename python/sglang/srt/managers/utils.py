import logging
import multiprocessing as mp
from http import HTTPStatus
from typing import Dict, List, Optional

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
            return error_msg

    return None


class DPBalanceMeta:
    """
    This class will be use in scheduler and dp controller
    """

    def __init__(self, num_workers: int):
        self.num_workers = num_workers
        self._manager = mp.Manager()
        self.mutex = self._manager.Lock()

        init_local_tokens = [0] * self.num_workers
        init_onfly_info = [self._manager.dict() for _ in range(self.num_workers)]

        self.shared_state = self._manager.Namespace()
        self.shared_state.local_tokens = self._manager.list(init_local_tokens)
        self.shared_state.onfly_info = self._manager.list(init_onfly_info)

    def destructor(self):
        # we must destructor this class manually
        self._manager.shutdown()

    def get_shared_onfly(self) -> List[Dict[int, int]]:
        return [dict(d) for d in self.shared_state.onfly_info]

    def set_shared_onfly_info(self, data: List[Dict[int, int]]):
        self.shared_state.onfly_info = data

    def get_shared_local_tokens(self) -> List[int]:
        return list(self.shared_state.local_tokens)

    def set_shared_local_tokens(self, data: List[int]):
        self.shared_state.local_tokens = data

    def __getstate__(self):
        state = self.__dict__.copy()
        del state["_manager"]
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._manager = None
