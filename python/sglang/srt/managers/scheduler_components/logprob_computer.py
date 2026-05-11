from __future__ import annotations  # noqa: F401

from typing import List, Optional, Tuple  # noqa: F401

import torch  # noqa: F401

from sglang.srt.layers.logits_processor import LogitsProcessorOutput  # noqa: F401
from sglang.srt.managers.schedule_batch import Req  # noqa: F401
from sglang.srt.server_args import MIS_DELIMITER_TOKEN_ID  # noqa: F401


class SchedulerLogprobComputer:
    """Pure-compute logprob accumulator helpers. Composition target on
    Scheduler (``self.logprob_computer``)."""

    def __init__(self, *, server_args, model_config) -> None:
        self.server_args = server_args
        self.model_config = model_config
