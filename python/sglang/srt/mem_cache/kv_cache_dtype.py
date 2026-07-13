import logging
from typing import Optional

import torch
from torch import nn

from sglang.srt.layers.quantization.fp8_kernel import fp8_dtype
from sglang.srt.server_args import ServerArgs
from sglang.srt.utils import is_hip

logger = logging.getLogger(__name__)

_is_hip = is_hip()
