import asyncio
import copy
import dataclasses
import logging
import os
import signal
import sys
import time
import uuid
from typing import Any, Awaitable, Dict, Generic, List, Optional, Tuple, TypeVar, Union

import fastapi
import uvloop
import zmq
import zmq.asyncio
from fastapi import BackgroundTasks

from sglang.communicator import create_receiver, create_sender
from sglang.srt.configs.model_config import ModelConfig
from sglang.srt.hf_transformers_utils import get_processor, get_tokenizer
from sglang.srt.managers.image_processor import (
    get_dummy_image_processor,
    get_image_processor,
)
from sglang.srt.metrics.collector import TokenizerMetricsCollector
from sglang.srt.sampling.sampling_params import SamplingParams
from sglang.srt.server_args import PortArgs, ServerArgs
from sglang.srt.utils import (
    dataclass_to_string_truncated,
    get_zmq_socket,
    kill_process_tree,
)


class TokenizerManagerCommunicator:
    def __init__(self, port_args: PortArgs):
        # Init inter-process communication
        self.recv_from_detokenizer = create_receiver(port_args.tokenizer_ipc_name)
        self.send_to_scheduler = create_sender(port_args.scheduler_input_ipc_name)
