import logging

import zmq

from sglang.srt.managers.detokenizer_manager import DetokenizerManager
from sglang.srt.server_args import PortArgs
from sglang.srt.utils import get_zmq_socket

logger = logging.getLogger(__name__)


class DetokenizerManagerCommunicator:
    def __init__(self, core: DetokenizerManager, port_args: PortArgs):
        self.core = core

        # Init inter-process communication
        context = zmq.Context(2)
        self.recv_from_scheduler = get_zmq_socket(
            context, zmq.PULL, port_args.detokenizer_ipc_name, True
        )
        self.send_to_tokenizer = get_zmq_socket(
            context, zmq.PUSH, port_args.tokenizer_ipc_name, False
        )
