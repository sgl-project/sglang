import logging

import zmq
from sglang.srt.managers.detokenizer_manager import DetokenizerManager
from sglang.srt.managers.io_struct import BatchTokenIDOut, BatchEmbeddingOut
from sglang.srt.server_args import PortArgs
from sglang.srt.utils import get_zmq_socket
from sglang.utils import TypeBasedDispatcher

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

        self._request_dispatcher = TypeBasedDispatcher(
            [
                (BatchEmbeddingOut, self.core.handle_batch_embedding_out),
                (BatchTokenIDOut, self.core.handle_batch_token_id_out),
            ]
        )
