import logging
import signal

import psutil
import setproctitle
import zmq
from sglang.srt.managers.detokenizer_manager import DetokenizerManager
from sglang.srt.managers.io_struct import BatchEmbeddingOut, BatchTokenIDOut
from sglang.srt.server_args import PortArgs, ServerArgs
from sglang.srt.utils import configure_logger, get_zmq_socket
from sglang.utils import get_exception_traceback, TypeBasedDispatcher

logger = logging.getLogger(__name__)


class DetokenizerManagerCommunicator:
    def __init__(self, core: DetokenizerManager, port_args: PortArgs):
        self.core = core

        # Init inter-process communication
        context = zmq.Context(2)
        self._recv_from_scheduler = get_zmq_socket(
            context, zmq.PULL, port_args.detokenizer_ipc_name
        )
        self._send_to_tokenizer = get_zmq_socket(
            context, zmq.PUSH, port_args.tokenizer_ipc_name
        )

        self._dispatcher = TypeBasedDispatcher(
            [
                (BatchEmbeddingOut, self.core.handle_batch_embedding_out),
                (BatchTokenIDOut, self.core.handle_batch_token_id_out),
            ]
        )


def run_detokenizer_process(
    server_args: ServerArgs,
    port_args: PortArgs,
):
    setproctitle.setproctitle("sglang::detokenizer")
    configure_logger(server_args)
    parent_process = psutil.Process().parent()

    try:
        manager = DetokenizerManager(server_args, port_args)
        manager.event_loop()
    except Exception:
        traceback = get_exception_traceback()
        logger.error(f"DetokenizerManager hit an exception: {traceback}")
        parent_process.send_signal(signal.SIGQUIT)
