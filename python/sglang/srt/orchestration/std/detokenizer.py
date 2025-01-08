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
        manager = DetokenizerManager(server_args)
        communicator = DetokenizerManagerCommunicator(core=manager, port_args=port_args)

        while True:
            recv_obj = self._recv_from_scheduler.recv_pyobj()
            if isinstance(recv_obj, BatchEmbeddingOut):
                self._send_to_tokenizer.send_pyobj(recv_obj)
            else:
                assert isinstance(recv_obj, BatchTokenIDOut)
                self._send_to_tokenizer.send_pyobj(self.handle_batch_token_id_out(recv_obj))
    except Exception:
        traceback = get_exception_traceback()
        logger.error(f"DetokenizerManager hit an exception: {traceback}")
        parent_process.send_signal(signal.SIGQUIT)
