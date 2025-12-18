from sglang.multimodal_gen.runtime.distributed.disagg_communicators.pytorch_communicator import (
    PyTorchDisaggCommunicator,
)
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger

logger = init_logger(__name__)

_DISAGG_COMMUNICATOR: PyTorchDisaggCommunicator | None = None


def init_disaggregated_topology(server_args) -> None:
    """
    Initialize the disaggregated topology if enabled.
    This creates the DisaggCommunicator and sets up the process groups.
    """
    global _DISAGG_COMMUNICATOR
    logger.info("Initializing Disaggregated Topology...")
    comm = PyTorchDisaggCommunicator()
    comm.initialize_topology(server_args)
    _DISAGG_COMMUNICATOR = comm


def get_disagg_communicator() -> PyTorchDisaggCommunicator:
    return _DISAGG_COMMUNICATOR
