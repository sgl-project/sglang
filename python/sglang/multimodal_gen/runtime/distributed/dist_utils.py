from sglang.multimodal_gen.runtime.communication.pytorch_communicator import (
    PyTorchDisaggCommunicator,
)
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger

logger = init_logger(__name__)

_DISAGG_COMMUNICATOR = None


def init_disaggregated_topology(server_args) -> None:
    """
    Initialize the disaggregated topology if enabled.
    This creates the DisaggCommunicator and sets up the process groups.
    """
    global _DISAGG_COMMUNICATOR
    if server_args.enable_disagg:
        # Auto-enable disaggregation logic if user didn't explicitly set flag
        # but environment suggests it?
        # For now, rely on explicit flag or the "num_gpus > 1 and odd" logic if we want to auto-detect.
        pass

    # We always initialize the communicator if we are in a distributed setting that *could* support it.
    # But strictly speaking, we only need it if enable_disagg is True.
    # To keep it simple, we initialize it if enabled.

    if server_args.enable_disagg:
        logger.info("Initializing Disaggregated Topology...")
        comm = PyTorchDisaggCommunicator()
        comm.initialize_topology(server_args)
        _DISAGG_COMMUNICATOR = comm


def get_disagg_communicator() -> PyTorchDisaggCommunicator:
    return _DISAGG_COMMUNICATOR
