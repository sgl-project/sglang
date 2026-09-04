"""NPU graph runner for linear UNO speculative decoding."""

from sglang.srt.hardware_backend.npu.graph_runner.npu_graph_runner import NPUGraphRunner
from sglang.srt.speculative.uno_cuda_graph_runner import UnoDecodeCudaGraphRunner


class UnoNPUGraphRunner(UnoDecodeCudaGraphRunner, NPUGraphRunner):
    """Combine UNO's two LoRA variants with the native NPU graph backend.

    ``UnoDecodeCudaGraphRunner`` owns UNO-specific capture metadata and graph
    keys, while ``NPUGraphRunner`` supplies NPUGraph capture/replay and Ascend
    sequence-length updates.  Linear mode is enforced during argument
    validation, so no CUDA-only tree path is reachable on NPU.
    """

    pass
