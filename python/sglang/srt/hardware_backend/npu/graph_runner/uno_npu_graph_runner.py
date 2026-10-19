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

    def capture_prepare(self, size, stream_idx=None, num_tokens=None):
        # Ascend metadata prep aliases the input seq_lens and adds the query
        # width in place. Reset before EACH bucket/LoRA variant so capture
        # always starts from the dummy prefix, rather than the previous
        # variant's final KV length. Replay still uses the normal padding ABI.
        self.buffers.seq_lens.fill_(self.seq_len_fill_value)
        self.buffers.seq_lens_cpu.fill_(self.seq_len_fill_value)
        return super().capture_prepare(
            size, stream_idx=stream_idx, num_tokens=num_tokens
        )
