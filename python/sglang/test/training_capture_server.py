"""Test-only server observer for KV snapshot parity, including spawned workers."""

import os
import sys

from sglang.srt.debug_utils import tensor_dump_forward_hook
from sglang.srt.layers.logits_processor import LogitsProcessor
from sglang.srt.layers.radix_attention import RadixAttention
from sglang.srt.model_executor.forward_batch_info import ForwardBatch


class CaptureReferenceDumper(tensor_dump_forward_hook.TensorDumper):
    def _dump_hook(self, tensor_name, do_dump):
        # Flush after logits_processor has produced raw scores and before
        # the caller's serving sampler can mutate them. ModelRunner invokes
        # ForCausalLM.forward directly, so a root module hook would not run.
        output_hook = super()._dump_hook(tensor_name, False)

        def observe(module, inputs, output):
            if do_dump:
                for value in inputs:
                    if isinstance(value, ForwardBatch):
                        self.add_tensor(tensor_name, value)
            if isinstance(module, RadixAttention):
                # Each attention layer is distinct; RotaryEmbedding instances
                # can be shared across layers and their hooks overwrite keys.
                self.add_tensor(tensor_name + ".input_k", inputs[1])
                self.add_tensor(tensor_name + ".input_v", inputs[2])
            output_hook(module, inputs, output)
            if isinstance(module, LogitsProcessor):
                self.dump_current_tensors()

        return observe


# multiprocessing spawn imports this module before constructing the worker.
# The observer is confined to this test entrypoint; normal serving is untouched.
tensor_dump_forward_hook.TensorDumper = CaptureReferenceDumper


if __name__ == "__main__":
    from sglang.launch_server import run_server
    from sglang.srt.server_args import prepare_server_args
    from sglang.srt.utils import kill_process_tree

    try:
        run_server(prepare_server_args(sys.argv[1:]))
    finally:
        kill_process_tree(os.getpid(), include_parent=False)
