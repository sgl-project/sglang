"""Launch a PD worker and record its metadata-buffer shape and device once.

Use launch_worker.sh, or set PD_TOP_LOGPROBS_OUTPUT_DIR and pass the normal
sglang.launch_server arguments to this script. The constructor wrapper only
reads host metadata after the original initialization has completed.
"""

import functools
import json
import os
import sys
from pathlib import Path

from sglang.srt.disaggregation.utils import MetadataBuffers
from sglang.srt.runtime_context import get_disagg

original_init = MetadataBuffers.__init__


@functools.wraps(original_init)
def record_metadata_buffers(self, *args, **kwargs):
    original_init(self, *args, **kwargs)
    role = get_disagg().disaggregation_mode
    record = {
        "role": role,
        "pid": os.getpid(),
        "visible_npus": os.environ.get("ASCEND_RT_VISIBLE_DEVICES"),
        "values_shape": list(self.output_top_logprobs_val.shape),
        "indices_shape": list(self.output_top_logprobs_idx.shape),
        "device": str(self.output_top_logprobs_val.device),
        "dtype_values": str(self.output_top_logprobs_val.dtype),
        "dtype_indices": str(self.output_top_logprobs_idx.dtype),
        "extra_device_readback": False,
        "extra_forward_synchronization": False,
    }
    output = (
        Path(os.environ.get("PD_TOP_LOGPROBS_OUTPUT_DIR", "pd-top-logprobs-results"))
        .expanduser()
        .resolve()
        / "evidence"
    )
    output.mkdir(parents=True, exist_ok=True)
    path = output / f"runtime_buffers_{role}_{os.getpid()}.json"
    path.write_text(json.dumps(record, indent=2) + "\n")


# Install in both the launcher and multiprocessing spawn children.
MetadataBuffers.__init__ = record_metadata_buffers

if __name__ == "__main__":
    from sglang.launch_server import run_server
    from sglang.srt.plugins import load_plugins
    from sglang.srt.server_args import prepare_server_args
    from sglang.srt.utils import kill_process_tree

    load_plugins()
    try:
        run_server(prepare_server_args(sys.argv[1:]))
    finally:
        kill_process_tree(os.getpid(), include_parent=False)
