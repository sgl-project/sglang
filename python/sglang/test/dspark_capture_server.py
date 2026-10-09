"""Test entrypoint observing raw compact rows and actual acceptance decisions."""

import json
import os
import sys
from pathlib import Path

from sglang.srt.distributed import get_tensor_model_parallel_rank
from sglang.srt.runtime_context import get_spec
from sglang.srt.speculative.dspark_components.dspark_verify import TargetVerifyExecutor
from sglang.test.dspark_capture_observer import install_capture_observer

_accept = TargetVerifyExecutor.accept_and_finalize


def observed_accept(self, **kwargs):
    result = _accept(self, **kwargs)
    layout = kwargs["layout"]
    rank = get_tensor_model_parallel_rank()
    path = Path(get_spec().speculative_draft_model_path) / f"acceptance-tp{rank}.jsonl"
    with path.open("a") as stream:
        stream.write(
            json.dumps(
                {
                    "tp_rank": rank,
                    "folded": kwargs["folded_accept"],
                    "verify_lens": layout.verify_lens.tolist()
                    if layout is not None
                    else None,
                    "commit_lens": result.commit_lens.tolist(),
                    "cap_trim_lens": result.cap_trim_lens.tolist(),
                }
            )
            + "\n"
        )
    return result


install_capture_observer()
TargetVerifyExecutor.accept_and_finalize = observed_accept

if __name__ == "__main__":
    from sglang.launch_server import run_server
    from sglang.srt.server_args import prepare_server_args
    from sglang.srt.utils import kill_process_tree

    try:
        run_server(prepare_server_args(sys.argv[1:]))
    finally:
        kill_process_tree(os.getpid(), include_parent=False)
