"""SGLang --forward-hooks factory for observing real MoE input layouts.

Use only during a separate layout-observation run; exclude its file writes from
reported serving performance. One record is written per distinct observed shape.
"""

import json
from pathlib import Path

import torch.distributed as dist


def make_hook(config):
    directory = Path(config["output_dir"])
    directory.mkdir(parents=True, exist_ok=True)
    seen = set()

    def hook(module, args, output):
        hidden, batch = args[:2]
        phase = batch.forward_mode.name
        global_tokens = getattr(batch, "global_num_token_non_padded_cpu", None)
        label_path = directory / "request_shape.json"
        request_shape = (
            json.loads(label_path.read_text()) if label_path.exists() else {}
        )
        input_length = request_shape.get("input_length")
        key = (input_length, phase, hidden.shape[0], global_tokens, batch.batch_size)
        if key in seen:
            return
        seen.add(key)
        record = dict(
            request_input_length=input_length,
            rank=dist.get_rank(),
            phase=phase,
            local_tokens=hidden.shape[0],
            global_tokens=global_tokens,
            batch_size=batch.batch_size,
            extend_num_tokens=batch.extend_num_tokens,
        )
        with (directory / f"rank{dist.get_rank()}.jsonl").open("a") as stream:
            stream.write(json.dumps(record) + "\n")

    return hook
