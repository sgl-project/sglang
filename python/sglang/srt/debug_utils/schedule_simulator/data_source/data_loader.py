import json
from pathlib import Path
from typing import List, Union

from sglang.srt.debug_utils.schedule_simulator.request import SimRequest


def load_from_request_logger(file_path: Union[str, Path]) -> List[SimRequest]:
    requests = []
    file_path = Path(file_path)

    with open(file_path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f):
            line = line.strip()
            if not line or not line.startswith("{"):
                continue

            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                continue

            if data.get("event") != "request.finished":
                continue

            rid = data.get("rid", f"req_{line_num}")
            out = data.get("out", {})
            meta_info = out.get("meta_info", {})

            prompt_tokens = meta_info.get("prompt_tokens")
            completion_tokens = meta_info.get("completion_tokens")

            if prompt_tokens is None or completion_tokens is None:
                continue

            requests.append(
                SimRequest(
                    request_id=rid,
                    input_len=prompt_tokens,
                    output_len=completion_tokens,
                )
            )

    return requests
