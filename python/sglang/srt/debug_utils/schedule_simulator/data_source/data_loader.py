import json
from pathlib import Path
from typing import List, Union

from sglang.srt.debug_utils.schedule_simulator.request import SimRequest


def load_from_request_logger(file_path: Union[str, Path]) -> List[SimRequest]:
    requests = []
    file_path = Path(file_path)

    with file_path.open(encoding="utf-8") as f:
        for line_num, line in enumerate(f):
            line = line.strip()
            if not line or not line.startswith("{"):
                continue

            data = json.loads(line)

            if data.get("event") != "request.finished":
                continue

            rid = data.get("rid", f"req_{line_num}")
            meta_info = data["out"]["meta_info"]

            requests.append(
                SimRequest(
                    request_id=rid,
                    input_len=meta_info["prompt_tokens"],
                    output_len=meta_info["completion_tokens"],
                )
            )

    return requests
