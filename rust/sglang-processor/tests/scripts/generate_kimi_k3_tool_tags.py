"""Fill tests/fixtures/kimi_k3_tool_tags.json with SGLang's Kimi-K3 auto
tool-call structural tags, as `serving_chat` puts them in `sampling_params`.

Each case names request tools, `parallel_tool_calls` and the reasoning
parser's think tokens `strip_structural_tag_excludes` removes; this records
`get_kimik3_auto_tool_call_structural_tag`'s tag as `to_sampling_params`
serializes it.

Run from rust/sglang-processor in a SGLang Python environment:
    python tests/scripts/generate_kimi_k3_tool_tags.py
"""

import json
import sys
from pathlib import Path

# Record this checkout's SGLang, not whichever one is installed.
sys.path.insert(0, str(Path(__file__).resolve().parents[4] / "python"))

from fixture_json import dump

from sglang.srt.entrypoints.openai.protocol import Tool
from sglang.srt.function_call.kimik3_structural_tag import (
    get_kimik3_auto_tool_call_structural_tag,
)
from sglang.srt.function_call.utils import strip_structural_tag_excludes
from sglang.utils import convert_json_schema_to_str

FIXTURE = Path(__file__).resolve().parents[1] / "fixtures/kimi_k3_tool_tags.json"

fixture = json.loads(FIXTURE.read_text())
for case in fixture["cases"]:
    tag = get_kimik3_auto_tool_call_structural_tag(
        [Tool(**tool) for tool in case["tools"]],
        thinking_mode=False,
        parallel_tool_calls=case["parallel_tool_calls"],
    )
    if tag is not None:
        strip_structural_tag_excludes(tag, case["strip"])
        tag = convert_json_schema_to_str(tag.model_dump(by_alias=True))
    case["structural_tag"] = tag
FIXTURE.write_text(dump(fixture))
