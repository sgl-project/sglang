"""Fill tests/fixtures/reasoning_parity/*.json with SGLang's reasoning splits.

Each fixture names a `--reasoning-parser` and chunked outputs; this records
`ReasoningParser`'s streaming steps and one-shot split for every combination of
forced reasoning and `stream_reasoning`.

Run from rust/sglang-processor in a SGLang Python environment:
    python tests/scripts/generate_reasoning_parity.py
"""

import json
import sys
from pathlib import Path

# Record this checkout's SGLang, not whichever one is installed.
sys.path.insert(0, str(Path(__file__).resolve().parents[4] / "python"))

from sglang.srt.parser.reasoning_parser import ReasoningParser

from fixture_json import dump

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures/reasoning_parity"


def split(parser, force, stream, chunks):
    streaming = ReasoningParser(parser, stream_reasoning=stream, force_reasoning=force)
    steps = [list(streaming.parse_stream_chunk(chunk)) for chunk in chunks]
    steps.append(list(streaming.parse_stream_end()))
    unary = ReasoningParser(parser, stream_reasoning=False, force_reasoning=force)
    whole = unary.parse_non_stream("".join(chunks))
    as_text = lambda pair: [part or "" for part in pair]
    return {"steps": [as_text(step) for step in steps], "unary": as_text(whole)}


for path in sorted(FIXTURES.glob("*.json")):
    fixture = json.loads(path.read_text())
    fixture["cases"] = [
        {
            "input": index,
            "force": force,
            "stream": stream,
            **split(fixture["parser"], force, stream, case["chunks"]),
        }
        for index, case in enumerate(fixture["inputs"])
        for force in (True, False)
        for stream in (True, False)
    ]
    path.write_text(dump(fixture))
