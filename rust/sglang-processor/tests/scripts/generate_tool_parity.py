"""Fill tests/fixtures/tool_parity/*.json with SGLang's tool-call parsing.

Each fixture names a `--tool-call-parser`, the request's tools and chunked
outputs; this records `FunctionCallParser`'s streaming steps (the last chunk
also flushed, as `serving_chat` does at finish) and its one-shot parse.

Run from rust/sglang-processor in a SGLang Python environment:
    python tests/scripts/generate_tool_parity.py
"""

import json
import sys
from pathlib import Path

# Record this checkout's SGLang, not whichever one is installed.
sys.path.insert(0, str(Path(__file__).resolve().parents[4] / "python"))

from sglang.srt.entrypoints.openai.protocol import Tool
from sglang.srt.function_call.function_call_parser import FunctionCallParser

from fixture_json import dump

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures/tool_parity"


def calls(items):
    return [[c.tool_index, c.name, c.parameters] for c in items]


def parse(parser_name, tools, chunks):
    streaming = FunctionCallParser(tools, parser_name)
    steps = []
    for i, chunk in enumerate(chunks):
        normal, found = streaming.parse_stream_chunk(chunk)
        if i == len(chunks) - 1:
            end_normal, end_found = streaming.parse_stream_end()
            normal, found = (normal or "") + end_normal, list(found) + end_found
        steps.append([normal or "", calls(found)])
    normal, found = FunctionCallParser(tools, parser_name).parse_non_stream(
        "".join(chunks)
    )
    return {"steps": steps, "unary": [normal, calls(found)]}


for path in sorted(FIXTURES.glob("*.json")):
    fixture = json.loads(path.read_text())
    tools = [Tool(**tool) for tool in fixture["tools"]]
    fixture["cases"] = [
        {"chunks": chunks, **parse(fixture["parser"], tools, chunks)}
        for chunks in fixture["inputs"]
    ]
    path.write_text(dump(fixture))
