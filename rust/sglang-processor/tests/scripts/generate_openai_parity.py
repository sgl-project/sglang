"""Fill tests/fixtures/openai_parity/*.json from SGLang's own OpenAI layer.

Each fixture names the engine's OpenAI `settings` and request `cases`. For every
case this records the `/generate` body Python's OpenAI handler builds, the
engine's `/generate` output for that body (recorded once from `--engine-url`),
and the OpenAI response the handler builds from that output.

Run from rust/sglang-processor in a SGLang Python environment:
    python tests/scripts/generate_openai_parity.py --tokenizer DIR \
        [--engine-url http://127.0.0.1:30000] [tests/fixtures/openai_parity/<name>.json ...]
"""

import argparse
import asyncio
import dataclasses
import json
import re
import sys
from http import HTTPStatus
from pathlib import Path
from types import SimpleNamespace

import requests
from fastapi.encoders import jsonable_encoder
from fastapi.responses import StreamingResponse
from starlette.requests import Request
from starlette.responses import Response

# Record this checkout's SGLang, not whichever one is installed.
sys.path.insert(0, str(Path(__file__).resolve().parents[4] / "python"))

from sglang.srt.entrypoints.openai.protocol import CompletionRequest
from sglang.srt.entrypoints.openai.serving_completions import OpenAIServingCompletion
from sglang.srt.managers.io_struct import GenerateReqInput
from sglang.srt.runtime_context import publish, reset_context
from sglang.srt.server_args import ServerArgs
from sglang.srt.utils.hf_transformers_utils import get_tokenizer

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures/openai_parity"
HANDLERS = {"completions": (CompletionRequest, OpenAIServingCompletion)}
SKIPPED_FIELDS = {"received_time"}


def generate_defaults():
    defaults = {}
    for field in dataclasses.fields(GenerateReqInput):
        if field.default is not dataclasses.MISSING:
            defaults[field.name] = field.default
        elif field.default_factory is not dataclasses.MISSING:
            defaults[field.name] = field.default_factory()
    return defaults


def generate_body(obj, defaults):
    """The fields Python set away from their defaults, as `/generate` JSON."""
    return {
        name: getattr(obj, name)
        for name in defaults
        if name not in SKIPPED_FIELDS and getattr(obj, name) != defaults[name]
    }


def record_frames(engine_url, body):
    response = requests.post(f"{engine_url}/generate", json=body, stream=body.get("stream"))
    if not body.get("stream") or response.status_code != 200:
        return {"status": response.status_code, "body": response.json()}
    frames = []
    for line in response.iter_lines(decode_unicode=True):
        if line.startswith("data: ") and line != "data: [DONE]":
            frames.append(json.loads(line[len("data: "):]))
    return {"status": 200, "frames": frames}


def replay(output):
    """`generate_request` as the OpenAI handler sees the recorded `/generate` output."""

    def internal(output):
        for frame in output if isinstance(output, list) else [output]:
            reason = frame.get("meta_info", {}).get("finish_reason") or {}
            if isinstance(reason.get("status_code"), int):
                reason["status_code"] = HTTPStatus(reason["status_code"])
        return output

    async def generate_request(obj, raw_request=None):
        if "frames" not in output:
            body = output["body"]
            if output["status"] != 200:
                raise ValueError(body["error"]["message"])
            yield internal(json.loads(json.dumps(body)))
            return
        for frame in json.loads(json.dumps(output["frames"])):
            if "error" in frame:
                raise ValueError(frame["error"]["message"])
            yield internal(frame)

    return generate_request


async def openai_response(handler, request, raw_request):
    response = await handler.handle_request(request, raw_request)
    if isinstance(response, StreamingResponse):
        events = [chunk async for chunk in response.body_iterator]
        return {"events": events}
    if isinstance(response, Response):
        return {"status": response.status_code, "body": json.loads(response.body)}
    return {"status": 200, "body": jsonable_encoder(response)}


def run_case(case, settings, tokenizer, engine_url, defaults):
    request_type, handler_type = HANDLERS[case["endpoint"]]
    reset_context()
    server_args = ServerArgs(model_path="dummy", **settings)
    publish(server_args, role="tokenizer")
    recorded = {}
    tokenizer_manager = SimpleNamespace(
        tokenizer=tokenizer,
        server_args=server_args,
        request_logger=SimpleNamespace(log_requests=False),
        create_abort_task=lambda obj: None,
    )
    template_manager = SimpleNamespace(completion_template_name=settings.get("completion_template"))
    handler = handler_type(tokenizer_manager, template_manager)
    raw_request = Request(
        {
            "type": "http",
            "headers": [(k.lower().encode(), v.encode()) for k, v in case.get("headers", {}).items()],
        }
    )
    request = request_type(**case["request"])
    obj, _ = handler._convert_to_internal_request(request, raw_request)
    recorded["generate"] = generate_body(obj, defaults)
    if case.get("lower_only"):
        return recorded
    if engine_url:
        case["engine"] = record_frames(engine_url, recorded["generate"])
    tokenizer_manager.generate_request = replay(case["engine"])
    recorded["engine"] = case["engine"]
    recorded["openai"] = asyncio.run(openai_response(handler, request, raw_request))
    return recorded


def generate(path, tokenizer, engine_url):
    fixture = json.loads(path.read_text())
    defaults = generate_defaults()
    fixture["generate_defaults"] = {
        name: value for name, value in defaults.items() if name not in SKIPPED_FIELDS
    }
    for case in fixture["cases"]:
        if case.get("unsupported"):
            continue
        case.update(run_case(case, fixture.get("settings", {}), tokenizer, engine_url, defaults))
        print(f"{path.name}: {case['name']}", file=sys.stderr)
    path.write_text(dump(fixture))


def dump(fixture):
    """Indented JSON, with each list of numbers on one line."""
    text = json.dumps(fixture, indent=2, ensure_ascii=False)
    join = lambda m: "[" + re.sub(r",\n\s+", ", ", m.group(1)) + "]"
    return re.sub(r"\[\n\s+([^\[\]{}\"]*?)\n\s+\]", join, text) + "\n"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--engine-url")
    parser.add_argument("fixtures", nargs="*", type=Path)
    args = parser.parse_args()
    tokenizer = get_tokenizer(args.tokenizer, trust_remote_code=True)
    for path in args.fixtures or sorted(FIXTURES.glob("*.json")):
        generate(path, tokenizer, args.engine_url)


if __name__ == "__main__":
    main()
