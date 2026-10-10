"""Fill tests/fixtures/openai_parity/*.json from SGLang's own OpenAI layer.

Each fixture names the engine's OpenAI `settings` and request `cases`. For every
case this records the `/generate` body Python's OpenAI handler builds, the
engine's `/generate` output for that body (recorded once from `--engine-url`,
except for `synthetic` cases, whose output is written by hand), and the OpenAI
response the handler builds from that output.

Run from rust/sglang-processor in a SGLang Python environment, with `--model`
the checkpoint directory (its tokenizer and config):
    python tests/scripts/generate_openai_parity.py --model DIR \
        [--engine-url http://127.0.0.1:30000] [tests/fixtures/openai_parity/<name>.json ...]
"""

import argparse
import asyncio
import copy
import dataclasses
import json
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

from fixture_json import dump

from sglang.srt.entrypoints.openai.protocol import (
    ChatCompletionRequest,
    CompletionRequest,
)
from sglang.srt.entrypoints.openai.serving_chat import OpenAIServingChat
from sglang.srt.entrypoints.openai.serving_completions import OpenAIServingCompletion
from sglang.srt.managers.io_struct import GenerateReqInput
from sglang.srt.runtime_context import publish, reset_context
from sglang.srt.server_args import ServerArgs
from sglang.srt.utils.hf_transformers.common import get_context_length
from sglang.srt.utils.hf_transformers_utils import get_tokenizer

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures/openai_parity"
HANDLERS = {
    "completions": (CompletionRequest, OpenAIServingCompletion),
    "chat": (ChatCompletionRequest, OpenAIServingChat),
}
SKIPPED_FIELDS = {"received_time"}
# `meta_info` keys the OpenAI layer never reads; `spec_*` feeds only unlowered `sglext`.
UNREAD_META = {
    "e2e_latency",
    "response_sent_to_client_ts",
    "num_retractions",
    "dp_rank",
}
# What each `/generate` stream frame repeats in full; fixtures keep only what a frame adds.
GROWING = {"text", "output_ids", "output_token_logprobs", "output_top_logprobs"}
# The bare request of each endpoint; cases keep only the sampling params they change.
BARE_REQUESTS = {
    "completions": {"prompt": "x"},
    "chat": {"messages": [{"role": "user", "content": "x"}]},
}


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


def without_sampling_defaults(body, sampling_defaults):
    """`body` as stored: only the sampling params a case changes from a bare request's."""
    params = body.get("sampling_params")
    if not isinstance(params, dict):
        return body
    params = {
        k: v
        for k, v in params.items()
        if k not in sampling_defaults or sampling_defaults[k] != v
    }
    return {**body, "sampling_params": params}


def trim_meta(output):
    """Drops `UNREAD_META` and `spec_*` keys from recorded `/generate` output."""
    items = output["frames"] if "frames" in output else output["body"]
    for item in items if isinstance(items, list) else [items]:
        meta = item.get("meta_info", {})
        for key in [k for k in meta if k in UNREAD_META or k.startswith("spec_")]:
            del meta[key]
    return output


def frame_delta(prev, frame):
    """`frame` as stored: `index`, what it appends to `GROWING` and the keys that changed."""
    delta = {}
    for key, value in frame.items():
        if key in GROWING:
            delta[key] = value[len(prev[key]) :]
        elif key == "meta_info":
            delta[key] = frame_delta(prev[key], value)
        elif key == "index" or prev.get(key) != value:
            delta[key] = value
    assert frame_merge(prev, delta) == frame
    return delta


def frame_merge(prev, delta):
    frame = dict(prev)
    for key, value in delta.items():
        if key in GROWING:
            frame[key] = prev[key] + value
        elif key == "meta_info":
            frame[key] = frame_merge(prev[key], value)
        else:
            frame[key] = value
    return frame


def stream_frames(frames, store):
    """Each choice's stream frames as stored (`store`) or as the engine sent them."""
    last, out = {}, []
    for frame in frames:
        prev = last.get(frame.get("index"))
        if prev is not None and "meta_info" in frame:
            out.append(frame_delta(prev, frame) if store else frame_merge(prev, frame))
        else:
            out.append(frame)
        last[frame.get("index")] = frame if store else out[-1]
    return out


def record_frames(engine_url, body):
    response = requests.post(
        f"{engine_url}/generate", json=body, stream=body.get("stream")
    )
    if not body.get("stream") or response.status_code != 200:
        return {"status": response.status_code, "body": response.json()}
    frames = []
    for line in response.iter_lines(decode_unicode=True):
        if line.startswith("data: ") and line != "data: [DONE]":
            frames.append(json.loads(line[len("data: ") :]))
    return {"status": 200, "frames": stream_frames(frames, store=True)}


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
        for frame in json.loads(json.dumps(stream_frames(output["frames"], False))):
            if "error" in frame:
                raise ValueError(frame["error"]["message"])
            yield internal(frame)

    return generate_request


def event_json(event):
    data = event.removeprefix("data: ").strip()
    return data if data == "[DONE]" else json.loads(data)


async def openai_response(handler, request, raw_request):
    response = await handler.handle_request(request, raw_request)
    if isinstance(response, StreamingResponse):
        events = [chunk async for chunk in response.body_iterator]
        return {"events": [event_json(event) for event in events]}
    if isinstance(response, Response):
        return {"status": response.status_code, "body": json.loads(response.body)}
    return {"status": 200, "body": jsonable_encoder(response)}


def lower(case, settings, model):
    """Python's handler for `case`, its raw request and the `GenerateReqInput` it builds."""
    request_type, handler_type = HANDLERS[case["endpoint"]]
    reset_context()
    server_args = ServerArgs(model_path=model.path, **settings)
    publish(server_args, role="tokenizer")
    tokenizer_manager = SimpleNamespace(
        tokenizer=model.tokenizer,
        server_args=server_args,
        model_path=model.path,
        model_config=SimpleNamespace(
            hf_config=model.hf_config,
            is_multimodal=False,
            get_default_sampling_params=lambda: model.sampling_defaults,
        ),
        config_value=lambda name: getattr(server_args, name),
        request_logger=SimpleNamespace(log_requests=False),
        create_abort_task=lambda obj: None,
    )
    # A checkpoint without a Jinja template, as DeepSeek-V4's.
    template_manager = SimpleNamespace(
        completion_template_name=settings.get("completion_template"),
        chat_template_name=None,
        jinja_template_content_format="string",
        jinja_template_may_reorder_tool_results=False,
        reasoning_config=None,
        force_reasoning=False,
    )
    handler = handler_type(tokenizer_manager, template_manager)
    raw_request = Request(
        {
            "type": "http",
            "headers": [
                (k.lower().encode(), v.encode())
                for k, v in case.get("headers", {}).items()
            ],
        }
    )
    request = request_type(**copy.deepcopy(case["request"]))
    if handler._validate_request(request):
        raise ValueError(f"{case['name']}: SGLang rejects this request")
    obj, _ = handler._convert_to_internal_request(request, raw_request)
    return handler, raw_request, obj


def run_case(case, settings, model, engine_url, defaults, sampling_defaults):
    handler, raw_request, obj = lower(case, settings, model)
    body = generate_body(obj, defaults)
    recorded = {"generate": without_sampling_defaults(body, sampling_defaults)}
    if case.get("lower_only"):
        return recorded
    # A bare request's sampling params are not `/generate`'s defaults, so the engine gets them all.
    if engine_url and not case.get("synthetic"):
        case["engine"] = record_frames(engine_url, body)
    recorded["engine"] = trim_meta(case["engine"])
    handler.tokenizer_manager.generate_request = replay(case["engine"])
    request_type, _ = HANDLERS[case["endpoint"]]
    request = request_type(**copy.deepcopy(case["request"]))
    recorded["openai"] = asyncio.run(openai_response(handler, request, raw_request))
    return recorded


def load_model(path):
    config = json.loads(Path(path, "config.json").read_text())
    generation_file = Path(path, "generation_config.json")
    generation = (
        json.loads(generation_file.read_text()) if generation_file.exists() else {}
    )
    keys = ("repetition_penalty", "temperature", "top_k", "top_p", "min_p")
    return SimpleNamespace(
        path=path,
        tokenizer=get_tokenizer(path, trust_remote_code=True),
        hf_config=SimpleNamespace(**config, to_dict=lambda: config),
        text_config=SimpleNamespace(**config.get("text_config", config)),
        sampling_defaults={
            k: generation[k] for k in keys if generation.get(k) is not None
        },
    )


def generate(path, model, engine_url):
    fixture = json.loads(path.read_text())
    # The model facts the host reads, so the test runs without the checkpoint.
    fixture["chat_model"] = {
        "context_length": get_context_length(model.text_config),
        "generation_config": model.sampling_defaults,
    }
    defaults = generate_defaults()
    fixture["generate_defaults"] = {
        name: value for name, value in defaults.items() if name not in SKIPPED_FIELDS
    }
    settings = fixture.get("settings", {})
    fixture["sampling_defaults"] = {
        endpoint: lower({"endpoint": endpoint, "request": bare}, settings, model)[
            2
        ].sampling_params
        for endpoint, bare in BARE_REQUESTS.items()
    }
    for case in fixture["cases"]:
        if case.get("unsupported"):
            continue
        sampling_defaults = fixture["sampling_defaults"][case["endpoint"]]
        case.update(
            run_case(case, settings, model, engine_url, defaults, sampling_defaults)
        )
        print(f"{path.name}: {case['name']}", file=sys.stderr)
    path.write_text(dump(fixture))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--engine-url")
    parser.add_argument("fixtures", nargs="*", type=Path)
    args = parser.parse_args()
    model = load_model(args.model)
    for path in args.fixtures or sorted(FIXTURES.glob("*.json")):
        generate(path, model, args.engine_url)


if __name__ == "__main__":
    main()
