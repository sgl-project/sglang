"""Text chat render endpoint for llm-d's HTTP token-producer."""

import argparse
import asyncio
import json
from pathlib import Path

from aiohttp import web

from . import ChatProcessor


def _text_message(message):
    if not isinstance(message, dict) or message.get("role") not in {
        "system",
        "user",
        "assistant",
    }:
        return False
    if message.keys() - {"role", "content"}:
        return False
    content = message.get("content")
    return isinstance(content, str) or (
        isinstance(content, list)
        and all(
            isinstance(part, dict)
            and part.get("type") == "text"
            and isinstance(part.get("text"), str)
            for part in content
        )
    )


def create_app(
    model_dir, model_name, chat_template=None, defaults=None, max_body_bytes=8 << 20
):
    if defaults is not None and not isinstance(defaults, dict):
        raise ValueError("default chat template kwargs must be an object")
    if defaults and "chat_template" in defaults:
        raise ValueError("select the renderer template with --chat-template")
    processor = ChatProcessor(
        str(model_dir), chat_template, default_kwargs_json=json.dumps(defaults or {})
    )

    async def render(request):
        try:
            raw = await request.read()
            body = json.loads(raw)
            if not isinstance(body, dict):
                raise ValueError("expected a Chat request object")
            if body.get("model") != model_name:
                raise web.HTTPNotFound(text="unknown model")
            messages = body.get("messages")
            if (
                not isinstance(messages, list)
                or not messages
                or not all(map(_text_message, messages))
            ):
                raise ValueError(
                    "expected plain-text system, user or assistant messages"
                )
            if (
                body.get("n") not in (None, 1)
                or body.get("tools")
                or body.get("tool_choice") not in (None, "none", "auto")
            ):
                raise ValueError("render supports single-candidate text chat")
            transfer = body.get("kv_transfer_params")
            if transfer is not None and not isinstance(transfer, dict):
                raise ValueError("kv_transfer_params must be an object")
            if body.get("input_ids") is not None or (
                transfer and "prompt_token_ids" in transfer
            ):
                raise ValueError(
                    "render expects messages without precomputed prompt IDs"
                )
            kwargs = body.get("chat_template_kwargs")
            if kwargs is not None and not isinstance(kwargs, dict):
                raise ValueError("chat_template_kwargs must be an object")
            if kwargs and "chat_template" in kwargs:
                raise ValueError("request template overrides are unsupported")
            ids = await asyncio.to_thread(processor.encode_chat, raw)
        except (TypeError, ValueError) as error:
            raise web.HTTPBadRequest(text=str(error)) from error
        return web.json_response({"token_ids": ids})

    async def health(_request):
        return web.Response(text="ok")

    app = web.Application(client_max_size=max_body_bytes)
    app.router.add_get("/health", health)
    app.router.add_post("/v1/chat/completions/render", render)
    return app


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument(
        "--model-name", required=True, help="Model name sent by the EPP"
    )
    parser.add_argument("--chat-template")
    parser.add_argument("--default-chat-template-kwargs", type=json.loads, default={})
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8090)
    parser.add_argument("--max-body-bytes", type=int, default=8 << 20)
    args = parser.parse_args()
    if (
        not isinstance(args.default_chat_template_kwargs, dict)
        or args.max_body_bytes <= 0
    ):
        parser.error(
            "expected an object for template defaults and a positive body size"
        )
    app = create_app(
        args.model_dir,
        args.model_name,
        args.chat_template,
        args.default_chat_template_kwargs,
        args.max_body_bytes,
    )
    web.run_app(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
