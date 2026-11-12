"""Measure multi-turn TTFT through a Dynamo SGLang PD deployment.

The client forces every decode step to emit one tokenizer-round-trip-stable
token. This makes the assistant text in the next turn reproduce the exact
generated token IDs, which is required for reverse KV to become a radix hit.
Restart the deployment between disabled and enabled runs so both start cold.
"""

import argparse
import asyncio
import json
import time
import unicodedata
from pathlib import Path

import aiohttp
import numpy as np
from transformers import AutoTokenizer


def find_round_trip_token(tokenizer, repeat: int) -> tuple[int, str]:
    special_ids = set(tokenizer.all_special_ids)
    for token_id in range(tokenizer.vocab_size):
        if token_id in special_ids:
            continue
        text = tokenizer.decode(
            [token_id],
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )
        if (
            not text
            or not text.strip()
            or len(text) > 16
            or "\ufffd" in text
            or "\n" in text
            or "\r" in text
            or any(unicodedata.category(char).startswith("C") for char in text)
        ):
            continue
        if tokenizer.encode(text, add_special_tokens=False) != [token_id]:
            continue
        if tokenizer.encode(text * 2, add_special_tokens=False) != [token_id] * 2:
            continue
        if (
            tokenizer.encode(text * repeat, add_special_tokens=False)
            == [token_id] * repeat
        ):
            return token_id, text
    raise RuntimeError("No tokenizer-round-trip-stable non-special token found")


def longest_run(values: list[int], target: int) -> int:
    best = current = 0
    for value in values:
        if value == target:
            current += 1
            best = max(best, current)
        else:
            current = 0
    return best


async def stream_turn(session, url, body):
    start = time.perf_counter()
    first_token_at = None
    content = []
    last_payload = None
    async with session.post(f"{url}/v1/chat/completions", json=body) as response:
        if response.status != 200:
            text = await response.text()
            raise RuntimeError(f"HTTP {response.status}: {text}")
        async for raw_line in response.content:
            line = raw_line.strip()
            if not line.startswith(b"data:"):
                continue
            data = line[5:].strip()
            if not data or data == b"[DONE]":
                continue
            payload = json.loads(data)
            last_payload = payload
            choices = payload.get("choices") or []
            if not choices:
                continue
            delta = choices[0].get("delta") or {}
            piece = delta.get("content") or delta.get("reasoning_content") or ""
            if piece:
                if first_token_at is None:
                    first_token_at = time.perf_counter()
                content.append(piece)
    if first_token_at is None:
        raise RuntimeError(f"Streaming response contained no text: {last_payload}")
    end = time.perf_counter()
    return {
        "ttft_s": first_token_at - start,
        "elapsed_s": end - start,
        "text": "".join(content),
        "last_payload": last_payload,
    }


async def run_conversation(args, session, tokenizer, token_id, index):
    marker = f"reverse-kv-{index}"
    messages = [
        {
            "role": "user",
            "content": (
                f"{marker}. Preserve this identifier. "
                + " ".join(f"context-{index}-{i}" for i in range(args.prompt_words))
            ),
        }
    ]
    records = []
    expected_output_ids = [token_id] * args.output_tokens

    for turn in range(args.turns):
        body = {
            "model": args.model,
            "messages": messages,
            "stream": True,
            "temperature": 0,
            "max_tokens": args.output_tokens,
            "ignore_eos": True,
            "logit_bias": {str(token_id): 100.0},
            "chat_template_kwargs": {"enable_thinking": False},
            "user": marker,
        }
        prompt_ids = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        result = await stream_turn(session=session, url=args.url, body=body)
        output_ids = tokenizer.encode(result["text"], add_special_tokens=False)
        if output_ids != expected_output_ids:
            raise RuntimeError(
                f"Turn {turn} output did not round-trip to the forced token: "
                f"expected {args.output_tokens} copies of {token_id}, "
                f"got {output_ids[:16]} (len={len(output_ids)})"
            )

        records.append(
            {
                "conversation": index,
                "turn": turn,
                "prompt_tokens": len(prompt_ids),
                "output_tokens": len(output_ids),
                "ttft_ms": result["ttft_s"] * 1000,
                "elapsed_ms": result["elapsed_s"] * 1000,
                "nvext": (result["last_payload"] or {}).get("nvext"),
            }
        )
        messages.append({"role": "assistant", "content": result["text"]})
        if turn + 1 < args.turns:
            messages.append(
                {
                    "role": "user",
                    "content": f"Continue {marker} at turn {turn + 1}.",
                }
            )
            next_prompt_ids = tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                enable_thinking=False,
            )
            if (
                longest_run(values=next_prompt_ids, target=token_id)
                < args.output_tokens
            ):
                raise RuntimeError(
                    "The next-turn chat template did not preserve the generated "
                    "assistant token IDs; this workload cannot validate reverse KV"
                )
            await asyncio.sleep(args.inter_turn_delay)
    return records


async def main(args):
    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer or args.model,
        trust_remote_code=args.trust_remote_code,
    )
    if args.forced_token_id is None:
        token_id, token_text = find_round_trip_token(
            tokenizer=tokenizer, repeat=args.output_tokens
        )
    else:
        token_id = args.forced_token_id
        token_text = tokenizer.decode(
            [token_id],
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )
        if (
            tokenizer.encode(token_text * args.output_tokens, add_special_tokens=False)
            != [token_id] * args.output_tokens
        ):
            raise ValueError("--forced-token-id is not repetition-stable")

    timeout = aiohttp.ClientTimeout(total=args.request_timeout)
    connector = aiohttp.TCPConnector(limit=args.conversations)
    async with aiohttp.ClientSession(timeout=timeout, connector=connector) as session:
        conversations = await asyncio.gather(
            *(
                run_conversation(
                    args=args,
                    session=session,
                    tokenizer=tokenizer,
                    token_id=token_id,
                    index=index,
                )
                for index in range(args.conversations)
            )
        )

    records = [record for conversation in conversations for record in conversation]
    measured = [record for record in records if record["turn"] > 0]
    ttfts = [record["ttft_ms"] for record in measured]
    summary = {
        "label": args.label,
        "args": vars(args),
        "forced_token_id": token_id,
        "forced_token_text": token_text,
        "measured_turns": len(measured),
        "ttft_mean_ms": float(np.mean(ttfts)),
        "ttft_p50_ms": float(np.percentile(ttfts, 50)),
        "ttft_p95_ms": float(np.percentile(ttfts, 95)),
        "records": records,
    }
    Path(args.result).write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(
        json.dumps({key: value for key, value in summary.items() if key != "records"}),
        flush=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    parser.add_argument("--model", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--tokenizer")
    parser.add_argument("--label", choices=("disabled", "enabled"), required=True)
    parser.add_argument("--turns", type=int, default=4)
    parser.add_argument("--conversations", type=int, default=1)
    parser.add_argument("--prompt-words", type=int, default=32)
    parser.add_argument("--output-tokens", type=int, default=256)
    parser.add_argument("--inter-turn-delay", type=float, default=1.0)
    parser.add_argument("--request-timeout", type=float, default=1800)
    parser.add_argument("--forced-token-id", type=int)
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--result", required=True)
    args = parser.parse_args()
    if (
        min(
            args.turns,
            args.conversations,
            args.prompt_words,
            args.output_tokens,
        )
        <= 0
    ):
        parser.error(
            "--turns, --conversations, --prompt-words, and --output-tokens "
            "must be positive"
        )
    if args.turns < 2:
        parser.error("--turns must be at least 2 to measure returned KV")
    if args.inter_turn_delay < 0:
        parser.error("--inter-turn-delay must be non-negative")
    asyncio.run(main(args))
