"""Run the StartLux-Decision protocol — thought loop included — on an SGLang server.

The checkpoint's own package owns the whole protocol: the prompt format, the letter
read-out, the per-type temperature, the "None of the above" candidate, and the
AutoThink loop that writes a thought and mixes a second read-out. It needs only two
primitives from an engine, so this file implements them on SGLang and the package
does the rest. `/v1/systemone` serves the one-pass read-out natively; this is for the
thought loop, which needs a generation.

    pip install "startlux-decide"
    python examples/startlux/sglang_backend.py \
        --model-path startlux-models/StartLux-Decision-v1.1-27B-AutoThink \
        --url http://127.0.0.1:30000 --gate 0.9

Start the server first, for example:

    python -m sglang.launch_server \
        --model-path startlux-models/StartLux-Decision-v1.1-27B-AutoThink --tp 2
"""

from __future__ import annotations

import argparse
import asyncio
import json
from typing import Any, Sequence

import aiohttp
from startlux_decide import modelcfg, pipeline
from startlux_decide.backends.base import (
    Backend,
    BackendError,
    BackendRequestError,
    Generation,
    Sampling,
)
from startlux_decide.controls import Controls

REQUEST = {
    "state": {
        "ticket": "I was charged twice for order #4411 and the app still shows it as unpaid.",
    },
    "questions": {
        "team": {
            "type": "choice",
            "instructions": "Which team should handle this ticket?",
            "criteria": {
                "billing": "Payments and refunds",
                "technical": "App and login",
            },
        },
        "urgent": {
            "type": "noul",
            "instructions": "Should this ticket be answered today?",
        },
    },
}


class SGLangBackend(Backend):
    """The two primitives the protocol layer needs, over one SGLang server."""

    name = "sglang"
    reads_images = False

    def __init__(
        self,
        url: str,
        tokenizer: Any,
        model: str = "default",
        timeout: float = 1800.0,
        context_length: int | None = None,
    ):
        self.url = url.rstrip("/")
        self.tokenizer = tokenizer
        self.model = model
        self.timeout = timeout
        self.context_length = context_length
        self._session: aiohttp.ClientSession | None = None

    async def start(self) -> None:
        self._session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=self.timeout),
            connector=aiohttp.TCPConnector(limit=256),
        )
        async with self._session.get(f"{self.url}/health") as response:
            if response.status != 200:
                raise BackendError(f"{self.url}: /health answered {response.status}")

    async def close(self) -> None:
        if self._session is not None:
            await self._session.close()
            self._session = None

    def describe(self) -> dict:
        return {"name": self.name, "url": self.url}

    async def _post(self, path: str, body: dict) -> dict:
        if self._session is None:
            raise BackendError("the backend is not started")
        try:
            async with self._session.post(f"{self.url}{path}", json=body) as response:
                if response.status == 200:
                    return await response.json()
                text = await response.text()
                if response.status in (400, 413, 422):
                    raise BackendRequestError(text[:300])
                raise BackendError(f"{path} answered {response.status}: {text[:200]}")
        except (aiohttp.ClientError, asyncio.TimeoutError) as error:
            raise BackendError(f"{path}: {type(error).__name__}: {error}") from error

    async def readout(self, prompt: str, letter_ids: Sequence[int]) -> list[float]:
        """Next-token log-probabilities of the answer letters after `prompt`.

        SGLang scores an arbitrary label set only through `/v1/score`, which compares a
        query against items; a plain prompt has no such call, so this reads the last
        prompt position's top log-probabilities from `/generate` instead. A letter
        outside that window raises rather than being read as -inf, which would silently
        change the answer.
        """
        ids = self.tokenizer.encode(prompt, add_special_tokens=False)
        body = {
            "model": self.model,
            "input_ids": ids,
            "sampling_params": {
                "max_new_tokens": 1,
                "temperature": 1.0,
                "top_k": 1,
                "top_p": 1.0,
            },
            "return_logprob": True,
            "top_logprobs_num": max(64, len(letter_ids) * 2),
            "logprob_start_len": len(ids) - 1,
        }
        meta = (await self._post("/generate", body)).get("meta_info") or {}
        top = meta.get("output_top_logprobs") or []
        if not top or not top[0]:
            raise BackendError("/generate returned no next-token logprobs")
        # Each entry is [logprob, token_id, token_text].
        by_id = {int(entry[1]): float(entry[0]) for entry in top[0] if len(entry) >= 2}
        missing = [token for token in letter_ids if token not in by_id]
        if missing:
            raise BackendError(
                f"the next-token window is missing answer letters {missing}; the "
                "engine does not expose enough log-probabilities"
            )
        return [by_id[token] for token in letter_ids]

    async def generate(
        self,
        prompt: str,
        max_tokens: int,
        sampling: Sampling,
        stop: Sequence[str],
        seed: int,
    ) -> Generation:
        """A sampled continuation of `prompt`, which is one thought."""
        ids = self.tokenizer.encode(prompt, add_special_tokens=False)
        body = {
            "model": self.model,
            "input_ids": ids,
            "sampling_params": {
                "max_new_tokens": max_tokens,
                "temperature": sampling.temperature,
                "top_p": sampling.top_p,
                "top_k": sampling.top_k,
                "stop": list(stop),
                # SGLang names the seed sampling_seed, not seed.
                "sampling_seed": int(seed),
                "skip_special_tokens": False,
            },
        }
        answer = await self._post("/generate", body)
        meta = answer.get("meta_info") or {}
        finish = meta.get("finish_reason") or {}
        reason = finish.get("type") if isinstance(finish, dict) else finish
        return Generation(
            text=answer.get("text") or "",
            tokens=int(meta.get("completion_tokens") or 0),
            finish_reason="stop" if reason == "stop" else "length",
        )


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True, help="the checkpoint folder")
    parser.add_argument("--url", default="http://127.0.0.1:30000")
    parser.add_argument("--gate", type=float, default=0.6)
    parser.add_argument(
        "--think-mode", default="auto", choices=("auto", "never", "always")
    )
    args = parser.parse_args()

    folder = modelcfg.load_model_folder(args.model_path)
    tokenizer = modelcfg.load_tokenizer(args.model_path)
    backend = SGLangBackend(args.url, tokenizer)
    await backend.start()
    try:
        decider = pipeline.Decider(folder, backend)
        controls = Controls(
            think_mode=args.think_mode,
            gate=args.gate,
            none_check=True,
            details=True,
        )
        decision = await decider.decide(
            REQUEST["state"], REQUEST["questions"], controls
        )
        for name, answer in decision.answers.items():
            thought = (answer.get("details") or {}).get("thought") or {}
            print(
                json.dumps(
                    {
                        "question": name,
                        "answer": {k: v for k, v in answer.items() if k != "details"},
                        "thought": {
                            "used": thought.get("used"),
                            "tokens": thought.get("tokens"),
                            "ended": thought.get("ended"),
                        },
                    }
                )
            )
    finally:
        await backend.close()


if __name__ == "__main__":
    asyncio.run(main())
