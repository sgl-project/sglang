from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

import requests


def post_parallel_generate(
    *,
    url: str,
    prompts: list[str],
    max_new_tokens: int,
    timeout: float,
) -> list[dict]:
    def _send(prompt: str) -> dict:
        try:
            resp = requests.post(
                url,
                json={
                    "text": prompt,
                    "sampling_params": {
                        "max_new_tokens": max_new_tokens,
                        "temperature": 0.0,
                    },
                },
                timeout=timeout,
            )
            return {"status_code": resp.status_code, "body": resp.text}
        except requests.RequestException as exc:
            return {"status_code": -1, "error": repr(exc)}

    with ThreadPoolExecutor(max_workers=max(1, len(prompts))) as pool:
        return list(pool.map(_send, prompts))
