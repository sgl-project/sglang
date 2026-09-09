# SPDX-License-Identifier: Apache-2.0
"""Remote prompt-enhancement (PE) client for SenseNova-U1.

SenseNova-U1 ships no local PE checkpoint. Enhancement is always a remote
call to an OpenAI-compatible `/chat/completions` endpoint, mirroring the
`U1_ENHANCE_*` contract from the upstream OpenSenseNova/SenseNova-U1
reference implementation (`src/sensenova_u1/prompt_enhance`).
"""

from __future__ import annotations

import requests

DEFAULT_TIMEOUT = 60.0


class SenseNovaU1PEClient:
    """OpenAI-compatible chat/completions client for prompt enhancement."""

    def __init__(
        self,
        endpoint: str,
        model: str,
        api_key: str,
        timeout: float = DEFAULT_TIMEOUT,
    ) -> None:
        self._endpoint = endpoint
        self._model = model
        self._api_key = api_key
        self._timeout = timeout

    def enhance(self, system_prompt: str, user_prompt: str) -> str:
        response = requests.post(
            self._endpoint,
            json={
                "model": self._model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
            },
            headers={
                "Authorization": f"Bearer {self._api_key}",
                "Content-Type": "application/json",
            },
            timeout=self._timeout,
        )
        response.raise_for_status()

        try:
            data = response.json()
        except requests.exceptions.JSONDecodeError as exc:
            raise RuntimeError(
                "PE chat/completions response is not valid JSON."
            ) from exc

        choices = data.get("choices")
        if not isinstance(choices, list) or not choices:
            raise RuntimeError("PE chat/completions response has no valid choices.")

        # RuntimeError, not TypeError, throughout: uniform for any malformed
        # response shape rather than distinguishing by which piece is wrong.
        choice = choices[0]
        if not isinstance(choice, dict):
            raise RuntimeError(  # noqa: TRY004
                "PE chat/completions response has an invalid choice."
            )

        message = choice.get("message")
        if not isinstance(message, dict):
            raise RuntimeError(  # noqa: TRY004
                "PE chat/completions response is missing message."
            )

        content = message.get("content")
        if not isinstance(content, str):
            raise RuntimeError(  # noqa: TRY004
                "PE chat/completions response is missing message.content."
            )

        content = content.strip()
        if not content:
            raise RuntimeError(
                "PE chat/completions response contains empty message.content."
            )
        return content
