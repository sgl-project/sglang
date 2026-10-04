# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import json
from dataclasses import asdict, dataclass

from sglang.multimodal_gen.runtime.pipelines_core.schedule_batch import Req
from sglang.multimodal_gen.runtime.pipelines_core.stages.base import PipelineStage
from sglang.multimodal_gen.runtime.server_args import ServerArgs
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger

from sglang.srt.models.yue2.protocol import SongRequest
from sglang.srt.models.yue2.tokenization_yue2 import YuE2TextTokenizer

logger = init_logger(__name__)


@dataclass
class Yue2Request:
    style: str
    lyrics: str
    cot: str = "full"
    seed: int = 831001
    abc: str | None = None
    cfg_scale: float | None = None
    id: str = "song"
    artifacts_dir: str = ""  # optional per-request artifact export (WSB-style scoring)

    @classmethod
    def from_sampling_params(cls, sampling_params) -> "Yue2Request":
        raw_prompt = sampling_params.prompt
        payload: dict | None = None
        if isinstance(raw_prompt, str) and raw_prompt.lstrip().startswith("{"):
            try:
                value = json.loads(raw_prompt)
                if isinstance(value, dict):
                    payload = value
            except Exception:
                logger.debug("YuE2 prompt is not valid JSON; using fallback parsing")
        if payload is None:
            payload = {
                "style": getattr(sampling_params, "style", None),
                "lyrics": getattr(sampling_params, "lyrics", None),
                "cot": getattr(sampling_params, "cot", "full"),
                "seed": getattr(sampling_params, "seed", 831001),
                "abc": getattr(sampling_params, "request_abc", None),
                "cfg_scale": getattr(sampling_params, "cfg_scale", None),
                "id": getattr(sampling_params, "output_file_name", None) or "song",
            }
        style = payload.get("style") or payload.get("tags")
        lyrics = payload.get("lyrics")
        if not style and not lyrics and isinstance(raw_prompt, str):
            lines = [line.strip() for line in raw_prompt.splitlines() if line.strip()]
            if len(lines) == 1:
                style = lines[0]
            elif len(lines) >= 2:
                style, lyrics = lines[0], "\n".join(lines[1:])
        if not style:
            style = payload.get("id") or "instrumental"
        if not lyrics:
            lyrics = ""
        return cls(
            style=str(style),
            lyrics=str(lyrics),
            cot=str(payload.get("cot", "full")),
            seed=int(payload.get("seed", 831001)),
            abc=payload.get("abc"),
            cfg_scale=payload.get("cfg_scale"),
            id=str(payload.get("id", "song")),
            artifacts_dir=str(payload.get("artifacts_dir") or ""),
        )

    def to_song_request(self) -> SongRequest:
        fields = {k: v for k, v in asdict(self).items() if k != "artifacts_dir"}
        return SongRequest(**fields)


class Yue2PrepareRequestStage(PipelineStage):
    """Build a checkpoint-native request and prompt prefix."""

    def __init__(self, tokenizer: YuE2TextTokenizer):
        super().__init__()
        self.tokenizer = tokenizer

    def forward(self, batch: Req, server_args: ServerArgs) -> Req:
        request = Yue2Request.from_sampling_params(batch.sampling_params)
        song_request = request.to_song_request()
        from sglang.srt.models.yue2.protocol import token_prefixes

        prefix = token_prefixes(song_request, self.tokenizer)
        if not batch.output_file_name:
            batch.output_file_name = f"{song_request.id}.wav"
        batch.extra.update(
            {
                "yue2_request": song_request,
                "yue2_prefix": list(prefix),
                "yue2_request_id": song_request.id,
                "yue2_artifacts_dir": request.artifacts_dir,
            }
        )
        return batch
