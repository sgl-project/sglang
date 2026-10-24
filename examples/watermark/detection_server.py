"""Minimal reference server for watermark detection."""

import argparse
from functools import lru_cache
from typing import Optional

import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field, model_validator
from transformers import AutoTokenizer, PreTrainedTokenizerBase

from sglang.srt.sampling.watermarking import (
    WatermarkDetection,
    WatermarkDetector,
    WatermarkStatistics,
)
from sglang.srt.sampling.watermarking.config import load_watermark_config


class DetectionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    text: Optional[str] = None
    token_ids: Optional[list[int]] = None
    prompt_token_ids: list[int] = Field(default_factory=list)
    tokenizer: Optional[str] = None

    @model_validator(mode="after")
    def validate_input(self) -> "DetectionRequest":
        if (self.text is None) == (self.token_ids is None):
            raise ValueError("exactly one of text or token_ids is required")
        return self


class DetectionStatistics(BaseModel):
    p_value: float
    z_score: float
    num_contexts: int


class DetectionResponse(DetectionStatistics):
    watermarked: bool
    per_key: Optional[dict[str, DetectionStatistics]] = None


@lru_cache(maxsize=4)
def get_tokenizer(tokenizer_id: str) -> PreTrainedTokenizerBase:
    return AutoTokenizer.from_pretrained(tokenizer_id, trust_remote_code=False)


def statistics_response(statistics: WatermarkStatistics) -> DetectionStatistics:
    return DetectionStatistics(
        p_value=statistics.p_value,
        z_score=statistics.z_score,
        num_contexts=statistics.num_contexts,
    )


def per_key_response(
    result: WatermarkDetection,
) -> Optional[dict[str, DetectionStatistics]]:
    if result.key_b_all_positions is None:
        return None
    return {
        "key_a_all_positions": statistics_response(result.key_a_all_positions),
        "key_b_all_positions": statistics_response(result.key_b_all_positions),
        "key_a_partition": statistics_response(result.key_a_partition),
        "key_b_partition": statistics_response(result.key_b_partition),
    }


def create_app(
    detector: WatermarkDetector,
    *,
    p_value_threshold: float,
    default_tokenizer: Optional[str],
) -> FastAPI:
    app = FastAPI()

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.post("/detect")
    def detect(request: DetectionRequest) -> DetectionResponse:
        if request.text is not None:
            tokenizer_id = request.tokenizer or default_tokenizer
            if tokenizer_id is None:
                raise HTTPException(
                    status_code=400,
                    detail="text detection requires a tokenizer",
                )
            try:
                token_ids = get_tokenizer(tokenizer_id).encode(
                    request.text, add_special_tokens=False
                )
            except (OSError, ValueError) as error:
                raise HTTPException(
                    status_code=400, detail="failed to load tokenizer"
                ) from error
        else:
            token_ids = request.token_ids

        try:
            result = detector.detect_tokens(
                token_ids, prompt_token_ids=request.prompt_token_ids
            )
        except ValueError as error:
            raise HTTPException(status_code=400, detail=str(error)) from error

        combined = statistics_response(result.combined)
        return DetectionResponse(
            **combined.model_dump(),
            watermarked=combined.p_value < p_value_threshold,
            per_key=per_key_response(result),
        )

    return app


def probability(value: str) -> float:
    probability_value = float(value)
    if not 0 < probability_value < 1:
        raise argparse.ArgumentTypeError("value must be strictly between 0 and 1")
    return probability_value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--watermark-config", required=True)
    parser.add_argument("--tokenizer")
    parser.add_argument("--max-contexts", type=int, default=4096)
    parser.add_argument("--p-value-threshold", type=probability, default=0.01)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    return parser.parse_args()


def main(args: argparse.Namespace) -> None:
    config = load_watermark_config(args.watermark_config)
    if config.key is None:
        raise ValueError("watermark config must set key")
    detector = WatermarkDetector(
        config.key,
        key_b=config.key_b,
        mixing_probability=(
            config.mixing_probability if config.mixing_probability is not None else 0.5
        ),
        context_window=(
            config.context_window if config.context_window is not None else 4
        ),
        max_contexts=args.max_contexts,
    )
    uvicorn.run(
        create_app(
            detector,
            p_value_threshold=args.p_value_threshold,
            default_tokenizer=args.tokenizer,
        ),
        host=args.host,
        port=args.port,
    )


if __name__ == "__main__":
    main(parse_args())
