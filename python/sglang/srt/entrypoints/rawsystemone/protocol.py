from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, StrictBool, StrictStr


class RawSystemOneRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    prefix: StrictStr
    suffixes: list[StrictStr] = Field(min_length=1)
    return_token_logprobs: StrictBool = False


class TokenLogprob(BaseModel):
    position: int
    token_id: int
    logprob: float | None


class CandidateScore(BaseModel):
    index: int
    score: float
    logprob_sum: float
    scored_token_count: int
    token_logprobs: list[TokenLogprob] | None = None


class Usage(BaseModel):
    input_tokens: int
    scored_tokens: int
    generated_tokens: Literal[0] = 0


class RawSystemOneResponse(BaseModel):
    id: str
    object: Literal["rawsystemone"] = "rawsystemone"
    model: str
    scoring: Literal["mean_logprob_full_sequence"] = "mean_logprob_full_sequence"
    tokenization: Literal["native_text_v1"] = "native_text_v1"
    data: list[CandidateScore]
    best_index: int
    usage: Usage


class RawSystemOneError(Exception):
    def __init__(self, code: str, message: str, status: int = 400):
        super().__init__(message)
        self.code = code
        self.status = status

    def response(self):
        from fastapi.responses import JSONResponse

        # Match the existing ErrorResponse envelope: integer HTTP code, stable
        # machine-readable type. Never include native errors containing inputs.
        return JSONResponse(
            status_code=self.status,
            content={
                "object": "error",
                "message": str(self),
                "type": self.code,
                "param": None,
                "code": self.status,
            },
        )
