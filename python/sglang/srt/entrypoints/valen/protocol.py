# SPDX-License-Identifier: Apache-2.0
"""Valen's native request shape, including its 2–255-level Score contract."""

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from sglang.srt.entrypoints.valen.encoding import valen_options


class ValenQuestionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    type: Literal["choice", "noul", "score"]
    instructions: str
    criteria: Any = None

    @model_validator(mode="after")
    def _valid_candidates(self):
        valen_options(self.model_dump())
        return self


class ValenRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    state: str | dict[str, Any]
    questions: dict[str, ValenQuestionRequest] = Field(min_length=1)
    model: str | None = None
    temperature: float = Field(default=1.0, gt=0, allow_inf_nan=False)
