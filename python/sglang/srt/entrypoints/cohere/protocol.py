# Copyright 2023-2026 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Cohere-style /v2/embed protocol, matching vLLM v0.32.0's wire format."""

import builtins
from typing import Literal, Optional
from uuid import uuid4

from pydantic import BaseModel, Field, model_validator
from starlette.responses import JSONResponse

CohereEmbeddingType = Literal["float", "binary", "ubinary", "base64"]
CohereTruncate = Literal["END", "START", "NONE"]


class CohereEmbedContent(BaseModel):
    type: Literal["text", "image_url"]
    text: Optional[str] = None
    image_url: Optional[dict[str, str]] = None

    @model_validator(mode="after")
    def validate_payload(self):
        if self.type == "text" and self.text is None:
            raise ValueError("Text content requires a text field")
        if self.type == "image_url" and (
            not self.image_url or not self.image_url.get("url")
        ):
            raise ValueError("Image content requires a non-empty image_url.url")
        if self.type != "text" and self.type != "image_url":
            raise ValueError("Content type must be either 'text' or 'image_url'")
        return self


class CohereEmbedInput(BaseModel):
    content: list[CohereEmbedContent] = Field(min_length=1)


class CohereEmbedRequest(BaseModel):
    model: Optional[str] = None
    texts: Optional[list[str]] = None
    images: Optional[list[str]] = None
    inputs: Optional[list[CohereEmbedInput]] = None
    # Keys are model-defined, so the serving layer validates input_type.
    input_type: Optional[str] = None
    output_dimension: Optional[int] = Field(default=None, gt=0)
    embedding_types: Optional[list[CohereEmbeddingType]] = None
    truncate: CohereTruncate = "END"
    # vLLM accepts -1 (model budget) and 0; the serving path resolves them.
    max_tokens: Optional[int] = Field(default=None, ge=-1)
    priority: int = Field(default=0, ge=-(2**63), le=2**63 - 1)

    @model_validator(mode="after")
    def validate_inputs(self):
        provided = [
            value
            for value in (self.texts, self.images, self.inputs)
            if value is not None
        ]
        if len(provided) != 1 or not provided[0]:
            raise ValueError(
                "Exactly one of texts, images, or inputs must be provided, "
                "and it must be non-empty"
            )
        if self.images is not None and any(not image for image in self.images):
            raise ValueError("images must contain non-empty image URLs")
        return self

    @property
    def has_images(self) -> bool:
        return self.images is not None or any(
            part.type == "image_url"
            for item in self.inputs or []
            for part in item.content
        )


class CohereApiVersion(BaseModel):
    version: Literal["2"] = "2"


class CohereBilledUnits(BaseModel):
    input_tokens: int
    image_tokens: int


class CohereMeta(BaseModel):
    api_version: CohereApiVersion = Field(default_factory=CohereApiVersion)
    billed_units: CohereBilledUnits


class CohereEmbedByTypeEmbeddings(BaseModel):
    # The field named float shadows the builtin in this class body.
    float: Optional[list[list[builtins.float]]] = None
    binary: Optional[list[list[int]]] = None
    ubinary: Optional[list[list[int]]] = None
    base64: Optional[list[str]] = None


class CohereEmbedResponse(BaseModel):
    id: str = Field(default_factory=lambda: f"embd-{uuid4().hex}")
    embeddings: CohereEmbedByTypeEmbeddings
    texts: Optional[list[str]] = None
    meta: CohereMeta
    response_type: Literal["embeddings_by_type"] = "embeddings_by_type"


class CohereErrorDetail(BaseModel):
    message: str
    type: str
    param: Optional[str] = None
    code: int


class CohereErrorResponse(BaseModel):
    error: CohereErrorDetail


def create_error_response(
    message: str,
    status_code: int = 400,
    error_type: str = "BadRequestError",
    param: Optional[str] = None,
) -> JSONResponse:
    error = CohereErrorResponse(
        error=CohereErrorDetail(
            message=message, type=error_type, param=param, code=status_code
        )
    )
    # Keep param: null in errors; successful responses instead omit unused types.
    return JSONResponse(content=error.model_dump(), status_code=status_code)
