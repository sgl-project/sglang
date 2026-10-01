"""Jev (TypeSafe System One) request and response models, served for decision model checkpoints."""

import math
from typing import Annotated, Any, Dict, List, Literal, Optional, Union

from pydantic import (
    BaseModel,
    ConfigDict,
    Discriminator,
    Field,
    PlainValidator,
    Tag,
    field_validator,
)

from sglang.srt.entrypoints.openai.protocol import DecisionRequest
from sglang.srt.entrypoints.systemone.protocol import SystemOneRequest

MAX_QUESTIONS = 16
MAX_IMAGES = 8
# The System One maximum; a family refuses more options than it has answer symbols.
MAX_OPTIONS = 255


# Unknown question keys are dropped, as the official service copies only these three.
class _Question(BaseModel):
    model_config = ConfigDict(extra="ignore")

    # Rendered with str(), so an explicit null reads "None" as in the official prompt.
    instructions: Any = ""


class JevChoiceQuestion(_Question):
    type: Literal["choice"]
    criteria: Dict[str, Any] = Field(min_length=1, max_length=MAX_OPTIONS)


class JevScoreQuestion(_Question):
    type: Literal["score"]
    criteria: Union[List[Any], Dict[str, Any]] = Field(
        min_length=1, max_length=MAX_OPTIONS
    )

    @field_validator("criteria")
    @classmethod
    def _numeric_keys(cls, criteria):
        if isinstance(criteria, dict):
            try:
                finite = all(math.isfinite(float(key)) for key in criteria)
            except ValueError:
                finite = False
            if not finite:
                raise ValueError("score option keys must be finite numbers")
        return criteria


class JevNoulQuestion(_Question):
    type: Literal["noul"]
    # yes/true/1 and no/false/0 keys override the default descriptions.
    criteria: Any = None


JevQuestion = Annotated[
    Union[JevChoiceQuestion, JevScoreQuestion, JevNoulQuestion],
    Field(discriminator="type"),
]


class JevImageUpload(BaseModel):
    """The upload object of the official service; a data URL in data overrides type."""

    type: str = ""
    data: str


# Base64 or a base64 data URL; paths and remote URLs are never loaded.
JevImage = Union[JevImageUpload, str]


class JevRequest(BaseModel):
    state: Any
    questions: Dict[str, JevQuestion] = Field(min_length=1, max_length=MAX_QUESTIONS)
    # In prompt order; decoded and normalized only after the server refusals pass.
    images: Optional[List[JevImage]] = Field(default=None, max_length=MAX_IMAGES)
    # Divides the candidate logits, softmax(log p / T); a value whose rounding would
    # change an argmax is refused.
    temperature: Optional[float] = Field(default=None, gt=0, allow_inf_nan=False)
    thinking: Optional[Dict[str, Any]] = None
    model: Optional[str] = None

    @field_validator("questions")
    @classmethod
    def _names_nonempty(cls, questions):
        if any(not name for name in questions):
            raise ValueError("question names must be nonempty strings")
        return questions

    @field_validator("thinking")
    @classmethod
    def _no_thinking_handoff(cls, thinking):
        if thinking and thinking.get("enabled"):
            raise ValueError("the thinking handoff is not served by this route")
        return thinking


class JevAnswer(BaseModel):
    type: Literal["choice", "score", "noul"]
    probabilities: Dict[str, float]
    decision: str
    confidence: float
    choice: Optional[str] = None
    noul: Optional[float] = None
    score: Optional[float] = None
    legend: Optional[Dict[str, str]] = None


class JevUsage(BaseModel):
    input_tokens: int
    output_tokens: int
    decision_count: int


class JevCalibration(BaseModel):
    method: Literal["temperature-scaling"] = "temperature-scaling"
    temperature: float


class JevResponse(BaseModel):
    model: str
    answers: Dict[str, JevAnswer]
    usage: JevUsage
    calibration: Optional[JevCalibration] = None


def _decisions_body_kind(body: Any) -> str:
    # The generic body has an input and a list of questions, so the shapes never overlap.
    if isinstance(body, dict) and (
        "state" in body or isinstance(body.get("questions"), dict)
    ):
        return "jev"
    return "generic"


DecisionsRouteRequest = Annotated[
    Union[
        Annotated[DecisionRequest, Tag("generic")],
        Annotated[JevRequest, Tag("jev")],
    ],
    Discriminator(_decisions_body_kind),
]


def _validated_by_route(body: Any) -> Any:
    return body


# The served checkpoint picks the shape, so the route validates; the schema documents both.
SystemOneRouteRequest = Annotated[
    Any,
    PlainValidator(
        _validated_by_route,
        json_schema_input_type=Union[SystemOneRequest, JevRequest],
    ),
]
