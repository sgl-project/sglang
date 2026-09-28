"""Pydantic models for the System One decision API.

Follows the published System One OpenAPI 0.2.0 request and response shapes:
a state, a map of noul, choice, and score questions keyed by caller ids, and
one answer per question id.
"""

from typing import Annotated, Any, Dict, List, Literal, Optional, Union

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    NonNegativeInt,
    ValidationInfo,
    field_validator,
    model_validator,
)
from pydantic.json_schema import SkipJsonSchema

from sglang.srt.entrypoints.openai.protocol import (
    DecisionText,
    RequiredDecisionText,
    check_option_names,
    is_blank_decision_text,
)

# The documented maximum. Options beyond 26 get two-letter labels, checked per request.
MAX_CHOICE_OPTIONS = 255
# Levels are labeled 0 to 9, so at most 10.
MAX_SCORE_LEVELS = 10


class _Question(BaseModel):
    # Misspelled keys inside a question would otherwise answer a different question.
    model_config = ConfigDict(extra="forbid")

    instructions: Optional[DecisionText] = None


class SystemOneNoulCriteria(BaseModel):
    model_config = ConfigDict(extra="forbid")

    true: Optional[DecisionText] = None
    false: Optional[DecisionText] = None


class SystemOneNoulQuestion(_Question):
    type: Literal["noul"]
    criteria: Optional[SystemOneNoulCriteria] = None

    @model_validator(mode="after")
    def _asks_something(self):
        criteria = self.criteria or SystemOneNoulCriteria()
        if all(
            is_blank_decision_text(value)
            for value in (self.instructions, criteria.true, criteria.false)
        ):
            raise ValueError(
                "a noul question needs instructions or a true or false "
                "description to decide on"
            )
        return self


class SystemOneChoiceQuestion(_Question):
    type: Literal["choice"]
    criteria: Dict[str, Optional[DecisionText]] = Field(
        min_length=1, max_length=MAX_CHOICE_OPTIONS
    )

    @field_validator("criteria")
    @classmethod
    def _option_names_distinct(cls, criteria):
        check_option_names(criteria)
        return criteria


class SystemOneScoreQuestion(_Question):
    type: Literal["score"]
    criteria: List[RequiredDecisionText] = Field(
        min_length=1, max_length=MAX_SCORE_LEVELS
    )


SystemOneQuestion = Annotated[
    Union[SystemOneNoulQuestion, SystemOneChoiceQuestion, SystemOneScoreQuestion],
    Field(discriminator="type"),
]


class SystemOneRequest(BaseModel):
    # Unknown top-level fields are ignored, as the published schema allows.
    state: DecisionText
    model: str
    questions: Dict[str, SystemOneQuestion] = Field(min_length=1)
    # SGLang extension, for chat templates whose reasoning toggle needs a kwarg.
    chat_template_kwargs: Dict[str, Any] = Field(default_factory=dict)

    # /v1/decisions fields, which would change the answers if honored or ignored.
    # Declared only to refuse them by name, and hidden from the schema.
    temperature: SkipJsonSchema[Any] = None
    prompt_format_version: SkipJsonSchema[Any] = None
    return_prompt_token_ids: SkipJsonSchema[Any] = None

    @field_validator("temperature", "prompt_format_version", "return_prompt_token_ids")
    @classmethod
    def _refuse_decisions_fields(cls, value, info: ValidationInfo):
        if value is not None:
            raise ValueError(
                f"{info.field_name} is not part of this API, use /v1/decisions for it"
            )
        return value


class SystemOneNoulAnswer(BaseModel):
    type: Literal["noul"] = "noul"
    noul: float
    # Full-vocabulary probability of the answer labels, an SGLang extension.
    x_label_mass: float


class SystemOneChoiceAnswer(BaseModel):
    type: Literal["choice"] = "choice"
    choice: str
    confidence: float
    probabilities: Dict[str, float]
    x_label_mass: float


class SystemOneScoreAnswer(BaseModel):
    type: Literal["score"] = "score"
    score: float
    confidence: float
    legend: Dict[str, Any]
    probabilities: Dict[str, float]
    x_label_mass: float


class SystemOneUsage(BaseModel):
    input_tokens: NonNegativeInt
    output_tokens: NonNegativeInt = 0


class SystemOneResponse(BaseModel):
    model: str
    answers: Dict[
        str, Union[SystemOneNoulAnswer, SystemOneChoiceAnswer, SystemOneScoreAnswer]
    ]
    usage: SystemOneUsage
