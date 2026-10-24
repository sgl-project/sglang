from typing import Annotated, Literal, Union

from pydantic import BaseModel, Field, TypeAdapter, ValidationError

DPRank = Annotated[int, Field(strict=True, ge=0)]
RequestId = Annotated[str, Field(strict=True)]


class RetryParams(BaseModel):
    pass


class ScaleDownParams(BaseModel):
    removed_dp_ranks: list[DPRank]


class RetryRequest(BaseModel):
    instruction: Literal["retry"]
    params: RetryParams = Field(default_factory=RetryParams)
    request_id: RequestId = ""


class ScaleDownRequest(BaseModel):
    instruction: Literal["scale_down"]
    params: ScaleDownParams
    request_id: RequestId = ""


FaultToleranceApplyRequest = Annotated[
    Union[RetryRequest, ScaleDownRequest],
    Field(discriminator="instruction"),
]

_APPLY_REQUEST_ADAPTER = TypeAdapter(FaultToleranceApplyRequest)


def parse_apply_request(body: bytes) -> FaultToleranceApplyRequest:
    try:
        return _APPLY_REQUEST_ADAPTER.validate_json(body)
    except ValidationError as exc:
        error = exc.errors()[0]
        if error["type"] == "union_tag_invalid":
            message = f"Invalid instruction: '{error['ctx']['tag']}'."
        else:
            message = error["msg"]
        raise ValueError(message) from None
