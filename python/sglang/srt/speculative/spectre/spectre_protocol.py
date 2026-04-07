from dataclasses import dataclass, fields
from enum import Enum
from typing import Any, Dict, List, Optional

from sglang.srt.sampling.sampling_params import SamplingParams


def _sampling_params_to_dict(sampling_params: SamplingParams) -> Dict[str, Any]:
    stop_token_ids = (
        list(sampling_params.stop_token_ids) if sampling_params.stop_token_ids else None
    )
    return {
        "max_new_tokens": sampling_params.max_new_tokens,
        "stop_strs": sampling_params.stop_strs,
        "stop_token_ids": stop_token_ids,
        "stop_regex_strs": sampling_params.stop_regex_strs,
        "temperature": sampling_params.temperature,
        "top_p": sampling_params.top_p,
        "top_k": sampling_params.top_k,
        "min_p": sampling_params.min_p,
        "frequency_penalty": sampling_params.frequency_penalty,
        "presence_penalty": sampling_params.presence_penalty,
        "repetition_penalty": sampling_params.repetition_penalty,
        "min_new_tokens": sampling_params.min_new_tokens,
        "n": sampling_params.n,
        "json_schema": sampling_params.json_schema,
        "regex": sampling_params.regex,
        "ebnf": sampling_params.ebnf,
        "structural_tag": sampling_params.structural_tag,
        "ignore_eos": sampling_params.ignore_eos,
        "skip_special_tokens": sampling_params.skip_special_tokens,
        "spaces_between_special_tokens": sampling_params.spaces_between_special_tokens,
        "no_stop_trim": sampling_params.no_stop_trim,
        "custom_params": sampling_params.custom_params,
        "stream_interval": sampling_params.stream_interval,
        "logit_bias": sampling_params.logit_bias,
        "sampling_seed": sampling_params.sampling_seed,
    }


class SpectreAction(Enum):
    """
    SpectreAction is the action to take for the Spectre request.
    draft: normal draft request (D->T & T->D)
    finish: when req is finished in target (T->D)
    abort: when req is aborted in target (T->D)
    reject: when draft is high overhead (D->T)
    """

    DRAFT = "draft"
    FINISH = "finish"
    ABORT = "abort"
    REJECT = "reject"


class SpecType(Enum):
    """
    SpecType is the type of the Spectre request. It is used to distinguish the type of the request.
    normal: normal request
    draft_request: draft request (D->T)
    draft_response: draft response (T->D)
    """

    NORMAL = "normal"
    DRAFT_REQUEST = "draft_request"
    DRAFT_RESPONSE = "draft_response"


@dataclass
class SpectreRequest:
    request_id: Optional[str] = None
    spec_cnt: Optional[int] = None
    action: SpectreAction = SpectreAction.FINISH
    spec_type: SpecType = SpecType.NORMAL
    draft_token_ids: Optional[List[int]] = None
    target_send_time: float = -1.0
    target_recv_time: float = -1.0

    input_ids: Optional[List[int]] = None
    output_ids: Optional[List[int]] = None
    num_draft_tokens: Optional[int] = None
    sampling_params: Optional[SamplingParams] = None
    grammar: Optional[str] = None

    draft_logprobs: Optional[List[float]] = None
    draft_recv_time: float = -1.0
    draft_send_time: float = -1.0

    def to_dict(self) -> Dict[str, Any]:
        result = {}
        for f in fields(self):
            val = getattr(self, f.name)
            if val is not None:
                if isinstance(val, Enum):
                    result[f.name] = val.value
                elif isinstance(val, SamplingParams):
                    result[f.name] = {
                        k: v
                        for k, v in _sampling_params_to_dict(val).items()
                        if v is not None
                    }
                else:
                    result[f.name] = val
        return result

    @classmethod
    def from_dict(cls, d: dict):
        d = dict(d)
        field_names = {f.name for f in fields(cls)}
        d = {k: v for k, v in d.items() if k in field_names}

        init_kwargs = {}
        for f in fields(cls):
            val = d.get(f.name)
            if val is None:
                init_kwargs[f.name] = None
                continue
            if isinstance(f.type, type) and issubclass(f.type, Enum):
                if not isinstance(val, f.type):
                    init_kwargs[f.name] = f.type(val)
                else:
                    init_kwargs[f.name] = val
            elif isinstance(val, dict) and "max_new_tokens" in val:
                stop_strs = val.pop("stop_strs", None)
                stop_regex_strs = val.pop("stop_regex_strs", None)
                sp = SamplingParams(**val)
                if stop_strs:
                    sp.stop_strs = stop_strs
                if stop_regex_strs:
                    sp.stop_regex_strs = stop_regex_strs
                init_kwargs[f.name] = sp
            else:
                init_kwargs[f.name] = val
        return cls(**init_kwargs)


def is_health_check_req(req) -> bool:
    return getattr(req, "rid", "").startswith("HEALTH_CHECK")
