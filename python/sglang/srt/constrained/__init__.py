import json
from typing import Dict, Optional, Union

from outlines.caching import cache as disk_cache
from outlines.caching import disable_cache
from outlines.fsm.fsm import RegexFSM
from outlines.fsm.regex import FSMInfo, make_deterministic_fsm
from outlines.models.transformers import TransformerTokenizer
from pydantic import BaseModel

try:
    from outlines.fsm.json_schema import build_regex_from_object
except ImportError:
    # Since outlines 0.0.32, build_regex_from_object is replaced by build_regex_from_schema,
    # which only accepts string schema as input.
    from outlines.fsm.json_schema import build_regex_from_schema

    def build_regex_from_object(
        object: Union[str, BaseModel, Dict], whitespace_pattern: Optional[str] = None
    ):
        if isinstance(object, type(BaseModel)):
            schema = json.dumps(object.model_json_schema())
        elif isinstance(object, Dict):
            schema = json.dumps(object)
        else:
            schema = object
        return build_regex_from_schema(schema, whitespace_pattern)


__all__ = [
    "RegexFSM",
    "FSMInfo",
    "make_deterministic_fsm",
    "build_regex_from_object",
    "TransformerTokenizer",
    "disk_cache",
    "disable_cache",
]
