"""
Copyright 2023-2024 SGLang Team
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

"""For constrained decoding."""

import json
from typing import Dict, Optional, Union

from pydantic import BaseModel

try:
    from outlines.caching import cache as disk_cache
    from outlines.caching import disable_cache
    from outlines.fsm.guide import RegexGuide
    from outlines.fsm.regex import FSMInfo, make_byte_level_fsm, make_deterministic_fsm
    from outlines.models.transformers import TransformerTokenizer
except ImportError as e:
    print(
        f'\nError: {e}. Please install a new version of outlines by `pip install "outlines>=0.0.44"`\n'
    )
    raise

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


try:
    from xgrammar import (
        GrammarMatcher,
        GrammarMatcherInitContext,
        GrammarMatcherInitContextCache,
    )
except ImportError as e:

    class Dummy:
        pass

    GrammarMatcher = Dummy
    GrammarMatcherInitContext = Dummy
    GrammarMatcherInitContextCache = Dummy

__all__ = [
    "RegexGuide",
    "FSMInfo",
    "make_deterministic_fsm",
    "build_regex_from_object",
    "TransformerTokenizer",
    "disk_cache",
    "disable_cache",
    "make_byte_level_fsm",
    "GrammarMatcher",
    "GrammarMatcherInitContext",
    "GrammarMatcherInitContextCache",
]
