# Copyright 2023-2025 SGLang Team
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

import inspect
import itertools
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Callable, List, Optional, Set, Tuple, Type


class OpPatternBase(ABC):
    @staticmethod
    @abstractmethod
    def pattern(*args, **kwargs):
        pass


@dataclass
class OpPattern:
    op_type: Type[OpPatternBase]
    pattern: Callable


class OpPatternRegistery(ABC):
    def __init__(self):
        self._patterns: List[OpPattern] = []
        self.build_op_pattern_registery()

    def register_op_pattern(self, op: Type[OpPattern]):
        self._patterns.append(OpPattern(op, op.pattern))

    @abstractmethod
    def build_op_pattern_registery(self):
        pass

    @property
    def patterns(self):
        return self._patterns


def pattern_builder(
    builder: Callable,
    op_pattern_registeries: List[OpPatternRegistery],
    ignore_combinations: Optional[List[Tuple[Type[OpPatternBase], ...]]] = None,
):
    num_args = sum(
        1
        for param in inspect.signature(builder).parameters.values()
        if param.kind in (param.POSITIONAL_ONLY, param.POSITIONAL_OR_KEYWORD)
        and param.default is param.empty
    )
    assert (
        len(op_pattern_registeries) == num_args
    ), f"Expected {num_args} op_pattern_registeries, got {len(op_pattern_registeries)}"

    patterns = list(map(lambda x: x.patterns, op_pattern_registeries))
    for ops in itertools.product(*patterns):
        if ignore_combinations:
            ops_key = set(map(lambda x: x.op_type, ops))
            if any(ops_key.issuperset(t) for t in ignore_combinations):
                continue
        builder(*ops)
