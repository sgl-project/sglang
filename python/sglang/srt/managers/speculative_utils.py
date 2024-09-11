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

from typing import Type

import torch


class SpecDraftInfo:
    pass


class SpecDraftInfoFactory:
    def __init__(self):
        self.factory = {}

    def register(self, name: str) -> SpecDraftInfo:
        def wrapper(info: Type[SpecDraftInfo]) -> Type[SpecDraftInfo]:
            self.factory[name] = info
            return info

        return wrapper

    def get(self, name):
        return self.factory[name]


DraftInfoFactory = SpecDraftInfoFactory()


@DraftInfoFactory.register("EAGLE")
class EAGLEDraftInfo(SpecDraftInfo):
    def __init__(
        self, hidden_states: torch.Tensor, input_ids: torch.Tensor, output_token
    ):
        hidden_states: torch.Tensor = hidden_states
        input_ids: torch.Tensor = input_ids

    def update_input(self, info: "EAGLEDraftInfo"):
        self.hidden_states = info.hidden_states
        self.input_ids = info.input_ids


class SpecInfoPipline:
    def __init__(self):
        self.draft_input_queue = torch.multiprocessing.Queue()
        self.draft_output_queue = torch.multiprocessing.Queue()
