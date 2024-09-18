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


class SpecDraftInput:
    pass


class SpecDraftInfoFactory:
    def __init__(self):
        self.factory = {}

    def register(self, name: str) -> SpecDraftInput:
        def wrapper(info: Type[SpecDraftInput]) -> Type[SpecDraftInput]:
            self.factory[name] = info
            return info

        return wrapper

    def get(self, name):
        return self.factory[name]


DraftInfoFactory = SpecDraftInfoFactory()


@DraftInfoFactory.register("EAGLE")
class EAGLEDraftInput(SpecDraftInput):
    hidden_states: torch.Tensor = None
    verified_id: torch.Tensor = None

    def prepare_for_extend(self, batch):
        seq_lens = [0] + batch.seq_lens.tolist()
        input_ids = batch.input_ids.tolist()
        verified_id = self.verified_id.tolist()
        model_input_ids = []
        for i in range(len(seq_lens) - 1):
            model_input_ids.extend(
                input_ids[seq_lens[i] + 1 : seq_lens[i + 1]] + [verified_id[i]]
            )
        batch.input_ids = torch.tensor(
            model_input_ids, dtype=torch.int32, device="cuda"
        )
        del verified_id


class SpecInfoPipline:
    def __init__(self):
        ctx = torch.multiprocessing.get_context("forkserver")
        self.draft_input_queue = ctx.Queue()
        self.draft_output_queue = ctx.Queue()
        self.max_total_num_tokens = ctx.Value("i", -1)
