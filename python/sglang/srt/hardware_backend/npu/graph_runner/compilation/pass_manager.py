# Copyright 2025 SGLang Team
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

import logging

import torch

logger = logging.getLogger(__name__)


class PassManager:
    def __init__(self, graph_module: torch.fx.GraphModule):
        self.graph_module = graph_module
        self.passes = []

    def add(self, pass_, **kwargs):
        self.passes.append((pass_, kwargs))

    def apply(self):
        updated = False
        for pass_, kwargs in self.passes:
            pass_instance = pass_(**kwargs)
            results = []
            try:
                if callable(pass_instance):
                    results = pass_instance(self.graph_module)
                else:
                    results = torch.fx.replace_pattern(
                        self.graph_module, pass_.pattern, pass_.replacement
                    )

                logger.debug(
                    f"PassManager::apply: pass_instance={type(pass_instance)}: results({len(results)})={results}"
                )
            except Exception as e:
                # pass was not applied
                logger.debug(
                    f"PassManager::apply: pass_instance={type(pass_instance)}: ignored={e}"
                )

            if not updated:
                updated = len(results) != 0

        if updated:
            self.graph_module.recompile()
