# Copyright 2023-2024 SGLang Team
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
"""Elastic EP failures a rank cannot serve through.

Its own module so the two raisers can share a base without importing each other:
``eplb_manager`` is heavy and ``expert_map_repair`` has no reason to pull it in.
"""


class ElasticLayoutFatal(RuntimeError):
    """This rank's expert map and the weights behind it no longer agree.

    Not survivable, and not reportable either. A rank in this state keeps answering,
    from one expert's weights under another's label, and nothing downstream can tell.
    Raised past the scale-down FSM so the scheduler goes down instead.
    """
