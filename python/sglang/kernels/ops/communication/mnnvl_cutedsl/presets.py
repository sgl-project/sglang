# Copyright (c) 2026 by FlashInfer team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Built-in routing configurations for the MNNVL CuTe DSL backend."""

import torch

from .config import (
    KernelTarget,
    MNNVLCuteDSLConfig,
    MRangeDispatch,
    ProtocolKind,
    StaticProfile,
)
from .kernel_bt import (
    BT_ALL_REDUCE_GB300_TP8_H8192_PRESET_0,
    BT_ALL_REDUCE_GB300_TP8_H8192_PRESET_1,
    BT_ALL_REDUCE_GB300_TP16_H8192_PRESET_0,
    BT_ALL_REDUCE_GB300_TP16_H8192_PRESET_1,
    BT_FINALIZE_GB300_TP8_H8192_K10_PRESET_0,
    BT_FINALIZE_GB300_TP8_H8192_K10_PRESET_1,
    BT_FINALIZE_GB300_TP16_H8192_K10_PRESET_0,
    BT_FINALIZE_GB300_TP16_H8192_K10_PRESET_1,
)
from .kernel_ht import (
    HT_ALL_REDUCE_GB300_TP8_H8192,
    HT_ALL_REDUCE_GB300_TP16_H8192,
    HT_FINALIZE_GB300_TP8_H8192_K10,
    HT_FINALIZE_GB300_TP16_H8192_K10,
)
from .kernel_ll import (
    LL_ALL_REDUCE_GB300_TP8_H8192,
    LL_ALL_REDUCE_GB300_TP16_H8192,
    LL_FINALIZE_GB300_TP8_H8192_K10,
    LL_FINALIZE_GB300_TP16_H8192_K10,
)

__all__ = [
    "BT_ONLY_CONFIG",
    "DEFAULT_CONFIG",
    "HT_ONLY_CONFIG",
    "LL_ONLY_CONFIG",
]


def _target(protocol: ProtocolKind, preset: object) -> KernelTarget[object]:
    return KernelTarget(protocol=protocol, preset=preset)


LL_ONLY_CONFIG = MNNVLCuteDSLConfig(
    profiles=(
        StaticProfile(
            tp_size=8,
            hidden_size=8192,
            top_k=10,
            dtype=torch.bfloat16,
            finalize_routes=MRangeDispatch(
                upper_bounds=(None,),
                targets=(
                    _target(
                        ProtocolKind.LL,
                        LL_FINALIZE_GB300_TP8_H8192_K10,
                    ),
                ),
            ),
            all_reduce_routes=MRangeDispatch(
                upper_bounds=(None,),
                targets=(
                    _target(
                        ProtocolKind.LL,
                        LL_ALL_REDUCE_GB300_TP8_H8192,
                    ),
                ),
            ),
        ),
        StaticProfile(
            tp_size=16,
            hidden_size=8192,
            top_k=10,
            dtype=torch.bfloat16,
            finalize_routes=MRangeDispatch(
                upper_bounds=(None,),
                targets=(
                    _target(
                        ProtocolKind.LL,
                        LL_FINALIZE_GB300_TP16_H8192_K10,
                    ),
                ),
            ),
            all_reduce_routes=MRangeDispatch(
                upper_bounds=(None,),
                targets=(
                    _target(
                        ProtocolKind.LL,
                        LL_ALL_REDUCE_GB300_TP16_H8192,
                    ),
                ),
            ),
        ),
    )
)


BT_ONLY_CONFIG = MNNVLCuteDSLConfig(
    profiles=(
        StaticProfile(
            tp_size=8,
            hidden_size=8192,
            top_k=10,
            dtype=torch.bfloat16,
            finalize_routes=MRangeDispatch(
                upper_bounds=(48, 1024),
                targets=(
                    _target(
                        ProtocolKind.BT,
                        BT_FINALIZE_GB300_TP8_H8192_K10_PRESET_0,
                    ),
                    _target(
                        ProtocolKind.BT,
                        BT_FINALIZE_GB300_TP8_H8192_K10_PRESET_1,
                    ),
                ),
            ),
            all_reduce_routes=MRangeDispatch(
                upper_bounds=(256, 1024),
                targets=(
                    _target(
                        ProtocolKind.BT,
                        BT_ALL_REDUCE_GB300_TP8_H8192_PRESET_0,
                    ),
                    _target(
                        ProtocolKind.BT,
                        BT_ALL_REDUCE_GB300_TP8_H8192_PRESET_1,
                    ),
                ),
            ),
        ),
        StaticProfile(
            tp_size=16,
            hidden_size=8192,
            top_k=10,
            dtype=torch.bfloat16,
            finalize_routes=MRangeDispatch(
                upper_bounds=(52, 1024),
                targets=(
                    _target(
                        ProtocolKind.BT,
                        BT_FINALIZE_GB300_TP16_H8192_K10_PRESET_0,
                    ),
                    _target(
                        ProtocolKind.BT,
                        BT_FINALIZE_GB300_TP16_H8192_K10_PRESET_1,
                    ),
                ),
            ),
            all_reduce_routes=MRangeDispatch(
                upper_bounds=(512, 1024),
                targets=(
                    _target(
                        ProtocolKind.BT,
                        BT_ALL_REDUCE_GB300_TP16_H8192_PRESET_0,
                    ),
                    _target(
                        ProtocolKind.BT,
                        BT_ALL_REDUCE_GB300_TP16_H8192_PRESET_1,
                    ),
                ),
            ),
        ),
    )
)


HT_ONLY_CONFIG = MNNVLCuteDSLConfig(
    profiles=(
        StaticProfile(
            tp_size=8,
            hidden_size=8192,
            top_k=10,
            dtype=torch.bfloat16,
            finalize_routes=MRangeDispatch(
                upper_bounds=(None,),
                targets=(
                    _target(
                        ProtocolKind.HT,
                        HT_FINALIZE_GB300_TP8_H8192_K10,
                    ),
                ),
            ),
            all_reduce_routes=MRangeDispatch(
                upper_bounds=(None,),
                targets=(
                    _target(
                        ProtocolKind.HT,
                        HT_ALL_REDUCE_GB300_TP8_H8192,
                    ),
                ),
            ),
        ),
        StaticProfile(
            tp_size=16,
            hidden_size=8192,
            top_k=10,
            dtype=torch.bfloat16,
            finalize_routes=MRangeDispatch(
                upper_bounds=(None,),
                targets=(
                    _target(
                        ProtocolKind.HT,
                        HT_FINALIZE_GB300_TP16_H8192_K10,
                    ),
                ),
            ),
            all_reduce_routes=MRangeDispatch(
                upper_bounds=(None,),
                targets=(
                    _target(
                        ProtocolKind.HT,
                        HT_ALL_REDUCE_GB300_TP16_H8192,
                    ),
                ),
            ),
        ),
    )
)


DEFAULT_CONFIG = MNNVLCuteDSLConfig(
    profiles=(
        StaticProfile(
            tp_size=8,
            hidden_size=8192,
            top_k=10,
            dtype=torch.bfloat16,
            finalize_routes=MRangeDispatch(
                upper_bounds=(23, 48, 703, None),
                targets=(
                    _target(
                        ProtocolKind.LL,
                        LL_FINALIZE_GB300_TP8_H8192_K10,
                    ),
                    _target(
                        ProtocolKind.BT,
                        BT_FINALIZE_GB300_TP8_H8192_K10_PRESET_0,
                    ),
                    _target(
                        ProtocolKind.BT,
                        BT_FINALIZE_GB300_TP8_H8192_K10_PRESET_1,
                    ),
                    _target(
                        ProtocolKind.HT,
                        HT_FINALIZE_GB300_TP8_H8192_K10,
                    ),
                ),
            ),
            all_reduce_routes=MRangeDispatch(
                upper_bounds=(15, 256, 1024, None),
                targets=(
                    _target(
                        ProtocolKind.LL,
                        LL_ALL_REDUCE_GB300_TP8_H8192,
                    ),
                    _target(
                        ProtocolKind.BT,
                        BT_ALL_REDUCE_GB300_TP8_H8192_PRESET_0,
                    ),
                    _target(
                        ProtocolKind.BT,
                        BT_ALL_REDUCE_GB300_TP8_H8192_PRESET_1,
                    ),
                    _target(
                        ProtocolKind.HT,
                        HT_ALL_REDUCE_GB300_TP8_H8192,
                    ),
                ),
            ),
        ),
        StaticProfile(
            tp_size=16,
            hidden_size=8192,
            top_k=10,
            dtype=torch.bfloat16,
            finalize_routes=MRangeDispatch(
                upper_bounds=(7, 52, 703, None),
                targets=(
                    _target(
                        ProtocolKind.LL,
                        LL_FINALIZE_GB300_TP16_H8192_K10,
                    ),
                    _target(
                        ProtocolKind.BT,
                        BT_FINALIZE_GB300_TP16_H8192_K10_PRESET_0,
                    ),
                    _target(
                        ProtocolKind.BT,
                        BT_FINALIZE_GB300_TP16_H8192_K10_PRESET_1,
                    ),
                    _target(
                        ProtocolKind.HT,
                        HT_FINALIZE_GB300_TP16_H8192_K10,
                    ),
                ),
            ),
            all_reduce_routes=MRangeDispatch(
                upper_bounds=(5, 512, 959, None),
                targets=(
                    _target(
                        ProtocolKind.LL,
                        LL_ALL_REDUCE_GB300_TP16_H8192,
                    ),
                    _target(
                        ProtocolKind.BT,
                        BT_ALL_REDUCE_GB300_TP16_H8192_PRESET_0,
                    ),
                    _target(
                        ProtocolKind.BT,
                        BT_ALL_REDUCE_GB300_TP16_H8192_PRESET_1,
                    ),
                    _target(
                        ProtocolKind.HT,
                        HT_ALL_REDUCE_GB300_TP16_H8192,
                    ),
                ),
            ),
        ),
    )
)
