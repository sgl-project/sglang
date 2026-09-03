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

"""MNNVL CuTe DSL AllReduce fusion backend internals."""

from importlib import import_module

from .config import (
    KernelTarget,
    MNNVLCuteDSLConfig,
    MRangeDispatch,
    ProtocolKind,
    StaticProfile,
)


def __getattr__(name: str):
    if name in {
        "BT_ONLY_CONFIG",
        "DEFAULT_CONFIG",
        "HT_ONLY_CONFIG",
        "LL_ONLY_CONFIG",
    }:
        presets = import_module(f"{__name__}.presets")
        value = getattr(presets, name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "BT_ONLY_CONFIG",
    "DEFAULT_CONFIG",
    "HT_ONLY_CONFIG",
    "LL_ONLY_CONFIG",
    "KernelTarget",
    "MNNVLCuteDSLConfig",
    "MRangeDispatch",
    "ProtocolKind",
    "StaticProfile",
]
