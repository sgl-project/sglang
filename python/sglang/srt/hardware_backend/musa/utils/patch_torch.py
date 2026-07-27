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
import re
from dataclasses import replace as _dataclass_replace

import torch
import torch.fx.graph as fx_graph

_DEVICE_REPR_RE = re.compile(r"\bdevice\(type='([^']+)'(?:,\s*index=(\d+))?\)")


def _replace_device_repr(m: re.Match) -> str:
    dev_type = m.group(1)
    dev_index = m.group(2)
    if dev_index is not None:
        return f"torch.device('{dev_type}:{dev_index}')"
    return f"torch.device('{dev_type}')"


def patch_fx_custom_device() -> None:
    """
    Fix FX codegen serialization for non-standard devices (e.g. torch_musa).

    Root cause:
    torch.device is registered as a custom builtin named 'device', imported
    via 'from torch import device'. repr(torch.device('musa', 0)) produces
    "device(type='musa', index=0)", which is syntactically valid but fails
    at runtime because torch.device does not recognize 'musa' as a type when
    invoked through the standard import path.

    Fix:
    Post-process the generated src string, replacing all occurrences of
    device(type='x', index=N) with torch.device('x:N'), and ensure 'torch'
    is present in the graph globals.

    Note:
    _get_repr is a closure inside _gen_python_code and cannot be patched
    directly, so we wrap _gen_python_code and rewrite its output instead.
    """
    original = fx_graph.CodeGen._gen_python_code

    def patched(self, nodes, root_module, namespace, **kwargs):
        result = original(self, nodes, root_module, namespace, **kwargs)
        new_src = _DEVICE_REPR_RE.sub(_replace_device_repr, result.src)
        if new_src is result.src:
            return result
        result.globals.setdefault("torch", torch)
        if hasattr(result, "_replace"):
            return result._replace(src=new_src)
        return _dataclass_replace(result, src=new_src)

    fx_graph.CodeGen._gen_python_code = patched
