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

from __future__ import annotations

import torch
from torch._dynamo.decorators import skip
from torch._dynamo.eval_frame import DisableContext, innermost_fn


def patch_dynamo_context():
    setattr(torch._dynamo.eval_frame.DisableContext, "compiled_function_args", {})
    setattr(torch._dynamo.eval_frame.DisableContext, "compiled_function", {})
    setattr(torch._dynamo.eval_frame.DisableContext, "batch_size", None)


original_disable_context_call = None
original_disable = None


def decorators_disable(fn=None, recursive=True):
    if recursive:
        if fn is not None:
            fn = innermost_fn(fn)
            assert callable(fn)

            DisableContext.compiled_function[DisableContext.batch_size] = fn
            return DisableContext()(fn)
        return DisableContext()
    else:
        return skip(fn)


def patch_dynamo_context_call():
    global original_disable
    original_disable = torch._dynamo.decorators.disable
    torch._dynamo.decorators.disable = decorators_disable


def restore_dynamo_context_call():
    global original_disable
    torch._dynamo.decorators.disable = original_disable
    original_disable = None