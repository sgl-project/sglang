# Copyright 2023-2026 SGLang Team
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
"""CUDA runtime binding utilities."""

try:
    from cuda.bindings import runtime as rt
except ImportError:
    rt = None


def _cudaGetErrorString(error):
    err, msg = rt.cudaGetErrorString(error)
    if err != rt.cudaError_t.cudaSuccess:
        return "<unknown>"
    if isinstance(msg, bytes):
        return msg.decode("utf-8", "replace")
    return str(msg)


def checkCudaErrors(result):
    if result[0] != rt.cudaError_t.cudaSuccess:
        raise RuntimeError(
            f"CUDA error {int(result[0])}({_cudaGetErrorString(result[0])})"
        )
    if len(result) == 1:
        return None
    elif len(result) == 2:
        return result[1]
    else:
        return result[1:]
