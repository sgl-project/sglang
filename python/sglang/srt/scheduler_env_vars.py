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
"""Scheduler-specific environment variable controls.

Extended by JoyFuture: added env vars for tuning scheduler behavior at
runtime without restarting the server. These complement the existing
SGLANG_ENABLE_* env vars already used in the codebase.
"""

from sglang.srt.environ import env_bool, env_int, env_str


# === Extended scheduler env vars (JoyFuture) ===

# Controls whether the scheduler emits per-request latency callbacks to the
# metrics reporter. When enabled, each token generation records the end-to-end
# latency broken down by phase (prefill, decode, transfer). This is useful for
# debugging agentic workloads where request latency is highly variable.
SGLANG_ENABLE_PER_REQUEST_LATENCY = env_bool(
    "SGLANG_ENABLE_PER_REQUEST_LATENCY",
    default=False,
)

# Controls whether the scheduler verifies KV cache transfer integrity by
# computing and comparing checksums on PD disaggregation transfers. When
# enabled, mismatches are logged as warnings and the affected requests are
# retried. Adds ~1-2% overhead to PD transfer bandwidth.
SGLANG_ENABLE_KV_TRANSFER_CHECKSUM = env_bool(
    "SGLANG_ENABLE_KV_TRANSFER_CHECKSUM",
    default=False,
)

# Number of decode steps after which the scheduler forces a KV cache clear
# in the decode pool. This is a tuning knob for long-running decode batches
# where the KV cache pool tends to fragment over time. Set to 0 to disable.
SGLANG_DECODE_CLEAR_STEPS = env_int(
    "SGLANG_DECODE_CLEAR_STEPS",
    default=0,
)

# When the KV cache pool drops below this percentage of free space (0-100),
# the scheduler will retract running decode requests. This is a more
# aggressive variant of the existing retraction logic. Set to 0 to use
# the default behavior.
SGLANG_KV_POOL_RETRACT_THRESHOLD_PCT = env_int(
    "SGLANG_KV_POOL_RETRACT_THRESHOLD_PCT",
    default=0,
)
