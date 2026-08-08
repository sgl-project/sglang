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

from enum import Enum


class AbortReason(str, Enum):
    UNSPECIFIED = "unspecified"
    HTTP_CLIENT_DISCONNECT_WAITING = "http_client_disconnect_waiting"
    HTTP_CLIENT_DISCONNECT_RUNNING = "http_client_disconnect_running"
    HTTP_CLIENT_DISCONNECT_STREAM = "http_client_disconnect_stream"
    HTTP_ABORT_REQUEST = "http_abort_request"
    RESPONSES_CANCEL = "responses_cancel"
    GRPC_BACKPRESSURE_TIMEOUT = "grpc_backpressure_timeout"
    GRPC_ABORT = "grpc_abort"
    PAUSE_GENERATION = "pause_generation"
    WEIGHT_UPDATE = "weight_update"
    WEIGHT_VERSION_UPDATE = "weight_version_update"
    REQUEST_CLEANUP = "request_cleanup"
    PRIORITY_DISABLED = "priority_disabled"
    QUEUE_FULL = "queue_full"
    PRIORITY_PREEMPTED = "priority_preempted"
    WAITING_TIMEOUT = "waiting_timeout"
    RUNNING_TIMEOUT = "running_timeout"
    KV_CACHE_EXHAUSTED = "kv_cache_exhausted"
    INVALID_REQUEST = "invalid_request"
    GRAMMAR_ERROR = "grammar_error"
    DISAGGREGATION_ERROR = "disaggregation_error"
    MULTIMODAL_ERROR = "multimodal_error"
    SESSION_CLEANUP = "session_cleanup"
