# Copyright 2026 SGLang Team
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

import dataclasses
import logging
from typing import Any, Dict, List, Optional, Set, Tuple

from sglang.srt.environ import envs
from sglang.srt.utils.log_utils import create_log_targets, log_json
from sglang.srt.utils.request_logger import (
    _dataclass_to_string_truncated,
    _transform_data_for_logging,
)

logger = logging.getLogger(__name__)

# Whitelist of Req fields that are meaningful for users.
_WHITELIST_FIELDS: Set[str] = {
    "prompt",
    "negative_prompt",
    "sampling_params",
    "output_file_ext",
    "original_condition_image_size",
    "max_sequence_length",
    "prompt_template",
    "do_classifier_free_guidance",
    "seeds",
    "is_prompt_processed",
    "generate_audio",
}


def _allow_to_skip_names(data: Any, allow_names: Set[str]) -> Set[str]:
    if dataclasses.is_dataclass(data):
        all_names = {f.name for f in dataclasses.fields(data)}
    elif isinstance(data, dict):
        all_names = set(data.keys())
    else:
        return set()
    return all_names - allow_names


def _extract_req_metadata(req: Any) -> Dict[str, Any]:
    return {
        "request_id": getattr(req, "request_id", None),
        "is_warmup": getattr(req, "is_warmup", False),
    }


class DiffusionRequestLogger:
    def __init__(
        self,
        log_requests: bool,
        log_requests_level: int,
        log_requests_format: str,
        log_requests_target: Optional[List[str]],
    ):
        self.log_requests = log_requests
        self.log_requests_level = log_requests_level
        self.log_requests_format = log_requests_format
        self.log_requests_target = log_requests_target

        self.metadata: Tuple[int, Set[str]] = self._compute_metadata()
        self.targets = create_log_targets(
            targets=self.log_requests_target, name_prefix=__name__
        )
        self.log_exceeded_ms = envs.SGLANG_LOG_REQUEST_EXCEEDED_MS.get()

    def _compute_metadata(self) -> Tuple[int, Set[str]]:
        """Compute max_length and allow_names based on log level.

        Uses a whitelist approach: only fields in _WHITELIST_FIELDS are logged.
        Different levels control which subset of the whitelist is allowed:

        - Level 0: metadata only (request_id, is_warmup)
        - Level 1: metadata + sampling_params
        - Level 2: metadata + sampling_params + partial prompt/negative_prompt
        - Level 3: all whitelisted fields (full prompt, full output)
        """
        max_length: int = 1 << 30
        allow_names: Set[str] = set()

        if self.log_requests:
            _LEVEL_0_FIELDS: Set[str] = set()
            _LEVEL_1_FIELDS = _LEVEL_0_FIELDS | {
                "sampling_params",
            }
            _LEVEL_2_FIELDS = _LEVEL_1_FIELDS | {
                "prompt",
                "negative_prompt",
            }
            _LEVEL_3_FIELDS = _WHITELIST_FIELDS

            if self.log_requests_level == 0:
                allow_names = _LEVEL_0_FIELDS
            elif self.log_requests_level == 1:
                allow_names = _LEVEL_1_FIELDS
            elif self.log_requests_level == 2:
                allow_names = _LEVEL_2_FIELDS
                max_length = 2048
            elif self.log_requests_level == 3:
                allow_names = _LEVEL_3_FIELDS
            else:
                raise ValueError(
                    f"Invalid --log-requests-level: {self.log_requests_level=}"
                )

        return max_length, allow_names

    def _log(self, msg: str) -> None:
        for target in self.targets:
            target.info(msg)

    def _format_req_text(self, req: Any, max_length: int, skip_names: Set[str]) -> str:
        obj_str = _dataclass_to_string_truncated(req, max_length, skip_names=skip_names)
        req_meta = _extract_req_metadata(req)
        meta_str = ", ".join(f"{k}={v!r}" for k, v in req_meta.items())
        prefix_len = len(f"{req.__class__.__name__}(")
        inner = obj_str[prefix_len:-1]
        if inner:
            return f"Req({meta_str}, {inner})"
        else:
            return f"Req({meta_str})"

    def log_received_request(self, req: Any) -> None:
        if not self.log_requests:
            return

        max_length, allow_names = self.metadata
        skip_names = _allow_to_skip_names(req, allow_names)
        req_meta = _extract_req_metadata(req)
        rid = req_meta["request_id"] or "unknown"

        if self.log_requests_format == "json":
            obj = _transform_data_for_logging(req, max_length, skip_names)
            if "sampling_params" in obj and isinstance(obj["sampling_params"], dict):
                obj["sampling_params"].pop("request_id", None)
            obj.update(req_meta)
            log_data = {
                "rid": rid,
                "obj": obj,
            }
            log_json(self.targets, "request.received", log_data)
        else:
            self._log(
                f"Receive: obj={self._format_req_text(req, max_length, skip_names)}"
            )

    def log_finished_request(self, req: Any, output_batch: Any) -> None:
        if not self.log_requests:
            return

        e2e_latency = 0.0
        if (
            output_batch is not None
            and hasattr(output_batch, "metrics")
            and output_batch.metrics is not None
        ):
            e2e_latency = getattr(output_batch.metrics, "total_duration_s", 0.0) or 0.0

        if self.log_exceeded_ms > 0 and e2e_latency * 1000 < self.log_exceeded_ms:
            return

        max_length, allow_names = self.metadata
        skip_names = _allow_to_skip_names(req, allow_names)
        req_meta = _extract_req_metadata(req)
        rid = req_meta["request_id"] or "unknown"

        sp = getattr(req, "sampling_params", None)
        out = {
            "meta_info": {
                "e2e_latency": e2e_latency,
                "num_inference_steps": (
                    getattr(sp, "num_inference_steps", None) if sp else None
                ),
                "seed": getattr(sp, "seed", None) if sp else None,
                "guidance_scale": getattr(sp, "guidance_scale", None) if sp else None,
                "width": getattr(sp, "width", None) if sp else None,
                "height": getattr(sp, "height", None) if sp else None,
                "num_frames": getattr(sp, "num_frames", None) if sp else None,
                "fps": getattr(sp, "fps", None) if sp else None,
            },
            "output_shape": (
                list(output_batch.output.shape)
                if output_batch
                and hasattr(output_batch, "output")
                and output_batch.output is not None
                else None
            ),
            "error": getattr(output_batch, "error", None) if output_batch else None,
        }

        if self.log_requests_format == "json":
            obj = _transform_data_for_logging(req, max_length, skip_names)
            if "sampling_params" in obj and isinstance(obj["sampling_params"], dict):
                obj["sampling_params"].pop("request_id", None)
            obj.update(req_meta)
            log_data = {
                "rid": rid,
                "obj": obj,
                "out": out,
            }
            log_json(self.targets, "request.finished", log_data)
        else:
            out_str = f", out={_dataclass_to_string_truncated(out, max_length)}"
            self._log(
                f"Finish: obj={self._format_req_text(req, max_length, skip_names)}{out_str}"
            )
