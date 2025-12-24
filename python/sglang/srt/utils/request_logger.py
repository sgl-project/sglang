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
"""Request logging utilities."""
from __future__ import annotations

import dataclasses
import logging
from functools import lru_cache
from typing import TYPE_CHECKING, Any, Optional, Set, Tuple, Union

from sglang.srt.utils.common import get_bool_env_var

if TYPE_CHECKING:
    from sglang.srt.managers.io_struct import EmbeddingReqInput, GenerateReqInput

logger = logging.getLogger(__name__)


@lru_cache(maxsize=2)
def disable_request_logging() -> bool:
    return get_bool_env_var("SGLANG_DISABLE_REQUEST_LOGGING")


def dataclass_to_string_truncated(
    data: Any, max_length: int = 2048, skip_names: Optional[Set[str]] = None
) -> str:
    if skip_names is None:
        skip_names = set()
    if isinstance(data, str):
        if len(data) > max_length:
            half_length = max_length // 2
            return f"{repr(data[:half_length])} ... {repr(data[-half_length:])}"
        else:
            return f"{repr(data)}"
    elif isinstance(data, (list, tuple)):
        if len(data) > max_length:
            half_length = max_length // 2
            return str(data[:half_length]) + " ... " + str(data[-half_length:])
        else:
            return str(data)
    elif isinstance(data, dict):
        return (
            "{"
            + ", ".join(
                f"'{k}': {dataclass_to_string_truncated(v, max_length)}"
                for k, v in data.items()
                if k not in skip_names
            )
            + "}"
        )
    elif dataclasses.is_dataclass(data):
        fields = dataclasses.fields(data)
        return (
            f"{data.__class__.__name__}("
            + ", ".join(
                f"{f.name}={dataclass_to_string_truncated(getattr(data, f.name), max_length)}"
                for f in fields
                if f.name not in skip_names
            )
            + ")"
        )
    else:
        return str(data)


class RequestLogger:
    def __init__(self, log_requests: bool, log_requests_level: int):
        self.log_requests = log_requests
        self.log_requests_level = log_requests_level
        self._metadata: Tuple[Optional[int], Optional[Set[str]], Optional[Set[str]]] = (
            self._compute_metadata()
        )

    def _compute_metadata(
        self,
    ) -> Tuple[Optional[int], Optional[Set[str]], Optional[Set[str]]]:
        max_length: Optional[int] = None
        skip_names: Optional[Set[str]] = None
        out_skip_names: Optional[Set[str]] = None
        if self.log_requests:
            if self.log_requests_level == 0:
                max_length = 1 << 30
                skip_names = {
                    "text",
                    "input_ids",
                    "input_embeds",
                    "image_data",
                    "audio_data",
                    "lora_path",
                    "sampling_params",
                }
                out_skip_names = {"text", "output_ids", "embedding"}
            elif self.log_requests_level == 1:
                max_length = 1 << 30
                skip_names = {
                    "text",
                    "input_ids",
                    "input_embeds",
                    "image_data",
                    "audio_data",
                    "lora_path",
                }
                out_skip_names = {"text", "output_ids", "embedding"}
            elif self.log_requests_level == 2:
                max_length = 2048
            elif self.log_requests_level == 3:
                max_length = 1 << 30
            else:
                raise ValueError(
                    f"Invalid --log-requests-level: {self.log_requests_level=}"
                )
        return max_length, skip_names, out_skip_names

    @property
    def metadata(self) -> Tuple[Optional[int], Optional[Set[str]], Optional[Set[str]]]:
        return self._metadata

    def configure(
        self,
        log_requests: Optional[bool] = None,
        log_requests_level: Optional[int] = None,
    ) -> None:
        if log_requests is not None:
            self.log_requests = log_requests
        if log_requests_level is not None:
            self.log_requests_level = log_requests_level
        self._metadata = self._compute_metadata()

    def log_received_request(
        self, obj: Union["GenerateReqInput", "EmbeddingReqInput"], tokenizer: Any = None
    ) -> None:
        if not self.log_requests:
            return

        max_length, skip_names, _ = self._metadata
        logger.info(
            f"Receive: obj={dataclass_to_string_truncated(obj, max_length, skip_names=skip_names)}"
        )

        # Decode input_ids to text if needed for logging
        if (
            self.log_requests_level >= 2
            and obj.text is None
            and obj.input_ids is not None
            and tokenizer is not None
        ):
            decoded = tokenizer.decode(obj.input_ids, skip_special_tokens=False)
            obj.text = decoded

    def log_finished_request(
        self,
        obj: Union["GenerateReqInput", "EmbeddingReqInput"],
        out: Any,
        is_multimodal_gen: bool = False,
    ) -> None:
        if not self.log_requests:
            return

        max_length, skip_names, out_skip_names = self._metadata
        if is_multimodal_gen:
            msg = f"Finish: obj={dataclass_to_string_truncated(obj, max_length, skip_names=skip_names)}"
        else:
            msg = f"Finish: obj={dataclass_to_string_truncated(obj, max_length, skip_names=skip_names)}, out={dataclass_to_string_truncated(out, max_length, skip_names=out_skip_names)}"
        logger.info(msg)

