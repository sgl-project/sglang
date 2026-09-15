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


from typing import Any, Optional

from sglang.srt.environ import envs
from sglang.srt.utils.log_utils import create_log_targets, log_json
from sglang.srt.utils.request_logger import (
    _dataclass_to_string_truncated,
    _transform_data_for_logging,
)

# Core generation knobs logged per record. Prompt text is logged separately,
# gated by the level, so it is excluded here.
_SAMPLING_CONFIG_FIELDS = (
    "data_type",
    "seed",
    "num_inference_steps",
    "guidance_scale",
    "true_cfg_scale",
    "width",
    "height",
    "num_frames",
    "fps",
    "num_outputs_per_prompt",
)

# Level 2 truncates prompt text; level 3 keeps it whole. Lower levels log no
# prompt at all.
_TRUNCATE_LENGTH = 2048
_UNLIMITED = 1 << 30


class DiffusionRequestLogger:
    def __init__(
        self,
        log_requests: bool,
        log_requests_level: int,
        log_requests_format: str,
        log_requests_target: Optional[list],
    ):
        self.log_requests = log_requests
        self.log_requests_level = log_requests_level
        self.log_requests_format = log_requests_format
        self.log_requests_target = log_requests_target
        self.targets = create_log_targets(
            targets=log_requests_target, name_prefix=__name__
        )
        self.log_exceeded_ms = envs.SGLANG_LOG_REQUEST_EXCEEDED_MS.get()
        self._max_length = (
            _TRUNCATE_LENGTH if self.log_requests_level == 2 else _UNLIMITED
        )

    @classmethod
    def from_server_args(cls, server_args: Any) -> "DiffusionRequestLogger":
        """Build a logger from server args."""
        return cls(
            log_requests=server_args.log_requests,
            log_requests_level=server_args.log_requests_level,
            log_requests_format=server_args.log_requests_format,
            log_requests_target=server_args.log_requests_target,
        )

    @staticmethod
    def _request_id(req: Any) -> Optional[str]:
        """The request's id, or ``None`` if absent."""
        return getattr(req, "request_id", None)

    def _config_view(self, req: Any, *, drop_seed: bool = False) -> dict:
        """Sampling config + (level >= 2) prompt, gated by the log level.
        Returns ``{}`` below level 1."""
        sp = getattr(req, "sampling_params", None)
        if sp is None or self.log_requests_level < 1:
            return {}
        cfg = {name: getattr(sp, name, None) for name in _SAMPLING_CONFIG_FIELDS}
        if drop_seed:
            cfg.pop("seed", None)
        view: dict = {"sampling_params": cfg}
        if self.log_requests_level >= 2:
            view["prompt"] = getattr(sp, "prompt", None)
            view["negative_prompt"] = getattr(sp, "negative_prompt", None)
        return view

    def _result_view(self, result: Any) -> dict:
        """Result-side fields for a finished record: latency and error."""
        e2e_latency = 0.0
        metrics = getattr(result, "metrics", None) if result is not None else None
        if metrics is not None:
            e2e_latency = getattr(metrics, "total_duration_s", 0.0) or 0.0
        return {
            "meta_info": {"e2e_latency": e2e_latency},
            "error": getattr(result, "error", None) if result is not None else None,
        }

    def _emit(self, msg: str) -> None:
        for target in self.targets:
            target.info(msg)

    def _per_request_view(self, req: Any) -> dict:
        """Per-output identity within a batch: ``request_id`` plus ``seed`` at
        level >= 1."""
        sp = getattr(req, "sampling_params", None)
        view: dict = {"request_id": self._request_id(req)}
        if self.log_requests_level >= 1:
            view["seed"] = getattr(sp, "seed", None) if sp is not None else None
        return view

    def _batch_record(self, reqs: list) -> tuple:
        """Build the ``rid`` / ``obj`` for one forward call"""
        rids = [self._request_id(req) or "unknown" for req in reqs]
        if len(reqs) == 1:
            # Single request: scalar rid + flat dict obj (id + config).
            req = reqs[0]
            obj = {"request_id": self._request_id(req), **self._config_view(req)}
            return rids[0], obj

        shared_views = [self._config_view(req, drop_seed=True) for req in reqs]
        if all(view == shared_views[0] for view in shared_views):
            obj = {
                **shared_views[0],
                "outputs": [self._per_request_view(req) for req in reqs],
            }
        else:
            # Configs genuinely differ: list each request's full payload verbatim.
            obj = [
                {"request_id": self._request_id(req), **self._config_view(req)}
                for req in reqs
            ]
        return rids, obj

    def _loggable(self, req: Any) -> bool:
        """Whether ``req`` should be recorded: logging is on, it's a real
        generation request (control messages -- LoRA / weight / stats / shutdown
        -- have no ``sampling_params`` and are skipped), and it's not a warmup."""
        return (
            self.log_requests
            and getattr(req, "sampling_params", None) is not None
            and not getattr(req, "is_warmup", False)
        )

    def _logged_reqs(self, batch: Any) -> list:
        """Normalize ``batch`` to a list and drop control / warmup requests."""
        reqs = batch if isinstance(batch, (list, tuple)) else [batch]
        return [r for r in reqs if self._loggable(r)]

    def log_received_request(self, batch: Any) -> None:
        reqs = self._logged_reqs(batch)
        if not reqs:
            return

        rid, obj = self._batch_record(reqs)
        max_length = self._max_length

        if self.log_requests_format == "json":
            log_json(
                self.targets,
                "request.received",
                {"rid": rid, "obj": _transform_data_for_logging(obj, max_length)},
            )
        else:
            self._emit(
                f"Receive: obj={_dataclass_to_string_truncated(obj, max_length)}"
            )

    def log_finished_request(self, batch: Any, result: Any) -> None:
        reqs = self._logged_reqs(batch)
        if not reqs:
            return

        out = self._result_view(result)
        e2e_latency_ms = out["meta_info"]["e2e_latency"] * 1000
        if self.log_exceeded_ms > 0 and e2e_latency_ms < self.log_exceeded_ms:
            return

        rid, obj = self._batch_record(reqs)
        max_length = self._max_length

        if self.log_requests_format == "json":
            log_json(
                self.targets,
                "request.finished",
                {
                    "rid": rid,
                    "obj": _transform_data_for_logging(obj, max_length),
                    "out": _transform_data_for_logging(out, max_length),
                },
            )
        else:
            self._emit(
                f"Finish: obj={_dataclass_to_string_truncated(obj, max_length)}"
                f", out={_dataclass_to_string_truncated(out, max_length)}"
            )
