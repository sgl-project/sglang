# Copied and adapted from: https://github.com/hao-ai-lab/FastVideo

# SPDX-License-Identifier: Apache-2.0
import os
import tempfile
from typing import Any, Awaitable, Callable

from tqdm.auto import tqdm

from sglang.multimodal_gen.runtime.pipelines_core.schedule_batch import (
    OutputBatch,
    Req,
)
from sglang.multimodal_gen.runtime.server_args import ServerArgs
from sglang.multimodal_gen.runtime.utils.image_io import save_base64_image_to_path
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger
from sglang.multimodal_gen.runtime.warmup_request_builder import (
    build_warmup_reqs,
    should_include_warmup_image,
)

logger = init_logger(__name__)

MINIMUM_PICTURE_BASE64_FOR_WARMUP = "data:image/jpg;base64,iVBORw0KGgoAAAANSUhEUgAAACAAAAAgCAYAAABzenr0AAAACXBIWXMAAA7EAAAOxAGVKw4bAAAAbUlEQVRYhe3VsQ2AMAxE0Y/lIgNQULD/OqyCMgCihCKSG4yRuKuiNH6JLsoEbMACOGBcua9HOR7Y6w6swBwMy0qLTpkeI77qdEBpBFAHBBDAGH8WrwJKI4AAegUCfAKgEgpQDvh3CR3oQCuav58qlAw73kKCSgAAAABJRU5ErkJggg=="


def get_first_generation_req(req_or_group: Any) -> Req | None:
    """Extract the first req"""
    if isinstance(req_or_group, Req):
        return req_or_group
    if isinstance(req_or_group, list) and req_or_group:
        first_req = req_or_group[0]
        if isinstance(first_req, Req):
            return first_req
    return None


def is_warmup_req(req_or_group: Any) -> bool:
    """either server-based or req-based"""
    req = get_first_generation_req(req_or_group)
    return req.is_warmup if req is not None else False


def is_server_based_warmup(req_or_group: Any) -> bool:
    req = get_first_generation_req(req_or_group)
    return (
        req is not None and req.is_warmup and bool(req.extra.get("server_based_warmup"))
    )


def should_return_warmup_result(req_or_group: Any) -> bool:
    # server-based warmup needs to return to the http server to finish the startup
    req = get_first_generation_req(req_or_group)
    return (
        req is not None
        and req.is_warmup
        and bool(req.extra.get("return_warmup_result"))
    )


def should_run_server_warmup(server_args: ServerArgs) -> bool:
    return server_args.warmup and server_args.server_warmup


def is_realtime_serving(server_args: ServerArgs) -> bool:
    """Synthetic warmup has no realtime session state."""
    try:
        from sglang.multimodal_gen.runtime.entrypoints.openai.realtime.registry import (
            get_realtime_model_adapter,
        )

        get_realtime_model_adapter(server_args)
        return True
    except Exception:
        return False


def should_run_synthetic_server_warmup(server_args: ServerArgs) -> bool:
    return should_run_server_warmup(server_args) and not is_realtime_serving(
        server_args
    )


def should_run_explicit_client_warmup(server_args: ServerArgs) -> bool:
    return server_args.warmup and server_args.warmup_resolutions is not None


def format_warmup_req(req_or_group: Any) -> str:
    req = get_first_generation_req(req_or_group)
    prefix = (
        "server warmup req" if is_server_based_warmup(req_or_group) else "warmup req"
    )
    if req is None:
        return prefix

    shape = f"{req.width}x{req.height}"
    if req.num_frames is not None and req.num_frames > 1:
        shape += f"x{req.num_frames}f"

    default_steps = req.extra.get("cache_dit_num_inference_steps")
    if default_steps is not None and default_steps != req.num_inference_steps:
        steps = f"{req.num_inference_steps}/{default_steps} steps"
    else:
        steps = f"{req.num_inference_steps} step"
        if req.num_inference_steps != 1:
            steps += "s"

    return f"{prefix} ({shape}, {steps})"


def build_client_warmup_reqs(
    server_args: ServerArgs,
    *,
    warmup_input_path: str | None = None,
) -> list[Req]:
    warmup_reqs = build_warmup_reqs(
        server_args,
        warmup_resolutions=server_args.warmup_resolutions,
        warmup_input_path=warmup_input_path,
        return_warmup_result=True,
        server_based_warmup=True,
    )
    warmup_total = len(warmup_reqs)
    for req in warmup_reqs:
        req.extra["warmup_total"] = warmup_total
    return warmup_reqs


async def run_async_client_warmup(
    server_args: ServerArgs,
    forward: Callable[[Req], Awaitable[OutputBatch]],
    *,
    fail_open: bool = False,
) -> None:
    try:
        warmup_input_path = None
        if should_include_warmup_image(server_args, server_based_warmup=True):
            warmup_input_path = prepare_warmup_image_path(server_args)

        for req in build_client_warmup_reqs(
            server_args, warmup_input_path=warmup_input_path
        ):
            response = await forward(req)
            if response.error is not None:
                raise RuntimeError(response.error)
    except Exception as e:
        if fail_open:
            logger.warning("Synthetic server warmup failed; continuing startup: %s", e)
            return
        raise


def run_sync_client_warmup(
    server_args: ServerArgs,
    forward: Callable[[Req], OutputBatch],
) -> None:
    warmup_input_path = None
    if should_include_warmup_image(server_args, server_based_warmup=True):
        warmup_input_path = prepare_warmup_image_path(server_args)

    for req in build_client_warmup_reqs(
        server_args, warmup_input_path=warmup_input_path
    ):
        response = forward(req)
        if response.error is not None:
            raise RuntimeError(response.error)


def prepare_warmup_image_path(server_args: ServerArgs) -> str:
    if server_args.input_save_path is not None:
        uploads_dir = server_args.input_save_path
        os.makedirs(uploads_dir, exist_ok=True)
    else:
        uploads_dir = tempfile.mkdtemp(prefix="sglang_input_")

    warmup_image_base = os.path.join(uploads_dir, "warmup_image")
    return save_base64_image_to_path(
        MINIMUM_PICTURE_BASE64_FOR_WARMUP, warmup_image_base
    )


class SchedulerWarmupMixin:
    @staticmethod
    def _format_warmup_req(req_or_group: Any) -> str:
        return format_warmup_req(req_or_group)

    def _warmup_progress_total(self, req_or_group: Any | None = None) -> int:
        req = get_first_generation_req(req_or_group)
        if req is not None:
            warmup_total = req.extra.get("warmup_total")
            if warmup_total is not None:
                return warmup_total

        return max(self._warmup_total, 1)

    def _ensure_warmup_progress_bar(self, req_or_group: Any) -> None:
        if not self._show_warmup_progress:
            return

        if self._warmup_progress_bar is None:
            self._warmup_progress_bar = tqdm(
                total=self._warmup_progress_total(req_or_group),
                desc="Warmup requests",
                unit="req",
            )
        self._warmup_progress_bar.set_postfix_str(
            self._format_warmup_req(req_or_group), refresh=False
        )

    def _advance_warmup_progress_bar(
        self, req_or_group: Any, output_batch: OutputBatch
    ) -> None:
        if not self._show_warmup_progress:
            return

        if self._warmup_progress_bar is None:
            self._ensure_warmup_progress_bar(req_or_group)

        if output_batch.metrics is not None:
            last_duration_s = output_batch.metrics.total_duration_s
            self._warmup_progress_bar.set_postfix_str(
                f"{self._format_warmup_req(req_or_group)}, last={last_duration_s:.2f}s",
                refresh=False,
            )
        self._warmup_progress_bar.update(1)

        if self._warmup_progress_bar.n >= self._warmup_progress_bar.total:
            self._warmup_progress_bar.close()
            self._warmup_progress_bar = None

    def _log_warmup_result(
        self,
        output_batch: OutputBatch,
        req_or_group: Any,
        is_warmup: bool,
    ) -> None:
        if not is_warmup:
            return

        server_based_warmup = is_server_based_warmup(req_or_group)
        self._warmup_processed += 1
        self._advance_warmup_progress_bar(req_or_group, output_batch)

        if output_batch.error is None:
            if (
                not server_based_warmup
                and not self._logged_server_ready_after_warmup
                and (
                    self._warmup_total <= 0
                    or self._warmup_processed >= self._warmup_total
                )
            ):
                logger.info("The server is fired up and ready to roll!")
                self._logged_server_ready_after_warmup = True
        else:
            warmup_desc = self._format_warmup_req(req_or_group)
            logger.info(f"{warmup_desc} processing failed")

    def process_received_reqs_with_req_based_warmup(
        self, recv_reqs: list[tuple[bytes, Any]]
    ) -> list[tuple[bytes, Any]]:
        if (
            self.req_based_warmup_scheduled
            or not self.server_args.warmup
            or not recv_reqs
            or self.server_args.warmup_resolutions is not None
            or self.server_args.server_warmup
        ):
            return recv_reqs

        identity, req_or_group = recv_reqs[0]
        req = get_first_generation_req(req_or_group)
        if req is not None:
            warmup_req = req.copy_as_warmup(self.server_args.warmup_steps)
            recv_reqs.insert(0, (identity, warmup_req))
            self._warmup_total = 1
            self._warmup_processed = 0
            self.req_based_warmup_scheduled = True
        return recv_reqs
