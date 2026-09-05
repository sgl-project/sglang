# Copied and adapted from: https://github.com/hao-ai-lab/FastVideo

# SPDX-License-Identifier: Apache-2.0
import asyncio
import os
import tempfile
from typing import Any, Awaitable, Callable

from tqdm.auto import tqdm

from sglang.multimodal_gen.runtime.entrypoints.openai.realtime.registry import (
    has_realtime_model_adapter,
)
from sglang.multimodal_gen.runtime.pipelines_core.schedule_batch import (
    OutputBatch,
    Req,
)
from sglang.multimodal_gen.runtime.server_args import ServerArgs
from sglang.multimodal_gen.runtime.server_args.auto_tune import (
    auto_residency_args_skip_reason,
    auto_residency_static_skip_reason,
)
from sglang.multimodal_gen.runtime.utils.image_io import save_base64_image_to_path
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger
from sglang.multimodal_gen.runtime.warmup_request_builder import (
    build_warmup_reqs,
    lighten_warmup_req,
    should_include_warmup_image,
    supports_synthetic_warmup,
)

logger = init_logger(__name__)

# a 64x64 image because some pipelines reject smaller inputs (e.g. FLUX.2's
# diffusers image processor requires both dimensions >= 64px)
MINIMUM_PICTURE_BASE64_FOR_WARMUP = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAEAAAABACAIAAAAlC+aJAAAAS0lEQVR42u3PMQ0AAAwDoEqv9ErYvQQckD4XAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAYHLAB8+AWnmfUycAAAAAElFTkSuQmCC"


def _is_ci_log_env() -> bool:
    return (
        os.environ.get("GITHUB_ACTIONS", "").lower() == "true"
        or os.environ.get("CI", "").lower() == "true"
    )


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
    return server_args.warmup_mode == "server"


def is_realtime_serving(server_args: ServerArgs) -> bool:
    """Synthetic warmup has no realtime session state."""
    return has_realtime_model_adapter(server_args)


def should_run_synthetic_server_warmup(server_args: ServerArgs) -> bool:
    return (
        should_run_server_warmup(server_args)
        and supports_synthetic_warmup(server_args)
        and not is_realtime_serving(server_args)
    )


def should_run_explicit_client_warmup(server_args: ServerArgs) -> bool:
    return (
        server_args.warmup_mode != "off"
        and server_args.warmup_resolutions is not None
        and supports_synthetic_warmup(server_args)
    )


def auto_residency_skip_reason(server_args: ServerArgs) -> str | None:
    """Final gate for warmup-calibrated residency placement.

    Only rules out paths the planner was not designed for; the workers
    re-check per component (explicit placement, FSDP modules, custom
    strategies, missing sizes) and per measurement.
    """
    args_reason = auto_residency_args_skip_reason(server_args)
    if args_reason is not None:
        return args_reason
    if not should_run_synthetic_server_warmup(server_args):
        return "no synthetic server warmup to calibrate from"
    return None


def should_apply_pre_warmup_auto_residency(server_args: ServerArgs) -> bool:
    return auto_residency_static_skip_reason(server_args) is None


def _auto_residency_status(response: OutputBatch) -> str | None:
    if isinstance(response.output, dict):
        return response.output.get("status")
    return None


async def maybe_apply_pre_warmup_auto_residency(
    server_args: ServerArgs,
    forward: Callable[[Req], Awaitable[OutputBatch]],
) -> None:
    """Choose a weight-feasible placement before the first serving forward."""
    from sglang.multimodal_gen.runtime.entrypoints.control_requests import (
        AutoResidencyReq,
    )
    from sglang.multimodal_gen.runtime.managers.memory_managers.auto_residency import (
        PLACEMENT_STATUS_ROLLBACK_FAILED,
    )

    if not should_apply_pre_warmup_auto_residency(server_args):
        return
    response = await forward(AutoResidencyReq(action="apply_static"))
    status = _auto_residency_status(response)
    if status == PLACEMENT_STATUS_ROLLBACK_FAILED:
        raise RuntimeError(f"auto residency rollback failed: {response.error}")
    if response.error is not None:
        logger.warning(
            "Pre-warmup residency placement was not applied; calibrating the "
            "original placement: %s",
            response.error,
        )


async def maybe_apply_auto_residency(
    server_args: ServerArgs,
    forward: Callable[[Req], Awaitable[OutputBatch]],
) -> None:
    """Adjust implicit component residency after warmup, then re-warm.

    Runs between the synthetic warmup and the ready signal, so the residency
    is frozen before ``/health`` turns 200. If an adjustment or its calibration
    fails, the workers roll back that round and retain the last calibrated
    placement; only a failed rollback raises and aborts startup.
    """
    from sglang.multimodal_gen.runtime.entrypoints.control_requests import (
        AutoResidencyReq,
    )
    from sglang.multimodal_gen.runtime.managers.memory_managers.auto_residency import (
        PLACEMENT_STATUS_ADJUSTED,
        PLACEMENT_STATUS_ROLLBACK_FAILED,
        PLACEMENT_STATUS_ROLLED_BACK,
        PLACEMENT_STATUS_VALIDATED,
    )

    skip_reason = auto_residency_skip_reason(server_args)
    if skip_reason is not None:
        # Whoever asked for auto needs to hear why it did nothing; on any other
        # server this is just a note about a mode they did not pick.
        log = logger.info if server_args.performance_mode == "auto" else logger.debug
        log("Auto residency: skipped (%s)", skip_reason)
        return

    logger.info(
        "Server warmup complete; adjusting component residency for the "
        "default workload (--performance-mode auto)."
    )
    # Same fail-open contract as the warmup itself: implicit warmups must
    # never abort startup, explicit --warmup-resolutions ones must succeed.
    fail_open = server_args.warmup_resolutions is None

    async def rollback_and_rewarm(error: Exception) -> None:
        logger.warning(
            "Post-adjustment calibration failed (%s); rolling back auto residency",
            error,
        )
        rollback = await forward(AutoResidencyReq(action="rollback"))
        if (
            rollback.error is not None
            or _auto_residency_status(rollback) != PLACEMENT_STATUS_ROLLED_BACK
        ):
            raise RuntimeError(
                f"auto residency rollback failed: {rollback.error}"
            ) from error
        # Restore warm caches for the previous calibrated placement before
        # turning ready. Once placement was mutated, failure to revalidate the
        # restored state is unsafe to ignore.
        await run_async_client_warmup(
            server_args, forward, fail_open=False, rewarm=True
        )

    try:
        response = await forward(AutoResidencyReq(action="apply"))
    except Exception as e:
        if not fail_open:
            raise
        logger.warning(
            "Auto residency apply request failed; continuing on the original "
            "strategy: %s",
            e,
        )
        return
    status = _auto_residency_status(response)
    recovering_from_oom = bool(
        isinstance(response.output, dict) and response.output.get("recovering_from_oom")
    )
    if status == PLACEMENT_STATUS_ROLLBACK_FAILED:
        raise RuntimeError(f"auto residency rollback failed: {response.error}")
    if response.error is not None:
        if status == PLACEMENT_STATUS_ROLLED_BACK:
            await run_async_client_warmup(
                server_args, forward, fail_open=False, rewarm=True
            )
        logger.warning(
            "Auto residency adjustment not applied; continuing on the original "
            "strategy: %s",
            response.error,
        )
        return
    if status != PLACEMENT_STATUS_ADJUSTED:
        if recovering_from_oom:
            raise RuntimeError(
                "default workload exceeds the VRAM budget and auto residency "
                "found no feasible placement"
            )
        return

    short_validation = bool(
        isinstance(response.output, dict) and response.output.get("short_validation")
    )
    # This pass physically realizes the selected placement and measures phases
    # that overlap under it. Resident-only changes need one full-shape step for
    # memory safety; other changes retain the longer regression timing sample.
    try:
        validation_options = (
            {"step_limit": _short_validation_step_limit(server_args)}
            if short_validation
            else {}
        )
        await run_async_client_warmup(
            server_args,
            forward,
            fail_open=False,
            rewarm=True,
            **validation_options,
        )
    except Exception as e:
        await rollback_and_rewarm(e)
        return

    try:
        validation = await forward(AutoResidencyReq(action="validate"))
    except Exception as e:
        await rollback_and_rewarm(e)
        return
    validation_status = _auto_residency_status(validation)
    if validation_status == PLACEMENT_STATUS_ROLLBACK_FAILED:
        raise RuntimeError(f"auto residency rollback failed: {validation.error}")
    if validation_status == PLACEMENT_STATUS_ROLLED_BACK:
        await run_async_client_warmup(
            server_args, forward, fail_open=False, rewarm=True
        )
        return
    if validation.error is not None or validation_status != PLACEMENT_STATUS_VALIDATED:
        await rollback_and_rewarm(
            RuntimeError(
                validation.error
                or "post-adjustment calibration returned no validation result"
            )
        )


# Enough to clear a probe that overshot the card, few enough that a failure
# which is not about probe size gives up quickly instead of walking the
# workload down to nothing.
MAX_WARMUP_DEGRADE_ATTEMPTS = 3


_OUT_OF_MEMORY_MARKERS = (
    "out of memory",
    "outofmemory",
    "cudaerrormemoryallocation",
    "cublas_status_alloc_failed",
    "cannot allocate memory",
    "unable to allocate",
)


def _is_out_of_memory(error: Any) -> bool:
    text = str(error).lower()
    return any(marker in text for marker in _OUT_OF_MEMORY_MARKERS)


def _degrade_after_oom(server_args: ServerArgs, req: Req) -> Req | None:
    """Next warmup probe to try after `req` ran the card out of memory.

    Only memory failures are worth retrying smaller; anything else fails the
    same way at every size and should surface instead of being shrunk away.
    """
    lighter = lighten_warmup_req(server_args, req)
    if lighter is None:
        return None
    logger.warning(
        "%s ran out of memory; retrying warmup at %s",
        format_warmup_req(req),
        format_warmup_req(lighter),
    )
    return lighter


def _short_validation_step_limit(server_args: ServerArgs) -> int:
    """Steps for the resident-only validation pass: one, unless the pipeline
    needs more (MiniMax H3 rejects fewer than two)."""
    config = getattr(server_args, "pipeline_config", None)
    return max(1, int(getattr(config, "minimum_inference_steps", 1) or 1))


def format_warmup_req(req_or_group: Any) -> str:
    req = get_first_generation_req(req_or_group)
    if req is not None and req.extra.get("auto_residency_full_shape_probe"):
        prefix = "auto residency probe"
    else:
        prefix = (
            "server warmup req"
            if is_server_based_warmup(req_or_group)
            else "warmup req"
        )
    if req is None:
        return prefix

    width = getattr(req, "width", None)
    height = getattr(req, "height", None)
    shape = "action" if width is None or height is None else f"{width}x{height}"
    num_frames = getattr(req, "num_frames", None)
    if num_frames is not None and num_frames > 1:
        shape += f"x{num_frames}f"

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
    rewarm: bool = False,
    step_limit: int | None = None,
) -> list[Req]:
    warmup_reqs = build_warmup_reqs(
        server_args,
        warmup_resolutions=server_args.warmup_resolutions,
        warmup_input_path=warmup_input_path,
        return_warmup_result=True,
        server_based_warmup=True,
    )
    warmup_total = sum(1 for req in warmup_reqs if req.is_warmup)
    for req in warmup_reqs:
        if req.is_warmup:
            req.extra["warmup_total"] = warmup_total
            if step_limit is not None:
                req.num_inference_steps = min(req.num_inference_steps, step_limit)
        if rewarm:
            # a repeat pass after an auto-residency change: keep it out of
            # the scheduler's warmup progress accounting (already at N/N)
            req.extra["server_warmup_rewarm"] = True
    return warmup_reqs


async def run_async_client_warmup(
    server_args: ServerArgs,
    forward: Callable[[Req], Awaitable[OutputBatch]],
    *,
    fail_open: bool = False,
    rewarm: bool = False,
    step_limit: int | None = None,
) -> None:
    try:
        auto_residency_handles_oom = auto_residency_skip_reason(server_args) is None
        warmup_input_path = None
        if should_include_warmup_image(server_args, server_based_warmup=True):
            warmup_input_path = prepare_warmup_image_path(server_args)

        for req in build_client_warmup_reqs(
            server_args,
            warmup_input_path=warmup_input_path,
            rewarm=rewarm,
            step_limit=step_limit,
        ):
            response = await forward(req)
            for _ in range(MAX_WARMUP_DEGRADE_ATTEMPTS):
                if response.error is None or not _is_out_of_memory(response.error):
                    break
                # The residency planner needs the first failed target-shape
                # measurement. Retrying smaller requests cannot fix a weight-
                # dominated OOM and can retain failed-forward allocations that
                # contaminate the phase model used by the planner.
                if auto_residency_handles_oom:
                    break
                lighter = _degrade_after_oom(server_args, req)
                if lighter is None:
                    break
                req = lighter
                response = await forward(req)
            if response.error is not None:
                raise RuntimeError(response.error)
    except Exception:
        if fail_open:
            logger.warning(
                "Synthetic server warmup failed; continuing startup", exc_info=True
            )
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
        for _ in range(MAX_WARMUP_DEGRADE_ATTEMPTS):
            if response.error is None or not _is_out_of_memory(response.error):
                break
            lighter = _degrade_after_oom(server_args, req)
            if lighter is None:
                break
            req = lighter
            response = forward(req)
        if response.error is not None:
            raise RuntimeError(response.error)


def run_sync_startup_warmup(
    server_args: ServerArgs,
    forward: Callable[[Req], OutputBatch],
) -> None:
    """Run the server-style startup sequence through a synchronous client.

    Offline ``DiffGenerator`` launches the same scheduler and workers as the
    HTTP entrypoint, but its client API is synchronous. Adapting that call into
    the shared async orchestration keeps static placement, one warmup probe,
    and measured refinement identical across both entrypoints.
    """

    async def async_forward(req: Req) -> OutputBatch:
        return forward(req)

    async def run() -> None:
        await maybe_apply_pre_warmup_auto_residency(server_args, async_forward)
        if not should_run_synthetic_server_warmup(server_args):
            return
        await run_async_client_warmup(server_args, async_forward, fail_open=True)
        await maybe_apply_auto_residency(server_args, async_forward)

    asyncio.run(run())


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

        ci_log_env = _is_ci_log_env()
        if self._warmup_progress_bar is None:
            self._warmup_progress_bar = tqdm(
                total=self._warmup_progress_total(req_or_group),
                desc="Warmup requests",
                unit="req",
                disable=ci_log_env,
            )
            if ci_log_env:
                logger.info(
                    "Warmup requests: 0/%s %s",
                    self._warmup_progress_bar.total,
                    self._format_warmup_req(req_or_group),
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
        progress_n = self._warmup_processed
        if _is_ci_log_env():
            logger.info(
                "Warmup requests: %s/%s %s",
                progress_n,
                self._warmup_progress_bar.total,
                self._format_warmup_req(req_or_group),
            )

        if progress_n >= self._warmup_progress_bar.total:
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

        req = get_first_generation_req(req_or_group)
        if req is not None and req.extra.get("server_warmup_rewarm"):
            # auto-residency re-warm passes repeat already-counted requests;
            # advancing the bar again would log N+1/N in CI
            if output_batch.error is not None:
                logger.warning(
                    "%s processing failed: %s",
                    self._format_warmup_req(req_or_group),
                    output_batch.error,
                )
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
            logger.warning("%s processing failed: %s", warmup_desc, output_batch.error)

    def process_received_reqs_with_req_based_warmup(
        self, recv_reqs: list[tuple[bytes, Any]]
    ) -> list[tuple[bytes, Any]]:
        if (
            self.req_based_warmup_scheduled
            or self.server_args.warmup_mode != "request"
            or not recv_reqs
            or self.server_args.warmup_resolutions is not None
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
