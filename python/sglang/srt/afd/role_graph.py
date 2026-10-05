"""Whole-role graph: one capture per step, spanning every layer and the transport.

Every layer, stage and inter-role exchange is captured into one graph. NCCL
side streams fork from and rejoin the capture stream. Both eager execution and
capture use stream waits; receive lookahead is restricted to capture.

A whole step cannot fall back independently on one rank. Only eager reasons
derived from the shared step descriptor are admitted; capture, HBM or replay
failures abort the role instead of posting an unmatched exchange round.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Callable, NoReturn, Protocol

import msgspec

from .cache import AFDCacheSelection, AFDShapeCache
from .config import AFDConfig
from .contracts import (
    AFDError,
    AFDExecutionKind,
    AFDExecutionResult,
    AFDMetadataGuard,
    AFDReason,
    AFDRole,
    AFDShapeIdentity,
)

logger = logging.getLogger(__name__)

# The only reasons both roles are guaranteed to reach together. Every one of them
# is a function of the step descriptor alone: the attention role puts the row plan
# and the eligibility flag on it, the FFN role builds its shape from those same
# fields, and neither consults the cache on a step the descriptor calls ineligible.
# Both roles use the startup-agreed capture plan for size eligibility.
_PEER_AGREED_EAGER_REASONS = frozenset(
    (
        AFDReason.FORWARD_MODE,
        AFDReason.BUCKET_LIMIT,
    )
)


class AFDGraphProgram(Protocol):
    def close(self) -> None: ...


class AFDGraphDriver(Protocol):
    def capture(self, *, spec: Any) -> AFDGraphProgram: ...

    def memory_usage(self, *, device: Any) -> tuple[int, int]: ...

    def close(self) -> None: ...


def copy_tensor_prefix(*, target: Any, source: Any, real_rows: int) -> None:
    if real_rows < 0 or real_rows > target.shape[0] or source.shape[0] != real_rows:
        raise AFDError(
            "AFD_GRAPH_COPY_ROWS_INVALID",
            f"real={real_rows} source={source.shape[0]} capacity={target.shape[0]}",
        )
    if real_rows:
        target[:real_rows].copy_(source)
    if real_rows < target.shape[0]:
        target[real_rows:].zero_()


def cuda_memory_diagnostic(device: Any) -> dict[str, Any]:
    try:
        import torch

        result = {
            "allocated": torch.cuda.memory_allocated(device),
            "reserved": torch.cuda.memory_reserved(device),
        }
        if hasattr(torch.cuda, "mem_get_info"):
            result["mem_get_info"] = torch.cuda.mem_get_info(device)
        return result
    # Exception, not BaseException: this only enriches a capture-failure log line, and
    # swallowing the exception is the point, so catching Ctrl-C here would convert the
    # interrupt into an "unavailable" string and lose it.
    except Exception as exc:
        return {"unavailable": f"{type(exc).__name__}: {exc}"}


class AFDRoleGraphProgramSpec(msgspec.Struct, kw_only=True):
    shape_digest: str
    device: Any
    bucket_rows: tuple[int, ...]
    stage_rows: tuple[int, ...]
    stage_args: tuple[tuple[Any, ...], ...]
    compute: Callable[[tuple[tuple[Any, ...], ...]], tuple[tuple[Any, ...], ...]]
    forward_batches: tuple[Any, ...]
    metadata_guards: tuple[AFDMetadataGuard | None, ...]


class TorchRoleGraphProgram:
    """One CUDA graph holding a whole role step, transport included."""

    def __init__(self, *, spec: AFDRoleGraphProgramSpec, backend: Any) -> None:
        import torch

        from sglang.srt.model_executor.runner.shape_key import ShapeKey
        from sglang.srt.model_executor.runner_utils.pool import (
            get_or_create_global_graph_capture_stream,
        )

        self._torch = torch
        self._guards = spec.metadata_guards
        self._closed = False
        self._backend = backend
        self._key = ShapeKey(
            size=sum(spec.bucket_rows), variant_label=spec.shape_digest
        )
        self._stage_inputs: tuple[tuple[Any, ...], ...] = ()
        self._sentinel = None
        self._sentinel_source = None
        device = spec.device
        if device is None:
            raise AFDError("AFD_ROLE_GRAPH_DEVICE_UNRESOLVED")
        self._stage_inputs = tuple(
            tuple(
                self._allocate(value=value, bucket_rows=rows, device=device)
                for value in values
            )
            for values, rows in zip(spec.stage_args, spec.bucket_rows)
        )
        self._copy_inputs(
            stage_args=spec.stage_args,
            stage_rows=spec.stage_rows,
        )
        try:
            for guard, forward_batch in zip(self._guards, spec.forward_batches):
                if guard is not None:
                    guard.capture(forward_batch)
            outputs = None

            def forward():
                nonlocal outputs
                for guard in self._guards:
                    if guard is not None:
                        guard.activate_in_graph()
                outputs = spec.compute(self._stage_inputs)
                self._capture_sentinel(outputs=outputs, device=device)
                return outputs

            torch.cuda.synchronize(device)
            # Native capture passes use one runtime stream so allocator scratch
            # can be reused across shapes. Warmup must use that stream too.
            stream = get_or_create_global_graph_capture_stream()
            with (
                torch.cuda.stream(stream),
                self._backend.capture_session(stream),
            ):
                self._backend.capture_one(
                    self._key,
                    forward,
                    capture_inputs=(
                        self._stage_inputs,
                        spec.forward_batches,
                        self._guards,
                    ),
                    post_warmup_hook=lambda: self._copy_inputs(
                        stage_args=spec.stage_args, stage_rows=spec.stage_rows
                    ),
                )
            self._validate_outputs(outputs=outputs)
            self._restore_metadata()
        except BaseException:
            self.close()
            raise

    def _allocate(self, *, value: Any, bucket_rows: int, device: Any) -> Any:
        shape = (bucket_rows,) + tuple(value.shape[1:])
        return self._torch.zeros(shape, dtype=value.dtype, device=device)

    def _copy_inputs(
        self,
        *,
        stage_args: tuple[tuple[Any, ...], ...],
        stage_rows: tuple[int, ...],
    ) -> None:
        if len(stage_args) != len(self._stage_inputs):
            raise AFDError(
                "AFD_ROLE_GRAPH_STAGE_COUNT_CHANGED",
                f"expected={len(self._stage_inputs)} actual={len(stage_args)}",
            )
        for targets, sources, rows in zip(
            self._stage_inputs,
            stage_args,
            stage_rows,
        ):
            if len(targets) != len(sources):
                raise AFDError(
                    "AFD_ROLE_GRAPH_INPUT_ARITY_CHANGED",
                    f"expected={len(targets)} actual={len(sources)}",
                )
            for target, source in zip(targets, sources):
                copy_tensor_prefix(
                    target=target,
                    source=source,
                    real_rows=rows,
                )

    def _capture_sentinel(
        self,
        *,
        outputs: tuple[tuple[Any, ...], ...],
        device: Any,
    ) -> None:
        """Record a value the captured work must write, to prove replay runs it.

        A graph that recorded nothing replays in about zero time and reads as an
        enormous speedup, so CAPTURE_OK alone is not evidence. Deriving the
        sentinel from the step's own output means an empty capture also skips the
        copy that fills it, and the poison value survives into the check.
        """

        source = None
        for values in outputs:
            for value in values:
                if value is not None:
                    source = value
        if source is None:
            raise AFDError("AFD_ROLE_GRAPH_SENTINEL_SOURCE_MISSING")
        self._sentinel = self._torch.zeros(
            1,
            dtype=self._torch.float32,
            device=device,
        )
        self._sentinel_source = source
        self._sentinel.copy_(source.detach()[(0,) * source.ndim].reshape(1).float())

    def _validate_outputs(
        self,
        *,
        outputs: tuple[tuple[Any, ...], ...],
    ) -> None:
        if not isinstance(outputs, tuple) or len(outputs) != len(self._stage_inputs):
            raise AFDError(
                "AFD_ROLE_GRAPH_OUTPUT_CONTRACT_INVALID",
                f"shape={self._key!r}",
            )
        for values in outputs:
            if not isinstance(values, tuple):
                raise AFDError(
                    "AFD_ROLE_GRAPH_OUTPUT_CONTRACT_INVALID",
                    f"shape={self._key!r}",
                )

    def arm_sentinel(self) -> None:
        """Poison the sentinel so the next replay has to overwrite it."""

        if self._sentinel is None:
            raise AFDError("AFD_ROLE_GRAPH_SENTINEL_MISSING")
        self._sentinel.fill_(float("nan"))

    def require_sentinel_written(self) -> None:
        """Require the arming replay to have written the sentinel.

        Checked around the step's own replay rather than by replaying a second
        time: every replay of this region performs its NCCL sends and receives,
        so a private self-test is an exchange round the peer does not perform and
        the pair stays one round apart for the rest of the run.
        """

        if self._sentinel is None:
            raise AFDError("AFD_ROLE_GRAPH_SENTINEL_MISSING")
        self._torch.cuda.synchronize(self._sentinel.device)
        if bool(self._torch.isnan(self._sentinel).all()):
            raise AFDError(
                "AFD_ROLE_GRAPH_SELF_TEST_SENTINEL_STATIC",
                f"shape={self._key!r}",
            )

    def replay(
        self,
        *,
        stage_args: tuple[tuple[Any, ...], ...],
        stage_rows: tuple[int, ...],
        forward_batches: tuple[Any, ...],
    ) -> tuple[tuple[Any, ...], ...]:
        if self._closed:
            raise AFDError("AFD_ROLE_GRAPH_PROGRAM_CLOSED")
        self._copy_inputs(stage_args=stage_args, stage_rows=stage_rows)
        try:
            for guard, forward_batch in zip(self._guards, forward_batches):
                if guard is not None:
                    guard.prepare_replay(forward_batch)
                    guard.assert_stable()
            outputs = self._backend.replay(self._key, static_forward_batch=None)
            return tuple(
                tuple(value[:rows] if value is not None else None for value in values)
                for values, rows in zip(outputs, stage_rows)
            )
        finally:
            self._restore_metadata()

    def _restore_metadata(self) -> None:
        first_error = None
        for guard in reversed(self._guards):
            if guard is not None:
                try:
                    guard.restore()
                except BaseException as exc:
                    if first_error is None:
                        first_error = exc
        if first_error is not None:
            raise first_error

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        first_error = None
        backend, guards = self._backend, self._guards
        self._backend, self._guards = None, ()
        # Release the graph before its static metadata and private backend state.
        cleanups = [lambda: backend.release_shape(self._key)] if backend else []
        cleanups.extend(guard.close for guard in reversed(guards) if guard is not None)
        for cleanup in cleanups:
            try:
                cleanup()
            except BaseException as exc:
                if first_error is None:
                    first_error = exc
        self._stage_inputs = ()
        self._sentinel = self._sentinel_source = None
        if first_error is not None:
            raise first_error


class TorchRoleGraphDriver:
    def __init__(self) -> None:
        self._backend = None

    @staticmethod
    def memory_usage(*, device: Any) -> tuple[int, int]:
        import torch

        return torch.cuda.memory_allocated(device), torch.cuda.memory_reserved(device)

    def close(self) -> None:
        if self._backend is not None:
            self._backend.cleanup()
            self._backend = None

    def capture(
        self,
        *,
        spec: AFDRoleGraphProgramSpec,
    ) -> TorchRoleGraphProgram:
        import torch

        if self._backend is None:
            from sglang.srt.model_executor.runner_backend.full_cuda_graph_backend import (
                FullCudaGraphBackend,
            )

            self._backend = FullCudaGraphBackend(device_module=torch.cuda)
        try:
            with torch.cuda.device(spec.device):
                return TorchRoleGraphProgram(spec=spec, backend=self._backend)
        except BaseException:
            # Also covers metadata restoration raising after a successful capture.
            from sglang.srt.model_executor.runner.shape_key import ShapeKey

            self._backend.release_shape(
                ShapeKey(size=sum(spec.bucket_rows), variant_label=spec.shape_digest)
            )
            raise


class AFDRoleGraphService:
    """One graph per bucket covering the whole role step, transport included."""

    strategy = "role"

    def __init__(
        self,
        *,
        role: AFDRole,
        config: AFDConfig,
        capture_sizes: tuple[int, ...],
        num_layers: int,
        driver: AFDGraphDriver,
        base_hbm_bytes: int = 0,
        device: Any = None,
        log_interval: int = 40,
    ) -> None:
        self.role = role
        self._config = config
        self._num_layers = num_layers
        self._driver = driver
        # FFN inputs arrive over the wire; bind the device before capture.
        self._device = device
        self._cache = AFDShapeCache(
            config=config,
            capture_sizes=capture_sizes,
            base_hbm_bytes=base_hbm_bytes,
        )
        self._selection: AFDCacheSelection | None = None
        self._shape: AFDShapeIdentity | None = None
        self._step_id: int | None = None
        self._graph_enabled = bool(capture_sizes)
        self._eligible = False
        self._capture_failure_reason: AFDReason | None = None
        self._estimated_graph_hbm_bytes = 0
        self._retained_backing_hbm_bytes = 0
        self._log_interval = max(1, log_interval)
        self._completed_steps = 0
        self._non_replay_steps = 0
        self._last_log_state = None
        self._eligible_steps = 0
        self._closed = False
        if self._device is None:
            raise AFDError("AFD_ROLE_GRAPH_DEVICE_REQUIRED")

    def _bytes_per_value(self, *, shape: AFDShapeIdentity) -> int:
        bytes_per_value = {"bfloat16": 2, "float16": 2}.get(shape.dtype)
        if bytes_per_value is None:
            raise AFDError(
                "AFD_GRAPH_DTYPE_UNSUPPORTED",
                f"dtype={shape.dtype!r}",
            )
        return bytes_per_value

    @property
    def capture_sizes(self) -> tuple[int, ...]:
        return self._cache.capture_sizes

    def finish_capture(self) -> None:
        self._cache.finish_capture()
        allocated, reserved = self._driver.memory_usage(device=self._device)
        logger.info(
            "AFD_STARTUP_CAPTURE_MEMORY role=%s allocated_bytes=%d reserved_bytes=%d "
            "accounted_bytes=%d",
            self.role.value,
            allocated,
            reserved,
            self._cache.retained_hbm_bytes,
        )

    def begin_step(
        self,
        *,
        step_id: int,
        shape: AFDShapeIdentity,
        eligible: bool,
        backing_hbm_bytes: int = 0,
        capture: bool = False,
    ) -> AFDReason | None:
        if self._closed:
            return AFDReason.CLOSED
        if self._selection is not None:
            raise AFDError(
                "AFD_GRAPH_STEP_ALREADY_ACTIVE",
                f"step={self._step_id}",
            )
        self._step_id = step_id
        self._shape = shape
        self._eligible = eligible and self._graph_enabled
        self._capture_failure_reason = None
        self._retained_backing_hbm_bytes = 0
        if not self._eligible:
            return AFDReason.FORWARD_MODE
        self._eligible_steps += 1
        self._estimated_graph_hbm_bytes = (
            self._estimate_hbm_bytes(shape=shape) if capture else 0
        )
        self._selection = self._cache.select(
            shape=shape,
            capture=capture,
            estimated_hbm_bytes=(self._estimated_graph_hbm_bytes + backing_hbm_bytes),
        )
        return self._selection.reason

    def ensure_backing_hbm(self, *, retained_hbm_bytes: int) -> bool:
        selection = self._selection
        if selection is None or selection.bucket is None or not self.retains_backing:
            return False
        if selection.replay:
            if retained_hbm_bytes > selection.bucket.retained_backing_hbm_bytes:
                self._cache.invalidate(
                    bucket=selection.bucket,
                    reason=AFDReason.HBM_LIMIT,
                )
                self._capture_failure_reason = AFDReason.HBM_LIMIT
                selection.reason = AFDReason.HBM_LIMIT
                selection.replay = False
                selection.arming = False
                return False
            self._retained_backing_hbm_bytes = retained_hbm_bytes
            return True
        required_hbm_bytes = self._estimated_graph_hbm_bytes + retained_hbm_bytes
        accepted = self._cache.ensure_reservation(
            bucket=selection.bucket,
            retained_hbm_bytes=required_hbm_bytes,
        )
        if not accepted:
            self._capture_failure_reason = AFDReason.HBM_LIMIT
            selection.reason = AFDReason.HBM_LIMIT
            selection.replay = False
            selection.arming = False
            self._cache.rollback(
                bucket=selection.bucket,
                reason=AFDReason.HBM_LIMIT,
            )
        else:
            selection.bucket.retained_backing_hbm_bytes = retained_hbm_bytes
            self._retained_backing_hbm_bytes = retained_hbm_bytes
        return accepted

    @property
    def capturing(self) -> bool:
        return bool(self._selection is not None and self._selection.arming)

    @property
    def retains_backing(self) -> bool:
        return (
            self._selection is not None
            and self._selection.bucket is not None
            and (self._selection.arming or self._selection.replay)
            and self._selection.reason
            not in (
                AFDReason.BUCKET_LIMIT,
                AFDReason.HBM_LIMIT,
                AFDReason.CAPTURE_FAILED,
                AFDReason.PARTIAL_CAPTURE,
                AFDReason.METADATA_DRIFT,
                AFDReason.REPLAY_FAILED,
            )
        )

    def end_step(self) -> AFDReason | None:
        selection = self._selection
        step_id = self._step_id
        eligible = self._eligible
        self._selection = None
        self._shape = None
        self._step_id = None
        self._eligible = False
        if selection is None or not selection.arming:
            result = (
                selection.reason if selection is not None else AFDReason.FORWARD_MODE
            )
        else:
            bucket = selection.bucket
            if bucket is None:
                raise AFDError("AFD_GRAPH_ARMING_BUCKET_MISSING")
            if self._capture_failure_reason is not None:
                result = self._capture_failure_reason
            else:
                try:
                    self._cache.install(
                        bucket=bucket,
                    )
                except Exception:
                    self._cache.rollback(
                        bucket=bucket,
                        reason=AFDReason.PARTIAL_CAPTURE,
                    )
                    result = AFDReason.PARTIAL_CAPTURE
                else:
                    logger.info(
                        "AFD_GRAPH_INSTALL %s",
                        json.dumps(
                            {
                                "role": self.role.value,
                                "strategy": self.strategy,
                                "step_id": step_id,
                                "usage": self._cache.usage(status="ACTIVE"),
                            },
                            sort_keys=True,
                        ),
                    )
                    result = None
        self._completed_steps += 1
        self._non_replay_steps += int(
            selection is None or selection.arming or selection.reason is not None
        )
        state = (eligible, result)
        if (
            state != self._last_log_state
            or self._completed_steps % self._log_interval == 0
        ):
            self._log_step_snapshot(step_id=step_id, eligible=eligible)
        self._last_log_state = state
        return result

    def _log_step_snapshot(
        self,
        *,
        step_id: int | None,
        eligible: bool,
    ) -> None:
        if not logger.isEnabledFor(logging.INFO):
            return
        usage = self._cache.usage(status="ACTIVE")
        buckets = tuple(usage["buckets"].values())
        logger.info(
            "AFD_GRAPH_USAGE_SNAPSHOT %s",
            json.dumps(
                {
                    "role": self.role.value,
                    "strategy": self.strategy,
                    "step_id": step_id,
                    "eligible": eligible,
                    "eligible_steps": self._eligible_steps,
                    "completed_steps": self._completed_steps,
                    "non_replay_steps": self._non_replay_steps,
                    "expected_operations": 1,
                    "bucket_count": usage["bucket_count"],
                    "installs": sum(item["usage"]["installs"] for item in buckets),
                    "replays": sum(item["usage"]["replays"] for item in buckets),
                    "typed_eager_failures": 0,  # Unsafe eager fallback is forbidden.
                    "overflow_eager": usage["overflow_eager"],
                    "terminal_buckets": sum(
                        item["phase"] == "terminal_eager" for item in buckets
                    ),
                    "retained_hbm_bytes": usage["retained_hbm_bytes"],
                    "max_hbm_bytes": usage["max_hbm_bytes"],
                },
                sort_keys=True,
            ),
        )

    def usage(self, *, status: str = "ACTIVE") -> dict[str, Any]:
        return self._cache.usage(status=status)

    def close(self) -> dict[str, Any]:
        if self._closed:
            return self._cache.usage(status="CLOSED")
        self._closed = True
        first_error = None
        try:
            if self._selection is not None and self._selection.bucket is not None:
                self._cache.rollback(
                    bucket=self._selection.bucket,
                    reason=AFDReason.PARTIAL_CAPTURE,
                )
        except BaseException as exc:
            first_error = exc
        finally:
            self._selection = None
            self._retained_backing_hbm_bytes = 0
        try:
            usage = self._cache.close()
        except BaseException as exc:
            if first_error is None:
                first_error = exc
        try:
            self._driver.close()
        except BaseException as exc:
            if first_error is None:
                first_error = exc
        if first_error is not None:
            raise first_error
        return usage

    def _estimate_hbm_bytes(self, *, shape: AFDShapeIdentity) -> int:
        bytes_per_value = self._bytes_per_value(shape=shape)
        rows = sum(self._own_bucket_rows(shape=shape))
        if self.role == AFDRole.FFN and shape.attention_lanes % shape.ffn_size:
            # Every F computes merged rows, including ranks with no ingress.
            # Budget the new layout conservatively; preserve legacy estimates.
            rows = sum(sum(vector) for vector in shape.lane_bucket_rows)
        # Entry hidden, entry residual, and every layer's two intermediates live
        # in the graph pool for the whole step rather than being reused per op.
        entry_bytes = rows * shape.hidden_size * bytes_per_value * 2
        per_layer_bytes = rows * shape.hidden_size * bytes_per_value * 2
        return entry_bytes + per_layer_bytes * self._num_layers

    def _own_bucket_rows(self, *, shape: AFDShapeIdentity) -> tuple[int, ...]:
        """Padded rows this rank runs per stage.

        An FFN rank serving k attention lanes runs their rows as one batch, so
        its width is the group total rather than any single lane's.
        """

        if self.role == AFDRole.ATTENTION:
            return shape.bucket_rows
        return shape.group_total_bucket_rows(ffn_ordinal=shape.lane)

    def _own_stage_rows(self, *, shape: AFDShapeIdentity) -> tuple[int, ...]:
        if self.role == AFDRole.ATTENTION:
            return shape.stage_rows
        return shape.group_total_stage_rows(ffn_ordinal=shape.lane)

    def execute_step(
        self,
        *,
        stage_args: tuple[tuple[Any, ...], ...],
        stage_rows: tuple[int, ...],
        forward_batches: tuple[Any, ...],
        compute: Callable[[tuple[tuple[Any, ...], ...]], tuple[tuple[Any, ...], ...]],
        metadata_guards: tuple[AFDMetadataGuard | None, ...],
    ) -> AFDExecutionResult:
        self._validate_step(stage_rows=stage_rows)
        if not self._eligible or self._selection is None:
            return self._eager_step(
                compute=compute,
                stage_args=stage_args,
                reason=AFDReason.FORWARD_MODE,
            )
        selection = self._selection
        bucket = selection.bucket
        if selection.replay:
            if bucket is None or bucket.program is None:
                self._raise_graph_failure(reason=AFDReason.CACHE_GUARD)
            try:
                value = bucket.program.replay(
                    stage_args=stage_args,
                    stage_rows=stage_rows,
                    forward_batches=forward_batches,
                )
            except Exception as exc:
                reason = (
                    AFDReason.METADATA_DRIFT
                    if isinstance(exc, AFDError)
                    else AFDReason.REPLAY_FAILED
                )
                self._cache.invalidate(bucket=bucket, reason=reason)
                selection.replay = False
                selection.reason = reason
                self._raise_graph_failure(reason=reason)
            bucket.usage.replays += 1
            return AFDExecutionResult(
                value=value,
                kind=AFDExecutionKind.REPLAY,
                bucket=bucket.shape.digest,
            )
        if selection.arming and self._capture_failure_reason is None:
            return self._capture_and_run(
                bucket=bucket,
                stage_args=stage_args,
                stage_rows=stage_rows,
                forward_batches=forward_batches,
                compute=compute,
                metadata_guards=metadata_guards,
            )
        reason = (
            self._capture_failure_reason or selection.reason or AFDReason.CAPTURE_FAILED
        )
        return self._eager_step(
            compute=compute,
            stage_args=stage_args,
            reason=reason,
        )

    def _validate_step(self, *, stage_rows: tuple[int, ...]) -> None:
        if self._shape is None or stage_rows != self._own_stage_rows(shape=self._shape):
            raise AFDError(
                "AFD_ROLE_GRAPH_STEP_ROWS_INVALID",
                f"stage_rows={stage_rows!r}",
            )

    def _eager_step(
        self,
        *,
        compute: Callable[[tuple[tuple[Any, ...], ...]], tuple[tuple[Any, ...], ...]],
        stage_args: tuple[tuple[Any, ...], ...],
        reason: AFDReason,
    ) -> AFDExecutionResult:
        """Run the step eagerly, at real rows.

        Only reached for a step both roles decline identically: a non-decode
        step or a size outside the agreed plan. Any other reason would make this role's width differ from
        its peer's, so the role fails before computing or communicating.
        """

        if reason not in _PEER_AGREED_EAGER_REASONS:
            self._raise_graph_failure(reason=reason)
        return AFDExecutionResult(
            value=compute(stage_args),
            kind=AFDExecutionKind.EAGER,
            reason=reason,
        )

    def _raise_graph_failure(self, *, reason: AFDReason) -> NoReturn:
        # A local replay/capture/HBM failure cannot authorize an extra eager
        # communication round while a peer may still replay padded rows.
        raise AFDError(
            "AFD_ROLE_GRAPH_STEP_NOT_GRAPHABLE",
            f"reason={reason.value} role={self.role.value}",
        )

    def _capture_and_run(
        self,
        *,
        bucket: Any,
        stage_args: tuple[tuple[Any, ...], ...],
        stage_rows: tuple[int, ...],
        forward_batches: tuple[Any, ...],
        compute: Callable[[tuple[tuple[Any, ...], ...]], tuple[tuple[Any, ...], ...]],
        metadata_guards: tuple[AFDMetadataGuard | None, ...],
    ) -> AFDExecutionResult:
        if bucket is None or self._shape is None:
            raise AFDError("AFD_ROLE_GRAPH_ARMING_BUCKET_MISSING")
        device = self._device
        program = None
        try:
            baseline = self._driver.memory_usage(device=device)
            try:
                program = self._driver.capture(
                    spec=AFDRoleGraphProgramSpec(
                        shape_digest=bucket.shape.digest,
                        device=device,
                        bucket_rows=self._own_bucket_rows(shape=self._shape),
                        stage_rows=stage_rows,
                        stage_args=stage_args,
                        compute=compute,
                        forward_batches=forward_batches,
                        metadata_guards=metadata_guards,
                    ),
                )
            finally:
                allocated, reserved = self._driver.memory_usage(device=device)
                within_budget = self._cache.account_capture(
                    bucket=bucket,
                    retained_hbm_bytes=self._retained_backing_hbm_bytes,
                    capture_growth_bytes=max(
                        0, allocated - baseline[0], reserved - baseline[1]
                    ),
                )
            bucket.program = program
            if not within_budget:
                self._capture_failure_reason = AFDReason.HBM_LIMIT
                self._cache.rollback(
                    bucket=bucket,
                    reason=AFDReason.HBM_LIMIT,
                )
                program = None
            else:
                program.arm_sentinel()
                value = program.replay(
                    stage_args=stage_args,
                    stage_rows=stage_rows,
                    forward_batches=forward_batches,
                )
        except Exception:
            if program is not None and bucket.program is not program:
                program.close()
            logger.exception(
                "AFD_ROLE_GRAPH_CAPTURE_EXCEPTION bucket=%s device=%r memory=%s",
                bucket.shape.digest,
                device,
                cuda_memory_diagnostic(device),
            )
            self._capture_failure_reason = AFDReason.CAPTURE_FAILED
            self._cache.rollback(
                bucket=bucket,
                reason=AFDReason.CAPTURE_FAILED,
            )
            return self._eager_step(
                compute=compute,
                stage_args=stage_args,
                reason=AFDReason.CAPTURE_FAILED,
            )
        if program is None:
            # Rollback already released the program. Refuse the unmatched eager
            # round outside capture handling, preserving the HBM failure reason.
            return self._eager_step(
                compute=compute, stage_args=stage_args, reason=AFDReason.HBM_LIMIT
            )
        # Outside the fallback: a graph that captured nothing replays in about
        # zero time and reads as an enormous speedup, and going eager from here
        # would cost an exchange round the peer does not perform. Refuse loudly.
        program.require_sentinel_written()
        return AFDExecutionResult(
            value=value,
            kind=AFDExecutionKind.EAGER,
            reason=AFDReason.ARMING,
            bucket=bucket.shape.digest,
        )
