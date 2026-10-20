"""Immutable CPU metadata from one native model forward."""

from dataclasses import dataclass

from sglang_simulator.time_predictor.base import (
    PredictorError,
    ScheduleBatch,
    ScheduleRequest,
)

SNAPSHOT_ATTRIBUTE = "_sglang_simulator_prepared_batch"


def _cpu_int(value, *, field):
    device = getattr(value, "device", None)
    if device is not None:
        if getattr(device, "type", None) != "cpu":
            raise PredictorError("invalid_batch", f"{field} must be on CPU")
        if getattr(value, "ndim", None) != 0:
            raise PredictorError(
                "invalid_batch", f"{field} must contain scalar integers"
            )
        value = value.item()
    if type(value) is not int:
        raise PredictorError("invalid_batch", f"{field} must contain integers")
    return value


def _cpu_lengths(batch, field, batch_size):
    values = getattr(batch, field, None)
    if values is None:
        raise PredictorError("invalid_batch", f"prepared forward is missing {field}")
    device = getattr(values, "device", None)
    if device is not None:
        if getattr(device, "type", None) != "cpu":
            raise PredictorError("invalid_batch", f"{field} must be on CPU")
        if getattr(values, "ndim", None) != 1:
            raise PredictorError("invalid_batch", f"{field} must be one-dimensional")
    elif not isinstance(values, (list, tuple)):
        raise PredictorError("invalid_batch", f"{field} must be a CPU vector")
    if len(values) != batch_size:
        raise PredictorError(
            "invalid_batch",
            f"{field} must match the prepared batch size",
            batch_size=batch_size,
            length=len(values),
        )
    return tuple(_cpu_int(value, field=field) for value in values)


@dataclass(frozen=True)
class PreparedBatchSnapshot:
    forward_mode: str
    seq_lens_cpu: tuple[int, ...]
    extend_seq_lens_cpu: tuple[int, ...] | None = None
    extend_prefix_lens_cpu: tuple[int, ...] | None = None

    def __post_init__(self):
        if self.forward_mode not in {"EXTEND", "MIXED", "DECODE"}:
            raise PredictorError(
                "unsupported_forward_mode",
                f"unsupported SGLang forward mode {self.forward_mode!r}",
                forward_mode=self.forward_mode,
            )
        if (
            type(self.seq_lens_cpu) is not tuple
            or not self.seq_lens_cpu
            or any(
                type(length) is not int or length < 1 for length in self.seq_lens_cpu
            )
        ):
            raise PredictorError(
                "invalid_batch", "prepared sequence lengths must be positive"
            )
        if self.forward_mode == "DECODE":
            if (
                self.extend_seq_lens_cpu is not None
                or self.extend_prefix_lens_cpu is not None
            ):
                raise PredictorError(
                    "invalid_batch", "decode snapshot must not contain extend lengths"
                )
            return
        for field in ("extend_seq_lens_cpu", "extend_prefix_lens_cpu"):
            lengths = getattr(self, field)
            minimum = 1 if field == "extend_seq_lens_cpu" else 0
            if (
                type(lengths) is not tuple
                or len(lengths) != len(self.seq_lens_cpu)
                or any(
                    type(length) is not int or length < minimum for length in lengths
                )
            ):
                raise PredictorError("invalid_batch", f"invalid prepared {field}")
        if any(
            sequence != prefix + extend
            for sequence, prefix, extend in zip(
                self.seq_lens_cpu,
                self.extend_prefix_lens_cpu,
                self.extend_seq_lens_cpu,
            )
        ):
            raise PredictorError(
                "invalid_batch",
                "prepared sequence length must equal prefix plus extend",
            )

    def predictor_batch(self, *, expected_batch_size=None):
        if expected_batch_size is not None and expected_batch_size != len(
            self.seq_lens_cpu
        ):
            raise PredictorError(
                "invalid_batch",
                "prepared snapshot does not match the result batch size",
            )
        if self.forward_mode == "DECODE":
            requests = [
                ScheduleRequest(extend_length=1, past_kv_length=sequence - 1)
                for sequence in self.seq_lens_cpu
            ]
        else:
            requests = [
                ScheduleRequest(extend_length=extend, past_kv_length=prefix)
                for extend, prefix in zip(
                    self.extend_seq_lens_cpu, self.extend_prefix_lens_cpu
                )
            ]
        return ScheduleBatch(reqs=requests, forward_mode=self.forward_mode)


def capture_prepared_batch(batch):
    """Copy CPU mirrors before model execution; never synchronize device metadata."""
    mode = getattr(getattr(batch, "forward_mode", None), "name", None)
    if mode not in {"EXTEND", "MIXED", "DECODE"}:
        raise PredictorError(
            "unsupported_forward_mode",
            f"unsupported SGLang forward mode {mode!r}",
            forward_mode=mode,
        )
    batch_size = getattr(batch, "batch_size", None)
    if type(batch_size) is not int or batch_size < 1:
        raise PredictorError("invalid_batch", "prepared batch size must be positive")
    sequence_lengths = _cpu_lengths(batch, "seq_lens_cpu", batch_size)
    if mode == "DECODE":
        return PreparedBatchSnapshot(mode, sequence_lengths)
    return PreparedBatchSnapshot(
        mode,
        sequence_lengths,
        _cpu_lengths(batch, "extend_seq_lens_cpu", batch_size),
        _cpu_lengths(batch, "extend_prefix_lens_cpu", batch_size),
    )


def predictor_batch_from_result(result, *, expected_batch_size):
    snapshot = getattr(getattr(result, "logits_output", None), SNAPSHOT_ATTRIBUTE, None)
    if not isinstance(snapshot, PreparedBatchSnapshot):
        raise PredictorError(
            "invalid_batch", "model result is missing a valid prepared snapshot"
        )
    return snapshot.predictor_batch(expected_batch_size=expected_batch_size)
