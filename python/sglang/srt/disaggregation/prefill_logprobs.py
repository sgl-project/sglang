"""Variable-sized prefill logprobs carried beside the P/D completion signal."""

import msgspec
import numpy as np

INPUT_FIELDS = (
    "input_token_logprobs_val",
    "input_token_logprobs_idx",
    "input_top_logprobs_val",
    "input_top_logprobs_idx",
    "input_token_ids_logprobs_val",
    "input_token_ids_logprobs_idx",
    "input_top_logprobs_val_flat",
    "input_top_logprobs_idx_flat",
    "input_top_logprobs_flat_null_prefix",
)
OUTPUT_FIELDS = (
    "output_token_logprobs_val",
    "output_token_logprobs_idx",
    "output_top_logprobs_val",
    "output_top_logprobs_idx",
    "output_token_ids_logprobs_val",
    "output_token_ids_logprobs_idx",
)
FIELDS = INPUT_FIELDS + OUTPUT_FIELDS


def encode(logprob) -> bytes:
    values = [getattr(logprob, name) for name in FIELDS]
    for index in (6, 7):
        array = values[index]
        if array is not None:
            values[index] = (array.shape, array.tobytes())
    # Scheduler output logprobs may still contain CPU scalar tensors.
    return msgspec.msgpack.encode((1, values), enc_hook=lambda value: value.tolist())


def decode(payload: bytes) -> list | None:
    message = msgspec.msgpack.decode(payload)
    if message is None:
        return None
    version, values = message
    if version != 1 or not isinstance(values, list) or len(values) != len(FIELDS):
        raise ValueError("Invalid P/D prompt logprob metadata")
    for index, dtype in ((6, np.float32), (7, np.int32)):
        if values[index] is not None:
            shape, data = values[index]
            values[index] = np.frombuffer(data, dtype=dtype).reshape(shape)
    return values


def restore_inputs(logprob, values: list) -> None:
    for name, value in zip(INPUT_FIELDS, values[: len(INPUT_FIELDS)], strict=True):
        setattr(logprob, name, value)


def append_output(logprob, values: list) -> None:
    for name, value in zip(OUTPUT_FIELDS, values[len(INPUT_FIELDS) :], strict=True):
        if value:
            getattr(logprob, name).extend(value)
