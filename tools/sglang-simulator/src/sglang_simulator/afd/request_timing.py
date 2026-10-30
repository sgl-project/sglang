"""Pure request-time checks shared by the Scheduler bridge and CPU tests."""

import math


def completion_offsets(predictor, request_ids, duration):
    offsets = predictor.completion_offsets()
    if set(offsets) != set(request_ids):
        raise ValueError("AFD_REQUEST_COMPLETION_IDENTITIES_MISMATCH")
    for value in offsets.values():
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            or not 0 < value <= duration
        ):
            raise ValueError("AFD_REQUEST_COMPLETION_OUTSIDE_STEP")
    return offsets


def decode_past_length(req):
    # A decode consumes the most recently generated token. All original input
    # tokens plus earlier output tokens precede it, including cache hits.
    if not hasattr(req, "origin_input_ids") or not req.output_ids:
        raise ValueError("AFD_EXACT_DECODE_CONTEXT_REQUIRED")
    return len(req.origin_input_ids) + len(req.output_ids) - 1
