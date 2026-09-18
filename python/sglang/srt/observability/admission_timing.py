"""Parse timing only on deployments explicitly trusting their SMG ingress."""

import math


def parse_admission_wait(value):
    try:
        seconds = float(value)
    except (TypeError, ValueError):
        return 0.0
    return seconds if math.isfinite(seconds) and seconds >= 0 else 0.0
