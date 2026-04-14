"""Shared WebSocket utilities for e2e tests and benchmarks."""


def gateway_ws_url(base_url: str) -> str:
    """Convert an HTTP gateway URL to the equivalent WebSocket URL."""
    ws_url = base_url.replace("https://", "wss://").replace("http://", "ws://")
    if not ws_url.startswith(("ws://", "wss://")):
        ws_url = f"ws://{ws_url}"
    return f"{ws_url}/v1/responses"


def percentile(values: list[float], p: float) -> float:
    """Linear-interpolation percentile (0-100 scale)."""
    if not values:
        return 0.0
    sorted_vals = sorted(values)
    n = len(sorted_vals)
    if n == 1:
        return sorted_vals[0]
    k = (p / 100.0) * (n - 1)
    f = int(k)
    c = f + 1
    if c >= n:
        return sorted_vals[-1]
    d = k - f
    return sorted_vals[f] + d * (sorted_vals[c] - sorted_vals[f])
