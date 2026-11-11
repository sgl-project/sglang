"""Load the Miles GPU-delta extension when Miles is installed.

Importing the extension registers its IPC types before the scheduler and
DP-controller receive loops start. Broken installed extensions fail at startup.
"""

try:
    from miles.backends.sglang_utils.gpu_delta import integration
except ModuleNotFoundError as exc:
    if exc.name != "miles":
        raise
    integration = None
