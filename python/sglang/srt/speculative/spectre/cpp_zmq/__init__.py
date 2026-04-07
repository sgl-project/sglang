import importlib
import subprocess
import warnings
from pathlib import Path

FORCE_BUILD = False


def build_cpp_zmq():
    build_script = Path(__file__).parent / "scripts" / "build_cpp_zmq.sh"
    if build_script.exists():
        print("[spectre_zmq] Compiling C++ extension...")
        ret = subprocess.run(
            ["bash", str(build_script)],
            capture_output=True,
            text=True,
        )
        if ret.returncode != 0:
            warnings.warn(f"Failed to build spectre_zmq:\n{ret.stdout}\n{ret.stderr}")


def _try_import():
    try:
        return importlib.import_module(".spectre_zmq", __package__)
    except ImportError as e:
        warnings.warn(f"spectre_zmq not found: {e}")
        return None


if FORCE_BUILD:
    build_cpp_zmq()
else:
    success = _try_import()
    if not success:
        build_cpp_zmq()

from .spectre_zmq import DealerEndpoint, RouterEndpoint

try:
    from .spectre_zmq import set_spectre_log_level
except ImportError:

    def set_spectre_log_level(level: int) -> None:
        return None


__all__ = ["DealerEndpoint", "RouterEndpoint", "set_spectre_log_level"]
