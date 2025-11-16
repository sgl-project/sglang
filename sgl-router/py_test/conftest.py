import sys
from importlib.util import find_spec
from pathlib import Path

# Only add bindings/python to path if the wheel is not installed (for local development)
# This ensures CI tests use the installed wheel which contains the Rust extension
_ROOT = Path(__file__).resolve().parents[1]
_SRC = _ROOT / "bindings" / "python"

# Check if sglang_router is already installed with the Rust extension
_wheel_installed = find_spec("sglang_router.sglang_router_rs") is not None

# Only add bindings/python if wheel is not installed (development mode)
if not _wheel_installed and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))
