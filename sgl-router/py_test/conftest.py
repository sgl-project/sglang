import sys
from pathlib import Path

# Only add py_src to path if the wheel is not installed (for local development)
# This ensures CI tests use the installed wheel which contains the Rust extension
_ROOT = Path(__file__).resolve().parents[1]
_SRC = _ROOT / "py_src"

# Check if sglang_router is already installed with the Rust extension
_wheel_installed = False
try:
    import sglang_router.sglang_router_rs
    _wheel_installed = True
except ImportError:
    pass

# Only add py_src if wheel is not installed (development mode)
if not _wheel_installed and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))
