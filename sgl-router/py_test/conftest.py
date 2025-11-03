import sys
from pathlib import Path

# Ensure local sources in py_src are importable ahead of any installed package
_ROOT = Path(__file__).resolve().parents[1]
_SRC = _ROOT / "py_src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))
