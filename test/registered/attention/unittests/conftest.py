import sys
from pathlib import Path

# Add this directory to sys.path so that test files can do
# `sys.path.insert(0, str(Path(__file__).resolve().parents[1]))` equivalently,
# and so pytest can import subpackages (dense/, mla/, etc.) without
# confusing this directory with the Python stdlib `unittest` module.
_here = str(Path(__file__).resolve().parent)
if _here not in sys.path:
    sys.path.insert(0, _here)
