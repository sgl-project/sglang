import sys
from pathlib import Path

# Put this directory on sys.path so pytest can import the subpackages
# (dense/, mla/, ...) without confusing this directory with the stdlib
# `unittest` module.
_here = str(Path(__file__).resolve().parent)
if _here not in sys.path:
    sys.path.insert(0, _here)
