import sys
import types
from pathlib import Path

# Ensure local sources in py_src are importable ahead of any installed package
_ROOT = Path(__file__).resolve().parents[1]
_SRC = _ROOT / "py_src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

# Provide lightweight stubs to avoid importing heavy/optional deps during collection
if "sglang_router.mini_lb" not in sys.modules:
    mini_lb = types.ModuleType("sglang_router.mini_lb")

    class MiniLoadBalancer:  # minimal placeholder for tests that don't use it
        def __init__(self, *args, **kwargs):
            pass

    mini_lb.MiniLoadBalancer = MiniLoadBalancer
    sys.modules["sglang_router.mini_lb"] = mini_lb

# Optional: stub setproctitle if present in import path
if "setproctitle" not in sys.modules:
    setpt = types.ModuleType("setproctitle")
    setpt.setproctitle = lambda *a, **k: None
    sys.modules["setproctitle"] = setpt
