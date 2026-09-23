"""Run checked-in bootstrap tests without importing the CUDA SGLang package."""

import importlib.util
import sys
import types
import unittest
from pathlib import Path

r = Path(__file__).resolve().parents[4]
name = "sglang.srt.disaggregation.draft_bootstrap"
spec = importlib.util.spec_from_file_location(
    name, r / "python/sglang/srt/disaggregation/draft_bootstrap.py"
)
m = importlib.util.module_from_spec(spec)
sys.modules[name] = m
spec.loader.exec_module(m)
ci = types.ModuleType("sglang.test.ci.ci_register")
ci.register_cpu_ci = lambda **kw: None
sys.modules[ci.__name__] = ci
spec = importlib.util.spec_from_file_location(
    "bootstrap_tests", r / "test/registered/unit/disaggregation/test_draft_bootstrap.py"
)
t = importlib.util.module_from_spec(spec)
spec.loader.exec_module(t)
result = unittest.TextTestRunner(verbosity=2).run(
    unittest.defaultTestLoader.loadTestsFromModule(t)
)
sys.exit(not result.wasSuccessful())
