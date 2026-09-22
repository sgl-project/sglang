"""CPU regression: a failing ``deep_ep`` import must not break dense-model servers.

``sgl-deep-ep`` is a hard dependency, and ``deep_ep/__init__.py`` runs host
checks at import time (NCCL runtime match, CUDA home lookup, JIT init) that
raise ``AssertionError``/``OSError``/``RuntimeError``. Every server process
imports the MoE token dispatchers (via ``runner_utils`` -> ``deepep_adapter``),
so when the DeepEP guards only caught ``ImportError``, one failed check (for
example a second ``libnccl`` loaded through ``LD_PRELOAD``) killed the launcher
at import time even for models that never use DeepEP.

Each case runs in a fresh interpreter with a stand-in ``deep_ep`` package on
``PYTHONPATH`` so the real package and the parent process's module cache are
never touched.
"""

import os
import subprocess
import sys
import tempfile
import textwrap
import unittest

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=30, suite="base-a-test-cpu")

_FAILURE = "Duplicate NCCL runtime found in the current system"

_CHILD = textwrap.dedent(
    f"""
    import torch

    from sglang.srt.layers.moe.token_dispatcher import deepep, deepep_v2

    assert deepep.use_deepep is False
    assert deepep_v2.use_deepep_v2 is False

    try:
        deepep._DeepEPDispatcherImplBase(
            group=None,
            router_topk=8,
            permute_fusion=False,
            num_experts=64,
            num_local_experts=8,
            hidden_size=4096,
            params_dtype=torch.bfloat16,
            deepep_mode=None,
        )
    except ImportError as exc:
        assert {_FAILURE!r} in str(exc), str(exc)
        assert isinstance(exc.__cause__, AssertionError), repr(exc.__cause__)
    else:
        raise SystemExit("DeepEP dispatcher was created without deep_ep")

    try:
        deepep_v2._ensure_deepep_v2_available()
    except ImportError as exc:
        assert {_FAILURE!r} in str(exc), str(exc)
    else:
        raise SystemExit("DeepEP v2 reported available without deep_ep")

    print("OK")
    """
)


class TestDeepEPImportGuard(CustomTestCase):
    def test_deep_ep_import_assertion_is_deferred(self):
        with tempfile.TemporaryDirectory() as fake_root:
            pkg = os.path.join(fake_root, "deep_ep")
            os.makedirs(pkg)
            with open(os.path.join(pkg, "__init__.py"), "w") as f:
                f.write(f"raise AssertionError({_FAILURE!r})\n")

            env = dict(os.environ)
            env["PYTHONPATH"] = os.pathsep.join(
                p for p in (fake_root, env.get("PYTHONPATH")) if p
            )
            result = subprocess.run(
                [sys.executable, "-c", _CHILD],
                env=env,
                capture_output=True,
                text=True,
                timeout=300,
            )

        self.assertEqual(
            result.returncode,
            0,
            f"stdout:\n{result.stdout}\nstderr:\n{result.stderr[-4000:]}",
        )
        self.assertIn("OK", result.stdout)


if __name__ == "__main__":
    unittest.main()
