"""With SGLANG_PRESPAWN_WORKERS=1 the `sglang` package applies the transformers
patches on demand and resolves its frontend-language API lazily (PEP 562);
by default it imports eagerly as before. The public names must resolve either
way, and the lazy import must not pull in torch or transformers."""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=15, suite="base-a-test-cpu")

import os
import subprocess
import sys
import unittest

import sglang
from sglang.test.test_utils import CustomTestCase


class TestLazyPublicApi(CustomTestCase):
    def test_all_public_names_resolve(self):
        for name in sglang.__all__:
            self.assertIsNotNone(getattr(sglang, name), name)

    def test_lazy_name_is_the_real_object(self):
        from sglang.lang.api import function

        self.assertIs(sglang.function, function)

    def test_from_import_works(self):
        from sglang import global_config  # noqa: F401

    def test_unknown_attribute(self):
        with self.assertRaises(AttributeError):
            sglang.no_such_name


class TestLazyMode(CustomTestCase):
    def _run(self, code, lazy):
        env = dict(os.environ)
        env.pop("SGLANG_PRESPAWN_WORKERS", None)
        if lazy:
            env["SGLANG_PRESPAWN_WORKERS"] = "1"
        return subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            text=True,
            check=True,
            env=env,
        ).stdout.strip()

    def test_lazy_import_stays_light_and_resolves(self):
        code = (
            "import sys, sglang\n"
            "light = [m for m in ('torch', 'transformers', 'sglang.lang.api')"
            " if m not in sys.modules]\n"
            "from sglang.lang.api import function\n"
            "print(len(light), sglang.function is function, sglang.gen is not None)\n"
        )
        self.assertEqual(self._run(code, lazy=True), "3 True True")

    def test_eager_import_is_the_default(self):
        code = "import sys, sglang; print('sglang.lang.api' in sys.modules)\n"
        self.assertEqual(self._run(code, lazy=False), "True")


if __name__ == "__main__":
    unittest.main()
