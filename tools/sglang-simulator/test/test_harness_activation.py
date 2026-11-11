"""External harnesses can activate simulator hooks before importing SGLang."""

import os
import shutil
import subprocess
import sys
import textwrap
import uuid
from pathlib import Path

import pytest

SIMULATOR_SRC = Path(__file__).parents[1] / "src"


def _write_module(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(textwrap.dedent(content), encoding="utf-8")


@pytest.fixture
def scratch_dir():
    scratch_root = Path.cwd() / ".pytest-sglang-simulator"
    path = scratch_root / uuid.uuid4().hex
    path.mkdir(parents=True)
    try:
        yield path
    finally:
        shutil.rmtree(path)
        try:
            scratch_root.rmdir()
        except OSError:
            pass


def _make_fake_dependencies(scratch_dir: Path) -> tuple[Path, Path]:
    fake_src = scratch_dir / "fake_src"
    hook_log = scratch_dir / "hook.log"

    _write_module(
        fake_src / "sitecustomize.py",
        """
        import os
        import sys
        from pathlib import Path
        from types import ModuleType

        import sglang_simulator
        import sglang_simulator.simulation.sglang as sglang_package

        def add_module(parent, fullname):
            module = ModuleType(fullname)
            sys.modules[fullname] = module
            setattr(parent, fullname.rsplit(".", 1)[-1], module)
            return module

        hook = add_module(sglang_simulator, "sglang_simulator.hook")

        def install_class_hooks(_hooks):
            hook_log = Path(os.environ["SGLANG_SIMULATOR_HOOK_LOG"])
            with hook_log.open("a", encoding="utf-8") as file:
                file.write("installed\\n")
        hook.install_class_hooks = install_class_hooks

        hook_modules = (
            "cache_controller",
            "hicache_storage",
            "mem_cache_allocator",
            "mem_pool_host",
            "model_runner",
            "scheduler",
            "unified_radix_cache",
        )
        for module_name in hook_modules:
            module = add_module(
                sglang_package,
                f"sglang_simulator.simulation.sglang.{module_name}",
            )
            module.__getattr__ = lambda _name: object()

        sgl_kernel_hook = add_module(
            sglang_package,
            "sglang_simulator.simulation.sglang.sgl_kernel_hook",
        )

        def install_load_utils_stub():
            return None
        sgl_kernel_hook.install_load_utils_stub = install_load_utils_stub
        """,
    )

    return fake_src, hook_log


def _make_subprocess_env(scratch_dir: Path) -> tuple[dict[str, str], Path]:
    fake_src, hook_log = _make_fake_dependencies(scratch_dir)
    env = os.environ.copy()
    env.pop("SGLANG_SIMULATOR_AUTO_ACTIVATE", None)
    env.pop("SGLANG_SIMULATOR_BOOTSTRAP", None)
    env["PYTHONPATH"] = os.pathsep.join(
        [
            str(fake_src),
            str(SIMULATOR_SRC),
            env.get("PYTHONPATH", ""),
        ]
    )
    env["SGLANG_SIMULATOR_HOOK_LOG"] = str(hook_log)
    return env, hook_log


def _run_python(env: dict[str, str], code: str) -> None:
    result = subprocess.run(
        [sys.executable, "-c", textwrap.dedent(code)],
        capture_output=True,
        env=env,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_external_harness_marker_activates_before_main(scratch_dir):
    env, hook_log = _make_subprocess_env(scratch_dir)
    env["SGLANG_SIMULATOR_AUTO_ACTIVATE"] = "1"

    _run_python(
        env,
        f"""
        import os
        from pathlib import Path

        assert os.environ["SGLANG_SIMULATOR_BOOTSTRAP"] == "1"
        assert Path({str(hook_log)!r}).read_text().splitlines() == ["installed"]
        """,
    )


def test_startup_without_marker_is_unchanged(scratch_dir):
    env, hook_log = _make_subprocess_env(scratch_dir)

    _run_python(
        env,
        f"""
        import os
        import sys
        from pathlib import Path

        assert "SGLANG_SIMULATOR_BOOTSTRAP" not in os.environ
        assert "sglang_simulator.simulation.sglang.hook_bootstrap" not in sys.modules
        assert not Path({str(hook_log)!r}).exists()
        """,
    )


def test_worker_bootstrap_marker_activates_child_interpreter(scratch_dir):
    env, hook_log = _make_subprocess_env(scratch_dir)
    env["SGLANG_SIMULATOR_AUTO_ACTIVATE"] = "1"
    child_code = """
    import os

    assert "SGLANG_SIMULATOR_AUTO_ACTIVATE" not in os.environ
    assert os.environ["SGLANG_SIMULATOR_BOOTSTRAP"] == "1"
    """

    _run_python(
        env,
        f"""
        import os
        import subprocess
        import sys
        from pathlib import Path

        os.environ.pop("SGLANG_SIMULATOR_AUTO_ACTIVATE")
        subprocess.run(
            [sys.executable, "-c", {textwrap.dedent(child_code)!r}],
            check=True,
            env=os.environ.copy(),
        )
        assert Path({str(hook_log)!r}).read_text().splitlines() == [
            "installed",
            "installed",
        ]
        """,
    )


def test_usercustomize_and_direct_bootstrap_are_idempotent(scratch_dir):
    env, hook_log = _make_subprocess_env(scratch_dir)
    env["SGLANG_SIMULATOR_AUTO_ACTIVATE"] = "1"

    _run_python(
        env,
        f"""
        from pathlib import Path

        from sglang_simulator.simulation.sglang.hook_bootstrap import (
            install_simulator_hooks,
        )

        install_simulator_hooks()
        install_simulator_hooks()
        assert Path({str(hook_log)!r}).read_text().splitlines() == ["installed"]
        """,
    )
