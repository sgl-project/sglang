"""External harnesses can activate simulator hooks before importing SGLang."""

import os
import shutil
import subprocess
import textwrap
import uuid
import venv
import zipfile
from pathlib import Path
from types import SimpleNamespace

import pytest
from sglang_simulator.compat import validate_simulator_server_args

SIMULATOR_ROOT = Path(__file__).parents[1]


def _write_module(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(textwrap.dedent(content), encoding="utf-8")


@pytest.fixture(scope="session")
def simulator_wheel(tmp_path_factory) -> Path:
    build_root = tmp_path_factory.mktemp("sglang-simulator-wheel") / "source"
    shutil.copytree(
        SIMULATOR_ROOT,
        build_root,
        ignore=shutil.ignore_patterns(".pytest_cache", "__pycache__", "*.pyc"),
    )
    dist_dir = build_root / "dist"
    uv = shutil.which("uv")
    assert uv is not None
    result = subprocess.run(
        [
            uv,
            "build",
            "--wheel",
            "--out-dir",
            str(dist_dir),
            str(build_root),
        ],
        capture_output=True,
        check=False,
        cwd=build_root,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    wheels = list(dist_dir.glob("*.whl"))
    assert len(wheels) == 1
    return wheels[0]


@pytest.fixture(scope="session")
def installed_simulator(simulator_wheel, tmp_path_factory) -> tuple[Path, Path, Path]:
    venv_dir = tmp_path_factory.mktemp("sglang-simulator-venv") / "venv"
    venv.EnvBuilder(with_pip=True).create(venv_dir)
    python = venv_dir / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    site_packages = Path(
        subprocess.check_output(
            [
                str(python),
                "-c",
                "import site; print(site.getsitepackages()[0])",
            ],
            text=True,
        ).strip()
    )

    sitecustomize = site_packages / "sitecustomize.py"
    _write_module(
        sitecustomize,
        """
        import os
        from pathlib import Path

        log_path = os.environ.get("SGLANG_SIMULATOR_SITECUSTOMIZE_LOG")
        if log_path:
            Path(log_path).write_text("loaded\\n", encoding="utf-8")
        """,
    )
    original_sitecustomize = sitecustomize.read_bytes()

    result = subprocess.run(
        [
            str(python),
            "-m",
            "pip",
            "install",
            "--no-deps",
            str(simulator_wheel),
        ],
        capture_output=True,
        check=False,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert sitecustomize.read_bytes() == original_sitecustomize

    _write_module(
        site_packages / "sglang_simulator_test_stubs.py",
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
            if os.environ.get("SGLANG_SIMULATOR_TEST_FAIL") == "1":
                raise RuntimeError("forced hook installation failure")
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
        sgl_kernel_hook.install_load_utils_stub = lambda: None
        """,
    )
    (site_packages / "00_sglang_simulator_test_stubs.pth").write_text(
        "import sglang_simulator_test_stubs\n",
        encoding="utf-8",
    )
    return python, site_packages, sitecustomize


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


def _make_subprocess_env(
    scratch_dir: Path, installed_simulator: tuple[Path, Path, Path]
) -> tuple[dict[str, str], Path, Path]:
    python, _site_packages, _sitecustomize = installed_simulator
    hook_log = scratch_dir / "hook.log"
    env = os.environ.copy()
    env.pop("PYTHONPATH", None)
    env.pop("SGLANG_SIMULATOR_BOOTSTRAP", None)
    env["SGLANG_SIMULATOR_HOOK_LOG"] = str(hook_log)
    return env, hook_log, python


def _run_python(
    python: Path, env: dict[str, str], code: str
) -> subprocess.CompletedProcess:
    return subprocess.run(
        [str(python), "-c", textwrap.dedent(code)],
        capture_output=True,
        check=False,
        env=env,
        text=True,
        timeout=30,
    )


def test_wheel_installs_namespaced_bootstrap_only(simulator_wheel):
    with zipfile.ZipFile(simulator_wheel) as wheel:
        names = set(wheel.namelist())

    assert "sglang_simulator_bootstrap.pth" in names
    assert "sglang_simulator/bootstrap.py" in names
    assert "sitecustomize.py" not in names
    assert "usercustomize.py" not in names
    assert "_sglang_simulator_startup.py" not in names


def test_external_harness_marker_activates_installed_wheel(
    scratch_dir, installed_simulator
):
    env, hook_log, python = _make_subprocess_env(scratch_dir, installed_simulator)
    _python, site_packages, _sitecustomize = installed_simulator
    env["SGLANG_SIMULATOR_BOOTSTRAP"] = "1"

    result = _run_python(
        python,
        env,
        f"""
        import site
        import sglang_simulator
        from pathlib import Path

        assert site.ENABLE_USER_SITE is False
        assert Path(sglang_simulator.__file__).parent == Path({str(site_packages)!r}) / "sglang_simulator"
        assert Path({str(hook_log)!r}).read_text().splitlines() == ["installed"]
        """,
    )

    assert result.returncode == 0, result.stdout + result.stderr


def test_startup_without_marker_is_unchanged(scratch_dir, installed_simulator):
    env, hook_log, python = _make_subprocess_env(scratch_dir, installed_simulator)

    result = _run_python(
        python,
        env,
        f"""
        import sys
        from pathlib import Path

        assert "sglang_simulator.bootstrap" in sys.modules
        assert "sglang_simulator.simulation.sglang.hook_bootstrap" not in sys.modules
        assert "torch" not in sys.modules
        assert not Path({str(hook_log)!r}).exists()
        """,
    )

    assert result.returncode == 0, result.stdout + result.stderr


def test_existing_sitecustomize_coexists_with_activation(
    scratch_dir, installed_simulator
):
    env, hook_log, python = _make_subprocess_env(scratch_dir, installed_simulator)
    _python, _site_packages, sitecustomize = installed_simulator
    sitecustomize_log = scratch_dir / "sitecustomize.log"
    env["SGLANG_SIMULATOR_BOOTSTRAP"] = "1"
    env["SGLANG_SIMULATOR_SITECUSTOMIZE_LOG"] = str(sitecustomize_log)

    result = _run_python(
        python,
        env,
        f"""
        from pathlib import Path

        assert Path({str(hook_log)!r}).read_text().splitlines() == ["installed"]
        assert Path({str(sitecustomize_log)!r}).read_text().splitlines() == ["loaded"]
        assert Path({str(sitecustomize)!r}).exists()
        """,
    )

    assert result.returncode == 0, result.stdout + result.stderr


def test_worker_bootstrap_marker_activates_child_interpreter(
    scratch_dir, installed_simulator
):
    env, hook_log, python = _make_subprocess_env(scratch_dir, installed_simulator)
    env["SGLANG_SIMULATOR_BOOTSTRAP"] = "1"
    child_code = """
    import os

    assert os.environ["SGLANG_SIMULATOR_BOOTSTRAP"] == "1"
    """

    result = _run_python(
        python,
        env,
        f"""
        import os
        import subprocess
        import sys
        from pathlib import Path

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

    assert result.returncode == 0, result.stdout + result.stderr


def test_standalone_bootstrap_sets_worker_marker(scratch_dir, installed_simulator):
    env, hook_log, python = _make_subprocess_env(scratch_dir, installed_simulator)

    result = _run_python(
        python,
        env,
        f"""
        import os
        from pathlib import Path

        assert "SGLANG_SIMULATOR_BOOTSTRAP" not in os.environ
        from sglang_simulator.simulation.sglang.hook_bootstrap import (
            install_simulator_hooks,
        )

        install_simulator_hooks()
        assert os.environ["SGLANG_SIMULATOR_BOOTSTRAP"] == "1"
        assert Path({str(hook_log)!r}).read_text().splitlines() == ["installed"]
        """,
    )

    assert result.returncode == 0, result.stdout + result.stderr


def test_pth_and_direct_bootstrap_are_idempotent(scratch_dir, installed_simulator):
    env, hook_log, python = _make_subprocess_env(scratch_dir, installed_simulator)
    env["SGLANG_SIMULATOR_BOOTSTRAP"] = "1"

    result = _run_python(
        python,
        env,
        f"""
        from pathlib import Path

        from sglang_simulator.bootstrap import activate, activate_from_env
        from sglang_simulator.simulation.sglang.hook_bootstrap import (
            install_simulator_hooks,
        )

        activate_from_env()
        activate()
        install_simulator_hooks()
        install_simulator_hooks()
        assert Path({str(hook_log)!r}).read_text().splitlines() == ["installed"]
        """,
    )

    assert result.returncode == 0, result.stdout + result.stderr


def test_activation_failure_stops_external_harness(scratch_dir, installed_simulator):
    env, _hook_log, python = _make_subprocess_env(scratch_dir, installed_simulator)
    env["SGLANG_SIMULATOR_BOOTSTRAP"] = "1"
    env["SGLANG_SIMULATOR_TEST_FAIL"] = "1"
    continued = scratch_dir / "continued"

    result = _run_python(
        python,
        env,
        f"""
        from pathlib import Path

        Path({str(continued)!r}).write_text("continued", encoding="utf-8")
        """,
    )

    assert result.returncode != 0
    assert not continued.exists()
    assert "SGLang Simulator activation failed" in result.stderr


def test_external_harness_server_args_satisfy_simulator_contract():
    server_args = SimpleNamespace(
        tp_size=1,
        ep_size=1,
        dp_size=1,
        pp_size=1,
        attn_cp_size=1,
        dcp_size=1,
        disable_overlap_schedule=True,
        disable_cuda_graph=True,
        attention_backend="torch_native",
        prefill_attention_backend="torch_native",
        decode_attention_backend="torch_native",
    )

    validate_simulator_server_args(server_args)
