import subprocess
import sys
import textwrap
from pathlib import Path

PKG_PARENT = str(Path(__file__).resolve().parents[1])


def test_package_imports_without_torch():
    code = textwrap.dedent(
        f"""
        import sys
        sys.path.insert(0, r"{PKG_PARENT}")
        import asc_bench
        import asc_bench.cleanup
        import asc_bench.config
        import asc_bench.diff_runs
        import asc_bench.expand
        import asc_bench.npu
        import asc_bench.provenance
        import asc_bench.report
        import asc_bench.runner
        import asc_bench.sla
        offenders = sorted(m for m in sys.modules if m == "torch" or m.startswith("torch."))
        assert not offenders, f"torch leaked into core package: {{offenders}}"
        print("portable")
        """
    )
    proc = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, timeout=120
    )
    assert proc.returncode == 0, proc.stderr
    assert "portable" in proc.stdout
