import os
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")

AMD_SCRIPTS = Path(__file__).resolve().parents[4] / "scripts/ci/amd"
CLEAR = "GPU[0] : GPU Memory Allocated (VRAM%): 0\n"
BUSY = "GPU[0] : GPU Memory Allocated (VRAM%): 6\n"


class TestAMDVramPreflight(CustomTestCase):
    def run_script(self, script, readings, *, sourced=False):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            commands = root / "bin"
            commands.mkdir()
            for tool in ("awk", "cat", "dirname", "grep", "sort", "tr", "xargs"):
                (commands / tool).symlink_to(shutil.which(tool))
            for tool in ("docker", "pgrep", "fuser", "lsof", "ps", "sleep"):
                command = commands / tool
                command.write_text(
                    '#!/bin/bash\nprintf "%s %s\\n" "${0##*/}" "$*" '
                    '>> "$FAKE_ROOT/actions"\n'
                )
                command.chmod(0o755)
            for tool in ("hostname", "date"):
                command = commands / tool
                command.write_text("#!/bin/bash\necho fixture\n")
                command.chmod(0o755)
            timeout = commands / "timeout"
            timeout.write_text('#!/bin/bash\nshift\nexec "$@"\n')
            timeout.chmod(0o755)
            if readings is not None:
                for index, (output, status) in enumerate(readings):
                    (root / f"{index}.output").write_text(output)
                    (root / f"{index}.status").write_text(str(status))
                command = commands / "rocm-smi"
                command.write_text(
                    '#!/bin/bash\n[[ "$1" == --showmemuse ]] || exit 0\n'
                    'index=0\nif [[ -f "$FAKE_ROOT/count" ]]; then\n'
                    '    read -r index < "$FAKE_ROOT/count"\nfi\n'
                    'printf "%s\\n" "$((index + 1))" > "$FAKE_ROOT/count"\n'
                    f"(( index < {len(readings)} )) || index={len(readings) - 1}\n"
                    'cat "$FAKE_ROOT/$index.output"\n'
                    'exit "$(cat "$FAKE_ROOT/$index.status")"\n'
                )
                command.chmod(0o755)
            argv = ["/bin/bash", str(AMD_SCRIPTS / script)]
            if sourced:
                # A caller's condition disables errexit within a sourced function.
                function = script.removesuffix(".sh")
                argv = [
                    "/bin/bash",
                    "-ec",
                    f'source "$1"; if {function}; then exit 0; else exit $?; fi',
                    "test",
                    str(AMD_SCRIPTS / script),
                ]
            completed = subprocess.run(
                argv,
                text=True,
                capture_output=True,
                timeout=10,
                env={**os.environ, "PATH": str(commands), "FAKE_ROOT": str(root)},
            )
            actions = root / "actions"
            return completed, actions.read_text() if actions.exists() else ""

    def test_check_rejects_unknown_telemetry(self):
        """Missing, failed, or malformed telemetry must not report free VRAM."""
        for readings, expected in (
            (None, 2),
            ([(CLEAR, 1)], 2),
            ([("", 0)], 2),
            ([("No GPUs found\n", 0)], 2),
            ([("GPU[0] : GPU Memory Allocated (VRAM%): N/A\n", 0)], 2),
            ([(CLEAR + "GPU[1] : GPU Memory Allocated (VRAM%): N/A\n", 0)], 2),
            ([("GPU[0] : GPU Memory Allocated (VRAM%): 101\n", 0)], 2),
            ([("GPU[0] : GPU Memory Allocated (VRAM%): 100\n", 0)], 1),
            ([(CLEAR + "GPU[1] : GPU Memory Allocated (VRAM%): 5\n", 0)], 0),
            ([(CLEAR + BUSY.replace("GPU[0]", "GPU[1]"), 0)], 1),
            ([("GPU[0] : GPU Memory Allocated (VRAM%): 5.5\n", 0)], 1),
        ):
            with self.subTest(readings=readings):
                completed, _ = self.run_script("check_vram_clear.sh", readings)
                self.assertEqual(completed.returncode, expected, completed.stdout)

    def test_unknown_telemetry_never_starts_cleanup(self):
        """An indeterminate preflight cannot authorize container/process cleanup."""
        for sourced in (False, True):
            for readings in (None, [(CLEAR, 1)], [("", 0)], [("unexpected\n", 0)]):
                with self.subTest(sourced=sourced, readings=readings):
                    completed, actions = self.run_script(
                        "ensure_vram_clear.sh", readings, sourced=sourced
                    )
                    self.assertEqual(completed.returncode, 2, completed.stdout)
                    self.assertEqual(actions, "")
                    self.assertNotIn("VRAM is already clear", completed.stdout)

    def test_rechecks_stop_on_unknown_and_keep_busy_recovery(self):
        """Losing telemetry during recovery must not trigger another cleanup."""
        for readings, expected, attempts in (
            ([(CLEAR, 0)], 0, 0),
            ([(BUSY, 0), (CLEAR, 0)], 0, 0),
            ([(BUSY, 0), ("", 0)], 2, 0),
            ([(BUSY, 0), (BUSY, 0), (CLEAR, 0)], 0, 1),
            ([(BUSY, 0), (BUSY, 0), ("", 0)], 2, 1),
            ([(BUSY, 0)], 1, 3),
        ):
            with self.subTest(readings=readings):
                completed, actions = self.run_script("ensure_vram_clear.sh", readings)
                self.assertEqual(completed.returncode, expected, completed.stdout)
                self.assertEqual(
                    completed.stdout.count("=== Cleanup Attempt"), attempts
                )
                self.assertEqual(actions.count("pgrep "), attempts)


if __name__ == "__main__":
    unittest.main()
