"""Exercise shell cache state flow with fake installers and real archives."""

import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest


SETUP_SCRIPT = Path(__file__).resolve().with_name("setup_evalscope.sh")
SPEC = "evalscope[ifeval]==1.11.1"
FAKE_COMMAND = r'''
import json
import os
from pathlib import Path
import shutil
import sys

args = sys.argv[1:]
command = Path(__file__).name
failure = os.environ.get("FAKE_FAIL_AT", "")


def record(event):
    with open(os.environ["FAKE_LOG"], "a") as stream:
        stream.write(json.dumps([event, args]) + "\n")


def fail_at(event):
    if failure == event:
        print("injected failure: " + event, file=sys.stderr)
        raise SystemExit(1)


if command == "mv":
    record("mv")
    if args[-2].endswith("/venv.tar.gz"):
        fail_at("publish")
    if sys.platform == "darwin":
        args = [arg.replace("T", "") if arg in ("-T", "-fT") else arg
                for arg in args]
        args = [arg for arg in args if arg != "-"]
    os.execv(os.environ["REAL_MV"], [os.environ["REAL_MV"], *args])

elif command == "python3" and args[0] == "-":
    record("key")
    source = sys.stdin.read()
    if sys.platform == "darwin":
        real_read_text = Path.read_text

        def read_text(path, *a, **kw):
            if str(path) == "/etc/os-release":
                return "ID=unittest-darwin\nVERSION_ID=fixture\n"
            return real_read_text(path, *a, **kw)

        Path.read_text = read_text
    sys.argv = args
    exec(compile(source, "<cache-key>", "exec"))

elif command == "python3" and args[:3] == ["-m", "venv", "--clear"]:
    record("venv")
    venv = Path(args[-1])
    if venv.exists():
        shutil.rmtree(venv)
    (venv / "bin").mkdir(parents=True)
    for name in ("python", "evalscope"):
        shutil.copy2(__file__, venv / "bin" / name)

elif command == "python" and args[:2] == ["-m", "pip"]:
    venv = Path(__file__).parent.parent
    operation = args[2]
    record("pip_" + operation)
    if operation == "install":
        if "--upgrade" in args:
            fail_at("bootstrap")
        else:
            fail_at("install")
            (venv / ".fake-spec").write_text(args[-1])
    elif operation == "check":
        fail_at("check")
        if not (venv / ".fake-spec").is_file():
            raise SystemExit(1)
    elif operation == "freeze":
        fail_at("freeze")
        print("evalscope==" + (venv / ".fake-spec").read_text().split("==", 1)[1])
    else:
        raise SystemExit("unexpected pip operation: " + operation)

elif command == "python" and args[0] == "-":
    record("verify")
    sys.stdin.read()
    fail_at("import")
    marker = Path(__file__).parent.parent / ".fake-spec"
    if not marker.is_file() or marker.read_text() != args[1]:
        raise SystemExit(1)
    print("fake EvalScope verification only")

elif command == "evalscope" and args == ["--help"]:
    record("help")
    fail_at("help")

else:
    raise SystemExit("unexpected fake invocation: " + command + " " + repr(args))
'''


@unittest.skipUnless(sys.platform in ("linux", "darwin"), "requires Linux or Darwin")
class TestEvalScopeCache(unittest.TestCase):
    def setUp(self):
        for command in ("bash", "tar", "gzip", "mktemp", "mv", "cp", "rm", "mkdir"):
            if not shutil.which(command):
                self.skipTest("missing command: " + command)
        self.tar = shutil.which("gtar" if sys.platform == "darwin" else "tar")
        if not self.tar:
            self.skipTest("Darwin needs GNU tar (gtar) for tar -I 'gzip -1'")
        temporary = tempfile.TemporaryDirectory(prefix="evalscope cache tests ")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.bin = self.root / "commands"
        self.bin.mkdir()
        (self.bin / "tar").symlink_to(self.tar)
        self.scratch = self.root / "scratch"
        self.scratch.mkdir()
        self.cache = self.root / "cache"
        self.venv = self.root / "venv one"
        self.log = self.root / "commands.jsonl"
        self.env = os.environ.copy()
        self.env.update({
            "PATH": str(self.bin) + os.pathsep + self.env.get("PATH", os.defpath),
            "TMPDIR": str(self.scratch),
            "EVALSCOPE_CACHE_DIR": str(self.cache),
            "EVALSCOPE_WHEELHOUSE": str(self.root / "missing wheelhouse"),
            "FAKE_LOG": str(self.log),
            "REAL_MV": shutil.which("mv"),
        })
        launcher = "#!" + str(Path(sys.executable).resolve()) + "\n" + FAKE_COMMAND
        for name in ("python3", "mv"):
            path = self.bin / name
            path.write_text(launcher)
            path.chmod(0o755)

    def run_setup(self, *, venv=None, spec=SPEC, failure="", cache=None):
        self.log.write_text("")
        env = self.env.copy()
        env.update({
            "EVALSCOPE_VENV": str(venv or self.venv),
            "EVALSCOPE_SPEC": spec,
            "FAKE_FAIL_AT": failure,
            "EVALSCOPE_CACHE_DIR": str(cache or self.cache),
        })
        return subprocess.run(
            ["bash", str(SETUP_SCRIPT)], env=env, text=True,
            capture_output=True, timeout=30, check=False,
        )

    def events(self, name):
        return [args for event, args in map(json.loads, self.log.read_text().splitlines())
                if event == name]

    def assert_success(self, result):
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("evalscope binary:", result.stdout)
        self.assertEqual(list(self.scratch.iterdir()), [])
        self.assertEqual(list(self.cache.glob(".*")), [])

    def archive_for(self, venv=None):
        key = ((venv or self.venv) / ".cache-key").read_text().strip()
        self.assertRegex(key, r"^[0-9a-f]{16}$")
        return self.cache / key / "venv.tar.gz"

    def assert_archive(self, archive, venv, spec=SPEC):
        with tempfile.TemporaryDirectory(dir=self.root) as stage:
            subprocess.run([self.tar, "-xzf", str(archive), "-C", stage], check=True)
            restored = Path(stage)
            self.assertEqual((restored / ".cache-key").read_text(),
                             (venv / ".cache-key").read_text())
            self.assertEqual((restored / ".fake-spec").read_text(), spec)
            self.assertTrue(os.access(restored / "bin/evalscope", os.X_OK))
            self.assertEqual((restored / ".cache-requirements.txt").read_text(),
                             "evalscope==" + spec.split("==", 1)[1] + "\n")

    def assert_zero_install(self):
        self.assertEqual(self.events("pip_install"), [])
        self.assertEqual(len(self.events("verify")), 1)
        self.assertEqual(len(self.events("help")), 1)
        self.assertEqual(len(self.events("pip_check")), 1)

    def test_miss_publishes_then_hit_without_install_or_venv_creation(self):
        result = self.run_setup()
        self.assert_success(result)
        self.assertIn("cache MISS:", result.stdout)
        self.assertIn("cache published:", result.stdout)
        self.assertEqual(len(self.events("venv")), 1)
        self.assertEqual(len(self.events("pip_install")), 2)
        archive = self.archive_for()
        self.assert_archive(archive, self.venv)
        original = archive.read_bytes()
        shutil.rmtree(self.venv)

        result = self.run_setup(failure="install")
        self.assert_success(result)
        self.assertIn("cache HIT:", result.stdout)
        self.assertNotIn("cache MISS:", result.stdout)
        self.assert_zero_install()
        self.assertEqual(self.events("venv"), [])
        self.assertEqual(archive.read_bytes(), original)

    def test_hit_clears_existing_invalid_local_directory(self):
        self.assert_success(self.run_setup())
        shutil.rmtree(self.venv)
        self.venv.mkdir()
        (self.venv / "stale-file").write_text("must be removed")
        result = self.run_setup(failure="install")
        self.assert_success(result)
        self.assertIn("cache HIT:", result.stdout)
        self.assert_zero_install()
        self.assertEqual(len(self.events("venv")), 1)
        self.assertFalse((self.venv / "stale-file").exists())

    def test_local_idempotence_without_shared_cache(self):
        self.assert_success(self.run_setup())
        shutil.rmtree(self.cache)
        result = self.run_setup(failure="install")
        self.assert_success(result)
        self.assertIn("environment already present", result.stdout)
        self.assertNotIn("cache HIT:", result.stdout)
        self.assert_zero_install()
        self.assertEqual(self.events("venv"), [])
        self.assertFalse(self.cache.exists())

    def test_spec_and_absolute_venv_path_separate_keys(self):
        self.assert_success(self.run_setup())
        first = self.archive_for()
        other_spec = "evalscope[ifeval]==1.11.2"
        result = self.run_setup(spec=other_spec)
        self.assert_success(result)
        self.assertIn("cache MISS:", result.stdout)
        second = self.archive_for()
        other_venv = self.root / "venv two"
        result = self.run_setup(venv=other_venv, spec=other_spec)
        self.assert_success(result)
        self.assertIn("cache MISS:", result.stdout)
        third = self.archive_for(other_venv)
        self.assertEqual(len({first.parent.name, second.parent.name, third.parent.name}), 3)
        self.assertTrue(all(path.is_file() for path in (first, second, third)))
        self.assert_archive(second, self.venv, other_spec)
        self.assert_archive(third, other_venv, other_spec)

        result = self.run_setup(failure="install")
        self.assert_success(result)
        self.assertIn("cache HIT:", result.stdout)
        self.assert_zero_install()
        self.assertEqual(self.archive_for(), first)
        self.assert_archive(first, self.venv)

    def test_corrupt_archive_is_cleaned_rebuilt_and_replaced(self):
        for corruption in ("gzip", "missing_key", "wrong_key", "invalid_environment"):
            with self.subTest(corruption=corruption):
                venv = self.root / corruption
                self.assert_success(self.run_setup(venv=venv))
                archive = self.archive_for(venv)
                if corruption == "gzip":
                    archive.write_bytes(b"not a gzip archive")
                else:
                    if corruption == "missing_key":
                        (venv / ".cache-key").unlink()
                    elif corruption == "wrong_key":
                        (venv / ".cache-key").write_text("wrong-key\n")
                    else:
                        (venv / ".fake-spec").write_text("invalid installed version")
                    subprocess.run([self.tar, "-czf", str(archive), "-C", str(venv), "."],
                                   check=True)
                corrupted = archive.read_bytes()
                shutil.rmtree(venv)
                venv.mkdir()
                (venv / "stale-file").write_text("must be removed")

                result = self.run_setup(venv=venv)
                self.assert_success(result)
                message = ("validation failed" if corruption == "invalid_environment"
                           else "restore failed")
                self.assertIn(message, result.stdout)
                self.assertIn("cache MISS:", result.stdout)
                self.assertIn("cache published:", result.stdout)
                self.assertEqual(len(self.events("pip_install")), 2)
                self.assertFalse((venv / "stale-file").exists())
                self.assertNotEqual(archive.read_bytes(), corrupted)
                self.assert_archive(archive, venv)

                shutil.rmtree(venv)
                result = self.run_setup(venv=venv, failure="install")
                self.assert_success(result)
                self.assertIn("cache HIT:", result.stdout)
                self.assert_zero_install()
                self.assertEqual(self.events("venv"), [])

    def test_install_and_verification_failures_never_publish(self):
        for failure in ("bootstrap", "install", "import", "help", "check"):
            with self.subTest(failure=failure):
                venv = self.root / failure
                result = self.run_setup(venv=venv, failure=failure)
                self.assertNotEqual(result.returncode, 0, result.stdout + result.stderr)
                self.assertIn("injected failure: " + failure, result.stderr)
                self.assertNotIn("cache published:", result.stdout)
                self.assertFalse((venv / ".cache-key").exists())
                self.assertEqual(list(self.cache.glob("*/venv.tar.gz")), [])
                self.assertEqual(list(self.scratch.iterdir()), [])

    def test_concurrent_publication_keeps_a_complete_archive(self):
        venvs = [self.venv, self.root / "second publisher"]
        for venv in venvs:
            self.assert_success(self.run_setup(venv=venv))
        key = "0123456789abcdef"
        archive = self.cache / key / "venv.tar.gz"
        for venv in venvs:
            (venv / ".cache-key").write_text(key + "\n")
        publication = 'if [ "${built}" = 1 ]; then' + SETUP_SCRIPT.read_text().split(
            'if [ "${built}" = 1 ]; then', 1
        )[1].rsplit('\necho "evalscope binary:', 1)[0]
        publishers = []
        for venv in venvs:
            env = self.env.copy()
            env.update({
                "EVALSCOPE_VENV": str(venv),
                "venv_python": str(venv / "bin/python"),
                "cache_key": key,
                "key_dir": str(archive.parent),
                "cache_archive": str(archive),
                "built": "1",
            })
            publishers.append(subprocess.Popen(
                ["bash", "-euo", "pipefail", "-c", publication],
                env=env, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            ))
        for process in publishers:
            stdout, stderr = process.communicate(timeout=30)
            self.assertEqual(process.returncode, 0, stdout + stderr)
            self.assertIn("cache published:", stdout)
            self.assertNotIn("::warning::", stdout)
        self.assert_archive(archive, venvs[0])
        self.assertEqual(list(self.scratch.iterdir()), [])
        self.assertEqual(list(self.cache.glob(".*")), [])

    def test_cache_write_failure_keeps_local_environment_usable(self):
        blocked_cache = self.root / "cache is a file"
        blocked_cache.write_text("cannot mkdir here")
        for failure in ("mkdir", "freeze", "publish"):
            with self.subTest(failure=failure):
                venv = self.root / ("write-failure-" + failure)
                cache = blocked_cache if failure == "mkdir" else self.cache
                result = self.run_setup(venv=venv, cache=cache, failure=failure)
                self.assert_success(result)
                self.assertIn("::warning::EvalScope cache publication failed", result.stdout)
                self.assertNotIn("cache published:", result.stdout)
                self.assertTrue((venv / ".cache-key").is_file())
                self.assertTrue(os.access(venv / "bin/evalscope", os.X_OK))
                self.assertEqual(list(self.cache.glob("*/venv.tar.gz")), [])
                result = self.run_setup(venv=venv, cache=cache, failure="install")
                self.assert_success(result)
                self.assertIn("environment already present", result.stdout)
                self.assert_zero_install()
                self.assertEqual(self.events("venv"), [])


if __name__ == "__main__":
    if sys.platform == "darwin":
        print("Darwin adapters: real gtar, mocked os-release, mv -T removed; "
              "native Linux behavior still requires a Linux run.", file=sys.stderr)
    unittest.main()
