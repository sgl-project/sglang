"""Tests for xpu_ci_start_container.sh against a stub docker; needs no XPU or daemon.

Run: python3 -m unittest scripts/ci/xpu/test_xpu_ci_start_container.py
"""

import json
import os
import shutil
import subprocess
import tempfile
import textwrap
import time
import unittest
from pathlib import Path

SCRIPT = Path(__file__).resolve().parent / "xpu_ci_start_container.sh"

# Stub docker: tracks containers in a state file and applies the busybox side
# effects the script relies on. `run -dt` waits FAKE_RUN_DELAY before the
# container exists, widening the window a concurrent start could race into.
DOCKER = textwrap.dedent(
    r"""
    #!/usr/bin/env python3
    import fcntl, json, os, shutil, sys, time
    from pathlib import Path

    args = sys.argv[1:]
    root = Path(os.environ["FAKE_ROOT"])
    with (root / "docker.jsonl").open("a") as log:
        log.write(json.dumps({"pid": os.getpid(), "args": args}) + "\n")
    state = root / "containers.txt"

    def names():
        return state.read_text().split() if state.exists() else []

    def update(fn):
        with (root / "state.lock").open("w") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            state.write_text("\n".join(fn(names())) + "\n")

    def mounts():
        out = {}
        for i, a in enumerate(args):
            if a == "-v" and ":" in args[i + 1]:
                src, dst = args[i + 1].split(":")[:2]
                out[dst] = src
        return out

    cmd = args[0] if args else ""
    if cmd == "image":
        print("sha256:" + "c" * 64)
    elif cmd == "ps":
        print("\n".join(names()))
    elif cmd == "rm":
        update(lambda n: [x for x in n if x != args[-1]])
    elif cmd == "run" and "--rm" in args:
        m, script = mounts(), " ".join(args)
        if "rm -rf /o/" in script:
            shutil.rmtree(Path(m["/o"]) / args[-1].split("/o/")[1], ignore_errors=True)
        elif "mknod" in script:
            for f in ("token", "stored_tokens"):
                (Path(m["/u"]) / f).touch()
    elif cmd == "run":
        time.sleep(float(os.environ.get("FAKE_RUN_DELAY", "0")))
        name = args[args.index("--name") + 1]
        update(lambda n: n + [name])
    """
).lstrip()


class StartContainerTest(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="xpu-start-test-"))
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)
        (self.tmp / "bin").mkdir()
        (self.tmp / "home").mkdir()
        (self.tmp / "ws").mkdir()
        docker = self.tmp / "bin" / "docker"
        docker.write_text(DOCKER)
        docker.chmod(0o755)
        self.overlay_root = self.tmp / "home/.cache/sglang-xpu-ci-pr-overlay"

    def env(self, **extra):
        env = {
            k: v
            for k, v in os.environ.items()
            if not k.startswith(("GITHUB_", "HF_", "XPU_CI_"))
        }
        env.update(
            PATH=f"{self.tmp / 'bin'}:{os.environ['PATH']}",
            HOME=str(self.tmp / "home"),
            GITHUB_WORKSPACE=str(self.tmp / "ws"),
            FAKE_ROOT=str(self.tmp),
            ZE_AFFINITY_MASK="0",
        )
        env.update(extra)
        return env

    def start(self, name, **extra):
        return subprocess.Popen(
            ["bash", str(SCRIPT), "--container-name", name],
            env=self.env(**extra),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )

    def finish(self, proc):
        out, _ = proc.communicate(timeout=60)
        self.assertEqual(proc.returncode, 0, out)

    def docker_runs(self):
        lines = (self.tmp / "docker.jsonl").read_text().splitlines()
        calls = [json.loads(line)["args"] for line in lines]
        return [a for a in calls if a[:2] == ["run", "-dt"]], calls

    def test_concurrent_start_keeps_other_jobs_overlay(self):
        a = self.start("job_a", XPU_CI_UNTRUSTED="1", FAKE_RUN_DELAY="3")
        a_overlay = self.overlay_root / "job_a"
        deadline = time.time() + 30
        while not (a_overlay / "upper").exists():
            self.assertLess(time.time(), deadline, "job_a never created its overlay")
            time.sleep(0.05)
        # job_a's overlay exists but its container does not yet.
        b = self.start("job_b", XPU_CI_UNTRUSTED="1")
        self.finish(b)
        self.finish(a)
        self.assertTrue((a_overlay / "upper").is_dir(), "job_b removed job_a's overlay")
        self.assertTrue((self.overlay_root / "job_b" / "upper").is_dir())

    def test_stale_overlay_is_removed(self):
        (self.overlay_root / "gone_job" / "upper").mkdir(parents=True)
        self.finish(self.start("job_c", XPU_CI_UNTRUSTED="1"))
        self.assertFalse((self.overlay_root / "gone_job").exists())

    def test_untrusted_gets_no_token_and_no_workspace_mount(self):
        self.finish(self.start("pr_job", XPU_CI_UNTRUSTED="1", HF_TOKEN="hf_secret"))
        runs, calls = self.docker_runs()
        args = runs[0]
        self.assertNotIn("HF_TOKEN", args)
        self.assertFalse(any(a.endswith(":/sglang-checkout") for a in args), args)
        self.assertIn(["cp", f"{self.tmp / 'ws'}/.", "pr_job:/sglang-checkout"], calls)
        self.assertNotIn("hf_secret", json.dumps(calls))

    def test_trusted_gets_token_by_name_and_workspace_mount(self):
        self.finish(self.start("main_job", XPU_CI_UNTRUSTED="0", HF_TOKEN="hf_secret"))
        runs, calls = self.docker_runs()
        args = runs[0]
        self.assertIn("HF_TOKEN", args)
        self.assertIn(f"{self.tmp / 'ws'}:/sglang-checkout", args)
        # Passed by name, so the value never appears in docker's argv.
        self.assertNotIn("hf_secret", json.dumps(calls))


if __name__ == "__main__":
    unittest.main()
