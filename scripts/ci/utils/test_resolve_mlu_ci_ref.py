"""Unit tests for resolve_mlu_ci_ref.sh.

The script turns a workflow `ref` input into an immutable commit SHA before it
is handed to the Cambricon runner. It accepts an empty input (fall back to the
default SHA), a full 40-character SHA, or a branch/tag name -- including fully
qualified `refs/heads/*` and `refs/tags/*` refs, which is the shape of
`github.ref`.
"""

import os
import subprocess
import tempfile
import unittest
from pathlib import Path

SCRIPT = Path(__file__).resolve().parent / "resolve_mlu_ci_ref.sh"


def _git(repo: Path, env: dict, *args: str) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=repo,
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


class TestResolveMluCiRef(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.TemporaryDirectory()
        cls.repo = Path(cls._tmp.name) / "repo"
        cls.repo.mkdir()
        env = {
            **os.environ,
            "GIT_AUTHOR_NAME": "t",
            "GIT_AUTHOR_EMAIL": "t@example.com",
            "GIT_COMMITTER_NAME": "t",
            "GIT_COMMITTER_EMAIL": "t@example.com",
            "GIT_CONFIG_GLOBAL": os.devnull,
            "GIT_CONFIG_SYSTEM": os.devnull,
        }
        cls.env = env

        _git(cls.repo, env, "init", "-q")
        _git(cls.repo, env, "symbolic-ref", "HEAD", "refs/heads/main")
        (cls.repo / "f.txt").write_text("first\n")
        _git(cls.repo, env, "add", "f.txt")
        _git(cls.repo, env, "commit", "-q", "-m", "first")
        cls.first = _git(cls.repo, env, "rev-parse", "HEAD")
        _git(cls.repo, env, "branch", "feature")
        _git(cls.repo, env, "tag", "light")

        (cls.repo / "f.txt").write_text("second\n")
        _git(cls.repo, env, "add", "f.txt")
        _git(cls.repo, env, "commit", "-q", "-m", "second")
        cls.second = _git(cls.repo, env, "rev-parse", "HEAD")
        _git(cls.repo, env, "tag", "-a", "annot", "-m", "annotated")

        cls.repo_url = str(cls.repo)

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()

    def _run(self, input_ref: str, default_sha: str):
        with tempfile.NamedTemporaryFile(delete=False) as handle:
            output_path = handle.name
        env = {
            **self.env,
            "INPUT_REF": input_ref,
            "DEFAULT_SHA": default_sha,
            "REPO_URL": self.repo_url,
            "GITHUB_OUTPUT": output_path,
        }
        try:
            result = subprocess.run(
                ["bash", str(SCRIPT)],
                env=env,
                capture_output=True,
                text=True,
            )
            resolved = ""
            for line in Path(output_path).read_text().splitlines():
                if line.startswith("commit_sha="):
                    resolved = line.split("=", 1)[1]
            return result, resolved
        finally:
            os.unlink(output_path)

    def test_empty_ref_uses_default_sha(self):
        result, resolved = self._run("", self.second)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(resolved, self.second)

    def test_full_sha_passthrough(self):
        result, resolved = self._run(self.first, "ignored")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(resolved, self.first)

    def test_branch_name(self):
        result, resolved = self._run("feature", self.second)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(resolved, self.first)

    def test_lightweight_tag(self):
        result, resolved = self._run("light", self.second)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(resolved, self.first)

    def test_annotated_tag_is_peeled_to_commit(self):
        result, resolved = self._run("annot", self.first)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(resolved, self.second)

    def test_fully_qualified_branch_ref(self):
        result, resolved = self._run("refs/heads/feature", self.second)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(resolved, self.first)

    def test_fully_qualified_annotated_tag_ref(self):
        result, resolved = self._run("refs/tags/annot", self.first)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(resolved, self.second)

    def test_fully_qualified_lightweight_tag_ref(self):
        result, resolved = self._run("refs/tags/light", self.second)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(resolved, self.first)

    def test_unknown_ref_fails(self):
        result, resolved = self._run("does-not-exist", self.second)
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(resolved, "")
        self.assertIn("Could not resolve Git ref", result.stdout)


if __name__ == "__main__":
    unittest.main()
