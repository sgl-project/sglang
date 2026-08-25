import ast
import json
import os
import re
import tomllib
import unittest
from pathlib import Path


def workspace_file(relative_path: str) -> Path:
    return (
        Path(os.environ["TEST_SRCDIR"]) / os.environ["TEST_WORKSPACE"] / relative_path
    )


def requirement_name(requirement: str) -> str:
    return re.split(r"[\[<>=!;]", requirement, maxsplit=1)[0].strip().lower()


def torch_pulling_packages() -> frozenset[str]:
    source = workspace_file("test/registered/core/test_srt_empty_deps.py").read_text()
    tree = ast.parse(source)
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        if not any(
            isinstance(target, ast.Name) and target.id == "TORCH_PULLING_PACKAGES"
            for target in node.targets
        ):
            continue
        if not isinstance(node.value, ast.Call) or not node.value.args:
            break
        return frozenset(ast.literal_eval(node.value.args[0]))
    raise AssertionError("TORCH_PULLING_PACKAGES was not found")


class PackageProfilesTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        profile_path = workspace_file("bazel/python/profiles.json")
        cls.profiles = json.loads(profile_path.read_text())["profiles"]

    def test_profiles_reference_existing_extras(self) -> None:
        names = [profile["name"] for profile in self.profiles]
        self.assertEqual(len(names), len(set(names)))

        for profile in self.profiles:
            manifest_path = workspace_file(profile["manifest"])
            self.assertTrue(manifest_path.is_file(), profile)
            with manifest_path.open("rb") as manifest_file:
                project = tomllib.load(manifest_file)["project"]
            extras = project.get("optional-dependencies", {})
            self.assertIn(profile["extra"], extras, profile)

    def test_srt_empty_remains_torch_free(self) -> None:
        manifest_path = workspace_file("python/pyproject_other.toml")
        with manifest_path.open("rb") as manifest_file:
            extras = tomllib.load(manifest_file)["project"]["optional-dependencies"]

        self.assertEqual(extras["srt_empty"], ["sglang[runtime_base]"])
        runtime_base = {requirement_name(dep) for dep in extras["runtime_base"]}
        self.assertFalse(runtime_base & torch_pulling_packages())

    def test_bootstrap_lock_is_a_runtime_base_subset(self) -> None:
        manifest_path = workspace_file("python/pyproject_other.toml")
        with manifest_path.open("rb") as manifest_file:
            extras = tomllib.load(manifest_file)["project"]["optional-dependencies"]
        runtime_base = {requirement_name(dep) for dep in extras["runtime_base"]}

        lock_path = workspace_file(
            "bazel/python/srt_empty_bootstrap.requirements.lock.txt"
        )
        locked = {
            requirement_name(line)
            for line in lock_path.read_text().splitlines()
            if line and not line.startswith((" ", "#", "-"))
        }
        self.assertTrue(locked)
        self.assertLessEqual(locked, runtime_base)
        self.assertFalse(locked & torch_pulling_packages())


if __name__ == "__main__":
    unittest.main()
