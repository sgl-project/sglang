import json
import tomllib
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
GATEWAY = ROOT / "sgl-model-gateway"


class GatewayLockParityTest(unittest.TestCase):
    def test_rendered_crates_are_pinned_by_gateway_cargo_lock(self):
        with (GATEWAY / "Cargo.lock").open("rb") as cargo_lock_file:
            cargo_lock = tomllib.load(cargo_lock_file)
        with (ROOT / "bazel/rust/sgl-model-gateway-crates.lock.json").open() as bazel_lock_file:
            bazel_lock = json.load(bazel_lock_file)

        cargo_packages = {
            f"{package['name']} {package['version']}"
            for package in cargo_lock["package"]
        }
        rendered_crates = set(bazel_lock["crates"])

        self.assertTrue(rendered_crates)
        self.assertEqual(rendered_crates - cargo_packages, set())
        self.assertTrue(
            {
                "pyo3 0.27.2",
                "tonic 0.12.3",
                "tonic 0.14.6",
            }.issubset(rendered_crates)
        )

    def test_gateway_manifests_keep_separate_edition_and_abi_contract(self):
        with (GATEWAY / "Cargo.toml").open("rb") as gateway_manifest_file:
            gateway_manifest = tomllib.load(gateway_manifest_file)
        with (GATEWAY / "bindings/python/Cargo.toml").open("rb") as python_manifest_file:
            python_manifest = tomllib.load(python_manifest_file)

        self.assertEqual(gateway_manifest["package"]["edition"], "2021")
        self.assertEqual(python_manifest["package"]["edition"], "2021")
        self.assertEqual(
            gateway_manifest["dependencies"]["tonic-v12"],
            {"version": "0.12.3", "package": "tonic"},
        )
        self.assertEqual(gateway_manifest["dependencies"]["tonic"]["version"], "0.14.2")
        self.assertEqual(
            set(python_manifest["dependencies"]["pyo3"]["features"]),
            {"abi3-py38", "extension-module"},
        )


if __name__ == "__main__":
    unittest.main()
