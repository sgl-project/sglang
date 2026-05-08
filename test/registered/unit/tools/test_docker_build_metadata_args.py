import importlib.util
import json
import subprocess
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[4]
CI_REGISTER_PATH = REPO_ROOT / "python" / "sglang" / "test" / "ci" / "ci_register.py"
HELPER_PATH = REPO_ROOT / "scripts" / "ci" / "utils" / "docker_build_metadata_args.py"
DOCKERFILE_PATH = REPO_ROOT / "docker" / "Dockerfile"
WORKFLOW_PATH = REPO_ROOT / ".github" / "workflows" / "_docker-build-and-publish.yml"


def _load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


register_cpu_ci = _load_module("ci_register", CI_REGISTER_PATH).register_cpu_ci
register_cpu_ci(est_time=0, suite="stage-a-test-cpu")


class TestDockerBuildMetadataArgs(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.helper = _load_module("docker_build_metadata_args", HELPER_PATH)

    def run_helper(
        self,
        *,
        cuda: str,
        tag_config: list[dict[str, object]],
        image_repo: str = "lmsysorg/sglang",
        version: str = "0.6.0",
        build_commit: str = "abcdef1234567890",
        build_url: str = "https://github.com/sgl-project/sglang/actions/runs/1",
        date: str = "20260429",
    ) -> list[str]:
        result = subprocess.run(
            [
                "python3",
                str(HELPER_PATH),
                "--cuda",
                cuda,
                "--tag-config",
                json.dumps(tag_config),
                "--image-repo",
                image_repo,
                "--sgl-version",
                version,
                "--build-commit",
                build_commit,
                "--build-url",
                build_url,
                "--date",
                date,
            ],
            check=True,
            stdout=subprocess.PIPE,
            text=True,
        )
        return result.stdout.splitlines()

    @staticmethod
    def option_values(args: list[str], option: str) -> list[str]:
        return [args[i + 1] for i, arg in enumerate(args[:-1]) if arg == option]

    def build_args(self, args: list[str]) -> dict[str, str]:
        values = {}
        for value in self.option_values(args, "--build-arg"):
            key, arg_value = value.split("=", 1)
            values[key] = arg_value
        return values

    def test_release_metadata_prefers_versioned_tag(self):
        args = self.run_helper(
            cuda="cu129",
            tag_config=[
                {"cuda": "cu129", "tags": ["v{version}", "latest"]},
                {"cuda": "cu130", "tags": ["v{version}-cu130", "latest-cu130"]},
            ],
        )

        self.assertEqual(
            self.build_args(args),
            {
                "SGLANG_BUILD_COMMIT": "abcdef1234567890",
                "SGLANG_BUILD_URL": (
                    "https://github.com/sgl-project/sglang/actions/runs/1"
                ),
                "SGLANG_IMAGE_TAG": "lmsysorg/sglang:v0.6.0",
            },
        )

    def test_runtime_metadata_uses_custom_repo_and_runtime_tag(self):
        args = self.run_helper(
            cuda="cu130",
            image_repo="lmsysorg/sglang-staging",
            tag_config=[
                {"cuda": "cu129", "tags": ["v{version}-runtime", "latest-runtime"]},
                {
                    "cuda": "cu130",
                    "tags": ["v{version}-cu130-runtime", "latest-cu130-runtime"],
                },
            ],
        )

        self.assertEqual(
            self.build_args(args)["SGLANG_IMAGE_TAG"],
            "lmsysorg/sglang-staging:v0.6.0-cu130-runtime",
        )

    def test_dev_nightly_metadata_prefers_unique_tag_from_checked_out_commit(self):
        args = self.run_helper(
            cuda="cu129",
            version="",
            build_commit="1234567890abcdef",
            tag_config=[
                {"cuda": "cu129", "tags": ["dev", "nightly-dev-{date}-{short_sha}"]},
                {
                    "cuda": "cu130",
                    "tags": ["dev-cu13", "nightly-dev-cu13-{date}-{short_sha}"],
                },
            ],
        )

        self.assertEqual(
            self.build_args(args)["SGLANG_IMAGE_TAG"],
            "lmsysorg/sglang:nightly-dev-20260429-12345678",
        )
        self.assertEqual(
            self.build_args(args)["SGLANG_BUILD_COMMIT"],
            "1234567890abcdef",
        )

    def test_custom_dev_tag_is_treated_as_specific(self):
        args = self.run_helper(
            cuda="cu130",
            version="",
            tag_config=[
                {"cuda": "cu129", "tags": ["dev-my-test"]},
                {"cuda": "cu130", "tags": ["dev-cu13-my-test"]},
            ],
        )

        self.assertEqual(
            self.build_args(args)["SGLANG_IMAGE_TAG"],
            "lmsysorg/sglang:dev-cu13-my-test",
        )

    def test_missing_cuda_entry_fails(self):
        with self.assertRaisesRegex(ValueError, "cu130"):
            self.helper.select_tag(
                json.dumps([{"cuda": "cu129", "tags": ["v{version}"]}]),
                "cu130",
                "0.6.0",
                "20260429",
                "abcdef12",
            )

    def test_final_dockerfile_stages_embed_metadata_contract(self):
        dockerfile = DOCKERFILE_PATH.read_text()
        framework_stage = dockerfile.split("FROM framework AS framework_final", 1)[
            1
        ].split("FROM nvidia/cuda:${CUDA_VERSION}-cudnn-devel-ubuntu24.04 AS runtime")[
            0
        ]
        runtime_stage = dockerfile.split(
            "FROM nvidia/cuda:${CUDA_VERSION}-cudnn-devel-ubuntu24.04 AS runtime", 1
        )[1]

        for stage in (framework_stage, runtime_stage):
            for expected in (
                "ARG SGLANG_BUILD_COMMIT=unknown",
                "ARG SGLANG_BUILD_URL=",
                "ARG SGLANG_IMAGE_TAG=local/sglang:dev",
                "SGLANG_BUILD_COMMIT=${SGLANG_BUILD_COMMIT:-unknown}",
                "SGLANG_BUILD_URL=${SGLANG_BUILD_URL:-}",
                "SGLANG_IMAGE_TAG=${SGLANG_IMAGE_TAG:-local/sglang:dev}",
                'org.opencontainers.image.source="https://github.com/sgl-project/sglang"',
                'org.opencontainers.image.revision="${SGLANG_BUILD_COMMIT}"',
                'org.opencontainers.image.version="${SGLANG_IMAGE_TAG}"',
                'org.opencontainers.image.url="${SGLANG_BUILD_URL}"',
                'ai.sglang.build.commit="${SGLANG_BUILD_COMMIT}"',
                'ai.sglang.build.url="${SGLANG_BUILD_URL}"',
                'ai.sglang.image.tag="${SGLANG_IMAGE_TAG}"',
            ):
                self.assertIn(expected, stage)

    def test_shared_docker_workflow_uses_checked_out_commit(self):
        workflow = WORKFLOW_PATH.read_text()

        self.assertIn("git rev-parse HEAD", workflow)
        self.assertIn("scripts/ci/utils/docker_build_metadata_args.py", workflow)
        self.assertIn("mapfile -t METADATA_ARGS", workflow)
        self.assertIn('"${METADATA_ARGS[@]}"', workflow)


if __name__ == "__main__":
    unittest.main()
