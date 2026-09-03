import pathlib
import tempfile
import unittest

from check_rust_ext_cache_prefix import check

ACTION = """
inputs:
  artifact_name:
    default: ''
runs:
  using: composite
  steps:
    - name: Resolve the cache key prefix for this runner
      id: prefix
      run: |
        resolved="rust-ext-$(uname -m)-cp310-cp312"
    - uses: actions/cache/restore@v4
      with:
        key: {key}-${{{{ hashFiles({hashed}) }}}}
"""

BUILD_WORKFLOW = """
on:
  workflow_call:
    inputs:
      runs_on:
        type: string
        required: true
      restore_runs_on:
        type: string
        default: ubuntu-latest
      collect_runs_on:
        type: string
        default: ubuntu-latest
      cache_key_prefix:
        type: string
        default: rust-ext-x86_64-cp310-cp312
jobs:
  compile:
    strategy:
      matrix:
        python-version: {versions}
    steps:
      - run: echo build
  collect:
    steps:
      - uses: actions/cache/save@v4
        with:
          key: ${{{{ inputs.cache_key_prefix }}}}-${{{{ hashFiles({hashed}) }}}}
"""

RUNNER_CONFIGS = """
runner_configs:
  1-gpu-large:  { install: x, runs_on: 1-gpu-h100 }
  4-gpu-gb300:  { install: x, runs_on: 4-gpu-gb300 }
"""

HASHED = "'rust/**', 'python/setup.py'"
DERIVED_KEY = "${{ steps.prefix.outputs.value }}"


def producer(name: str, **with_: str) -> str:
    body = "\n".join(f"      {key}: {value}" for key, value in with_.items())
    return (
        f"  {name}:\n"
        "    uses: ./.github/workflows/_pr-test-rust-ext-build.yml\n"
        "    with:\n"
        f"{body}\n"
    )


def stage(name: str, **with_: str) -> str:
    body = "\n".join(f"      {key}: {value}" for key, value in with_.items())
    return (
        f"  {name}:\n"
        "    uses: ./.github/workflows/_pr-test-stage.yml\n"
        "    with:\n"
        f"{body}\n"
    )


X86 = producer("build", runs_on="ubuntu-22.04")
ARM = producer(
    "build-aarch64",
    runs_on="ubuntu-22.04-arm",
    collect_runs_on="ubuntu-22.04-arm",
    cache_key_prefix="rust-ext-aarch64-cp310-cp312",
)
X86_STAGE = stage(
    "test-1-gpu-large",
    runner_config="1-gpu-large",
    rust_ext_artifact="${{ needs.build.outputs.artifact_name }}",
)
ARM_STAGE = stage(
    "test-4-gpu-gb300",
    runner_config="4-gpu-gb300",
    rust_ext_artifact="${{ needs.build-aarch64.outputs.artifact_name }}",
)


class TestCheckRustExtCachePrefix(unittest.TestCase):
    def check_tree(
        self,
        *,
        jobs: str,
        key: str = DERIVED_KEY,
        action_hashed: str = HASHED,
        build_hashed: str = HASHED,
        versions: str = "['3.10', '3.12']",
        extra_workflow: str = "",
    ) -> list[str]:
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            action = root / ".github/actions/download-rust-ext"
            workflows = root / ".github/workflows"
            configs = root / "scripts/ci"
            for path in (action, workflows, configs):
                path.mkdir(parents=True, exist_ok=True)
            (action / "action.yml").write_text(
                ACTION.format(key=key, hashed=action_hashed), encoding="utf-8"
            )
            (workflows / "_pr-test-rust-ext-build.yml").write_text(
                BUILD_WORKFLOW.format(hashed=build_hashed, versions=versions),
                encoding="utf-8",
            )
            (workflows / "caller.yml").write_text(f"jobs:\n{jobs}", encoding="utf-8")
            if extra_workflow:
                (workflows / "extra.yml").write_text(extra_workflow, encoding="utf-8")
            (configs / "runner_configs.yml").write_text(
                RUNNER_CONFIGS, encoding="utf-8"
            )
            return check(root)

    def assert_one(self, *, problems: list[str], fragment: str) -> None:
        self.assertEqual(len(problems), 1, problems)
        self.assertIn(fragment, problems[0])

    def test_accepts_an_x86_and_an_aarch64_producer_with_their_stages(self):
        problems = self.check_tree(jobs=X86 + ARM + X86_STAGE + ARM_STAGE)
        self.assertEqual(problems, [])

    def test_rejects_a_stage_taking_the_other_architecture_s_artifact(self):
        """The silent one: the modules land, EXT_SUFFIX never matches, install compiles."""
        crossed = stage(
            "test-4-gpu-gb300",
            runner_config="4-gpu-gb300",
            rust_ext_artifact="${{ needs.build.outputs.artifact_name }}",
        )
        problems = self.check_tree(jobs=X86 + ARM + crossed)
        self.assert_one(
            problems=problems,
            fragment="runs on aarch64 (4-gpu-gb300) but takes its artifact",
        )

    def test_rejects_a_stage_on_an_architecture_with_no_producer(self):
        problems = self.check_tree(jobs=X86 + X86_STAGE + ARM_STAGE)
        self.assertEqual(len(problems), 2, problems)
        self.assertIn("not a producer in that file", problems[0])
        self.assertIn("stages run on ['aarch64'] with no producer", problems[1])

    def test_accepts_a_stage_that_opts_out_of_the_prebuild(self):
        """skip_prebuilt_rust_ext is what makes a producerless arch deliberate."""
        skipped = stage(
            "test-4-gpu-gb300",
            runner_config="4-gpu-gb300",
            skip_prebuilt_rust_ext="true",
        )
        self.assertEqual(self.check_tree(jobs=X86 + X86_STAGE + skipped), [])

    def test_expands_a_matrix_runner_config_into_one_caller_per_pool(self):
        matrixed = (
            "  test-matrix:\n"
            "    strategy:\n"
            "      matrix:\n"
            "        include:\n"
            "          - { runner_config: 1-gpu-large }\n"
            "          - { runner_config: 4-gpu-gb300 }\n"
            "    uses: ./.github/workflows/_pr-test-stage.yml\n"
            "    with:\n"
            "      runner_config: ${{ matrix.runner_config }}\n"
        )
        self.assert_one(
            problems=self.check_tree(jobs=X86 + matrixed),
            fragment="stages run on ['aarch64'] with no producer",
        )

    def test_rejects_a_runner_config_it_cannot_resolve(self):
        """A caller whose pool is unknown gets reported, never skipped."""
        opaque = stage("test-opaque", runner_config="${{ inputs.runner_config }}")
        self.assert_one(
            problems=self.check_tree(jobs=X86 + opaque),
            fragment="cannot resolve runner_config",
        )

    def test_rejects_an_aarch64_prefix_compiled_on_an_x86_runner(self):
        wrong = producer(
            "build-aarch64",
            runs_on="ubuntu-22.04",
            collect_runs_on="ubuntu-22.04-arm",
            cache_key_prefix="rust-ext-aarch64-cp310-cp312",
        )
        self.assert_one(
            problems=self.check_tree(jobs=wrong), fragment="runs_on is 'ubuntu-22.04'"
        )

    def test_rejects_an_aarch64_producer_left_on_the_collect_default(self):
        """collect_runs_on decides which objdump reads the module's GLIBC symbols."""
        wrong = producer(
            "build-aarch64",
            runs_on="ubuntu-22.04-arm",
            cache_key_prefix="rust-ext-aarch64-cp310-cp312",
        )
        self.assert_one(
            problems=self.check_tree(jobs=wrong),
            fragment="collect_runs_on is 'ubuntu-latest'",
        )

    def test_rejects_a_runner_label_with_no_architecture_on_record(self):
        """A label is not self-describing: 4-gpu-gb300 is Grace and says no arm."""
        unknown = producer(
            "build-aarch64",
            runs_on="gb300-build-node",
            collect_runs_on="ubuntu-22.04-arm",
            cache_key_prefix="rust-ext-aarch64-cp310-cp312",
        )
        self.assert_one(
            problems=self.check_tree(jobs=unknown),
            fragment="has no architecture on record",
        )

    def test_rejects_a_prefix_no_consumer_can_derive(self):
        wrong = producer(
            "build-mips",
            runs_on="ubuntu-22.04",
            cache_key_prefix="rust-ext-mips64-cp310-cp312",
        )
        self.assert_one(
            problems=self.check_tree(jobs=wrong), fragment="no consumer can derive"
        )

    def test_rejects_a_prefix_naming_interpreters_the_matrix_does_not_build(self):
        problems = self.check_tree(jobs=X86, versions="['3.10', '3.13']")
        self.assert_one(
            problems=problems, fragment="the compile matrix builds cp310-cp313"
        )

    def test_rejects_a_restore_key_that_drops_the_derived_prefix(self):
        """The derived literal can sit in the file while nothing interpolates it."""
        problems = self.check_tree(jobs=X86, key="rust-ext-x86_64-cp310-cp312")
        self.assert_one(
            problems=problems, fragment="does not interpolate that step's output"
        )

    def test_rejects_a_file_hashed_by_only_one_side(self):
        problems = self.check_tree(jobs=X86, action_hashed=f"{HASHED}, 'python/new.py'")
        self.assert_one(problems=problems, fragment="cache key inputs do not match")

    def test_reports_a_dead_check_when_no_producer_calls_the_build_workflow(self):
        self.assert_one(
            problems=self.check_tree(jobs=X86_STAGE), fragment="this check is dead"
        )

    def test_reports_an_unparseable_workflow_instead_of_raising(self):
        problems = self.check_tree(jobs=X86, extra_workflow="jobs:\n  x: [unclosed\n")
        self.assert_one(problems=problems, fragment="cannot read")


if __name__ == "__main__":
    unittest.main()
