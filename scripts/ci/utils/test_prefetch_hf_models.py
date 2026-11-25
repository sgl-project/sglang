"""Unit tests for prefetch_hf_models.py, stdlib only:

python -m unittest discover -s scripts/ci/utils -p 'test_prefetch_hf_models.py'
"""

import json
import os
import sys
import tempfile
import textwrap
import types
import unittest
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import prefetch_hf_models as phm  # noqa: E402

NO_OVERRIDES = {
    "by_file": {},
    "by_suite": {},
    "deny": [],
    "prefetch": {},
    "unused_by_file": {},
}


class _RepoTestCase(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = self._tmp.name

    def tearDown(self):
        self._tmp.cleanup()

    def write(self, rel, source):
        path = os.path.join(self.root, rel)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            f.write(textwrap.dedent(source))
        return rel

    def register(self, rel, est_time, suite="base-b-test-1-gpu-small", extra=""):
        stage, runner_config = suite.split("-test-")
        return self.write(
            rel,
            f"""
            from sglang.test.ci.ci_register import register_cuda_ci
            register_cuda_ci(est_time={est_time}, stage="{stage}", runner_config="{runner_config}"{extra})
            if __name__ == "__main__":
                pass
            """,
        )


class TestSelectPartitionFiles(_RepoTestCase):
    def test_matches_run_suite_selection_and_lpt(self):
        a = self.register("test/registered/a/test_a.py", 100)
        b = self.register("test/registered/a/test_b.py", 60)
        c = self.register("test/registered/b/test_c.py", 50)
        self.register(
            "test/registered/test_other.py", 500, suite="base-c-test-1-gpu-small"
        )
        self.register("test/registered/test_off.py", 500, extra=', disabled="flaky"')
        self.write("test/registered/conftest.py", "x = 1\n")

        def select(pid, model=None):
            return phm.select_partition_files(
                self.root, ["base-b-test-1-gpu-small"], pid, 2, model
            )

        self.assertEqual(select(0), [a])
        self.assertEqual(sorted(select(1)), sorted([b, c]))

        # Live estimates reorder LPT the same way run_suite.py's do.
        model = os.path.join(self.root, "model.json")
        with open(model, "w") as f:
            json.dump({"est": {"base-b-test-1-gpu-small": {c: 1000}}}, f)
        self.assertEqual(select(0, model), [c])

    def test_unparsable_file_is_skipped_with_a_warning(self):
        a = self.register("test/registered/test_a.py", 10)
        self.write("test/registered/test_bad.py", "def (:\n")
        with mock.patch.object(phm, "warn") as warn:
            files = phm.select_partition_files(
                self.root, ["base-b-test-1-gpu-small"], None, None, None
            )
        self.assertEqual(files, [a])
        warn.assert_called_once()


class TestResolveRepoIds(_RepoTestCase):
    def test_import_source_prefers_test_file_directory(self):
        local = self.write("test/registered/helper_mod.py", "X = 1\n")
        self.write("python/helper_mod.py", "X = 2\n")
        self.write("test/registered/test_a.py", "from helper_mod import X\n")
        path = os.path.join(self.root, "test/registered/test_a.py")
        source = phm.import_source(self.root, path, "helper_mod", 0)
        self.assertEqual(source, os.path.join(self.root, local))

    def setUp(self):
        super().setUp()
        self.write(
            "python/sglang/test/test_utils.py",
            """
            DEFAULT_SMALL = "org/small"
            DEFAULT_ALIAS = DEFAULT_SMALL
            DEFAULT_TIMEOUT = 600
            """,
        )
        # Same bare name as the test's own constant: must not leak into it.
        self.write("python/sglang/test/other_recipe.py", 'MODEL = "org/huge"\n')
        self.write(
            "python/sglang/test/kits/server_kit.py",
            """
            from sglang.test.test_utils import DEFAULT_ALIAS

            class ServerBase:
                model = DEFAULT_ALIAS

            class DraftBase(ServerBase):
                draft = "org/draft"

            def launch_default():
                return "org/helper"
            """,
        )
        self.write(
            "python/sglang/srt/models_list.py",
            'def pick():\n    return "org/product"\n',
        )

    def resolve(self, files, overrides=NO_OVERRIDES, suites=("s",)):
        return phm.resolve_repo_ids(self.root, files, [], list(suites), overrides)

    def test_names_resolve_through_their_imports(self):
        rel = self.write(
            "test/registered/test_x.py",
            """
            from sglang.srt.models_list import pick
            from sglang.test.kits.server_kit import DraftBase, launch_default
            from sglang.test.test_utils import DEFAULT_SMALL as SMALL

            MODEL = "org/local"
            URL = f"https://x/{MODEL}/a"

            class TestX(DraftBase):
                def test(self):
                    launch_default()
                    run(SMALL, MODEL, pick())
            """,
        )
        self.assertEqual(
            self.resolve([rel])[rel],
            ["org/draft", "org/helper", "org/local", "org/small"],
        )

    def test_helper_next_to_the_test_and_relative_imports(self):
        self.write("test/registered/lora/lora_cases.py", 'BASE = "org/lora-base"\n')
        rel = self.write(
            "test/registered/lora/test_lora.py",
            "from lora_cases import BASE\nfrom .lora_cases import BASE as B2\nuse(BASE, B2)\n",
        )
        self.assertEqual(self.resolve([rel])[rel], ["org/lora-base"])

    def test_overrides_and_filters(self):
        os.makedirs(os.path.join(self.root, "local", "dir"))
        rel = self.write(
            "test/registered/test_y.py",
            'use("org/denied", "local/dir", "bad--id/x", "org/kept", "org/unused")\n',
        )
        overrides = {
            "by_file": {rel: ["org/extra"]},
            "by_suite": {"s": ["org/suite-extra"]},
            "deny": ["org/denied"],
            "prefetch": {},
            "unused_by_file": {rel: ["org/unused"]},
        }
        self.assertEqual(
            self.resolve([rel], overrides),
            {rel: ["org/extra", "org/kept"], "suite:s": ["org/suite-extra"]},
        )


class TestDiffusion(_RepoTestCase):
    def test_plan_part_cases_and_standalone_files(self):
        self.write(
            "python/sglang/multimodal_gen/test/test_utils.py",
            'DEFAULT_IMAGE = "org/image"\n',
        )
        self.write(
            "python/sglang/multimodal_gen/test/server/gpu_cases.py",
            """
            from sglang.multimodal_gen.test.test_utils import DEFAULT_IMAGE

            def _make_case(case_id, model_path):
                return DiffusionTestCase(case_id, Args(model_path=model_path))

            ONE_GPU_CASES = [
                DiffusionTestCase("t2i", Args(model_path=DEFAULT_IMAGE)),
                _make_case("fp8", model_path="org/fp8"),
                DiffusionTestCase("other", Args(model_path="org/unused")),
            ]
            """,
        )
        standalone = self.write(
            "python/sglang/multimodal_gen/test/single_test_file/test_s.py",
            'use("org/standalone")\n',
        )
        plan = json.dumps(
            {
                "suite": "1-gpu",
                "partition_count": 2,
                "partitions": [
                    {"part": 0, "case_ids": ["other"]},
                    {
                        "part": 1,
                        "case_ids": ["t2i", "fp8"],
                        "standalone_files": ["../single_test_file/test_s.py"],
                    },
                ],
            }
        )
        part = phm.diffusion_plan_part(self.root, plan, 1)
        self.assertEqual(part["standalone_files"], [standalone])
        ids = phm.resolve_repo_ids(
            self.root, part["standalone_files"], part["case_ids"], [], NO_OVERRIDES
        )
        self.assertEqual(
            ids,
            {
                "case:t2i": ["org/image"],
                "case:fp8": ["org/fp8"],
                standalone: ["org/standalone"],
            },
        )

    def test_standalone_files_stay_inside_the_repo(self):
        plan = json.dumps(
            {
                "partitions": [
                    {
                        "part": 0,
                        "standalone_files": ["../../../../../../etc/x.py", "ok.py"],
                    }
                ]
            }
        )
        with mock.patch.object(phm, "warn") as warn:
            part = phm.diffusion_plan_part(self.root, plan, 0)
        self.assertEqual(
            part["standalone_files"],
            [
                os.path.join(
                    "python", "sglang", "multimodal_gen", "test", "server", "ok.py"
                )
            ],
        )
        self.assertEqual(warn.call_count, 1)


class TestFilesToDownload(unittest.TestCase):
    def test_default_skips_unloaded_formats_and_duplicate_weights(self):
        files = [
            "config.json",
            "model.safetensors",
            "pytorch_model.bin",
            "original/consolidated.00.pth",
            "model.gguf",
            "onnx/model.onnx",
            "text_encoder/pytorch_model.bin",
            "unet/diffusion_pytorch_model.safetensors",
            "unet/diffusion_pytorch_model.bin",
        ]
        self.assertEqual(
            phm.files_to_download(files, None),
            [
                "config.json",
                "model.safetensors",
                "text_encoder/pytorch_model.bin",
                "unet/diffusion_pytorch_model.safetensors",
            ],
        )

    def test_patterns_select_the_only_files(self):
        files = ["a-q4.gguf", "a-q8.gguf", "config.json"]
        self.assertEqual(phm.files_to_download(files, ["*-q4.gguf"]), ["a-q4.gguf"])


class TestIsHubRepoId(unittest.TestCase):
    def test_rejects_non_ids(self):
        for value in ("org/model", "Org-1/Model_2.5-it"):
            self.assertTrue(phm.is_hub_repo_id(value, "/nonexistent"), value)
        for value in ("org/a--b", "org/a..b", "../model", "org/" + "x" * 97, "o/r/x"):
            self.assertFalse(phm.is_hub_repo_id(value, "/nonexistent"), value)


class TestPrefetch(unittest.TestCase):
    def setUp(self):
        self.cache = tempfile.TemporaryDirectory()
        self.downloads = []
        infos = {
            ("org/public", "main"): dict(private=False, files=["config.json", "a.pth"]),
            ("org/secret", "main"): dict(private=True, files=["config.json"]),
            ("org/pinned", "test"): dict(private=False, files=["w.safetensors"]),
            ("org/unknown", "main"): dict(private=None, files=["config.json"]),
            ("org/nolist", "main"): dict(private=False, files=None),
        }

        class FakeApi:
            def model_info(self, repo_id, revision):
                info = infos.get((repo_id, revision))
                if info is None:
                    raise LookupError(repo_id)
                return types.SimpleNamespace(
                    private=info["private"],
                    siblings=(
                        None
                        if info["files"] is None
                        else [types.SimpleNamespace(rfilename=f) for f in info["files"]]
                    ),
                )

        def snapshot_download(repo_id, revision, allow_patterns, max_workers):
            self.downloads.append((repo_id, revision, allow_patterns))

        hub = types.ModuleType("huggingface_hub")
        hub.HfApi = FakeApi
        hub.snapshot_download = snapshot_download
        hub.constants = types.SimpleNamespace(HF_HUB_CACHE=self.cache.name)
        filelock = types.ModuleType("filelock")
        filelock.FileLock = lambda path, mode: mock.MagicMock()
        self.modules = mock.patch.dict(
            sys.modules, {"huggingface_hub": hub, "filelock": filelock}
        )
        self.modules.start()

    def tearDown(self):
        self.modules.stop()
        self.cache.cleanup()

    def test_skips_unconfirmed_public_missing_and_listed_repos(self):
        specs = {
            "org/pinned": {"revisions": ["test"]},
            "org/skipped": {"skip": "own token"},
        }
        with mock.patch.object(phm, "warn") as warn:
            phm.prefetch(
                [
                    "org/missing",
                    "org/nolist",
                    "org/pinned",
                    "org/public",
                    "org/secret",
                    "org/skipped",
                    "org/unknown",
                ],
                specs,
            )
        self.assertEqual(
            self.downloads,
            [
                ("org/pinned", "test", ["w.safetensors"]),
                ("org/public", "main", ["config.json"]),
            ],
        )
        self.assertEqual(warn.call_count, 4)


if __name__ == "__main__":
    unittest.main()
