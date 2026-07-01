"""Unit tests for list_stage_models extraction logic.

Pure-logic tests (stdlib only, no GPU, no sglang import) so they run in the
ci-model-inventory workflow without installing dependencies:

    python -m unittest discover -s scripts/ci -p 'test_list_stage_models.py'

Guards the recall/precision contract of the static model extractor: model-id
shape filtering, constant-table resolution (incl. tuple values), inline +
name-reference extraction, override merge semantics, and the suite->files +
inventory assembly that drives cache-warming.
"""

import os
import shutil
import sys
import tempfile
import unittest
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import list_stage_models as lsm  # noqa: E402

# Repo root inferred from this file's location: <root>/scripts/ci/<this>.
_REPO_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
_REAL_CI_REGISTER = os.path.join(
    _REPO_ROOT, "python", "sglang", "test", "ci", "ci_register.py"
)


def _write(root, rel, content):
    path = os.path.join(root, rel)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)


def _make_fake_repo(root, registered, helpers=None):
    """Build a temp repo: copy the real ci_register.py, write test + helper files.

    ``registered`` maps ``test/registered/...`` relpaths to file content;
    ``helpers`` maps ``python/sglang/test/...`` relpaths to content.
    """
    dst = os.path.join(root, "python", "sglang", "test", "ci", "ci_register.py")
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    shutil.copy(_REAL_CI_REGISTER, dst)
    for rel, content in registered.items():
        _write(root, os.path.join("test", "registered", rel), content)
    for rel, content in (helpers or {}).items():
        _write(root, os.path.join("python", "sglang", "test", rel), content)


class LooksLikeModelId(unittest.TestCase):
    def test_accepts_real_model_ids(self):
        for value in (
            "meta-llama/Llama-3.1-8B-Instruct",
            "RedHatAI/Llama-3.2-3B-quantized.w8a8",  # dotted suffix is real
            "cross-encoder/ms-marco-MiniLM-L6-v2",
            "nvidia/DeepSeek-V3-0324-FP4",
            "lmsys/sglang-ci-dsv3-test",
            "Qwen/Qwen2-1.5B-Instruct-GGUF",  # -GGUF suffix, not .gguf extension
        ):
            self.assertTrue(lsm.looks_like_model_id(value), value)

    def test_rejects_non_models(self):
        for value in (
            "text/plain",  # MIME
            "application/json",  # MIME
            "image/png",  # MIME
            "Text/Plain",  # MIME, case-insensitive
            "2/3",  # numeric ratio, no letters
            "N/A",  # single-char sides
            "configs/model.json",  # path with file extension
            "org/weights.safetensors",  # weight-file extension
            "org/model.bin",  # weight-file extension
            "a/b/c",  # too many slashes
            "/abs/path",  # leading slash
            "./relative",  # leading dot
            "just-a-string",  # no slash
        ):
            self.assertFalse(lsm.looks_like_model_id(value), value)

    def test_deny_set_overrides(self):
        value = "org/looks-like-a-model"
        self.assertTrue(lsm.looks_like_model_id(value))
        self.assertFalse(lsm.looks_like_model_id(value, deny={value}))


class ConstantTable(unittest.TestCase):
    def test_single_and_tuple_values(self):
        source = (
            'DEFAULT_MODEL = "meta-llama/Llama-3.1-8B-Instruct"\n'
            'PAIR = ("OPEA/Qwen2.5-0.5B-int4", "Intel/Qwen2-0.5B-int4")\n'
            'NOT_A_MODEL = "https://example.com/x"\n'
            "PORT = 30000\n"
        )
        table = lsm.extract_constants_from_source(source)
        self.assertEqual(table["DEFAULT_MODEL"], {"meta-llama/Llama-3.1-8B-Instruct"})
        self.assertEqual(
            table["PAIR"], {"OPEA/Qwen2.5-0.5B-int4", "Intel/Qwen2-0.5B-int4"}
        )
        self.assertNotIn("NOT_A_MODEL", table)
        self.assertNotIn("PORT", table)

    def test_annotated_assignment(self):
        source = 'DEFAULT: str = "meta-llama/Llama-3.1-8B-Instruct"\n'
        table = lsm.extract_constants_from_source(source)
        self.assertEqual(table["DEFAULT"], {"meta-llama/Llama-3.1-8B-Instruct"})

    def test_implicit_concatenation(self):
        # Python folds adjacent string literals at parse time into one Constant.
        source = 'M = (\n    "meta-llama/"\n    "Llama-3.1-8B-Instruct"\n)\n'
        table = lsm.extract_constants_from_source(source)
        self.assertEqual(table["M"], {"meta-llama/Llama-3.1-8B-Instruct"})

    def test_deny_excludes_from_constant_table(self):
        source = 'BAD = "weird/thing"\nGOOD = "meta-llama/Llama-3.1-8B-Instruct"\n'
        table = lsm.extract_constants_from_source(source, deny={"weird/thing"})
        self.assertNotIn("BAD", table)
        self.assertIn("GOOD", table)


class ExtractModels(unittest.TestCase):
    def test_inline_and_name_reference(self):
        const_table = {
            "DEFAULT_MODEL_NAME_FOR_TEST": {"meta-llama/Llama-3.1-8B-Instruct"}
        }
        source = (
            "from sglang.test.test_utils import DEFAULT_MODEL_NAME_FOR_TEST\n"
            "class T:\n"
            "    model = DEFAULT_MODEL_NAME_FOR_TEST\n"
            '    draft = "lmsys/sglang-EAGLE3-LLaMA3.1-Instruct-8B"\n'
        )
        models = lsm.extract_models_from_source(source, const_table)
        self.assertEqual(
            models,
            {
                "meta-llama/Llama-3.1-8B-Instruct",
                "lmsys/sglang-EAGLE3-LLaMA3.1-Instruct-8B",
            },
        )

    def test_fstring_fragments_are_skipped(self):
        # An f-string with a placeholder yields a truncated, non-existent id;
        # it must NOT be picked up.
        source = 'msg = f"meta-llama/Llama-3.1-8B-Instruct-{suffix}"\n'
        self.assertEqual(lsm.extract_models_from_source(source, {}), set())

    def test_fstring_name_placeholder_still_resolves(self):
        # The literal fragment is skipped, but a constant referenced inside the
        # f-string placeholder is still resolved.
        const_table = {"DRAFT": {"lmsys/sglang-EAGLE-llama2-chat-7B"}}
        source = 'path = f"prefix/{DRAFT}/topk"\n'
        self.assertEqual(
            lsm.extract_models_from_source(source, const_table),
            {"lmsys/sglang-EAGLE-llama2-chat-7B"},
        )

    def test_model_less_file_yields_nothing(self):
        source = (
            "import torch\n"
            "def test_kernel():\n"
            "    assert torch.cuda.is_available() or True\n"
        )
        self.assertEqual(lsm.extract_models_from_source(source, {}), set())

    def test_deny_removes_name_resolved_model(self):
        # deny must apply to constant-resolved ids too, not just inline literals.
        const_table = {"M": {"weird/thing"}}
        source = "x = M\n"
        self.assertIn(
            "weird/thing", lsm.extract_models_from_source(source, const_table)
        )
        self.assertEqual(
            lsm.extract_models_from_source(source, const_table, deny={"weird/thing"}),
            set(),
        )


class Overrides(unittest.TestCase):
    def test_load_defaults_when_missing(self):
        ov = lsm.load_overrides(None)
        self.assertEqual(ov, {"by_file": {}, "by_suite": {}, "deny": []})

    def test_null_values_fall_back_to_defaults(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "ov.json")
            with open(path, "w", encoding="utf-8") as f:
                f.write('{"deny": null, "by_file": null}')
            ov = lsm.load_overrides(path)
            self.assertEqual(ov, {"by_file": {}, "by_suite": {}, "deny": []})


class CollectSuiteFiles(unittest.TestCase):
    REG = (
        "import unittest\n"
        "from sglang.test.ci.ci_register import register_cuda_ci, register_amd_ci\n"
        "{calls}\n"
        'MODEL = "{model}"\n'
        'if __name__ == "__main__":\n    unittest.main()\n'
    )

    def _repo(self, tmp):
        _make_fake_repo(
            tmp,
            registered={
                # enabled CUDA, two suites in one file -> dedupe per suite
                "a/test_a.py": self.REG.format(
                    calls=(
                        'register_cuda_ci(est_time=1, stage="base-x", '
                        'runner_config="1-gpu")\n'
                        'register_cuda_ci(est_time=1, suite="nightly-y", '
                        "nightly=True)"
                    ),
                    model="meta-llama/Llama-3.1-8B-Instruct",
                ),
                # disabled CUDA -> excluded unless include_disabled
                "b/test_b.py": self.REG.format(
                    calls=(
                        'register_cuda_ci(est_time=1, stage="base-z", '
                        'runner_config="1-gpu", disabled="flaky")'
                    ),
                    model="Qwen/Qwen3-8B",
                ),
                # AMD only -> never in CUDA inventory
                "c/test_c.py": self.REG.format(
                    calls='register_amd_ci(est_time=1, suite="nightly-amd")',
                    model="google/gemma-3-4b-it",
                ),
            },
        )

    def test_enabled_only_by_default(self):
        with tempfile.TemporaryDirectory() as tmp:
            self._repo(tmp)
            suites, nightly, errors = lsm.collect_suite_files(tmp, "cuda")
            self.assertEqual(set(suites), {"base-x-test-1-gpu", "nightly-y"})
            self.assertEqual(errors, {})
            self.assertTrue(nightly["nightly-y"])
            self.assertFalse(nightly["base-x-test-1-gpu"])
            # AMD suite never appears for the CUDA backend.
            self.assertNotIn("nightly-amd", suites)

    def test_include_disabled(self):
        with tempfile.TemporaryDirectory() as tmp:
            self._repo(tmp)
            suites, _, _ = lsm.collect_suite_files(tmp, "cuda", include_disabled=True)
            self.assertIn("base-z-test-1-gpu", suites)

    def test_unparsable_registry_is_surfaced(self):
        with tempfile.TemporaryDirectory() as tmp:
            _make_fake_repo(
                tmp,
                registered={
                    # est_time missing -> RegistryVisitor raises ValueError
                    "d/test_bad.py": (
                        "from sglang.test.ci.ci_register import register_cuda_ci\n"
                        'register_cuda_ci(suite="base-x")\n'
                    ),
                },
            )
            suites, _, errors = lsm.collect_suite_files(tmp, "cuda")
            self.assertEqual(suites, {})
            self.assertIn("test/registered/d/test_bad.py", errors)


class BuildInventory(unittest.TestCase):
    def _repo(self, tmp):
        _make_fake_repo(
            tmp,
            registered={
                # resolves a model via an imported constant
                "a/test_a.py": (
                    "import unittest\n"
                    "from sglang.test.ci.ci_register import register_cuda_ci\n"
                    "from sglang.test.test_utils import DEFAULT_MODEL\n"
                    'register_cuda_ci(est_time=1, stage="base-x", '
                    'runner_config="1-gpu")\n'
                    "MODEL = DEFAULT_MODEL\n"
                    'if __name__ == "__main__":\n    unittest.main()\n'
                ),
                # model-less -> lands in unresolved_files (same suite as a)
                "b/test_b.py": (
                    "import unittest\n"
                    "from sglang.test.ci.ci_register import register_cuda_ci\n"
                    'register_cuda_ci(est_time=1, stage="base-x", '
                    'runner_config="1-gpu")\n'
                    'if __name__ == "__main__":\n    unittest.main()\n'
                ),
            },
            helpers={"test_utils.py": 'DEFAULT_MODEL = "meta-llama/Llama-3.1-8B"\n'},
        )

    def test_resolution_and_unresolved(self):
        with tempfile.TemporaryDirectory() as tmp:
            self._repo(tmp)
            inv = lsm.build_inventory(tmp, "cuda", lsm.load_overrides(None), "sha123")
            self.assertEqual(inv["generated_at_commit"], "sha123")
            suite = inv["suites"]["base-x-test-1-gpu"]
            self.assertEqual(suite["models"], ["meta-llama/Llama-3.1-8B"])
            self.assertEqual(suite["test_file_count"], 2)
            self.assertEqual(suite["unresolved_files"], ["test/registered/b/test_b.py"])
            self.assertEqual(inv["model_count"], 1)
            self.assertEqual(inv["parse_failures"], {})

    def test_by_file_override_adds_and_clears_unresolved(self):
        with tempfile.TemporaryDirectory() as tmp:
            self._repo(tmp)
            overrides = {
                "by_file": {"test/registered/b/test_b.py": ["org/extra-model"]},
                "by_suite": {},
                "deny": [],
            }
            inv = lsm.build_inventory(tmp, "cuda", overrides, "sha")
            suite = inv["suites"]["base-x-test-1-gpu"]
            self.assertIn("org/extra-model", suite["models"])
            # by_file supplied a model for test_b -> no longer unresolved.
            self.assertEqual(suite["unresolved_files"], [])

    def test_by_suite_override_adds_model(self):
        with tempfile.TemporaryDirectory() as tmp:
            self._repo(tmp)
            overrides = {
                "by_file": {},
                "by_suite": {"base-x-test-1-gpu": ["org/suite-model"]},
                "deny": [],
            }
            inv = lsm.build_inventory(tmp, "cuda", overrides, "sha")
            self.assertIn(
                "org/suite-model", inv["suites"]["base-x-test-1-gpu"]["models"]
            )


class RenderMarkdown(unittest.TestCase):
    def test_table_and_glyphs(self):
        inv = {
            "backend": "cuda",
            "generated_at_commit": "sha",
            "suite_count": 2,
            "model_count": 1,
            "parse_failures": {},
            "suites": {
                "n-suite": {
                    "nightly": True,
                    "models": ["org/model"],
                    "test_file_count": 1,
                    "unresolved_files": [],
                },
                "empty": {
                    "nightly": False,
                    "models": [],
                    "test_file_count": 2,
                    "unresolved_files": ["x.py", "y.py"],
                },
            },
        }
        md = lsm.render_markdown(inv)
        self.assertIn("| `n-suite` | ✓ | org/model | 0 |", md)
        self.assertIn("| `empty` |  | _(none)_ | 2 |", md)

    def test_parse_failures_line(self):
        inv = {
            "backend": "cuda",
            "generated_at_commit": "sha",
            "suite_count": 0,
            "model_count": 0,
            "parse_failures": {"x.py": "SyntaxError: bad"},
            "suites": {},
        }
        self.assertIn("Unparsable files", lsm.render_markdown(inv))


class ResolveCommit(unittest.TestCase):
    def test_explicit_arg_wins(self):
        self.assertEqual(lsm.resolve_commit("abc", "/nonexistent"), "abc")

    def test_env_fallback(self):
        with mock.patch.dict(os.environ, {"GITHUB_SHA": "deadbeef"}, clear=True):
            self.assertEqual(lsm.resolve_commit(None, "/nonexistent"), "deadbeef")

    def test_unknown_when_no_git(self):
        with tempfile.TemporaryDirectory() as tmp, mock.patch.dict(
            os.environ, {}, clear=True
        ):
            self.assertEqual(lsm.resolve_commit(None, tmp), "unknown")


if __name__ == "__main__":
    unittest.main()
