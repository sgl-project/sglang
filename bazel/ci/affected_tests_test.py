from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import affected_tests


class FakeQuery:
    def __init__(self, responses):
        self.responses = responses
        self.calls = []

    def labels(self, expression, *, sky=False, missing_label=None):
        self.calls.append((expression, sky))
        for predicate, labels in self.responses:
            if predicate(expression):
                return sorted(labels)
        raise AssertionError(f"Unexpected query: {expression}")


class InvocationDirectoryTest(unittest.TestCase):
    def test_prefers_bazel_workspace_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            with mock.patch.dict(
                os.environ,
                {"BUILD_WORKSPACE_DIRECTORY": directory},
            ):
                self.assertEqual(
                    affected_tests.invocation_directory(),
                    Path(directory),
                )


class ParseNameStatusTest(unittest.TestCase):
    def test_parses_nul_delimited_changes(self):
        self.assertEqual(
            affected_tests.parse_name_status(
                b"M\0python/current.py\0D\0python/deleted.py\0"
            ),
            [
                affected_tests.Change("M", "python/current.py"),
                affected_tests.Change("D", "python/deleted.py"),
            ],
        )

    def test_rejects_incomplete_output(self):
        with self.assertRaisesRegex(ValueError, "Unexpected git"):
            affected_tests.parse_name_status(b"M\0")


class LabelMappingTest(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.repo = Path(self.temp_dir.name)
        (self.repo / "BUILD.bazel").write_text("", encoding="utf-8")
        (self.repo / "pkg" / "nested").mkdir(parents=True)
        (self.repo / "pkg" / "BUILD").write_text("", encoding="utf-8")
        (self.repo / "pkg" / "nested" / "source.py").write_text("", encoding="utf-8")

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_uses_deepest_package(self):
        self.assertEqual(
            affected_tests.source_and_build_labels(self.repo, "pkg/nested/source.py"),
            ("//pkg:nested/source.py", "//pkg:BUILD"),
        )

    def test_uses_root_package(self):
        (self.repo / "README.md").write_text("", encoding="utf-8")
        self.assertEqual(
            affected_tests.source_and_build_labels(self.repo, "README.md"),
            ("//:README.md", "//:BUILD.bazel"),
        )

    def test_rejects_path_escape(self):
        with self.assertRaisesRegex(ValueError, "escapes"):
            affected_tests.source_and_build_labels(self.repo, "../outside")


class GlobalChangeTest(unittest.TestCase):
    def test_global_configuration(self):
        self.assertEqual(
            affected_tests.global_reason(affected_tests.Change("M", "MODULE.bazel")),
            "global Bazel or CI configuration",
        )

    def test_dependency_locks(self):
        for path in (
            "rust/Cargo.lock",
            "bazel/python/requirements.lock.txt",
            "uv.lock",
        ):
            with self.subTest(path=path):
                self.assertEqual(
                    affected_tests.global_reason(affected_tests.Change("M", path)),
                    "dependency lock",
                )

    def test_deleted_build_file_is_global(self):
        self.assertEqual(
            affected_tests.global_reason(
                affected_tests.Change("D", "old/package/BUILD.bazel")
            ),
            "deleted package definition",
        )

    def test_deleted_starlark_file_is_global(self):
        self.assertEqual(
            affected_tests.global_reason(
                affected_tests.Change("D", "bazel/rules/old_rule.bzl")
            ),
            "deleted Starlark definition",
        )

    def test_ordinary_build_file_is_package_scoped(self):
        self.assertIsNone(
            affected_tests.global_reason(affected_tests.Change("M", "pkg/BUILD.bazel"))
        )


class SelectionTest(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.repo = Path(self.temp_dir.name)
        (self.repo / "BUILD.bazel").write_text("", encoding="utf-8")
        (self.repo / "pkg").mkdir()
        (self.repo / "pkg" / "BUILD.bazel").write_text("", encoding="utf-8")
        (self.repo / "pkg" / "owned.py").write_text("", encoding="utf-8")
        (self.repo / "pkg" / "uncovered.py").write_text("", encoding="utf-8")

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_maps_owners_selects_tests_and_reports_uncovered(self):
        query = FakeQuery(
            [
                (
                    lambda expression: expression.startswith('kind(".* rule"')
                    and "owned.py" in expression,
                    ["//pkg:library"],
                ),
                (
                    lambda expression: expression.startswith('kind(".* rule"')
                    and "uncovered.py" in expression,
                    [],
                ),
                (
                    lambda expression: 'kind(".*_test rule", rdeps' in expression,
                    [
                        "//tests:cpu_test",
                        "//tests:cuda_test",
                        "//tests:rocm_test",
                    ],
                ),
                (
                    lambda expression: expression.startswith(
                        'attr(tags, "manual", set'
                    ),
                    ["//tests:cuda_test"],
                ),
                (
                    lambda expression: "requires-cuda" in expression,
                    ["//tests:cuda_test"],
                ),
                (
                    lambda expression: "requires-rocm" in expression,
                    ["//tests:rocm_test"],
                ),
            ]
        )

        result = affected_tests.select(
            self.repo,
            [
                affected_tests.Change("M", "pkg/owned.py"),
                affected_tests.Change("A", "pkg/uncovered.py"),
            ],
            query,
            base="base",
            head="head",
        )

        self.assertFalse(result["global_change"])
        self.assertEqual(result["uncovered_files"], ["pkg/uncovered.py"])
        self.assertEqual(
            result["classification"],
            {
                "cpu": ["//tests:cpu_test"],
                "cuda": ["//tests:cuda_test"],
                "rocm": ["//tests:rocm_test"],
                "manual": ["//tests:cuda_test"],
            },
        )
        test_queries = [
            expression
            for expression, _ in query.calls
            if 'kind(".*_test rule", rdeps' in expression
        ]
        self.assertEqual(len(test_queries), 1)
        self.assertIn("//pkg:owned.py", test_queries[0])
        self.assertNotIn("//pkg:uncovered.py", test_queries[0])
        self.assertIn('tests(kind("test_suite rule"', test_queries[0])

    def test_starlark_change_uses_sky_query_load_edges(self):
        (self.repo / "pkg" / "defs.bzl").write_text("", encoding="utf-8")
        query = FakeQuery(
            [
                (
                    lambda expression: expression.startswith('kind(".* rule"'),
                    ["//consumer:target"],
                ),
                (
                    lambda expression: 'kind(".*_test rule", rdeps' in expression,
                    ["//consumer:target_test"],
                ),
                (
                    lambda expression: expression.startswith(
                        'attr(tags, "manual", set'
                    ),
                    [],
                ),
                (lambda expression: "requires-cuda" in expression, []),
                (lambda expression: "requires-rocm" in expression, []),
            ]
        )

        result = affected_tests.select(
            self.repo,
            [affected_tests.Change("M", "pkg/defs.bzl")],
            query,
            base="base",
            head="head",
        )

        self.assertEqual(result["changed_files"][0]["mapping"], "starlark-load")
        self.assertTrue(
            all(sky for expression, sky in query.calls if "rbuildfiles" in expression)
        )
        self.assertTrue(
            all(
                'set("//pkg:defs.bzl")' not in expression
                for expression, _ in query.calls
                if "rbuildfiles" in expression
            )
        )

    def test_global_change_selects_every_test(self):
        query = FakeQuery(
            [
                (
                    lambda expression: expression == "tests(//...)",
                    ["//tests:cpu_test", "//tests:cuda_test"],
                ),
                (
                    lambda expression: expression.startswith(
                        'attr(tags, "manual", set'
                    ),
                    ["//tests:cuda_test"],
                ),
                (
                    lambda expression: "requires-cuda" in expression,
                    ["//tests:cuda_test"],
                ),
                (lambda expression: "requires-rocm" in expression, []),
            ]
        )

        result = affected_tests.select(
            self.repo,
            [affected_tests.Change("M", "MODULE.bazel.lock")],
            query,
            base="base",
            head="head",
        )

        self.assertTrue(result["global_change"])
        self.assertEqual(result["uncovered_files"], [])
        self.assertEqual(result["classification"]["cpu"], ["//tests:cpu_test"])


class MarkdownTest(unittest.TestCase):
    def test_summary_highlights_uncovered_files(self):
        markdown = affected_tests.render_markdown(
            {
                "base": "a",
                "head": "b",
                "global_change": False,
                "changed_files": [
                    {
                        "path": "not-owned.txt",
                        "mapping": "uncovered",
                        "owners": [],
                    }
                ],
                "classification": {
                    "cpu": [],
                    "cuda": [],
                    "rocm": [],
                    "manual": [],
                },
                "uncovered_files": ["not-owned.txt"],
            }
        )
        self.assertIn("### Uncovered files", markdown)
        self.assertIn("`not-owned.txt`", markdown)
        self.assertIn("| CPU | 0 |", markdown)


if __name__ == "__main__":
    unittest.main()
