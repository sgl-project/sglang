"""
Test Selector - Precision test selector based on coverage data (line, function, file granularity)

Workflow:
1. Build 'test case -> covered lines' mapping from coverage SQLite data
2. Parse code changes (supports GitHub PR or local file hash comparison)
3. Select affected test cases (by line, function, file granularity)
"""

import argparse
import ast
import base64
import hashlib
import json
import os
import sqlite3
import ssl
import subprocess
import tempfile
import textwrap
import time
import urllib.error
import urllib.parse
import urllib.request
from collections import defaultdict
from pathlib import Path

import regex as re

# ==================== Configuration ====================
BASE_DIR = Path(__file__).resolve().parent

# Repository name: used for filtering and path normalization
REPO_NAME = "sglang"

# Product code path prefix in coverage data / diff paths.
# sglang: coverage path is /__w/sglang/sglang/python/sglang/xxx.py, diff path is python/sglang/xxx.py
#   -> PRODUCT_PREFIX = "python/sglang/"
# After stripping this prefix, both sides produce the same relative path (e.g. srt/models/qwen3_vl.py)
PRODUCT_PREFIX = "python/sglang/"


# Directory prefix of test case folders under coverage data dir.
TEST_CASE_DIR_PREFIX = "____w__sglang__sglang__test__"

# Coverage density threshold: proportion of changed lines covered
# Range: 0.0 ~ 1.0, higher value = stricter filtering
# Example: 0.05 means at least 5% of changed lines must be covered
# Recommendation: start at 0.05, increase to 0.10/0.15/0.20 if too many results
COVERAGE_DENSITY_THRESHOLD = 0.0

# Minimum affected lines threshold
MIN_AFFECTED_LINES = 1


def _get_test_files_from_pr_diff(diff_file: str) -> list[str]:
    """
    Extract new/modified test files from PR diff.
    Test files must be in tests/ directory and start with test_

    Args:
        diff_file: Path to the PR diff file

    Returns:
        List of test case names that correspond to new/modified test files
    """
    test_files_found = []

    try:
        with open(diff_file, encoding="utf-8-sig") as f:
            diff_content = f.read()
    except Exception as e:
        print(f"  Warning: Failed to read diff file for test file detection: {e}")
        return test_files_found

    # Pattern to match test file paths: test/registered/ directory
    # In diff output:
    #   - +++ b/test/registered/unit/xxx/test_xxx.py (new/modified test file)
    #   - rename to test/registered/unit/xxx/test_xxx.py (renamed test file)
    # Test files must be in test/ directory and start with test_
    test_file_pattern = re.compile(
        r"^(?:\+\+\+ [ab]/|rename to )((?:test/registered(?:/.+)?/test_\w+\.py|test/(?:unit|e2e|integration)(?:/.+)?/test_\w+\.py))",
        re.MULTILINE,
    )

    changed_test_files = set()
    for match in test_file_pattern.finditer(diff_content):
        test_file_path = match.group(1)
        changed_test_files.add(test_file_path)

    if not changed_test_files:
        return test_files_found

    print(
        f"  Found {len(changed_test_files)} changed test file(s): {changed_test_files}"
    )

    # Add all changed test files directly to recommended list (no matching with test_case_map)
    for changed_file in changed_test_files:
        if changed_file not in test_files_found:
            test_files_found.append(changed_file)

    return test_files_found


def _get_deleted_test_files_from_pr(diff_file: str, test_case_map: dict) -> list[str]:
    """
    Extract deleted test files from PR diff.
    Test files are in test/registered/ directory with test_*.py pattern.

    Args:
        diff_file: Path to the PR diff file
        test_case_map: Mapping of test case names to their coverage info

    Returns:
        List of test case names that correspond to deleted test files
    """
    deleted_test_files = []

    try:
        with open(diff_file, encoding="utf-8-sig") as f:
            diff_content = f.read()
    except Exception as e:
        print(f"  Warning: Failed to read diff file for deleted test detection: {e}")
        return deleted_test_files

    # Pattern to match deleted test files (test/registered/ directory)
    # Match --- a/test/... followed by +++ /dev/null (deleted file marker)
    deleted_pattern = re.compile(
        r"^--- a/(test/registered(?:/.+)?/test_\w+\.py|test/(?:unit|e2e|integration)(?:/.+)?/test_\w+\.py)\s*\n\s*\+\+\+ [ab]?/dev/null",
        re.MULTILINE,
    )

    for match in deleted_pattern.finditer(diff_content):
        test_file_path = match.group(1)
        deleted_test_files.append(test_file_path)

    if deleted_test_files:
        print(
            f"  Found {len(deleted_test_files)} deleted test file(s): {deleted_test_files}"
        )

    return deleted_test_files


class CoverageSelector:
    """Coverage-based test selector"""

    def __init__(
        self, coverage_data_dir: str | None = None, source_dir: str | None = None
    ):
        """
        Args:
            coverage_data_dir: Coverage data directory (only needed for building map)
            source_dir: Source code directory (only needed for function-level matching)
        """
        self.coverage_data_dir = Path(coverage_data_dir) if coverage_data_dir else None
        self.source_dir = Path(source_dir) if source_dir else None
        self.test_case_map = {}  # test_case_name -> {files: {filepath: {lines}}}
        self._noise_lines_cache = {}  # filepath -> set of noise lines (import + def)

    def scan_test_cases(self) -> list[str]:
        """
        Scan all test case directories.
        """
        test_cases = []
        if not self.coverage_data_dir or not self.coverage_data_dir.exists():
            print(
                f"  Warning: Coverage data directory not found: {self.coverage_data_dir}"
            )
            return test_cases
        for item in self.coverage_data_dir.iterdir():
            if not item.is_dir():
                continue
            name = item.name
            # sglang naming: ____w__sglang__sglang__test__... (GitHub Actions encoded path)
            is_sglang_layout = TEST_CASE_DIR_PREFIX and name.startswith(
                TEST_CASE_DIR_PREFIX
            )
            if not is_sglang_layout:
                continue
            # coverage.* files directly under test case dir (sglang)
            has_cov_files = any(item.glob("coverage.*"))
            if has_cov_files:
                test_cases.append(name)
        return sorted(test_cases)

    @staticmethod
    def normalize_test_name(test_name: str) -> str:
        """
        Convert test case directory name to standard script name format.
        sglang (GitHub Actions encoded dir name: /__w/sglang/sglang/test/... -> ____w__sglang__sglang__test__...):
        - ____w__sglang__sglang__test__registered__npu__xxx__test_foo.py
          -> test/registered/npu/xxx/test_foo.py (file-level)
        - ...--test_foo -> test/registered/npu/xxx/test_foo.py::test_foo (function-level)
        """
        # sglang layout: strip ____w__sglang__sglang__test__ prefix (encoded /__w/sglang/sglang/test/)
        if not TEST_CASE_DIR_PREFIX or not test_name.startswith(TEST_CASE_DIR_PREFIX):
            return test_name
        # rest: encoded path after /test/ (e.g. registered__npu__xxx__test_foo.py)
        rest = test_name[len(TEST_CASE_DIR_PREFIX) :]
        # Restore test/ prefix (TEST_CASE_DIR_PREFIX ends with test__, __ encodes /)
        result = "test/" + rest.replace("__", "/")
        # Handle function-level marker: .../test_foo.py--test_bar -> .../test_foo.py::test_bar
        result = result.replace("--", "::")
        # File-level tests need .py suffix; avoid double .py when name already ends with .py
        if "::" not in result and not result.endswith(".py"):
            result = result + ".py"
        return result

    def get_covered_lines_from_file(self, cov_file: str, filename: str) -> set[int]:
        """
        Get covered line numbers for a file from a single coverage SQLite file
        """
        lines = set()
        try:
            conn = sqlite3.connect(cov_file)
            cursor = conn.cursor()

            # Find file ID (fuzzy path matching)
            cursor.execute("SELECT id FROM file WHERE path LIKE ?", (f"%{filename}",))
            row = cursor.fetchone()
            if not row:
                conn.close()
                return lines
            file_id = row[0]

            # Get all arcs, calculate covered line numbers
            cursor.execute(
                "SELECT DISTINCT fromno, tono FROM arc WHERE file_id = ?", (file_id,)
            )
            for fromno, tono in cursor.fetchall():
                if fromno > 0:
                    lines.add(fromno)
                if tono > 0:
                    lines.add(tono)

            conn.close()
        except Exception as e:
            print(f"  Warning: Error reading {cov_file}: {e}")
        return lines

    def get_covered_files_from_file(self, cov_file: str) -> set[str]:
        """Get all covered files from a single coverage file"""
        files = set()
        try:
            conn = sqlite3.connect(cov_file)
            cursor = conn.cursor()
            cursor.execute("SELECT path FROM file")
            for (path,) in cursor.fetchall():
                # Product code paths contain PRODUCT_PREFIX
                # (e.g. /__w/sglang/sglang/python/sglang/srt/xxx.py -> srt/xxx.py)
                if PRODUCT_PREFIX in path:
                    rel_path = path.split(PRODUCT_PREFIX)[-1]
                    files.add(rel_path)
            conn.close()
        except Exception as e:
            print(f"  Warning: Error reading {cov_file}: {e}")
        return files

    def _get_function_def_lines(self, filepath: str) -> set[int]:
        """
        Get function definition line numbers (def line only, not function body).

        Args:
            filepath: Source file path

        Returns:
            Set of line numbers where function definitions occur
        """
        def_lines = set()
        try:
            with open(filepath, encoding="utf-8") as f:
                source = f.read()
                lines = source.splitlines()

            tree = ast.parse(source, filename=filepath)

            TARGET_DECORATORS = {"staticmethod", "classmethod", "property"}
            for node in ast.walk(tree):
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    # Add decorator lines (only @staticmethod, @classmethod, @property)
                    for decorator in node.decorator_list:
                        if (
                            isinstance(decorator, ast.Name)
                            and decorator.id in TARGET_DECORATORS
                        ):
                            if hasattr(decorator, "lineno") and decorator.lineno:
                                def_lines.add(decorator.lineno)
                                # Handle multi-line decorator expressions
                                if (
                                    hasattr(decorator, "end_lineno")
                                    and decorator.end_lineno
                                ):
                                    for i in range(
                                        decorator.lineno, decorator.end_lineno + 1
                                    ):
                                        def_lines.add(i)

                    def_lines.add(node.lineno)

                    # Bracket counting to find header end
                    start_idx = node.lineno - 1
                    paren_count = lines[start_idx].count("(") - lines[start_idx].count(
                        ")"
                    )

                    line_idx = start_idx
                    while paren_count > 0 and line_idx < len(lines):
                        line_idx += 1
                        paren_count += lines[line_idx].count("(") - lines[
                            line_idx
                        ].count(")")

                    header_end = line_idx + 1  # Convert to 1-indexed

                    # Extend to return type annotation if present
                    if node.returns:
                        header_end = max(header_end, node.returns.end_lineno)

                    # Record all lines from def to header end
                    for i in range(node.lineno, header_end + 1):
                        def_lines.add(i)
        except Exception:
            pass
        return def_lines

    def _get_class_def_lines(self, filepath: str) -> set[int]:
        """
        Get line numbers of all class definition lines.

        Args:
            filepath: Source file path

        Returns:
            Set of line numbers where class definitions occur
        """
        class_lines = set()
        try:
            with open(filepath, encoding="utf-8") as f:
                tree = ast.parse(f.read(), filename=filepath)

            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef):
                    class_lines.add(node.lineno)
        except Exception:
            pass
        return class_lines

    def _get_docstring_lines(self, filepath: str) -> set[int]:
        """
        Get line numbers of all docstring lines (module, class, and function).

        Docstrings are string literals that appear as the first statement
        in a module, class, or function body.

        Args:
            filepath: Source file path

        Returns:
            Set of line numbers where docstrings occur
        """
        docstring_lines = set()
        try:
            with open(filepath, encoding="utf-8") as f:
                tree = ast.parse(f.read(), filename=filepath)

            # Module-level docstring
            if (
                tree.body
                and isinstance(tree.body[0], ast.Expr)
                and isinstance(tree.body[0].value, ast.Constant)
            ):
                docstring_lines.add(tree.body[0].lineno)

            # Class and function docstrings
            for node in ast.walk(tree):
                if isinstance(
                    node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)
                ):
                    if (
                        node.body
                        and isinstance(node.body[0], ast.Expr)
                        and isinstance(node.body[0].value, ast.Constant)
                    ):
                        docstring_lines.add(node.body[0].lineno)
        except Exception:
            pass
        return docstring_lines

    @staticmethod
    def _get_blank_lines(filepath: str) -> set[int]:
        """
        Get line numbers of all blank/whitespace-only lines in file.

        Coverage arc data can record blank lines as control-flow nodes
        (e.g., block boundaries after if/return statements). These lines
        are not executable and must be filtered out to avoid false matches.

        Args:
            filepath: Source file path

        Returns:
            Set of line numbers that are blank or whitespace-only
        """
        blank_lines = set()
        try:
            with open(filepath, encoding="utf-8") as f:
                for line_no, line in enumerate(f, start=1):
                    if not line.strip():
                        blank_lines.add(line_no)
        except Exception:
            pass
        return blank_lines

    def _filter_noise_lines(self, filepath: str, lines: set[int]) -> set[int]:
        """
        Filter out invalid noise lines from coverage data:
        1. import/from...import statement lines
        2. Function definition lines (def line only)
        3. Class definition lines
        4. Docstring lines
        5. Blank/whitespace-only lines

        Args:
            filepath: Source file path
            lines: Original set of covered line numbers

        Returns:
            Filtered set with noise lines removed
        """
        if not lines:
            return lines

        # Use cache to avoid re-parsing the same file multiple times
        if filepath not in self._noise_lines_cache:
            import_lines = FunctionParser._get_import_lines(filepath)
            def_lines = self._get_function_def_lines(filepath)
            class_lines = self._get_class_def_lines(filepath)
            docstring_lines = self._get_docstring_lines(filepath)
            blank_lines = self._get_blank_lines(filepath)
            self._noise_lines_cache[filepath] = (
                import_lines | def_lines | class_lines | docstring_lines | blank_lines
            )

        return lines - self._noise_lines_cache[filepath]

    def _resolve_source_file(self, filename: str) -> Path | None:
        """
        Resolve source file path from relative filename.

        Args:
            filename: Relative file path (e.g., 'covstub/sglang/srt/models/qwen3_vl.py')

        Returns:
            Path object if found, None otherwise
        """
        if not self.source_dir:
            return None

        source_path = self.source_dir / REPO_NAME / filename
        return source_path if source_path.exists() else None

    def build_test_case_map(self) -> dict:
        """Build test case -> covered files mapping (with line numbers)"""
        print("Scanning test cases...")
        test_cases = self.scan_test_cases()
        print(f"  Found {len(test_cases)} test cases")

        for i, test_case in enumerate(test_cases):
            print(f"  [{i + 1}/{len(test_cases)}] Processing {test_case}...")
            test_case_dir = self.coverage_data_dir / test_case
            covdata_dir = test_case_dir / "covdata"

            file_lines_map = defaultdict(set)  # filepath -> set of lines

            # Coverage data files (coverage.*) are stored directly under the test case dir
            cov_dirs = [covdata_dir] if covdata_dir.exists() else [test_case_dir]

            for cov_dir in cov_dirs:
                for cov_file in cov_dir.glob("coverage.*"):
                    covered_files = self.get_covered_files_from_file(str(cov_file))

                    for filename in covered_files:
                        lines = self.get_covered_lines_from_file(
                            str(cov_file), filename
                        )
                        if lines:
                            # Filter noise lines if source_dir is available
                            if self.source_dir:
                                source_file = self._resolve_source_file(filename)
                                if source_file and source_file.exists():
                                    lines = self._filter_noise_lines(
                                        str(source_file), lines
                                    )
                            # Skip files with no coverage after filtering
                            if lines:
                                file_lines_map[filename].update(lines)

            normalized_name = self.normalize_test_name(test_case)
            self.test_case_map[normalized_name] = {
                "files": dict(file_lines_map),
                "file_count": len(file_lines_map),
                "line_count": sum(len(v) for v in file_lines_map.values()),
            }

            print(
                f"    -> {len(file_lines_map)} files, {sum(len(v) for v in file_lines_map.values())} lines"
            )

        return self.test_case_map

    def save_map(self, output_path: str = "test_case_map.json"):
        """Save test case mapping to file"""
        serializable_map = {}
        for test_case, data in self.test_case_map.items():
            serializable_map[test_case] = {
                "files": {k: list(v) for k, v in data["files"].items()},
                "file_count": data["file_count"],
                "line_count": data["line_count"],
            }

        with open(output_path, "w", encoding="utf-8", newline="\n") as f:
            json.dump(serializable_map, f, indent=2, ensure_ascii=False)
        print(f"\nTest case mapping saved to: {output_path}")

    def load_map(self, input_path: str = "test_case_map.json"):
        """Load test case mapping from file"""
        with open(input_path, encoding="utf-8") as f:
            serializable_map = json.load(f)

        self.test_case_map = {}
        for test_case, data in serializable_map.items():
            self.test_case_map[test_case] = {
                "files": {k: set(v) for k, v in data["files"].items()},
                "file_count": data["file_count"],
                "line_count": data["line_count"],
            }
        print(f"Loaded {len(self.test_case_map)} test case mappings from {input_path}")
        return self.test_case_map


class CodeChangeDetector:
    """Code change detector"""

    def __init__(self, source_dir: str):
        self.source_dir = Path(source_dir)
        self.file_hashes = {}

    def _product_code_root(self) -> Path:
        """
        Hash scanning uses this root so relative paths (e.g. srt/xxx.py) match
        the keys in test_case_map.json (which are relative to python/sglang/).
        """
        return self.source_dir / REPO_NAME

    def compute_file_hash(self, filepath: str) -> str:
        """Calculate MD5 hash of file"""
        hasher = hashlib.md5()
        try:
            with open(filepath, "rb") as f:
                hasher.update(f.read())
            return hasher.hexdigest()
        except Exception as e:
            print(f"  Warning: Error computing file hash: {filepath}: {e}")
            return ""

    def scan_source_files(self) -> dict[str, str]:
        """Scan product code files, compute hashes"""
        self.file_hashes = {}
        root = self._product_code_root()
        if not root.exists():
            print(f"  Warning: Product code root not found: {root}")
            return self.file_hashes
        for py_file in root.rglob("*.py"):
            rel_path = py_file.relative_to(root).as_posix()
            self.file_hashes[rel_path] = self.compute_file_hash(str(py_file))
        return self.file_hashes

    def detect_changes_by_comparison(self) -> dict[str, set[int]]:
        """Detect changes by file hash comparison (return all lines for changed files)"""
        changed_files = {}
        current_hashes = {}

        root = self._product_code_root()
        if not root.exists():
            print(f"  Warning: Product code root not found: {root}")
            return changed_files

        for py_file in root.rglob("*.py"):
            rel_path = py_file.relative_to(root).as_posix()
            current_hashes[rel_path] = self.compute_file_hash(str(py_file))

        baseline_path = self.source_dir / ".file_hashes.json"
        if baseline_path.exists():
            with open(baseline_path) as f:
                old_hashes = json.load(f)

            for rel_path, current_hash in current_hashes.items():
                old_hash = old_hashes.get(rel_path, "")
                if current_hash != old_hash:
                    # File has changes, return all line numbers (conservative estimate)
                    changed_files[rel_path] = set(
                        range(1, 10000)
                    )  # Conservative: assume all lines may have changed
        else:
            changed_files = {
                rel_path: set(range(1, 10000)) for rel_path in current_hashes
            }
            with open(baseline_path, "w") as f:
                json.dump(current_hashes, f)

        return changed_files

    def parse_git_diff(
        self,
        diff_output: str,
        base_content_getter=None,
    ) -> dict[str, set[int]]:
        """
        Parse git diff output, extract affected base (pre-change) line numbers.

        Rules:
        - Deleted lines: record the deleted base line itself, nothing more.
        - Pure comment/docstring changes are excluded (needs base content):
          a deletion group where every deleted line is a comment/docstring line
          and the additions are comments or doc prose; an insertion inside a
          docstring or consisting of comment lines only.
        - Isolated blank-line deletion (neighbours not deleted): treated as a
          one-line insertion -> candidate pair (line above, line below).
        - Pure insertions and blank-deletion pairs are classified via ast of
          the base file (needs base_content_getter):
          1. modifies an existing function -> record the line above only;
          2. sits between two function/class definitions -> excluded;
          3. inserted text belongs to a newly added def/class -> excluded;
          4. otherwise (module-level statements) -> record the line above only.
        - Without base content (or non-parseable Python) pairs fall back to
          counting both sides, bounded by the hunk's base range.

        Args:
            diff_output: diff content
            base_content_getter: optional callable(repo-relative-path -> str | None)
                returning the base file content for ast classification

        Returns:
            {filepath: {lineno, ...}} - set of affected base line numbers,
            .py files under '{PRODUCT_PREFIX}' only, with the prefix stripped.
            Renamed and deleted files are excluded: they are matched at file
            level via detect_renames() (see parse_pr_diff_file/main).
        """
        filter_prefix = PRODUCT_PREFIX
        renamed_files, deleted_files = self.detect_renames(diff_output)
        renamed_new_paths = set(renamed_files.values())
        deleted_paths = set(deleted_files)

        files, pending, del_groups = _parse_diff_base_lines(diff_output)

        changed_files = {}
        for path, lines in files.items():
            # Renamed/deleted files go through file-level matching, skip line-level parsing
            if path in renamed_new_paths or path in deleted_paths:
                continue
            # Filter: only keep product code (exclude test files, etc.)
            if not path.startswith(filter_prefix):
                continue
            if not path.endswith(".py"):
                continue
            # Normalize path: remove the '{PRODUCT_PREFIX}' prefix
            key = path[len(filter_prefix) :]
            changed_files[key] = lines
            pairs = pending.get(path) or []
            groups = del_groups.get(path) or []
            if pairs or groups:
                base_text = base_content_getter(path) if base_content_getter else None
                _classify_candidate_pairs(lines, pairs, groups, base_text, path)

        # Drop files that end up with no affected code lines (e.g. pure comment changes)
        return {k: v for k, v in changed_files.items() if v}

    def detect_renames(self, diff_output: str) -> tuple[dict[str, str], list[str]]:
        """
        Detect renamed and deleted files in git diff output (product code only,
        under PRODUCT_PREFIX). Both are handled the same way: file-level matching
        with the base path, excluded from line-level parsing.

        Args:
            diff_output: diff content

        Returns:
            Tuple of (rename_mapping, deleted_files)
            - rename_mapping: {old_path: new_path}
            - deleted_files: [path, ...] (base paths)
        """
        renames = {}
        deleted = []
        current_old_path = None
        current_new_path = None
        header_old_path = None

        for raw_line in diff_output.split("\n"):
            line = raw_line.rstrip("\r")

            # Detect rename marker
            if line.startswith("rename from "):
                current_old_path = line[12:].strip()
                continue
            if line.startswith("rename to "):
                current_new_path = line[10:].strip()
                # When we have both old and new path, record the rename
                if current_old_path and current_new_path:
                    # Remove a/ or b/ prefix if present
                    old_path = (
                        current_old_path[2:]
                        if current_old_path.startswith("a/")
                        else current_old_path
                    )
                    new_path = (
                        current_new_path[2:]
                        if current_new_path.startswith("b/")
                        else current_new_path
                    )
                    # Only record product code renames (under PRODUCT_PREFIX)
                    if old_path.startswith(PRODUCT_PREFIX):
                        renames[old_path] = new_path
                    current_old_path = None
                    current_new_path = None
                continue

            # Detect deleted file via '--- a/path' + '+++ /dev/null'
            if line.startswith("--- "):
                header_old_path = line[4:].strip()
                if header_old_path.startswith("a/"):
                    header_old_path = header_old_path[2:]
            elif line.startswith("+++ "):
                if (
                    line[4:].strip() == "/dev/null"
                    and header_old_path
                    and header_old_path.startswith(PRODUCT_PREFIX)
                ):
                    deleted.append(header_old_path)
                header_old_path = None

        return renames, deleted

    def parse_pr_diff_file(
        self, diff_file_path: str, base_content_getter=None
    ) -> tuple[dict[str, set[int]], dict[str, str], list[str]]:
        """
        Parse changed line numbers, renames and deleted files from PR diff file.

        Args:
            diff_file_path: diff file path
            base_content_getter: optional callable(repo-relative-path -> str | None)
                returning the base file content for ast classification

        Returns:
            Tuple of (changed_files_with_lines, rename_mapping, deleted_files)
            - changed_files_with_lines: {filepath: {lineno, ...}}
            - rename_mapping: {old_path: new_path}
            - deleted_files: [path, ...]
        """
        try:
            with open(diff_file_path, encoding="utf-8-sig") as f:
                diff_content = f.read()
            changed_files = self.parse_git_diff(
                diff_content, base_content_getter=base_content_getter
            )
            renames, deleted_files = self.detect_renames(diff_content)
            return changed_files, renames, deleted_files
        except Exception as e:
            print(f"Warning: Failed to read diff file: {e}")
            return {}, {}, []


_HUNK_RE = re.compile(r"@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@")
_DEF_RE = re.compile(r"(async\s+def|def|class)\s")


def _parse_diff_base_lines(
    diff_output: str,
) -> tuple[dict[str, set[int]], dict[str, list[tuple]], dict[str, list[tuple]]]:
    """Parse unified diff text into affected base (pre-change) line numbers.

    Returns (files, pending, del_groups):
      files[path]     : set of base line numbers recorded directly
                        (blank lines inside contiguous deletion blocks)
      pending[path]   : candidate pairs needing base-content classification;
                        tuple = (a, b, kind, add_indent, introduces_def,
                                 adds_all_comment, hunk_base_end)
                        kind='insert' -> pure insertion between a and b
                        kind='blank'  -> isolated blank deletion at a+1 (b = a+2)
      del_groups[path]: deletion groups needing comment/docstring filtering;
                        tuple = ([(base_line, deleted_text), ...], [added_text, ...])
    """
    files, pending, del_groups = {}, {}, {}
    current = None
    base_no = None
    hunk_base_end = 0
    old_path = None  # path from the last '--- a/...' line (used for deleted files)
    group_del = []  # (base_line, text) of '-' lines in the current change group
    group_add = []  # texts of '+' lines in the current change group

    def flush_group():
        if not group_del and not group_add:
            return
        if group_del:
            del_set = {n for n, _ in group_del}
            del_lines = []
            for n, text in group_del:
                if text.strip():
                    del_lines.append((n, text))
                elif (n - 1) in del_set or (n + 1) in del_set:
                    # blank inside a contiguous deletion block: classify with
                    # the group (dropped too if the block is pure comment/docstring)
                    del_lines.append((n, text))
                else:
                    pending[current].append(
                        (n - 1, n + 1, "blank", None, False, False, hunk_base_end)
                    )
            if del_lines:
                del_groups[current].append((del_lines, list(group_add)))
        else:
            # base_no is the next unprocessed base line = the line below the insertion
            if base_no is None or base_no < 1:
                # New file (hunk '@@ -0,0 ...'): there is no base version at all,
                # so there is nothing to classify the insertion against. Skip the
                # pair so callers never attempt to fetch a base file.
                return
            a = base_no - 1
            indent = min(
                ((len(t) - len(t.lstrip())) for t in group_add if t.strip()), default=0
            )
            introduces_def = any(
                t.strip().startswith("@") or _DEF_RE.match(t.strip())
                for t in group_add
                if t.strip()
            )
            adds_all_comment = all(
                t.strip().startswith("#") for t in group_add if t.strip()
            )
            pending[current].append(
                (
                    a,
                    a + 1,
                    "insert",
                    indent,
                    introduces_def,
                    adds_all_comment,
                    hunk_base_end,
                )
            )

    for raw_line in diff_output.split("\n"):
        line = raw_line.rstrip("\r")
        if line.startswith("diff --git"):
            flush_group()
            group_del, group_add = [], []
            current, base_no = None, None
            continue
        if line.startswith("--- "):
            old_path = line[4:]
            if old_path.startswith("a/"):
                old_path = old_path[2:]
            continue
        if line.startswith("+++ "):
            flush_group()
            group_del, group_add = [], []
            path = line[4:]
            if path == "/dev/null":
                # deleted file: keep the '--- a/...' path so deletions are recorded
                path = old_path
                old_path = None
                if path is None or path == "/dev/null":
                    current = None
                    continue
            if path.startswith("b/"):
                path = path[2:]
            current = path
            files.setdefault(path, set())
            pending.setdefault(path, [])
            del_groups.setdefault(path, [])
            continue
        if line.startswith("@@"):
            flush_group()
            group_del, group_add = [], []
            if current is None:
                continue
            m = _HUNK_RE.search(line)
            base_no = int(m.group(1))
            hunk_base_end = base_no + int(m.group(2) or "1") - 1
            continue
        if current is None or base_no is None:
            continue
        if line.startswith("-"):
            group_del.append((base_no, line[1:]))
            base_no += 1
        elif line.startswith("+"):
            group_add.append(line[1:])
        elif line.startswith("\\"):
            continue
        else:
            flush_group()
            group_del, group_add = [], []
            base_no += 1
    flush_group()
    return files, pending, del_groups


def _collect_defs(source: str) -> tuple[list, set, set, set]:
    """Parse Python source, return (ranges, end_lines, start_lines, blanks).

    ranges      : [(lineno, end_lineno, col_offset)] of every function/method
    end_lines   : line numbers where a function/class definition ends
    start_lines : def/class lines and their decorator lines
    blanks      : blank line numbers
    """
    tree = ast.parse(source)
    ranges, end_lines, start_lines = [], set(), set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            end_lines.add(node.end_lineno)
            start_lines.add(node.lineno)
            for deco in node.decorator_list:
                start_lines.add(deco.lineno)
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                ranges.append((node.lineno, node.end_lineno, node.col_offset))
    blanks = {i for i, text in enumerate(source.splitlines(), 1) if not text.strip()}
    return ranges, end_lines, start_lines, blanks


def _get_docstring_lines(source: str) -> set[int]:
    """Line numbers covered by docstrings (module/class/function docstring nodes)."""
    tree = ast.parse(source)
    lines = set()
    for node in ast.walk(tree):
        body = getattr(node, "body", None)
        if (
            isinstance(body, list)
            and body
            and isinstance(body[0], ast.Expr)
            and isinstance(body[0].value, ast.Constant)
            and isinstance(body[0].value.value, str)
        ):
            lines.update(range(body[0].lineno, body[0].end_lineno + 1))
    return lines


def _looks_like_code(texts: list) -> bool:
    """True if the added lines are real code: parseable as Python and not
    solely string-literal expressions (docstring prose)."""
    block = textwrap.dedent("\n".join(t for t in texts if t.strip()))
    if not block.strip():
        return False
    try:
        tree = ast.parse(block)
    except (SyntaxError, ValueError):
        return False
    return any(
        not (
            isinstance(stmt, ast.Expr)
            and isinstance(stmt.value, ast.Constant)
            and isinstance(stmt.value.value, str)
        )
        for stmt in tree.body
    )


def _innermost_func(ranges: list, n: int):
    """The innermost function whose body contains line n (None if module level)."""
    best = None
    for start, end, col in ranges:
        if start <= n <= end and (best is None or start >= best[0]):
            best = (start, end, col)
    return best


def _between_definitions(
    a: int, b: int, end_lines: set, start_lines: set, blanks: set
) -> bool:
    """True if the pair (a, b) sits between two definitions: the upper line is
    the end of a function/class (if a itself is blank, walk up past consecutive
    blank lines and check the nearest non-blank line instead) and the lower
    line is the start of a function/class (def/class line or decorator)."""
    if b not in start_lines:
        return False
    upper = a
    while upper in blanks:
        upper -= 1
    return upper in end_lines


def _classify_candidate_pairs(
    affected: set[int],
    pairs: list[tuple],
    del_groups: list[tuple],
    base_text: str | None,
    path: str,
) -> None:
    """Classify deletion groups and candidate pairs of one file using its base
    content and update the affected line set in place.

    Deletion groups: a group is dropped entirely when every deleted line is a
    comment/docstring line in the base file AND the added lines are comments or
    doc prose (not parseable Python), i.e. a pure comment/docstring change.
    Candidate pairs: without base content (or non-parseable Python) both sides
    of each pair are counted, bounded by the hunk."""
    info = None
    docstr_lines = set()
    comment_lines = set()
    if base_text is not None:
        try:
            info = _collect_defs(base_text)
            docstr_lines = _get_docstring_lines(base_text)
            comment_lines = {
                i
                for i, t in enumerate(base_text.splitlines(), 1)
                if t.strip().startswith("#")
            }
        except (SyntaxError, ValueError):
            info = None
    if base_text is None:
        print(
            f"  Warning: no base content for {path}, counting candidate pairs on both sides"
        )

    # Deleted non-blank lines: comment/docstring lines are never changes by
    # themselves; a group made entirely of them is dropped unless its lines are
    # replaced by real code (then they are kept as the only base anchors).
    noise_lines = comment_lines | docstr_lines
    for del_lines, add_texts in del_groups:
        if info is None:
            affected.update(n for n, _ in del_lines)
            continue
        code_dels = [n for n, _ in del_lines if n not in noise_lines]
        if code_dels:
            affected.update(code_dels)
            dropped = [n for n, _ in del_lines if n in noise_lines]
            if dropped:
                print(
                    f"  Skipped {path}:{dropped} (comment/docstring lines, not counted)"
                )
            continue
        adds = [t for t in add_texts if t.strip()]
        pure = (
            not adds
            or all(t.strip().startswith("#") for t in adds)
            or not _looks_like_code(adds)
        )
        if pure:
            print(
                f"  Skipped {path}:{[n for n, _ in del_lines]} (pure comment/docstring change, not counted)"
            )
        else:
            # comment/docstring lines replaced by real code: keep as change anchors
            affected.update(n for n, _ in del_lines)

    for (
        a,
        b,
        kind,
        add_indent,
        introduces_def,
        adds_all_comment,
        hunk_base_end,
    ) in pairs:
        if info is None:
            if a >= 1:
                affected.add(a)
            if b <= hunk_base_end:
                affected.add(b)
            continue
        ranges, end_lines, start_lines, blanks = info
        reason = None
        if kind == "insert":
            if a in docstr_lines:
                reason = "inside a docstring"
            elif adds_all_comment:
                reason = "pure comment insertion"
            else:
                func = _innermost_func(ranges, a)
                modifies = func is not None and (
                    b <= func[1] or (add_indent is not None and add_indent > func[2])
                )
                if not modifies:
                    if _between_definitions(a, b, end_lines, start_lines, blanks):
                        reason = "between function/class definitions"
                    elif introduces_def:
                        reason = "belongs to a newly added function/class"
        # kind == 'blank': isolated blank deletion == one-line insertion
        elif _between_definitions(a, b, end_lines, start_lines, blanks):
            reason = "between function/class definitions"
        if reason is None:  # kept: record the line above only
            if a >= 1:
                affected.add(a)
        else:
            print(f"  Skipped {path}:{a}-{b} ({reason}, not counted)")


class FunctionParser:
    """Python function parser - used to get line number ranges of functions and branches"""

    @staticmethod
    def get_function_ranges(filepath: str) -> dict[str, list[tuple[int, int]]]:
        """
        Parse Python file, return function name -> [(start_line, end_line), ...] mapping
        Supports multiple occurrences of the same function name (returns all matching ranges)
        """
        function_ranges = defaultdict(list)
        try:
            with open(filepath, encoding="utf-8") as f:
                tree = ast.parse(f.read(), filename=filepath)

            for node in ast.walk(tree):
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    function_ranges[node.name].append(
                        (node.lineno, node.end_lineno or node.lineno)
                    )
        except Exception as e:
            print(f"  Warning: Failed to parse function definition {filepath}: {e}")

        return function_ranges

    @staticmethod
    def _get_import_lines(filepath: str) -> set[int]:
        """
        Get line numbers of all import statements in file
        """
        import_lines = set()
        try:
            with open(filepath, encoding="utf-8") as f:
                tree = ast.parse(f.read(), filename=filepath)
            for node in ast.walk(tree):
                if isinstance(node, (ast.Import, ast.ImportFrom)):
                    import_lines.add(node.lineno)
                    if hasattr(node, "end_lineno") and node.end_lineno:
                        import_lines.update(range(node.lineno, node.end_lineno + 1))
        except Exception:
            pass
        return import_lines

    @staticmethod
    def get_lines_functions(
        filepath: str,
        lines: set[int],
        skip_imports: bool = False,
        function_ranges: dict[str, list[tuple[int, int]]] | None = None,
    ) -> dict[int, str]:
        """
        Get function name for each line

        Args:
            filepath: source file path
            lines: set of line numbers to query
            skip_imports: whether to skip import statement lines
            function_ranges: pre-parsed function ranges to reuse, avoiding
                re-parsing the file. When None, the file is parsed internally.
        """
        line_to_function = {}
        if not lines:
            return line_to_function

        if function_ranges is None:
            function_ranges = FunctionParser.get_function_ranges(filepath)
        if not function_ranges:
            return line_to_function

        # Flatten all function ranges into a single interval list sorted by
        # start line, then match each queried line with one linear scan.
        # This avoids per-function set expansion and the O(lines x functions)
        # nested loop. Semantics are preserved: for nested functions the outer
        # one has the smaller start line and is found first, matching the
        # original ast.walk (parent-before-child) order.
        intervals = []
        for func_name, ranges in function_ranges.items():
            for start, end in ranges:
                intervals.append((start, end, func_name))
        intervals.sort(key=lambda x: x[0])

        for line in lines:
            for start, end, func_name in intervals:
                if start > line:
                    break
                if line <= end:
                    line_to_function[line] = func_name
                    break

        return line_to_function


class TestSelector:
    """Test selector - select test cases to run based on code changes (line granularity)"""

    def __init__(self, test_case_map: dict):
        self.test_case_map = test_case_map

    def select_tests(
        self,
        changed_files_with_lines: dict[str, set[int]],
        min_affected_lines: int = 1,
        source_dir: str | None = None,
        enable_line_match: bool = True,
        enable_function_match: bool = True,
        enable_file_match: bool = True,
        enable_skip_imports: bool = False,
        enable_dedup: bool = False,
    ) -> tuple[list[tuple[str, dict[str, set[int]], int]], str]:
        """
        Select affected test cases based on changed files, supports 3 independent matching granularities:
        - Line-level matching: precise intersection of changed lines and covered lines
        - Function-level matching: entire function body range matching
        - File-level matching: any covered line in file matching

        Each granularity cascades: only when current granularity finds no tests, try the next.

        Args:
            changed_files_with_lines: changed files and their line numbers {filepath: {lineno, ...}}
            min_affected_lines: minimum affected lines, below this value will not be selected
            source_dir: source code directory, used for function/file-level expansion
            enable_line_match: whether to enable line-level matching
            enable_function_match: whether to enable function-level matching
            enable_file_match: whether to enable file-level matching
            enable_skip_imports: whether to skip import statement lines (only effective for function-level matching)
            enable_dedup: whether to enable deduplication

        Returns:
            (selected_tests, expand_reason)
            - selected_tests: [(test_case_name, {filepath: {covered_lines}}, total_affected_lines), ...]
            - expand_reason: expansion reason
                ('' means no expansion, 'line'/'function'/'file' indicates the granularity used)
        """
        selected = []
        expand_reason = ""

        # Normalize changed file paths: remove PRODUCT_PREFIX or REPO_NAME/ prefix
        normalized_changed = {}
        for f, lines in changed_files_with_lines.items():
            if PRODUCT_PREFIX and f.startswith(PRODUCT_PREFIX):
                normalized_changed[f[len(PRODUCT_PREFIX) :]] = lines
            elif f.startswith(f"{REPO_NAME}/"):
                normalized_changed[f[len(f"{REPO_NAME}/") :]] = lines
            else:
                normalized_changed[f] = lines

        total_changed_lines = sum(len(lines) for lines in normalized_changed.values())

        # ===== Line-level matching + Function-level matching (parallel execution, merge deduplication) =====
        line_results = []  # [(test_case, affected_detail, total_lines)]
        func_results = []  # [(test_case, affected_detail, total_lines)]

        # ----- Stage 1: Line-level matching -----
        if enable_line_match:
            for test_case, data in self.test_case_map.items():
                covered_files = data["files"]  # {filepath: {lineno, ...}}

                # Line-level matching: calculate which changed lines are covered by this test
                affected_detail = {}  # {filepath: set of covered changed lines}
                all_intersected_lines = set()  # union of intersections across all files

                for changed_file, changed_lines in normalized_changed.items():
                    if changed_file in covered_files:
                        covered_lines = covered_files[changed_file]
                        # Calculate intersection of changed lines and covered lines
                        intersected_lines = changed_lines & covered_lines
                        if intersected_lines:
                            affected_detail[changed_file] = intersected_lines
                            all_intersected_lines.update(intersected_lines)

                # Calculate overall coverage density: intersected lines / total changed lines
                overall_density = (
                    len(all_intersected_lines) / total_changed_lines
                    if total_changed_lines
                    else 0
                )

                # Filter by coverage density and minimum affected lines
                if (
                    all_intersected_lines
                    and overall_density >= COVERAGE_DENSITY_THRESHOLD
                    and len(all_intersected_lines) >= min_affected_lines
                ):
                    line_results.append(
                        (test_case, affected_detail, len(all_intersected_lines))
                    )

            # Sort by affected lines (more first)
            line_results.sort(key=lambda x: x[2], reverse=True)

            # Line-level deduplication: for same covered lines, only select one test
            if line_results and enable_dedup:
                claimed_lines = set()
                deduplicated = []
                for test_case, affected_detail, total_lines in line_results:
                    # Collect all lines covered by this test
                    test_lines = set()
                    for lines in affected_detail.values():
                        test_lines.update(lines)
                    # Only keep tests with new lines
                    unclaimed = test_lines - claimed_lines
                    if unclaimed:
                        deduplicated.append(
                            (test_case, affected_detail, len(unclaimed))
                        )
                        claimed_lines.update(test_lines)
                line_results = deduplicated

        # ----- Stage 2: Function-level matching -----
        if enable_function_match and source_dir:
            # Collect functions that changed lines belong to
            changed_functions = {}  # {filepath: {func_name: Set[linenos]}}
            changed_function_ranges = {}  # {filepath: function_ranges} - parsed once per file

            for changed_file, changed_lines in normalized_changed.items():
                source_path = Path(source_dir) / REPO_NAME / changed_file
                source_file = str(source_path) if source_path.exists() else None

                if not source_file:
                    continue

                # Parse function ranges ONCE per changed file and reuse the result
                # for both line-to-function mapping and later range lookups
                function_ranges = FunctionParser.get_function_ranges(source_file)

                # Get function mapping for changed lines (reuses pre-parsed ranges)
                line_to_function = FunctionParser.get_lines_functions(
                    source_file,
                    changed_lines,
                    skip_imports=enable_skip_imports,
                    function_ranges=function_ranges,
                )

                # Group by function name
                func_to_lines = defaultdict(set)
                for line, func_name in line_to_function.items():
                    func_to_lines[func_name].add(line)

                if func_to_lines:
                    changed_functions[changed_file] = func_to_lines
                    changed_function_ranges[changed_file] = function_ranges

            if changed_functions:
                # Build function -> tests covering that function mapping
                func_to_tests = defaultdict(list)

                for test_case, data in self.test_case_map.items():
                    covered_files = data["files"]

                    for changed_file, func_to_lines in changed_functions.items():
                        if changed_file not in covered_files:
                            continue

                        covered_lines = covered_files[changed_file]

                        # Resolve source file once per changed file
                        source_path = Path(source_dir) / REPO_NAME / changed_file
                        source_file = str(source_path) if source_path.exists() else None

                        if not source_file:
                            continue

                        # Reuse function ranges parsed in the collection phase
                        func_ranges = changed_function_ranges.get(changed_file, {})

                        # Filter out import statement lines (for display), computed once
                        if enable_skip_imports:
                            import_lines = FunctionParser._get_import_lines(source_file)
                            display_changed_lines = (
                                normalized_changed.get(changed_file, set())
                                - import_lines
                            )
                        else:
                            display_changed_lines = normalized_changed.get(
                                changed_file, set()
                            )

                        for func_name in func_to_lines:
                            if func_name not in func_ranges:
                                continue

                            # Merge all matched function ranges
                            func_all_lines = set()
                            for func_start, func_end in func_ranges[func_name]:
                                func_all_lines.update(range(func_start, func_end + 1))

                            if not func_all_lines:
                                continue

                            # Check if this test covers any line of this function
                            covered_in_func = covered_lines & func_all_lines
                            if covered_in_func:
                                # Get intersection of test covered lines and actual changed lines (for display)
                                covered_changed_lines = (
                                    covered_lines & display_changed_lines
                                )
                                func_to_tests[func_name].append(
                                    (test_case, covered_in_func, covered_changed_lines)
                                )

                # Select tests that cover other lines of changed functions (deduplication)
                for changed_file, func_to_lines in changed_functions.items():
                    for func_name in func_to_lines:
                        if func_name in func_to_tests:
                            for (
                                test_case,
                                covered_in_func,
                                covered_changed_lines,
                            ) in func_to_tests[func_name]:
                                existing = [s[0] for s in func_results]
                                if test_case not in existing and covered_in_func:
                                    # Display changed lines coverage if available, otherwise function coverage
                                    display_lines = (
                                        covered_changed_lines
                                        if covered_changed_lines
                                        else set()
                                    )
                                    func_results.append(
                                        (
                                            test_case,
                                            {changed_file: display_lines},
                                            len(display_lines) or len(covered_in_func),
                                        )
                                    )
                                    print(
                                        f"  [Function match] {test_case} covers function '{func_name}' in"
                                        f" {changed_file}"
                                    )

                func_results.sort(key=lambda x: x[2], reverse=True)

        # ===== Merge line-level and function-level results, deduplicate =====
        if line_results or func_results:
            # Deduplicate by test_case, keep line-level results (more precise)
            seen = set()
            for test_case, affected_detail, total_lines in line_results:
                if test_case not in seen:
                    seen.add(test_case)
                    selected.append((test_case, affected_detail, total_lines))

            # Add function-level exclusive results
            for test_case, affected_detail, total_lines in func_results:
                if test_case not in seen:
                    seen.add(test_case)
                    selected.append((test_case, affected_detail, total_lines))

            # Sort by affected lines
            selected.sort(key=lambda x: x[2], reverse=True)

            if selected:
                print(
                    f"  Line match: {len(line_results)} tests, Function match: {len(func_results)} tests, "
                    f"Total: {len(selected)} tests"
                )
                return selected, "line+function"

        # ===== Stage 3: File-level matching =====
        if not selected and enable_file_match:
            print("  Using file-level matching (renamed/deleted files)...")
            expand_reason = "file"

            # File-level matching: any test covering the changed file is selected
            for test_case, data in self.test_case_map.items():
                covered_files = data["files"]

                for changed_file in normalized_changed:
                    if changed_file in covered_files:
                        covered_lines = covered_files[changed_file]
                        if covered_lines:
                            selected.append(
                                (
                                    test_case,
                                    {changed_file: covered_lines},
                                    len(covered_lines),
                                )
                            )
                            break

            # Deduplicate: same test case only selected once
            if selected:
                seen = set()
                deduplicated = []
                for s in selected:
                    if s[0] not in seen:
                        seen.add(s[0])
                        deduplicated.append(s)
                selected = deduplicated

            selected.sort(key=lambda x: x[2], reverse=True)

        return selected, expand_reason

    def print_selection(
        self,
        selected: list[tuple[str, dict[str, set[int]], int]],
        changed_files: dict[str, set[int]],
        min_affected_lines: int = 1,
        expand_reason: str = "",
    ):
        """Print selection results"""
        total_changed_lines = sum(len(v) for v in changed_files.values())

        print("\n" + "=" * 70)
        print(f"Code changes: {len(changed_files)} files, {total_changed_lines} lines")

        # Display expansion reason
        gran_names = {
            "line": "Line match",
            "function": "Function match",
            "file": "File match",
            "line+function": "Line+Function match",
        }
        gran_detail_titles = {
            "line": "Details (Line match)",
            "function": "Details (Function match)",
            "file": "Details (File match)",
            "line+function": "Details (Line+Function match)",
        }
        if expand_reason and expand_reason in gran_names:
            print(f"Selected: {len(selected)} test cases ({gran_names[expand_reason]})")
        else:
            print(
                f"Selected: {len(selected)} test cases (min affected: {min_affected_lines} lines)"
            )
        print("=" * 70)

        if not selected:
            print("\nNo test cases cover the changed code lines!")
            print(f"Change details: {self._format_changed_files(changed_files)}")
            return

        print(f"\n{'#':<4} {'Test Case':<50} {'Affected Lines'}")
        print("-" * 70)

        for i, (test_case, affected_detail, total_lines) in enumerate(selected, 1):
            # Build coverage line display
            line_parts = []
            for filepath, lines in sorted(affected_detail.items()):
                line_parts.append(self._format_line_range(sorted(lines)))
            line_display = f" ({', '.join(line_parts)})" if line_parts else ""
            print(f"{i:<4} {test_case:<50} {total_lines}{line_display}")

        print(f"\n{gran_detail_titles.get(expand_reason, 'Details')}:")
        for test_case, affected_detail, total_lines in selected[:10]:
            print(f"\n  {test_case} ({total_lines} lines):")
            for filepath, lines in sorted(affected_detail.items()):
                line_str = self._format_line_range(sorted(lines))
                print(f"    - {filepath}: {line_str}")

    @staticmethod
    def _format_line_range(lines: list[int]) -> str:
        """Compress line number list into range representation"""
        if not lines:
            return ""

        lines = sorted(set(lines))
        ranges = []
        start = lines[0]
        end = lines[0]

        for line in lines[1:]:
            if line == end + 1:
                end = line
            else:
                if start == end:
                    ranges.append(str(start))
                else:
                    ranges.append(f"{start}-{end}")
                start = end = line

        if start == end:
            ranges.append(str(start))
        else:
            ranges.append(f"{start}-{end}")

        return ", ".join(ranges)

    def _format_changed_files(self, changed_files: dict[str, set[int]]) -> str:
        """Format changed files"""
        result = []
        for f, lines in sorted(changed_files.items()):
            if len(lines) > 10:
                result.append(f"{f}: {len(lines)} lines")
            else:
                result.append(f"{f}: {sorted(lines)}")
        return ", ".join(result[:5]) + ("..." if len(changed_files) > 5 else "")


def main():
    parser = argparse.ArgumentParser(
        description="Coverage-based precision test selector (line, function, file granularity)"
    )
    parser.add_argument(
        "--github-pr", "-pr", help="GitHub PR, format: owner/repo#pr_number"
    )
    parser.add_argument(
        "--source-dir",
        "-s",
        default="covstub",
        help="Source code directory (default: covstub)",
    )
    parser.add_argument(
        "--map-file",
        "-m",
        default="test_case_map.json",
        help="Test case map file (default: test_case_map.json)",
    )
    parser.add_argument(
        "--coverage-dir",
        "-c",
        default="coverage",
        help="Coverage data directory (default: ./coverage)",
    )
    parser.add_argument(
        "--build-map", "-b", action="store_true", help="Rebuild test case mapping"
    )
    parser.add_argument(
        "--min-affected",
        "-a",
        type=int,
        default=1,
        help="Minimum affected lines threshold (default: 1)",
    )
    parser.add_argument(
        "--dedup",
        action="store_true",
        help="Enable deduplication (keep only one test for same covered lines, default off)",
    )
    parser.add_argument(
        "--skip-imports",
        action="store_true",
        help="Skip import statement lines (only effective for function-level matching, default off)",
    )

    args = parser.parse_args()

    # Resolve relative paths against BASE_DIR (fixed structure), keep absolute paths as-is
    def _resolve_abs(base: Path, p: str) -> Path:
        path = Path(p)
        return path if path.is_absolute() else base / path

    coverage_dir = (
        _resolve_abs(BASE_DIR, args.coverage_dir) if args.coverage_dir else None
    )
    source_dir = _resolve_abs(BASE_DIR, args.source_dir)
    map_file = _resolve_abs(BASE_DIR, args.map_file)

    # 1. Build or load test case mapping
    selector = CoverageSelector(
        str(coverage_dir) if coverage_dir else None, str(source_dir)
    )

    if args.build_map or not map_file.exists():
        # Coverage data dir is required only when building the map
        if not coverage_dir:
            print(
                "Error: --coverage-dir is required when building the test case map (no map file found)"
            )
            exit(1)
        print("\n=== Building Test Case Mapping ===")
        selector.build_test_case_map()
        selector.save_map(str(map_file))
    else:
        print("\n=== Loading Test Case Mapping ===")
        selector.load_map(str(map_file))

    # If only need to generate map file, exit directly
    if args.build_map and not args.github_pr:
        print("\n=== Map file generated, done ===")
        return

    # 2. Parse code changes
    print("\n=== Parsing Code Changes ===")
    change_detector = CodeChangeDetector(str(source_dir))

    diff_file = None
    if args.github_pr:
        # Fetch changes from GitHub PR
        pr_spec = args.github_pr
        repo = None
        pr_num = None

        # Parse owner/repo#pr_number format
        if "#" in pr_spec:
            parts = pr_spec.split("#")
            repo = parts[0]
            pr_num = parts[1]
        else:
            pr_num = pr_spec
            # Try to get current repository
            try:
                result = subprocess.run(
                    ["git", "remote", "get-url", "origin"],
                    capture_output=True,
                    text=True,
                )
                if result.returncode == 0:
                    url = result.stdout.strip()
                    if "github.com" in url:
                        match = re.search(
                            r"github\.com[/:]([^/]+/[^/]+?)(?:\.git)?$", url
                        )
                        if match:
                            repo = match.group(1)
            except Exception as e:
                print(e)
                pass

        if not repo or not pr_num:
            print("Error: Cannot parse PR info, please use owner/repo#pr_number format")
            exit(1)

        print(f"Fetching changes from GitHub PR: {repo}#{pr_num}")

        github_token = os.environ.get("GITHUB_TOKEN") or os.environ.get("GH_TOKEN")

        def _github_request(url: str) -> urllib.request.Request:
            headers = {"Accept": "application/vnd.github.v3+json"}
            if github_token:
                headers["Authorization"] = f"Bearer {github_token}"
            return urllib.request.Request(url, headers=headers)

        # Create context that does not verify SSL certificates
        ssl_context = ssl.create_default_context()
        ssl_context.check_hostname = False
        ssl_context.verify_mode = ssl.CERT_NONE

        # Use cross-platform temp directory
        diff_file = os.path.join(tempfile.gettempdir(), "pr.diff")
        max_retries = 3
        base_sha = None

        for attempt in range(1, max_retries + 1):
            print(f"  Attempt {attempt}/{max_retries} to get PR diff via GitHub API...")
            try:
                pr_url = f"https://api.github.com/repos/{repo}/pulls/{pr_num}"
                req = _github_request(pr_url)
                with urllib.request.urlopen(
                    req, timeout=30, context=ssl_context
                ) as response:
                    pr_data = json.loads(response.read().decode())
                    diff_url = pr_data.get("diff_url")
                    base_sha = pr_data.get("base", {}).get("sha")

                if not diff_url:
                    raise Exception("Cannot get diff URL")

                # Download diff (use binary mode to avoid line ending conversion)
                req = _github_request(diff_url)
                with urllib.request.urlopen(
                    req, timeout=60, context=ssl_context
                ) as response:
                    diff_bytes = response.read()
                    with open(diff_file, "wb") as f:
                        f.write(diff_bytes)
                print("  Using GitHub API to get diff")
                break
            except Exception as e:
                print(f"  Attempt {attempt} failed: {e}")
                if attempt == max_retries:
                    print(f"  All {max_retries} attempts failed, exiting")
                    exit(1)
                time.sleep(1)

        print(f"  PR diff saved to: {diff_file}")

        def _fetch_base_content(path: str) -> str | None:
            """Fetch base (pre-change) file content via GitHub contents API"""
            content_url = f"https://api.github.com/repos/{repo}/contents/{urllib.parse.quote(path)}?ref={base_sha}"
            try:
                req = _github_request(content_url)
                with urllib.request.urlopen(
                    req, timeout=30, context=ssl_context
                ) as response:
                    data = json.loads(response.read().decode())
                if data.get("encoding") == "base64":
                    return base64.b64decode(data["content"]).decode("utf-8")
            except Exception as e:
                print(f"  Warning: Failed to fetch base content for {path}: {e}")
            return None

        # Only usable when the PR base sha was fetched successfully
        base_content_getter = _fetch_base_content if base_sha else None
    else:
        # Get from file comparison (default)
        change_detector.scan_source_files()
        changed_files_with_lines = change_detector.detect_changes_by_comparison()
        print(f"Detected {len(changed_files_with_lines)} changed files")

    # ===== Action 1: Extract new/deleted test files =====
    new_test_files: list[str] = []
    deleted_test_files: list[str] = []
    if args.github_pr and diff_file:
        new_test_files = _get_test_files_from_pr_diff(diff_file)
        deleted_test_files = _get_deleted_test_files_from_pr(
            diff_file, selector.test_case_map
        )

    # ===== Action 2: Detect Python product code changes -> Precision matching =====
    selected: list[tuple[str, dict[str, set[int]], int]] = []
    expand_reason = ""
    changed_files_with_lines: dict[str, set[int]] = {}

    renames: dict[str, str] = {}
    deleted_files: list[str] = []
    if args.github_pr and diff_file:
        changed_files_with_lines, renames, deleted_files = (
            change_detector.parse_pr_diff_file(
                diff_file, base_content_getter=base_content_getter
            )
        )
        print(f"Parsed {len(changed_files_with_lines)} changed files:")
        for file_path, line_set in changed_files_with_lines.items():
            print(f"  {file_path}: {TestSelector._format_line_range(list(line_set))}")

        # detect_renames already filters to PRODUCT_PREFIX only (product code renames)
        if renames:
            print(
                f"\n=== Detected {len(renames)} Product Code Renamed File(s) - Using File-Level Matching ==="
            )
            for old_path, new_path in renames.items():
                print(f"  {old_path} -> {new_path}")

        if deleted_files:
            print(
                f"\n=== Detected {len(deleted_files)} Product Code Deleted File(s) - Using File-Level Matching ==="
            )
            for path in deleted_files:
                print(f"  {path}")

    if changed_files_with_lines or renames or deleted_files:
        # Select test cases by precision matching
        print("\n=== Selecting Affected Test Cases ===")
        test_selector = TestSelector(selector.test_case_map)

        # Renamed/deleted files are already excluded from changed_files by
        # parse_git_diff; they are matched at file level below
        normal_files = changed_files_with_lines

        # Process normal files with precision matching
        selected: list[tuple[str, dict, int]] = []
        expand_reason = ""
        if normal_files:
            selected, expand_reason = test_selector.select_tests(
                normal_files,
                min_affected_lines=args.min_affected,
                source_dir=str(source_dir),
                enable_line_match=True,
                enable_function_match=True,
                enable_file_match=False,  # File-level matching reserved for renamed/deleted files only
                enable_skip_imports=args.skip_imports,
                enable_dedup=args.dedup,
            )

        # Process renamed/deleted files: file-level matching with the base path
        file_level_paths = [(p, f"{p} -> {n}") for p, n in renames.items()]
        file_level_paths += [(p, p) for p in deleted_files]
        for path, label in file_level_paths:
            fl_selected, fl_expand = test_selector.select_tests(
                {path: set()},
                min_affected_lines=args.min_affected,
                source_dir=str(source_dir),
                enable_line_match=False,  # Disable line match for file-level matching
                enable_function_match=False,  # Disable function match for file-level matching
                enable_file_match=True,  # Enable file match for renamed/deleted files
                enable_skip_imports=args.skip_imports,
                enable_dedup=args.dedup,
            )
            selected.extend(fl_selected)
            expand_reason += fl_expand
            # Print file-level matched test cases (even when empty, for diagnosis)
            print(f"\n=== File-Level Matched Tests for {label} ===")
            if fl_selected:
                for test_name, _, _ in fl_selected:
                    print(f"  {test_name}")
            else:
                print(
                    "  (0 tests matched: no coverage data for this path in test_case_map)"
                )

        # Deduplicate
        seen = set()
        deduped = []
        for item in selected:
            if item[0] not in seen:
                seen.add(item[0])
                deduped.append(item)
        selected = deduped
        test_selector.print_selection(
            selected,
            changed_files_with_lines,
            min_affected_lines=args.min_affected,
            expand_reason=expand_reason,
        )
    else:
        print("\n=== No product source code changes found ===")

    # ===== Merge results =====
    # Base set: precision matching results
    base_selected = selected

    # Add new test files
    existing_test_names = {s[0] for s in base_selected}
    for test_name in new_test_files:
        if test_name not in existing_test_names:
            base_selected.append((test_name, {}, 0))
            existing_test_names.add(test_name)

    if new_test_files:
        print(f"\n=== New Test Files Added: {len(new_test_files)} ===")
        print(f"  {new_test_files}")

    # Remove deleted test files
    if deleted_test_files:
        print(f"\n=== Deleted Test Files Removed: {len(deleted_test_files)} ===")
        print(f"  {deleted_test_files}")
        deleted_set = set(deleted_test_files)
        base_selected = [
            (name, detail, count)
            for name, detail, count in base_selected
            if name not in deleted_set
            and not any(name.startswith(d) for d in deleted_set)
        ]

    # ===== Output results =====
    test_names = [s[0] for s in base_selected]
    if test_names:
        print(f"\n=== Recommended Test Cases ({len(test_names)} tests) ===")
        print(test_names)
    else:
        print("\n=== No Test Cases Recommended ===")

    # Always write output file (even if empty), next to the script
    output_file = BASE_DIR / "recommended_pytest_paths.txt"
    with open(output_file, "w", encoding="utf-8") as f:
        for test_name in test_names:
            f.write(test_name + "\n")
    print(f"\nResults saved to: {output_file}")


if __name__ == "__main__":
    main()
