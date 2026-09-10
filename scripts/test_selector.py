"""
Test Selector - Precision test selector based on coverage data (line, function, file granularity)

Workflow:
1. Build 'test case -> covered lines' mapping from coverage SQLite data
2. Parse code changes (supports GitHub PR or local file hash comparison)
3. Select affected test cases (by line, function, file granularity)
"""

import argparse
import ast
import hashlib
import json
import os
import sqlite3
import ssl
import subprocess
import tempfile
import time
import urllib.error
import urllib.request
from collections import defaultdict
from pathlib import Path

import regex as re

# ==================== Configuration ====================
# Repository name: used for filtering and path normalization
# vllm-ascend: "vllm_ascend"  |  sglang: "sglang"
REPO_NAME = "sglang"

# Product code path prefix in coverage data / diff paths.
# - vllm-ascend: coverage path is /__w/vllm_ascend/vllm_ascend/xxx.py, diff path is vllm_ascend/xxx.py
#   -> PRODUCT_PREFIX = "vllm_ascend/"
# - sglang: coverage path is /__w/sglang/sglang/python/sglang/xxx.py, diff path is python/sglang/xxx.py
#   -> PRODUCT_PREFIX = "python/sglang/"
# After stripping this prefix, both sides produce the same relative path (e.g. srt/models/qwen3_vl.py)
PRODUCT_PREFIX = "python/sglang/"

# Directory prefix of test case folders under coverage data dir.
# - vllm-ascend: "tests__" (e.g. tests__e2e__pull_request__...)
# - sglang: "____w__sglang__sglang__test__..." (GitHub Actions encoded path /__w/sglang/sglang/test/...)
# Empty string "" means accept any folder that contains coverage files.
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

    # Pattern to match test file paths.
    # vllm-ascend: tests/e2e/pull_request/ or tests/ut/ directory
    # sglang: test/registered/ directory
    # In diff output:
    #   - +++ b/test/registered/unit/xxx/test_xxx.py (new/modified test file)
    #   - rename to test/registered/unit/xxx/test_xxx.py (renamed test file)
    # Test files must be in test/ directory and start with test_
    test_file_pattern = re.compile(
        r"^(?:\+\+\+ [ab]/|rename to )((?:test/registered(?:/.+)?/test_\w+\.py|test/(?:unit|e2e|integration)(?:/.+)?/test_\w+\.py|tests/(?:e2e|ut)(?:/.+)?/test_\w+\.py))",
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
    Test files are in test/registered/ (sglang) or vllm_ascend/tests/ directory with test_*.py pattern.

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

    # Pattern to match deleted test files (test/registered/ or tests/ directory)
    # Match --- a/test/... followed by +++ /dev/null (deleted file marker)
    deleted_pattern = re.compile(
        r"^--- a/(test/registered(?:/.+)?/test_\w+\.py|test/(?:unit|e2e|integration)(?:/.+)?/test_\w+\.py|tests/(?:e2e|ut)(?:/.+)?/test_\w+\.py)\s*\n\s*\+\+\+ [ab]?/dev/null",
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

    def __init__(self, coverage_data_dir: str = None, source_dir: str = None):
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

        Two coverage directory layouts are supported:
        - vllm-ascend: <test_case>/covdata/coverage.* (covdata subdirectory)
        - sglang: <test_case>/coverage.* (coverage files directly under test case dir)
        """
        test_cases = []
        for item in self.coverage_data_dir.iterdir():
            if not item.is_dir():
                continue
            name = item.name
            # vllm-ascend naming: tests__* or cpu-ut
            is_vllm_layout = name.startswith("tests__") or name == "cpu-ut"
            # sglang naming: ____w__sglang__sglang__test__... (GitHub Actions encoded path)
            is_sglang_layout = TEST_CASE_DIR_PREFIX and name.startswith(
                TEST_CASE_DIR_PREFIX
            )
            if not (is_vllm_layout or is_sglang_layout):
                continue
            # covdata subdirectory (vllm-ascend) OR coverage.* files directly (sglang)
            covdata = item / "covdata"
            has_covdata = covdata.exists()
            has_cov_files = any(item.glob("coverage.*"))
            if has_covdata or has_cov_files:
                test_cases.append(name)
        return sorted(test_cases)

    @staticmethod
    def normalize_test_name(test_name: str) -> str:
        """
        Convert test case directory name to standard script name format.

        vllm-ascend:
        - tests__e2e__... -> tests/e2e/... (file-level) -> tests/e2e/....py (with .py suffix)
        - tests__e2e__...--test_foo -> tests/e2e/...::test_foo (function-level, no .py suffix)
        - cpu-ut -> cpu-ut (unchanged)

        sglang (GitHub Actions encoded dir name: /__w/sglang/sglang/test/... -> ____w__sglang__sglang__test__...):
        - ____w__sglang__sglang__test__registered__npu__xxx__test_foo.py
          -> test/registered/npu/xxx/test_foo.py (file-level)
        - ...--test_foo -> test/registered/npu/xxx/test_foo.py::test_foo (function-level)
        """
        # sglang layout: strip ____w__sglang__sglang__test__ prefix (encoded /__w/sglang/sglang/test/)
        if TEST_CASE_DIR_PREFIX and test_name.startswith(TEST_CASE_DIR_PREFIX):
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

        # vllm-ascend layout: tests__e2e__... -> tests/e2e/...
        # First convert -- to :: (function-level test marker)
        result = test_name.replace("--", "::")
        # Then convert __ to /
        result = result.replace("__", "/")
        # If no :: (file-level test), add .py suffix
        if "::" not in result:
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
                # Fallback: vllm-ascend style /__w/vllm_ascend/vllm_ascend/xxx.py -> xxx.py
                elif REPO_NAME + "/" in path:
                    rel_path = path.split(f"{REPO_NAME}/")[-1]
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

    def _filter_noise_lines(self, filepath: str, lines: set[int]) -> set[int]:
        """
        Filter out invalid noise lines from coverage data:
        1. import/from...import statement lines
        2. Function definition lines (def line only)
        3. Class definition lines
        4. Docstring lines

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
            self._noise_lines_cache[filepath] = (
                import_lines | def_lines | class_lines | docstring_lines
            )

        return lines - self._noise_lines_cache[filepath]

    def _resolve_source_file(self, filename: str) -> Path | None:
        """
        Resolve source file path from relative filename.

        Args:
            filename: Relative file path (e.g., 'srt/models/qwen3_vl.py')

        Returns:
            Path object if found, None otherwise
        """
        if not self.source_dir:
            return None

        # vllm-ascend: source_dir/vllm_ascend/xxx.py
        # sglang: source_dir/python/sglang/xxx.py  OR  source_dir/sglang/xxx.py (flat covstub layout)
        candidates = [
            self.source_dir / PRODUCT_PREFIX.rstrip("/") / filename,
            self.source_dir / PRODUCT_PREFIX.rstrip("/").split("/")[-1] / filename,
            self.source_dir / filename,
        ]
        for source_path in candidates:
            if source_path.exists():
                return source_path
        return None

    def build_test_case_map(self) -> dict:
        """Build test case -> covered files mapping (with line numbers)"""
        print("Scanning test cases...")
        test_cases = self.scan_test_cases()
        print(f"  Found {len(test_cases)} test cases")

        for i, test_case in enumerate(test_cases):
            print(f"  [{i + 1}/{len(test_cases)}] Processing {test_case}...")
            test_case_dir = self.coverage_data_dir / test_case

            # Locate coverage files:
            # - vllm-ascend: <test_case>/covdata/coverage.*
            # - sglang: <test_case>/coverage.* (directly under test case dir)
            covdata_dir = test_case_dir / "covdata"
            if covdata_dir.exists():
                cov_dir = covdata_dir
            else:
                cov_dir = test_case_dir

            file_lines_map = defaultdict(set)  # filepath -> set of lines

            for cov_file in cov_dir.glob("coverage.*"):
                covered_files = self.get_covered_files_from_file(str(cov_file))

                for filename in covered_files:
                    lines = self.get_covered_lines_from_file(str(cov_file), filename)
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

        with open(output_path, "w", encoding="utf-8") as f:
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
        """Scan source files, compute hashes"""
        self.file_hashes = {}
        for py_file in self.source_dir.rglob("*.py"):
            rel_path = py_file.relative_to(self.source_dir).as_posix()
            self.file_hashes[rel_path] = self.compute_file_hash(str(py_file))
        return self.file_hashes

    def detect_changes_by_comparison(self) -> dict[str, set[int]]:
        """Detect changes by file hash comparison (return all lines for changed files)"""
        changed_files = {}
        current_hashes = {}

        for py_file in self.source_dir.rglob("*.py"):
            rel_path = py_file.relative_to(self.source_dir).as_posix()
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
        self, diff_output: str, filter_prefix: str | None = None
    ) -> dict[str, set[int]]:
        """
        Parse git diff output, extract changed line numbers

        Supports two diff formats:
        1. unified diff: @@ -10,3 +10,4 @@ context
        2. PR diff

        Args:
            diff_output: diff content
            filter_prefix: only keep files with this prefix
                (e.g., '{PRODUCT_PREFIX}' filters product code, defaults to PRODUCT_PREFIX)

        Returns:
            {filepath: {lineno, ...}} - set of changed line numbers in the new file
        """
        changed_files = {}
        current_file = None

        # Default to PRODUCT_PREFIX as filter prefix (product code only)
        if filter_prefix is None:
            filter_prefix = PRODUCT_PREFIX

        # Parse mode: line by line, precisely calculate each changed line number in the new file
        for raw_line in diff_output.split("\n"):
            line = raw_line.rstrip("\r")
            # New file starts
            if line.startswith("diff --git"):
                continue

            # File path
            elif line.startswith("+++ b/") or line.startswith("--- a/"):
                path = line[6:].strip()
                # Remove a/ or b/ prefix
                if path.startswith("a/") or path.startswith("b/"):
                    path = path[2:]
                # Filter: only keep paths with specified prefix (exclude test files, etc.)
                if filter_prefix and not path.startswith(filter_prefix):
                    current_file = None
                    continue
                # Normalize path: remove filter_prefix prefix
                if filter_prefix and path.startswith(filter_prefix):
                    path = path[len(filter_prefix) :]
                if not path.endswith(".py"):
                    continue
                current_file = path
                if current_file not in changed_files:
                    changed_files[current_file] = set()

            # hunk header: @@ -old_start,old_count +new_start,new_count @@
            elif line.startswith("@@") and current_file:
                # Parse: @@ -100,10 +100,12 @@
                match = re.search(r"@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@", line)
                if match:
                    old_start = int(match.group(1))
                    old_count = int(match.group(2)) if match.group(2) else 1
                    # Rule: start line = old_start + 2, end line = old_start + old_count - 3
                    start_line = old_start + 2
                    end_line = old_start + old_count - 3
                    if end_line <= start_line:
                        end_line = old_start + old_count
                    # Collect all lines in hunk, check if there are new lines (starting with +)
                    hunk_lines = []
                    for hunk_line in diff_output.split("\n")[
                        diff_output.split("\n").index(line) + 1 :
                    ]:
                        if (
                            hunk_line.startswith("@@")
                            or hunk_line.startswith("diff --git")
                            or hunk_line.startswith("--- a/")
                            or hunk_line.startswith("+++ b/")
                        ):
                            break
                        hunk_lines.append(hunk_line)
                    # If no new lines starting with +, changes only include deletions, shrink range by one line
                    has_addition = any(
                        hline.lstrip().startswith("+") for hline in hunk_lines
                    )
                    if not has_addition:
                        start_line += 1
                        end_line -= 1
                    for line_no in range(start_line, end_line + 1):
                        changed_files[current_file].add(line_no)

        return changed_files

    def detect_renames(self, diff_output: str) -> dict[str, str]:
        """
        Detect file renames in git diff output (product code only, under PRODUCT_PREFIX).

        Args:
            diff_output: diff content

        Returns:
            {old_path: new_path} - mapping from old file path to new file path
        """
        renames = {}
        current_old_path = None
        current_new_path = None

        for raw_line in diff_output.split("\n"):
            line = raw_line.rstrip("\r")

            # Detect rename marker
            if line.startswith("rename from "):
                current_old_path = line[12:].strip()
            elif line.startswith("rename to "):
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

        return renames

    def parse_pr_diff_file(
        self, diff_file_path: str
    ) -> tuple[dict[str, set[int]], dict[str, str]]:
        """
        Parse changed line numbers and detect renames from PR diff file.

        Args:
            diff_file_path: diff file path

        Returns:
            Tuple of (changed_files_with_lines, rename_mapping)
            - changed_files_with_lines: {filepath: {lineno, ...}}
            - rename_mapping: {old_path: new_path}
        """
        try:
            with open(diff_file_path, encoding="utf-8-sig") as f:
                diff_content = f.read()
            changed_files = self.parse_git_diff(diff_content)
            renames = self.detect_renames(diff_content)
            return changed_files, renames
        except Exception as e:
            print(f"Warning: Failed to read diff file: {e}")
            return {}, {}


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
        filepath: str, lines: set[int], skip_imports: bool = False
    ) -> dict[int, str]:
        """
        Get function name for each line

        Args:
            filepath: source file path
            lines: set of line numbers to query
            skip_imports: whether to skip import statement lines
        """
        line_to_function = {}
        function_ranges = FunctionParser.get_function_ranges(filepath)

        func_to_covered_lines = {}
        for func_name, ranges in function_ranges.items():
            func_to_covered_lines[func_name] = set()
            for start, end in ranges:
                func_to_covered_lines[func_name].update(range(start, end + 1))

        for line in lines:
            for func_name, covered_lines in func_to_covered_lines.items():
                if line in covered_lines:
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

            for changed_file, changed_lines in normalized_changed.items():
                possible_paths = [
                    Path(source_dir) / changed_file,
                    Path(source_dir) / REPO_NAME / changed_file,
                    Path(source_dir) / PRODUCT_PREFIX.rstrip("/") / changed_file,
                    Path(source_dir) / "covstub" / REPO_NAME / changed_file,
                    Path(source_dir) / changed_file.replace("/", os.sep),
                    Path(source_dir) / REPO_NAME / changed_file.replace("/", os.sep),
                    Path(source_dir)
                    / PRODUCT_PREFIX.rstrip("/")
                    / changed_file.replace("/", os.sep),
                    Path(source_dir)
                    / "covstub"
                    / REPO_NAME
                    / changed_file.replace("/", os.sep),
                ]

                source_file = None
                for p in possible_paths:
                    if p.exists():
                        source_file = str(p)
                        break

                if not source_file:
                    continue

                # Get function mapping for changed lines
                line_to_function = FunctionParser.get_lines_functions(
                    source_file, changed_lines, skip_imports=enable_skip_imports
                )

                # Group by function name
                func_to_lines = defaultdict(set)
                for line, func_name in line_to_function.items():
                    func_to_lines[func_name].add(line)

                if func_to_lines:
                    changed_functions[changed_file] = func_to_lines

            if changed_functions:
                # Build function -> tests covering that function mapping
                func_to_tests = defaultdict(list)

                for test_case, data in self.test_case_map.items():
                    covered_files = data["files"]

                    for changed_file, func_to_lines in changed_functions.items():
                        if changed_file not in covered_files:
                            continue

                        covered_lines = covered_files[changed_file]

                        for func_name in func_to_lines:
                            # Get full line range of this function
                            possible_paths = [
                                Path(source_dir) / changed_file,
                                Path(source_dir) / REPO_NAME / changed_file,
                                Path(source_dir)
                                / PRODUCT_PREFIX.rstrip("/")
                                / changed_file,
                                Path(source_dir) / "covstub" / REPO_NAME / changed_file,
                                Path(source_dir) / changed_file.replace("/", os.sep),
                                Path(source_dir)
                                / REPO_NAME
                                / changed_file.replace("/", os.sep),
                                Path(source_dir)
                                / "covstub"
                                / REPO_NAME
                                / changed_file.replace("/", os.sep),
                            ]

                            source_file = None
                            for p in possible_paths:
                                if p.exists():
                                    source_file = str(p)
                                    break

                            if not source_file:
                                continue

                            # Filter out import statement lines (for display)
                            if enable_skip_imports:
                                import_lines = FunctionParser._get_import_lines(
                                    source_file
                                )
                                display_changed_lines = (
                                    normalized_changed.get(changed_file, set())
                                    - import_lines
                                )
                            else:
                                display_changed_lines = normalized_changed.get(
                                    changed_file, set()
                                )

                            func_ranges = FunctionParser.get_function_ranges(
                                source_file
                            )

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

            # ===== Stage 3: Function-level matching (only when first two levels are empty) =====
            print("  Line-level matching empty, trying function-level matching...")
            expand_reason = "function"

            # Collect functions that changed lines belong to
            changed_functions = {}  # {filepath: {func_name: Set[linenos]}}

            for changed_file, changed_lines in normalized_changed.items():
                possible_paths = [
                    Path(source_dir) / changed_file,
                    Path(source_dir) / REPO_NAME / changed_file,
                    Path(source_dir) / PRODUCT_PREFIX.rstrip("/") / changed_file,
                    Path(source_dir) / "covstub" / REPO_NAME / changed_file,
                    Path(source_dir) / changed_file.replace("/", os.sep),
                    Path(source_dir) / REPO_NAME / changed_file.replace("/", os.sep),
                    Path(source_dir)
                    / PRODUCT_PREFIX.rstrip("/")
                    / changed_file.replace("/", os.sep),
                    Path(source_dir)
                    / "covstub"
                    / REPO_NAME
                    / changed_file.replace("/", os.sep),
                ]

                source_file = None
                for p in possible_paths:
                    if p.exists():
                        source_file = str(p)
                        break

                if not source_file:
                    continue

                # Get function mapping for changed lines
                line_to_function = FunctionParser.get_lines_functions(
                    source_file, changed_lines, skip_imports=enable_skip_imports
                )

                # Group by function name
                func_to_lines = defaultdict(set)
                for line, func_name in line_to_function.items():
                    func_to_lines[func_name].add(line)

                if func_to_lines:
                    changed_functions[changed_file] = func_to_lines

            if not changed_functions:
                return selected, expand_reason

            # Build function -> tests covering that function mapping
            func_to_tests = defaultdict(list)

            for test_case, data in self.test_case_map.items():
                covered_files = data["files"]

                for changed_file, func_to_lines in changed_functions.items():
                    if changed_file not in covered_files:
                        continue

                    covered_lines = covered_files[changed_file]

                    for func_name in func_to_lines:
                        # Get full line range of this function
                        possible_paths = [
                            Path(source_dir) / changed_file,
                            Path(source_dir) / REPO_NAME / changed_file,
                            Path(source_dir)
                            / PRODUCT_PREFIX.rstrip("/")
                            / changed_file,
                            Path(source_dir) / "covstub" / REPO_NAME / changed_file,
                            Path(source_dir) / changed_file.replace("/", os.sep),
                            Path(source_dir)
                            / REPO_NAME
                            / changed_file.replace("/", os.sep),
                            Path(source_dir)
                            / "covstub"
                            / REPO_NAME
                            / changed_file.replace("/", os.sep),
                        ]

                        source_file = None
                        for p in possible_paths:
                            if p.exists():
                                source_file = str(p)
                                break

                        if not source_file:
                            continue

                        # Filter out import statement lines (for display)
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

                        func_ranges = FunctionParser.get_function_ranges(source_file)

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
                                (test_case, covered_changed_lines)
                            )

            # Select tests that cover other lines of changed functions (deduplication)
            for changed_file, func_to_lines in changed_functions.items():
                for func_name in func_to_lines:
                    if func_name in func_to_tests:
                        for test_case, covered_changed_lines in func_to_tests[
                            func_name
                        ]:
                            existing = [s[0] for s in selected]
                            if test_case not in existing and covered_changed_lines:
                                # Display actual changed line coverage, not full function coverage
                                display_lines = (
                                    covered_changed_lines
                                    if covered_changed_lines
                                    else set()
                                )
                                selected.append(
                                    (
                                        test_case,
                                        {changed_file: display_lines},
                                        len(display_lines),
                                    )
                                )

            selected.sort(key=lambda x: x[2], reverse=True)

            if selected:
                return selected, expand_reason

        # ===== Stage 4: File-level matching =====
        if not selected and enable_file_match:
            print("  Function-level matching empty, trying file-level matching...")
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

    def _format_line_range(self, lines: list[int]) -> str:
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
        "--enable-line-match",
        action="store_true",
        default=False,
        help="Enable line-level matching (default off)",
    )
    parser.add_argument(
        "--disable-line-match", action="store_true", help="Disable line-level matching"
    )
    parser.add_argument(
        "--enable-function-match",
        action="store_true",
        default=True,
        help="Enable function-level matching (default on)",
    )
    parser.add_argument(
        "--disable-function-match",
        action="store_true",
        help="Disable function-level matching",
    )
    parser.add_argument(
        "--enable-file-match",
        action="store_true",
        default=True,
        help="Enable file-level matching (default on)",
    )
    parser.add_argument(
        "--disable-file-match", action="store_true", help="Disable file-level matching"
    )
    parser.add_argument(
        "--skip-imports",
        action="store_true",
        help="Skip import statement lines (only effective for function-level matching, default off)",
    )

    args = parser.parse_args()

    # Process granularity switches: disable takes precedence over enable
    args.enable_line_match = not args.disable_line_match
    args.enable_function_match = not args.disable_function_match
    args.enable_file_match = not args.disable_file_match

    # 1. Build or load test case mapping
    selector = CoverageSelector(args.coverage_dir, args.source_dir)

    if args.build_map or not Path(args.map_file).exists():
        print("\n=== Building Test Case Mapping ===")
        selector.build_test_case_map()
        selector.save_map(args.map_file)
    else:
        print("\n=== Loading Test Case Mapping ===")
        selector.load_map(args.map_file)

    # If only need to generate map file, exit directly
    if args.build_map and not args.github_pr:
        print("\n=== Map file generated, done ===")
        return

    # 2. Parse code changes
    print("\n=== Parsing Code Changes ===")
    change_detector = CodeChangeDetector(args.source_dir)

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
    if args.github_pr and diff_file:
        changed_files_with_lines, renames = change_detector.parse_pr_diff_file(
            diff_file
        )
        print(f"Parsed {len(changed_files_with_lines)} changed files:")
        for file_path, line_set in changed_files_with_lines.items():
            print(f"  {file_path}")

        # detect_renames already filters to PRODUCT_PREFIX only (product code renames)
        if renames:
            print(
                f"\n=== Detected {len(renames)} Product Code Renamed File(s) - Using File-Level Matching ==="
            )
            for old_path, new_path in renames.items():
                print(f"  {old_path} -> {new_path}")

    if changed_files_with_lines:
        # Select test cases by precision matching
        print("\n=== Selecting Affected Test Cases ===")
        test_selector = TestSelector(selector.test_case_map)

        # For renamed files, we need file-level matching to cover both old and new paths
        # Filter out renamed new paths from changed files
        if renames:
            renamed_new_paths = set(renames.values())
            normal_files = {
                k: v
                for k, v in changed_files_with_lines.items()
                if k not in renamed_new_paths
            }
        else:
            normal_files = changed_files_with_lines

        # Process normal files with precision matching
        selected: list[tuple[str, dict, int]] = []
        expand_reason = ""
        if normal_files:
            selected, expand_reason = test_selector.select_tests(
                normal_files,
                min_affected_lines=args.min_affected,
                source_dir=args.source_dir,
                enable_line_match=args.enable_line_match,
                enable_function_match=args.enable_function_match,
                enable_file_match=args.enable_file_match,
                enable_skip_imports=args.skip_imports,
                enable_dedup=args.dedup,
            )

        # Process renamed files: use OLD path for file-level matching
        for old_path, new_path in renames.items():
            rename_files = {old_path: set()}
            rename_selected, rename_expand = test_selector.select_tests(
                rename_files,
                min_affected_lines=args.min_affected,
                source_dir=args.source_dir,
                enable_line_match=False,  # Disable line match for renames
                enable_function_match=False,  # Disable function match for renames
                enable_file_match=True,  # Enable file match for renames
                enable_skip_imports=args.skip_imports,
                enable_dedup=args.dedup,
            )
            selected.extend(rename_selected)
            expand_reason += rename_expand
            # Print file-level matched test cases for this rename
            if rename_selected:
                print(
                    f"\n=== File-Level Matched Tests for Renamed File: {old_path} -> {new_path} ==="
                )
                for test_name, _, _ in rename_selected:
                    print(f"  {test_name}")

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

    # Always write output file (even if empty)
    with open("recommended_pytest_paths.txt", "w", encoding="utf-8") as f:
        for test_name in test_names:
            f.write(test_name + "\n")
    print("\nResults saved to: recommended_pytest_paths.txt")


if __name__ == "__main__":
    main()
