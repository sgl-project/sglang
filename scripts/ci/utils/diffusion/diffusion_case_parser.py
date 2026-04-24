#!/usr/bin/env python3
"""
AST-based parser for diffusion test cases.

This module parses the diffusion case source and run_suite.py using AST to
extract test case information without requiring sglang dependencies. The case
source file is discovered from ONE_GPU_CASES/TWO_GPU_CASES imports in
run_suite.py so CI keeps a single source of truth.

Usage:
    # From sibling scripts in this directory:
    from diffusion_case_parser import collect_diffusion_suites, resolve_case_config_path
    case_config_path = resolve_case_config_path(repo_root, run_suite_path)
    suites = collect_diffusion_suites(case_config_path, run_suite_path, baseline_path)
"""

import ast
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

# Mapping from list variable names to suite names
CASE_LIST_TO_SUITE = {
    "ONE_GPU_CASES": "1-gpu",
    "ONE_GPU_CASES_A": "1-gpu",
    "ONE_GPU_CASES_B": "1-gpu",
    "ONE_GPU_CASES_C": "1-gpu-b200",
    "ONE_GPU_MODELOPT_CASES": "1-gpu-b200",
    "TWO_GPU_CASES": "2-gpu",
    "TWO_GPU_CASES_A": "2-gpu",
    "TWO_GPU_CASES_B": "2-gpu",
}

# Default estimated time for cases without baseline (5 minutes)
DEFAULT_EST_TIME_SECONDS = 300.0

# Fixed overhead for server startup when estimated_full_test_time_s is not set
STARTUP_OVERHEAD_SECONDS = 120.0

# Paths relative to repository root
BASELINE_REL_PATH = "python/sglang/multimodal_gen/test/server/perf_baselines.json"
RUN_SUITE_REL_PATH = "python/sglang/multimodal_gen/test/run_suite.py"


@dataclass
class DiffusionCaseInfo:
    """Information about a single diffusion test case."""

    case_id: str  # e.g., "qwen_image_t2i"
    suite: str  # "1-gpu" or "2-gpu"
    est_time: float  # estimated time in seconds


@dataclass
class DiffusionSuiteInfo:
    """Complete information for a test suite."""

    suite: str  # "1-gpu" or "2-gpu"
    cases: List[DiffusionCaseInfo]  # parametrized test cases
    standalone_files: List[str]  # standalone test files
    standalone_est_times: Dict[str, float]  # standalone file -> estimated seconds
    missing_standalone_estimates: List[
        str
    ]  # standalone files without configured estimate


class DiffusionTestCaseVisitor(ast.NodeVisitor):
    """
    AST visitor to extract DiffusionTestCase definitions from the case config.

    Parses assignments like:
        ONE_GPU_CASES_A: list[DiffusionTestCase] = [
            DiffusionTestCase("case_id", ...),
            ...
        ]
    """

    def __init__(self):
        self.cases: Dict[str, List[str]] = {}  # list_name -> [case_id, ...]

    def visit_Assign(self, node: ast.Assign):
        self._process_assignment(node.targets, node.value)
        self.generic_visit(node)

    def visit_AnnAssign(self, node: ast.AnnAssign):
        if node.target and node.value:
            self._process_assignment([node.target], node.value)
        self.generic_visit(node)

    def visit_AugAssign(self, node: ast.AugAssign):
        self._process_aug_assignment(node.target, node.op, node.value)
        self.generic_visit(node)

    def _process_assignment(self, targets: List[ast.AST], value: ast.AST):
        """Process an assignment to extract case IDs."""
        for target in targets:
            if isinstance(target, ast.Name):
                list_name = target.id
                case_ids = self._extract_case_ids(value)
                if case_ids is not None:
                    self.cases[list_name] = case_ids

    def _process_aug_assignment(self, target: ast.AST, op: ast.AST, value: ast.AST):
        """Process `+=` style assignment to merge case lists."""
        if not isinstance(target, ast.Name) or not isinstance(op, ast.Add):
            return

        rhs_case_ids = self._extract_case_ids(value)
        if rhs_case_ids is None:
            return

        lhs_case_ids = self.cases.get(target.id, [])
        self.cases[target.id] = [*lhs_case_ids, *rhs_case_ids]

    def _extract_case_ids(self, node: ast.AST) -> Optional[List[str]]:
        """Extract case IDs from a supported expression."""
        if isinstance(node, ast.List):
            return self._extract_case_ids_from_list(node)

        if isinstance(node, ast.Name):
            # Reference to a previously parsed list variable.
            if node.id not in self.cases:
                return None
            return list(self.cases[node.id])

        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
            left_ids = self._extract_case_ids(node.left)
            right_ids = self._extract_case_ids(node.right)
            if left_ids is None or right_ids is None:
                return None
            return [*left_ids, *right_ids]

        return None

    def _extract_case_ids_from_list(self, node: ast.List) -> List[str]:
        """Extract case IDs from a literal list of DiffusionTestCase calls."""
        case_ids = []
        for elt in node.elts:
            if isinstance(elt, ast.Starred):
                starred_case_ids = self._extract_case_ids(elt.value)
                if starred_case_ids:
                    case_ids.extend(starred_case_ids)
                continue
            case_id = self._extract_case_id_from_call(elt)
            if case_id:
                case_ids.append(case_id)
        return case_ids

    def _extract_case_id_from_call(self, node: ast.AST) -> Optional[str]:
        """Extract case_id from DiffusionTestCase(...) call."""
        if not isinstance(node, ast.Call):
            return None

        # Check if it's a DiffusionTestCase call
        if isinstance(node.func, ast.Name) and node.func.id == "DiffusionTestCase":
            # First positional argument is the case_id
            if node.args and isinstance(node.args[0], ast.Constant):
                return node.args[0].value

        return None


def resolve_case_config_path(repo_root: Path, run_suite_path: Path) -> Path:
    """
    Resolve the diffusion case config path from run_suite imports.

    run_suite.py must import BOTH ONE_GPU_CASES and TWO_GPU_CASES from the same
    module. That imported module is treated as the single source of truth.
    """
    with open(run_suite_path, "r", encoding="utf-8") as f:
        content = f.read()

    tree = ast.parse(content, filename=str(run_suite_path))
    one_gpu_module: Optional[str] = None
    two_gpu_module: Optional[str] = None

    for node in ast.walk(tree):
        if not isinstance(node, ast.ImportFrom) or not node.module:
            continue
        imported_names = {alias.name for alias in node.names}
        if "ONE_GPU_CASES" in imported_names:
            one_gpu_module = node.module
        if "TWO_GPU_CASES" in imported_names:
            two_gpu_module = node.module

    if one_gpu_module is None or two_gpu_module is None:
        raise RuntimeError(
            "run_suite.py must import BOTH ONE_GPU_CASES and TWO_GPU_CASES."
        )
    if one_gpu_module != two_gpu_module:
        raise RuntimeError(
            "run_suite.py imports ONE_GPU_CASES and TWO_GPU_CASES from different "
            f"modules: {one_gpu_module} vs {two_gpu_module}"
        )

    rel_path = Path(*one_gpu_module.split(".")).with_suffix(".py")
    candidates = [repo_root / rel_path, repo_root / "python" / rel_path]
    case_config_path = next((path for path in candidates if path.exists()), None)
    if case_config_path is None:
        raise FileNotFoundError(
            "Resolved case config from run_suite does not exist. Checked: "
            + ", ".join(str(path) for path in candidates)
        )
    return case_config_path


class RunSuiteVisitor(ast.NodeVisitor):
    """
    AST visitor to extract standalone metadata from run_suite.py.

    Parses:
        STANDALONE_FILES = {
            "1-gpu": ["test_lora_format_adapter.py"],
            "2-gpu": [],
        }
    """

    def __init__(self):
        self.standalone_files: Dict[str, List[str]] = {}
        self.standalone_est_times: Dict[str, Dict[str, float]] = {}

    def visit_Assign(self, node: ast.Assign):
        for target in node.targets:
            if isinstance(target, ast.Name) and target.id == "STANDALONE_FILES":
                self.standalone_files = self._extract_file_dict(node.value)
            if (
                isinstance(target, ast.Name)
                and target.id == "STANDALONE_FILE_EST_TIMES"
            ):
                self.standalone_est_times = self._extract_est_time_dict(node.value)
        self.generic_visit(node)

    def _extract_file_dict(self, node: ast.AST) -> Dict[str, List[str]]:
        """Extract dictionary of suite -> file list."""
        result = {}
        if isinstance(node, ast.Dict):
            for key, value in zip(node.keys, node.values):
                if isinstance(key, ast.Constant) and isinstance(value, ast.List):
                    suite = key.value
                    files = [
                        elt.value for elt in value.elts if isinstance(elt, ast.Constant)
                    ]
                    result[suite] = files
        return result

    def _extract_est_time_dict(self, node: ast.AST) -> Dict[str, Dict[str, float]]:
        """Extract dictionary of suite -> standalone file -> estimated seconds."""
        result = {}
        if not isinstance(node, ast.Dict):
            return result

        for key, value in zip(node.keys, node.values):
            if not isinstance(key, ast.Constant) or not isinstance(value, ast.Dict):
                continue

            suite = key.value
            suite_est_times = {}
            for inner_key, inner_value in zip(value.keys, value.values):
                if not (
                    isinstance(inner_key, ast.Constant)
                    and isinstance(inner_value, ast.Constant)
                ):
                    continue
                suite_est_times[inner_key.value] = float(inner_value.value)
            result[suite] = suite_est_times

        return result


def load_baselines(baseline_path: Path) -> Dict[str, float]:
    """
    Load performance baselines from JSON file.

    Returns:
        Dictionary mapping case_id to estimated time in seconds.
    """
    if not baseline_path.exists():
        return {}

    with open(baseline_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    baselines = {}
    scenarios = data.get("scenarios", {})

    for case_id, scenario in scenarios.items():
        if scenario.get("estimated_full_test_time_s") is not None:
            baselines[case_id] = scenario["estimated_full_test_time_s"]
        else:
            expected_e2e_ms = scenario.get("expected_e2e_ms", 0)
            baselines[case_id] = expected_e2e_ms / 1000.0 + STARTUP_OVERHEAD_SECONDS

    return baselines


def get_case_est_time(case_id: str, baselines: Dict[str, float]) -> float:
    """Get estimated time for a case, with fallback to default."""
    return baselines.get(case_id, DEFAULT_EST_TIME_SECONDS)


def parse_testcase_configs(config_path: Path) -> Dict[str, List[str]]:
    """
    Parse a diffusion case config file to extract case IDs.

    Returns:
        Dictionary mapping list name to case IDs.
        e.g., {"ONE_GPU_CASES_A": ["qwen_image_t2i", ...], ...}
    """
    with open(config_path, "r", encoding="utf-8") as f:
        content = f.read()

    tree = ast.parse(content, filename=str(config_path))
    visitor = DiffusionTestCaseVisitor()
    visitor.visit(tree)

    return visitor.cases


def parse_run_suite_standalone_data(
    run_suite_path: Path,
) -> tuple[Dict[str, List[str]], Dict[str, Dict[str, float]]]:
    """
    Parse run_suite.py to extract standalone file metadata.

    Returns:
        Tuple of:
          - suite -> standalone file list
          - suite -> standalone file -> estimated seconds
    """
    with open(run_suite_path, "r", encoding="utf-8") as f:
        content = f.read()

    tree = ast.parse(content, filename=str(run_suite_path))
    visitor = RunSuiteVisitor()
    visitor.visit(tree)

    return visitor.standalone_files, visitor.standalone_est_times


def validate_standalone_est_times(
    standalone_files: Dict[str, List[str]],
    standalone_est_times: Dict[str, Dict[str, float]],
) -> Dict[str, List[str]]:
    missing_by_suite = {}
    for suite, files in standalone_files.items():
        suite_est_times = standalone_est_times.get(suite, {})
        missing = [
            standalone_file
            for standalone_file in files
            if standalone_file not in suite_est_times
        ]
        if missing:
            missing_by_suite[suite] = missing
    return missing_by_suite


def collect_diffusion_suites(
    case_config_path: Path,
    run_suite_path: Path,
    baseline_path: Path,
) -> Dict[str, DiffusionSuiteInfo]:
    """
    Collect all diffusion test suite information using AST parsing.

    Args:
        case_config_path: Path to case config (resolved from run_suite.py)
        run_suite_path: Path to run_suite.py
        baseline_path: Path to perf_baselines.json

    Returns:
        Dictionary mapping suite name to DiffusionSuiteInfo.
    """
    # Parse case IDs from the single source case config.
    case_lists = parse_testcase_configs(case_config_path)

    # Parse standalone files from run_suite.py
    standalone_files, standalone_est_times = parse_run_suite_standalone_data(
        run_suite_path
    )
    missing_standalone_estimates = validate_standalone_est_times(
        standalone_files, standalone_est_times
    )

    # Load baselines for time estimation
    baselines = load_baselines(baseline_path)

    # Build suite info
    suites = {}
    for list_name, suite in CASE_LIST_TO_SUITE.items():
        case_ids = case_lists.get(list_name, [])
        cases = [
            DiffusionCaseInfo(
                case_id=cid,
                suite=suite,
                est_time=get_case_est_time(cid, baselines),
            )
            for cid in case_ids
        ]

        if suite not in suites:
            suites[suite] = DiffusionSuiteInfo(
                suite=suite,
                cases=[],
                standalone_files=standalone_files.get(suite, []),
                standalone_est_times=dict(standalone_est_times.get(suite, {})),
                missing_standalone_estimates=list(
                    missing_standalone_estimates.get(suite, [])
                ),
            )
        suites[suite].cases.extend(cases)

    # Dedupe duplicated case IDs while preserving first-seen order.
    for suite_info in suites.values():
        seen_case_ids = set()
        deduped_cases = []
        for case in suite_info.cases:
            if case.case_id in seen_case_ids:
                continue
            seen_case_ids.add(case.case_id)
            deduped_cases.append(case)
        suite_info.cases = deduped_cases

    return suites
