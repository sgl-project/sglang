"""
Auto-discover models needing DeepGEMM warmup from CI test suite config.

This script:
1. Discovers test files for a given CI suite using collect_tests()
2. Extracts model names and TP sizes from test files via AST parsing
3. Filters to MoE models present in HF cache that need DeepGEMM
4. Deduplicates models sharing the same kernel dimensions + TP
5. Runs compile_deep_gemm for each unique (architecture, tp) group

Usage (from test/ directory):
    python3 ../scripts/ci/cuda/warmup_deep_gemm.py --suite stage-c-test-8-gpu-h200 --tp 8
    python3 ../scripts/ci/cuda/warmup_deep_gemm.py --suite stage-c-test-8-gpu-h200 --tp 8 --dry-run
"""

import argparse
import ast
import glob
import json
import os
import re
import subprocess
import sys
from pathlib import Path


def get_test_utils_constants():
    """Pre-resolve model name constants from test_utils.py via AST parsing."""
    test_utils_path = os.path.join(
        os.path.dirname(__file__),
        "..",
        "..",
        "..",
        "python",
        "sglang",
        "test",
        "test_utils.py",
    )
    test_utils_path = os.path.normpath(test_utils_path)

    if not os.path.exists(test_utils_path):
        print(f"Warning: test_utils.py not found at {test_utils_path}")
        return {}

    with open(test_utils_path) as f:
        tree = ast.parse(f.read(), test_utils_path)

    constants = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            target = node.targets[0]
            if isinstance(target, ast.Name) and isinstance(node.value, ast.Constant):
                if isinstance(node.value.value, str) and "/" in node.value.value:
                    constants[target.id] = node.value.value
    return constants


def _resolve_value(node, all_constants):
    """Resolve an AST node to a string model path, or None."""
    if (
        isinstance(node, ast.Constant)
        and isinstance(node.value, str)
        and "/" in node.value
    ):
        return node.value
    if isinstance(node, ast.Name) and node.id in all_constants:
        return all_constants[node.id]
    return None


def _extract_tp_from_list(node):
    """Extract --tp value from a List AST node (e.g. other_args = ["--tp", "8"]).

    Handles both "--tp", "N" (two elements) and "--tp=N" (single element).
    """
    if not isinstance(node, ast.List):
        return None
    for i, elt in enumerate(node.elts):
        if isinstance(elt, ast.Constant) and isinstance(elt.value, str):
            # Pattern: "--tp=N"
            if elt.value.startswith("--tp="):
                try:
                    return int(elt.value.split("=")[1])
                except (ValueError, IndexError):
                    pass
            # Pattern: "--tp", "N"
            if elt.value == "--tp" and i + 1 < len(node.elts):
                next_elt = node.elts[i + 1]
                if isinstance(next_elt, ast.Constant):
                    try:
                        return int(next_elt.value)
                    except (ValueError, TypeError):
                        pass
    return None


def _extract_tp_from_class(class_node):
    """Extract --tp value from a class body by scanning list assignments and calls."""
    for node in ast.walk(class_node):
        # Pattern: other_args = ["--tp", "8", ...]
        if isinstance(node, ast.Assign):
            tp = _extract_tp_from_list(node.value)
            if tp is not None:
                return tp
        # Pattern: popen_launch_server(..., other_args=[...])
        if isinstance(node, ast.Call):
            for kw in node.keywords:
                if kw.arg == "other_args":
                    tp = _extract_tp_from_list(kw.value)
                    if tp is not None:
                        return tp
            # Also check positional list args
            for arg in node.args:
                tp = _extract_tp_from_list(arg)
                if tp is not None:
                    return tp
    return None


def extract_models_from_file(filepath, external_constants, default_tp):
    """Extract (model_name, tp_size) pairs from a test file using AST parsing.

    Returns a list of (model_name, tp_size) tuples.
    """
    with open(filepath) as f:
        source = f.read()

    try:
        tree = ast.parse(source, filepath)
    except SyntaxError:
        return []

    # Build local constant map from module-level assignments
    local_constants = {}
    for node in ast.iter_child_nodes(tree):
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            target = node.targets[0]
            if isinstance(target, ast.Name) and isinstance(node.value, ast.Constant):
                if isinstance(node.value.value, str) and "/" in node.value.value:
                    local_constants[target.id] = node.value.value

    all_constants = {**external_constants, **local_constants}
    results = []  # list of (model, tp)
    seen = set()

    def add_result(model, tp):
        key = (model, tp)
        if key not in seen:
            seen.add(key)
            results.append(key)

    # Walk classes to extract (model, tp) pairs with correct per-class TP
    for node in ast.iter_child_nodes(tree):
        if isinstance(node, ast.ClassDef):
            class_models = set()

            for child in ast.walk(node):
                # cls.model = ... or model = ...
                if isinstance(child, ast.Assign) and len(child.targets) == 1:
                    target = child.targets[0]
                    if isinstance(target, ast.Name) and target.id == "model":
                        val = _resolve_value(child.value, all_constants)
                        if val:
                            class_models.add(val)
                    elif (
                        isinstance(target, ast.Attribute)
                        and target.attr == "model"
                        and isinstance(target.value, ast.Name)
                        and target.value.id == "cls"
                    ):
                        val = _resolve_value(child.value, all_constants)
                        if val:
                            class_models.add(val)

                # ModelLaunchSettings("org/model", tp_size=N, ...)
                if isinstance(child, ast.Call):
                    func = child.func
                    is_mls = (
                        isinstance(func, ast.Name) and func.id == "ModelLaunchSettings"
                    ) or (
                        isinstance(func, ast.Attribute)
                        and func.attr == "ModelLaunchSettings"
                    )
                    if is_mls and child.args:
                        val = _resolve_value(child.args[0], all_constants)
                        if val:
                            # Get tp_size from keyword arg or second positional
                            mls_tp = None
                            for kw in child.keywords:
                                if kw.arg == "tp_size" and isinstance(
                                    kw.value, ast.Constant
                                ):
                                    try:
                                        mls_tp = int(kw.value.value)
                                    except (ValueError, TypeError):
                                        pass
                            if mls_tp is None and len(child.args) >= 2:
                                if isinstance(child.args[1], ast.Constant):
                                    try:
                                        mls_tp = int(child.args[1].value)
                                    except (ValueError, TypeError):
                                        pass
                            add_result(val, mls_tp or default_tp)

            # Extract TP from the class body (other_args, popen_launch_server, etc.)
            class_tp = _extract_tp_from_class(node) or default_tp

            for model in class_models:
                add_result(model, class_tp)

    # Also check module-level constant assignments with MODEL in name
    for node in ast.iter_child_nodes(tree):
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            target = node.targets[0]
            if isinstance(target, ast.Name) and "MODEL" in target.id.upper():
                val = _resolve_value(node.value, all_constants)
                if val:
                    # These are just constant definitions; the actual TP comes
                    # from wherever they're used (class context above).
                    # Only add if not already found in a class context.
                    if not any(m == val for m, _ in results):
                        add_result(val, default_tp)

    return results


def check_deepgemm_disabled(filepath):
    """Check if a test file explicitly disables DeepGEMM."""
    with open(filepath) as f:
        source = f.read()
    if "SGLANG_ENABLE_JIT_DEEPGEMM" not in source:
        return False
    # Match patterns like: SGLANG_ENABLE_JIT_DEEPGEMM...False/0/false
    # Covers: .set(False), .override(False), env["..."] = "0", "...": "False"
    return bool(re.search(r'SGLANG_ENABLE_JIT_DEEPGEMM.*(?:False|"0"|"false")', source))


def model_in_hf_cache(model_name):
    """Check if a model exists in the HF cache."""
    cache_dir = os.environ.get(
        "HF_HOME", os.path.join(os.path.expanduser("~"), ".cache", "huggingface")
    )
    hub_dir = os.path.join(cache_dir, "hub")

    safe_name = "models--" + model_name.replace("/", "--")
    model_cache_path = os.path.join(hub_dir, safe_name)

    if os.path.isdir(model_cache_path):
        snapshots_dir = os.path.join(model_cache_path, "snapshots")
        if os.path.isdir(snapshots_dir) and os.listdir(snapshots_dir):
            return True
    return False


def get_config_json(model_name):
    """Load config.json for a cached model, returning the parsed dict or None."""
    cache_dir = os.environ.get(
        "HF_HOME", os.path.join(os.path.expanduser("~"), ".cache", "huggingface")
    )
    hub_dir = os.path.join(cache_dir, "hub")
    safe_name = "models--" + model_name.replace("/", "--")
    snapshots_dir = os.path.join(hub_dir, safe_name, "snapshots")

    if not os.path.isdir(snapshots_dir):
        return None

    snapshots = sorted(
        Path(snapshots_dir).iterdir(), key=lambda p: p.stat().st_mtime, reverse=True
    )
    for snapshot in snapshots:
        config_path = snapshot / "config.json"
        if config_path.exists():
            with open(config_path) as f:
                return json.load(f)

    return None


def is_moe_model(config):
    """Check if a model config indicates an MoE architecture."""
    if config is None:
        return False

    moe_fields = ["num_local_experts", "n_routed_experts", "num_experts"]
    for field in moe_fields:
        val = config.get(field, 0)
        if isinstance(val, int) and val > 0:
            return True

    return False


def get_architecture_key(config, tp):
    """Get a key that identifies the kernel dimensions for deduplication.

    Models with the same (num_experts, hidden_size, intermediate_size, tp) share
    the same DeepGEMM kernel shapes. TP matters because it determines
    experts-per-GPU and weight shard sizes.
    """
    if config is None:
        return None

    num_experts = (
        config.get("num_local_experts")
        or config.get("n_routed_experts")
        or config.get("num_experts")
        or 0
    )
    hidden_size = config.get("hidden_size", 0)
    intermediate_size = config.get("intermediate_size") or config.get(
        "moe_intermediate_size", 0
    )

    return (num_experts, hidden_size, intermediate_size, tp)


def discover_suite_files(suite):
    """Discover test files for a suite using collect_tests()."""
    files = glob.glob("registered/**/*.py", recursive=True)
    if not files:
        print(
            "No test files found in registered/. "
            "Are you running from the test/ directory?"
        )
        return []

    sys.path.insert(
        0, os.path.join(os.path.dirname(__file__), "..", "..", "..", "python")
    )
    from sglang.test.ci.ci_register import collect_tests

    all_tests = collect_tests(files, sanity_check=True)
    suite_tests = [t for t in all_tests if t.suite == suite]

    if not suite_tests:
        print(f"No tests found for suite '{suite}'")
        return []

    return suite_tests


def main():
    parser = argparse.ArgumentParser(
        description="Warmup DeepGEMM JIT compilation for CI test suites"
    )
    parser.add_argument("--suite", type=str, required=True, help="CI test suite name")
    parser.add_argument(
        "--tp",
        type=int,
        default=8,
        help="Default tensor parallelism degree (default: 8). "
        "Overridden by per-model TP extracted from test files.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only print what would be compiled, don't run",
    )
    args = parser.parse_args()

    print(f"=== DeepGEMM Warmup for suite: {args.suite} ===\n")

    # Step 1: Discover test files for the suite
    suite_tests = discover_suite_files(args.suite)
    if not suite_tests:
        print("No test files to process. Exiting.")
        return

    print(f"Found {len(suite_tests)} test file(s) in suite '{args.suite}':")
    for t in suite_tests:
        status = "DISABLED" if t.disabled else "enabled"
        print(f"  {t.filename} ({status})")
    print()

    # Step 2: Extract (model, tp) pairs from test files
    external_constants = get_test_utils_constants()
    print(f"Loaded {len(external_constants)} model constants from test_utils.py")

    all_model_tp_pairs = []  # list of (model, tp)
    file_models = {}  # filepath -> set of model names (for deepgemm disable check)
    seen_pairs = set()
    for t in suite_tests:
        pairs = extract_models_from_file(t.filename, external_constants, args.tp)
        models_in_file = set()
        for model, tp in pairs:
            models_in_file.add(model)
            key = (model, tp)
            if key not in seen_pairs:
                seen_pairs.add(key)
                all_model_tp_pairs.append((model, tp))
        if models_in_file:
            file_models[t.filename] = models_in_file

    print(f"\nDiscovered {len(all_model_tp_pairs)} unique (model, tp) pair(s):")
    for model, tp in sorted(all_model_tp_pairs):
        print(f"  {model} (tp={tp})")
    print()

    # Step 3: Filter to models that need DeepGEMM warmup
    warmup_candidates = []
    for model, tp in sorted(all_model_tp_pairs):
        if not model_in_hf_cache(model):
            print(f"  SKIP {model} (tp={tp}): not in HF cache")
            continue

        config = get_config_json(model)
        if not is_moe_model(config):
            print(f"  SKIP {model} (tp={tp}): not an MoE model")
            continue

        # Check if any test file using this model explicitly disables DeepGEMM
        disabled = False
        for filepath, models in file_models.items():
            if model in models and check_deepgemm_disabled(filepath):
                disabled = True
                break

        if disabled:
            print(f"  SKIP {model} (tp={tp}): DeepGEMM explicitly disabled in test")
            continue

        warmup_candidates.append((model, tp, config))
        print(f"  NEED {model} (tp={tp}): MoE model in cache, DeepGEMM enabled")

    print()

    if not warmup_candidates:
        print("No models need DeepGEMM warmup. Done.")
        return

    # Step 4: Deduplicate by architecture + TP (same kernel dimensions)
    arch_groups = {}
    for model, tp, config in warmup_candidates:
        key = get_architecture_key(config, tp)
        if key not in arch_groups:
            arch_groups[key] = []
        arch_groups[key].append((model, tp))

    print(
        f"Architecture deduplication: {len(warmup_candidates)} candidate(s) "
        f"-> {len(arch_groups)} unique architecture(s)"
    )
    models_to_warmup = []  # list of (model, tp)
    for key, group in arch_groups.items():
        representative = group[0]
        models_to_warmup.append(representative)
        if len(group) > 1:
            others = ", ".join(f"{m}" for m, _ in group[1:])
            print(
                f"  {key}: using {representative[0]} tp={representative[1]} "
                f"(also covers: {others})"
            )
        else:
            print(f"  {key}: {representative[0]} tp={representative[1]}")
    print()

    # Step 5: Run compile_deep_gemm for each unique architecture
    if args.dry_run:
        print("DRY RUN - would compile for:")
        for model, tp in models_to_warmup:
            print(
                f"  python3 -m sglang.compile_deep_gemm "
                f"--model {model} --tp {tp} --trust-remote-code"
            )
        return

    for i, (model, tp) in enumerate(models_to_warmup, 1):
        print(f"\n{'=' * 60}")
        print(f"[{i}/{len(models_to_warmup)}] Warming up: {model} (tp={tp})")
        print(f"{'=' * 60}")
        cmd = [
            sys.executable,
            "-m",
            "sglang.compile_deep_gemm",
            "--model",
            model,
            "--tp",
            str(tp),
            "--trust-remote-code",
        ]
        result = subprocess.run(cmd)
        if result.returncode != 0:
            print(
                f"Warning: compile_deep_gemm failed for {model} "
                f"(exit code {result.returncode})"
            )
            print("Continuing with remaining models...")
        else:
            print(f"Successfully warmed up: {model}")

    print(f"\nDeepGEMM warmup complete for suite '{args.suite}'.")


if __name__ == "__main__":
    main()
