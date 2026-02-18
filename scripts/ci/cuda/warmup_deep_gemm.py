"""
Auto-discover models needing DeepGEMM warmup from CI test suite config.

This script:
1. Discovers test files for a given CI suite using collect_tests()
2. Extracts model names from test files via AST parsing
3. Filters to MoE models present in HF cache that need DeepGEMM
4. Deduplicates models sharing the same kernel dimensions
5. Runs compile_deep_gemm for each unique architecture

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
    # This script runs from test/ directory; test_utils is at ../python/sglang/test/test_utils.py
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


def extract_models_from_file(filepath, external_constants):
    """Extract model names from a test file using AST parsing.

    Patterns:
    - Module-level: NAME_WITH_MODEL = "org/model"
    - Class attribute: model = "org/model"
    - cls.model = "org/model" or cls.model = CONSTANT (in setUpClass)
    - ModelLaunchSettings("org/model", ...) or ModelLaunchSettings(CONSTANT, ...)
    """
    with open(filepath) as f:
        source = f.read()

    try:
        tree = ast.parse(source, filepath)
    except SyntaxError:
        return set()

    models = set()

    # Build local constant map from module-level assignments
    local_constants = {}
    for node in ast.iter_child_nodes(tree):
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            target = node.targets[0]
            if isinstance(target, ast.Name) and isinstance(node.value, ast.Constant):
                if isinstance(node.value.value, str) and "/" in node.value.value:
                    local_constants[target.id] = node.value.value

    # Combined lookup: local constants first, then external (test_utils)
    all_constants = {**external_constants, **local_constants}

    def resolve_value(node):
        """Resolve a node to a string model path, or None."""
        if (
            isinstance(node, ast.Constant)
            and isinstance(node.value, str)
            and "/" in node.value
        ):
            return node.value
        if isinstance(node, ast.Name) and node.id in all_constants:
            return all_constants[node.id]
        return None

    for node in ast.walk(tree):
        # Pattern 1: Module-level constant assignment with MODEL in name
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            target = node.targets[0]
            if isinstance(target, ast.Name) and "MODEL" in target.id.upper():
                val = resolve_value(node.value)
                if val:
                    models.add(val)

        # Pattern 2: Class-level attribute: model = "org/model"
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            target = node.targets[0]
            if isinstance(target, ast.Name) and target.id == "model":
                val = resolve_value(node.value)
                if val:
                    models.add(val)

        # Pattern 3: cls.model = ... in setUpClass
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            target = node.targets[0]
            if (
                isinstance(target, ast.Attribute)
                and target.attr == "model"
                and isinstance(target.value, ast.Name)
                and target.value.id == "cls"
            ):
                val = resolve_value(node.value)
                if val:
                    models.add(val)

        # Pattern 4: ModelLaunchSettings("org/model", ...) or ModelLaunchSettings(CONST, ...)
        if isinstance(node, ast.Call):
            func = node.func
            if (isinstance(func, ast.Name) and func.id == "ModelLaunchSettings") or (
                isinstance(func, ast.Attribute) and func.attr == "ModelLaunchSettings"
            ):
                if node.args:
                    val = resolve_value(node.args[0])
                    if val:
                        models.add(val)

    return models


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

    # HF cache stores models as models--org--name
    safe_name = "models--" + model_name.replace("/", "--")
    model_cache_path = os.path.join(hub_dir, safe_name)

    if os.path.isdir(model_cache_path):
        # Check that there's at least one snapshot
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

    # Use the most recent snapshot
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


def get_architecture_key(config):
    """Get a key that identifies the kernel dimensions for deduplication.

    Models with the same (num_experts, hidden_size, intermediate_size) share
    the same DeepGEMM kernel shapes.
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

    return (num_experts, hidden_size, intermediate_size)


def discover_suite_files(suite):
    """Discover test files for a suite using collect_tests()."""
    # We run from the test/ directory, same as run_suite.py
    files = glob.glob("registered/**/*.py", recursive=True)
    if not files:
        print(
            f"No test files found in registered/. Are you running from the test/ directory?"
        )
        return []

    # Import collect_tests from ci_register
    sys.path.insert(
        0, os.path.join(os.path.dirname(__file__), "..", "..", "..", "python")
    )
    from sglang.test.ci.ci_register import collect_tests

    all_tests = collect_tests(files, sanity_check=True)

    # Filter to our suite (both per-commit and nightly)
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
        "--tp", type=int, default=8, help="Tensor parallelism degree (default: 8)"
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

    # Step 2: Extract model names from test files
    external_constants = get_test_utils_constants()
    print(f"Loaded {len(external_constants)} model constants from test_utils.py")

    all_models = set()
    file_models = {}
    for t in suite_tests:
        models = extract_models_from_file(t.filename, external_constants)
        if models:
            file_models[t.filename] = models
            all_models.update(models)

    print(f"\nDiscovered {len(all_models)} unique model(s) across all test files:")
    for m in sorted(all_models):
        print(f"  {m}")
    print()

    # Step 3: Filter to models that need DeepGEMM warmup
    warmup_candidates = []
    for model in sorted(all_models):
        # Check if model is in HF cache
        if not model_in_hf_cache(model):
            print(f"  SKIP {model}: not in HF cache")
            continue

        # Check config for MoE
        config = get_config_json(model)
        if not is_moe_model(config):
            print(f"  SKIP {model}: not an MoE model")
            continue

        # Check if any test file that uses this model explicitly disables DeepGEMM
        disabled = False
        for filepath, models in file_models.items():
            if model in models and check_deepgemm_disabled(filepath):
                disabled = True
                break

        if disabled:
            print(f"  SKIP {model}: DeepGEMM explicitly disabled in test")
            continue

        warmup_candidates.append((model, config))
        print(f"  NEED {model}: MoE model in cache, DeepGEMM enabled")

    print()

    if not warmup_candidates:
        print("No models need DeepGEMM warmup. Done.")
        return

    # Step 4: Deduplicate by architecture (same kernel dimensions)
    arch_groups = {}
    for model, config in warmup_candidates:
        key = get_architecture_key(config)
        if key not in arch_groups:
            arch_groups[key] = []
        arch_groups[key].append(model)

    print(
        f"Architecture deduplication: {len(warmup_candidates)} model(s) -> {len(arch_groups)} unique architecture(s)"
    )
    models_to_warmup = []
    for key, group in arch_groups.items():
        representative = group[0]
        models_to_warmup.append(representative)
        if len(group) > 1:
            print(
                f"  Architecture {key}: using {representative} (also covers: {', '.join(group[1:])})"
            )
        else:
            print(f"  Architecture {key}: {representative}")
    print()

    # Step 5: Run compile_deep_gemm for each unique architecture
    if args.dry_run:
        print("DRY RUN - would compile for:")
        for model in models_to_warmup:
            print(
                f"  python3 -m sglang.compile_deep_gemm --model {model} --tp {args.tp} --trust-remote-code"
            )
        return

    for i, model in enumerate(models_to_warmup, 1):
        print(f"\n{'='*60}")
        print(f"[{i}/{len(models_to_warmup)}] Warming up: {model}")
        print(f"{'='*60}")
        cmd = [
            sys.executable,
            "-m",
            "sglang.compile_deep_gemm",
            "--model",
            model,
            "--tp",
            str(args.tp),
            "--trust-remote-code",
        ]
        result = subprocess.run(cmd)
        if result.returncode != 0:
            print(
                f"Warning: compile_deep_gemm failed for {model} (exit code {result.returncode})"
            )
            print("Continuing with remaining models...")
        else:
            print(f"Successfully warmed up: {model}")

    print(f"\nDeepGEMM warmup complete for suite '{args.suite}'.")


if __name__ == "__main__":
    main()
