"""Exhaustive, import-free contracts for the PR2 model loader migration."""

import ast
import builtins
import copy
from pathlib import Path
from types import SimpleNamespace

import pytest

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=8, suite="base-b-test-cpu")

REPO_ROOT = Path(__file__).resolve().parents[4]
MODEL_ROOT = REPO_ROOT / "python/sglang/srt/models"

# Keep this explicit: adding or removing a migrated production file must be reviewed
# together with these contracts. These are the 69 model files in the PR2 change.
CHANGED_MODEL_FILES = (
    "afmoe.py",
    "apertus.py",
    "arcee.py",
    "baichuan.py",
    "bailing_moe.py",
    "cohere2_moe.py",
    "commandr.py",
    "dbrx.py",
    "deepseek.py",
    "ernie4.py",
    "ernie4_eagle.py",
    "exaone.py",
    "exaone4.py",
    "exaone_moe.py",
    "exaone_moe_mtp.py",
    "gemma.py",
    "gemma2.py",
    "gemma2_reward.py",
    "gpt_bigcode.py",
    "gpt_j.py",
    "granite.py",
    "granitemoe.py",
    "hrm_text.py",
    "hunyuan.py",
    "hunyuan_v3.py",
    "hunyuan_v3_nextn.py",
    "internlm2.py",
    "internlm2_reward.py",
    "iquest_loopcoder.py",
    "jet_nemotron.py",
    "laguna.py",
    "llama_classification.py",
    "llama_eagle.py",
    "llama_eagle3.py",
    "llama_embedding.py",
    "llama_reward.py",
    "mimo.py",
    "mimo_mtp.py",
    "minicpm.py",
    "minimax_m2.py",
    "minimax_m3.py",
    "mistral.py",
    "mistral_eagle.py",
    "mixtral.py",
    "mixtral_quant.py",
    "nemotron_nas.py",
    "olmo.py",
    "olmo2.py",
    "olmoe.py",
    "orion.py",
    "phi3_small.py",
    "phimoe.py",
    "qwen.py",
    "qwen2_classification.py",
    "qwen2_eagle.py",
    "qwen2_moe.py",
    "qwen2_rm.py",
    "qwen3.py",
    "qwen3_classification.py",
    "qwen3_embedding.py",
    "qwen3_moe.py",
    "qwen3_moe_mtp.py",
    "sdar.py",
    "sdar_moe.py",
    "solar.py",
    "stablelm.py",
    "starcoder2.py",
    "xverse.py",
    "xverse_moe.py",
)

PROTECTED_MODEL_PREFIXES = (
    "chatglm",
    "glm",
    "qwen3_5",
    "qwen3_next",
    # The production MLA loaders are intentionally outside this migration.
    "deepseek_v2",
    "deepseek_nextn",
    "deepseek_v4",
)

# Root MoE v2 methods either dispatch fused expert checkpoint names themselves,
# delegate to a model-local child loader, or load model-local expert parameters.
MOE_V2_PATHS = (
    ("afmoe.py", "AfmoeForCausalLM", "local"),
    ("bailing_moe.py", "BailingMoEForCausalLM", "dispatch"),
    ("cohere2_moe.py", "Cohere2MoeForCausalLM", "dispatch"),
    ("dbrx.py", "DbrxForCausalLM", "local"),
    ("deepseek.py", "DeepseekForCausalLM", "local"),
    ("ernie4.py", "Ernie4_5_MoeForCausalLM", "dispatch"),
    ("exaone_moe.py", "ExaoneMoEForCausalLM", "dispatch"),
    ("granitemoe.py", "GraniteMoeForCausalLM", "delegated"),
    ("hunyuan.py", "HunYuanMoEV1ForCausalLM", "dispatch"),
    ("hunyuan_v3.py", "HYV3ForCausalLM", "dispatch"),
    ("hunyuan_v3_nextn.py", "HYV3ForCausalLMNextN", "dispatch"),
    ("laguna.py", "LagunaForCausalLM", "dispatch"),
    ("minicpm.py", "MiniCPMForCausalLM", "local"),
    ("minimax_m2.py", "MiniMaxM2ForCausalLM", "dispatch"),
    ("minimax_m3.py", "MiniMaxM3SparseForCausalLM", "dispatch"),
    ("mixtral.py", "MixtralForCausalLM", "delegated"),
    ("mixtral_quant.py", "QuantMixtralForCausalLM", "local"),
    ("olmoe.py", "OlmoeForCausalLM", "delegated"),
    ("phimoe.py", "PhiMoEForCausalLM", "delegated"),
    ("qwen2_moe.py", "Qwen2MoeForCausalLM", "delegated"),
    ("qwen3_moe.py", "Qwen3MoeForCausalLM", "delegated"),
    ("sdar_moe.py", "SDARMoeForCausalLM", "dispatch"),
    ("xverse_moe.py", "XverseMoeForCausalLM", "local"),
)


def _tree(filename: str) -> ast.Module:
    path = MODEL_ROOT / filename
    return ast.parse(path.read_text(), filename=str(path))


def _classes(filename: str) -> dict[str, ast.ClassDef]:
    return {
        node.name: node
        for node in _tree(filename).body
        if isinstance(node, ast.ClassDef)
    }


def _methods(class_node: ast.ClassDef) -> dict[str, ast.FunctionDef]:
    return {
        node.name: node
        for node in class_node.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }


def _source(filename: str, node: ast.AST) -> str:
    return ast.get_source_segment((MODEL_ROOT / filename).read_text(), node) or ""


def _loader_cases() -> tuple[tuple[str, str], ...]:
    cases = []
    for filename in CHANGED_MODEL_FILES:
        for class_name, class_node in _classes(filename).items():
            methods = _methods(class_node)
            if {"_legacy_load_weights", "_load_weights_v2"} <= methods.keys():
                cases.append((filename, class_name))
    return tuple(cases)


LOADER_CASES = _loader_cases()


class _DropImports(ast.NodeTransformer):
    def visit_Import(self, node):
        return None

    def visit_ImportFrom(self, node):
        return None


def _compile_gate(filename: str, class_name: str):
    method = copy.deepcopy(_methods(_classes(filename)[class_name])["load_weights"])
    method.decorator_list = []
    method = _DropImports().visit(method)
    ast.fix_missing_locations(method)
    synthetic_class = ast.ClassDef(
        name="_SyntheticLoader",
        bases=[],
        keywords=[],
        body=[method],
        decorator_list=[],
    )
    module = ast.Module(
        body=[
            ast.ImportFrom(
                module="__future__", names=[ast.alias("annotations")], level=0
            ),
            synthetic_class,
        ],
        type_ignores=[],
    )
    ast.fix_missing_locations(module)
    namespace = {}
    exec(compile(module, filename, "exec"), namespace)
    return namespace["_SyntheticLoader"].load_weights


def _run_gate(filename: str, class_name: str, enabled: bool):
    calls = []

    def record(kind):
        def helper(*args, **kwargs):
            calls.append((kind, args, kwargs))
            return kind

        return helper

    instance = SimpleNamespace(
        _legacy_load_weights=record("legacy"),
        _load_weights_v2=record("v2"),
    )
    method = _compile_gate(filename, class_name)
    method.__globals__["envs"] = SimpleNamespace(
        SGLANG_ENABLE_WEIGHT_LOADER_V2=SimpleNamespace(get=lambda: enabled)
    )
    method(instance, iter(()))
    return calls


def _called_name_helpers(function: ast.FunctionDef) -> set[str]:
    return {
        call.func.id
        for call in ast.walk(function)
        if isinstance(call, ast.Call) and isinstance(call.func, ast.Name)
    }


def _bound_names(module: ast.Module, function: ast.FunctionDef) -> set[str]:
    bound = set(dir(builtins))
    for node in module.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            bound.add(node.name)
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            for alias in node.names:
                bound.add(alias.asname or alias.name.split(".")[0])
        elif isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            bound.update(
                child.id
                for target in targets
                for child in ast.walk(target)
                if isinstance(child, ast.Name)
            )
    bound.update(arg.arg for arg in function.args.args)
    bound.update(arg.arg for arg in function.args.kwonlyargs)
    if function.args.vararg:
        bound.add(function.args.vararg.arg)
    if function.args.kwarg:
        bound.add(function.args.kwarg.arg)
    for node in ast.walk(function):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            for alias in node.names:
                bound.add(alias.asname or alias.name.split(".")[0])
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            bound.add(node.name)
        elif isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
            bound.add(node.id)
    return bound


def test_pr2_scope_is_exact_and_excludes_protected_model_families():
    assert len(CHANGED_MODEL_FILES) == len(set(CHANGED_MODEL_FILES)) == 69
    assert all((MODEL_ROOT / filename).is_file() for filename in CHANGED_MODEL_FILES)
    assert not {
        filename
        for filename in CHANGED_MODEL_FILES
        if filename.startswith(PROTECTED_MODEL_PREFIXES)
    }


def test_removed_standard_moe_root_helper_has_no_production_references():
    references = [
        path.relative_to(REPO_ROOT)
        for path in (REPO_ROOT / "python/sglang").rglob("*.py")
        if "load_standard_moe_root_weights_v2" in path.read_text()
    ]
    assert references == []


def test_all_changed_parent_loaders_are_discovered():
    assert len(LOADER_CASES) == 58


@pytest.mark.parametrize(
    ("filename", "class_name"),
    LOADER_CASES,
    ids=lambda value: value.removesuffix(".py"),
)
def test_parent_loader_has_structural_v2_gate(filename, class_name):
    methods = _methods(_classes(filename)[class_name])
    gate = methods["load_weights"]
    attributes = {
        node.attr for node in ast.walk(gate) if isinstance(node, ast.Attribute)
    }
    source = _source(filename, gate)

    assert "SGLANG_ENABLE_WEIGHT_LOADER_V2" in source
    assert {"_legacy_load_weights", "_load_weights_v2"} <= attributes


@pytest.mark.parametrize(
    ("filename", "class_name"),
    LOADER_CASES,
    ids=lambda value: value.removesuffix(".py"),
)
@pytest.mark.parametrize("enabled", [False, True], ids=["legacy", "v2"])
def test_parent_loader_runtime_gate_selects_one_path(filename, class_name, enabled):
    calls = _run_gate(filename, class_name, enabled)
    assert [kind for kind, _, _ in calls] == ["v2" if enabled else "legacy"]


@pytest.mark.parametrize(
    ("filename", "class_name", "strategy"),
    MOE_V2_PATHS,
    ids=lambda value: value.removesuffix(".py"),
)
def test_moe_v2_path_uses_dispatch_or_model_local_loader(
    filename, class_name, strategy
):
    module = _tree(filename)
    methods = _methods(_classes(filename)[class_name])
    v2_source = _source(filename, methods["_load_weights_v2"])
    file_source = (MODEL_ROOT / filename).read_text()

    if strategy == "dispatch":
        assert "ExpertParamsDispatch" in v2_source
    elif strategy == "delegated":
        assert "AutoWeightsLoader" in v2_source
        assert "ExpertParamsDispatch" in file_source
    else:
        assert (
            "AutoWeightsLoader" in v2_source
            or "weight_loader" in v2_source
            or "expert_mapping" in v2_source
        )

    undefined_helpers = _called_name_helpers(
        methods["_load_weights_v2"]
    ) - _bound_names(module, methods["_load_weights_v2"])
    assert undefined_helpers == set()
