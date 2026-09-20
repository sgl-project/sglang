"""Run DSpark checkpoint unit methods on hosts without the serving dependency stack.

    .venv/bin/python test/registered/unit/spec/run_dspark_checkpoint_isolated.py

This executes the actual production function/class definitions extracted by AST,
with real Torch, msgspec and CLI metadata modules. The normal tests' small fake
backbone/vocabulary modules replace distributed model construction. Runtime imports,
distributed initialization, safetensors loading and GPU execution are NOT tested.
Use normal pytest on the serving environment for those integration checks.
"""

import argparse
import ast
import dataclasses
import importlib.util
import logging
import math
import numbers
import sys
import types
import typing
import unittest
from pathlib import Path
from unittest.mock import patch as mock_patch

import msgspec
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[4]
SOURCE = ROOT / "python/sglang/srt"


def main() -> int:
    module = types.ModuleType("dspark_checkpoint_isolated")
    sys.modules[module.__name__] = module
    namespace = module.__dict__
    namespace.update(
        torch=torch,
        nn=nn,
        math=math,
        msgspec=msgspec,
        logger=logging.getLogger(__name__),
        Integral=numbers.Integral,
        dataclass=dataclasses.dataclass,
        Any=typing.Any,
        Optional=typing.Optional,
        List=typing.List,
        Tuple=typing.Tuple,
        Iterable=typing.Iterable,
        Callable=typing.Callable,
        unittest=unittest,
        SimpleNamespace=types.SimpleNamespace,
        argparse=argparse,
    )

    def load_module(name, path):
        spec = importlib.util.spec_from_file_location(name, path)
        loaded = importlib.util.module_from_spec(spec)
        sys.modules[name] = loaded
        spec.loader.exec_module(loaded)
        return loaded

    for name in (
        "sglang",
        "sglang.srt",
        "sglang.srt.arg_groups",
        "sglang.srt.arg_groups.fields",
    ):
        package = types.ModuleType(name)
        package.__path__ = []
        sys.modules[name] = package
    cli_utils = load_module(
        "sglang.srt.arg_groups.arg_utils", SOURCE / "arg_groups/arg_utils.py"
    )
    load_module("sglang.srt.arg_groups.choices", SOURCE / "arg_groups/choices.py")
    fields = load_module(
        "sglang.srt.arg_groups.fields.spec", SOURCE / "arg_groups/fields/spec.py"
    )
    namespace.update(
        Spec=fields.Spec,
        add_cli_args_from_dataclass=cli_utils.add_cli_args_from_dataclass,
    )

    def load_definitions(path, names=None):
        tree = ast.parse(path.read_text())
        definitions = [
            node
            for node in tree.body
            if isinstance(node, (ast.FunctionDef, ast.ClassDef))
            and (names is None or node.name in names)
        ]
        body = [
            ast.ImportFrom(
                module="__future__", names=[ast.alias(name="annotations")], level=0
            )
        ] + definitions
        # 以下为安全注释COSEC：只执行下面固定清单内的仓库源码；不接受路径或代码输入。
        compiled = compile(
            ast.fix_missing_locations(ast.Module(body=body, type_ignores=[])),
            str(path),
            "exec",
        )
        exec(compiled, namespace)

    load_definitions(SOURCE / "configs/dspark.py")
    load_definitions(
        SOURCE / "speculative/dflash_utils.py",
        {
            "_cfg_get",
            "_get_text_config",
            "_get_dflash_config",
            "_parse_optional_int",
            "DFlashDraftConfig",
            "parse_dflash_draft_config",
        },
    )
    namespace["DEFAULT_DFLASH_MASK_TOKEN"] = "<mask>"
    load_definitions(SOURCE / "speculative/dspark_components/dspark_config.py")
    namespace.update(
        SUPPORTED_DSPARK_MARKOV_HEAD_TYPES=("vanilla", "gated", "rnn"),
        StepSampler=typing.Callable,
        DFlashDraftModel=nn.Module,
        RaggedVerifyMode=types.SimpleNamespace(STATIC=0),
        read_ragged_verify_mode=lambda: 0,
        _DSPARK_SKIPPED_WEIGHT_PREFIXES=("rotary_emb.",),
    )

    def default_weight_loader(parameter, value):
        with torch.no_grad():
            parameter.copy_(value)

    namespace["default_weight_loader"] = default_weight_loader
    load_definitions(
        SOURCE / "models/dspark.py",
        {
            "run_markov_block",
            "VanillaMarkov",
            "Nemotron35VanillaMarkov",
            "GatedMarkovHead",
            "RNNHead",
            "build_markov_head",
            "build_nemotron_35_markov_head",
            "DSparkConfidenceHead",
            "build_confidence_head",
            "validate_dspark_d2t",
            "DSparkDraftMixin",
        },
    )
    load_definitions(
        SOURCE / "arg_groups/speculative_hook.py", {"handle_speculative_decoding"}
    )
    load_definitions(
        SOURCE / "speculative/dspark_components/dspark_planner.py",
        {"build_markov_embed_stack"},
    )

    def patch(name, value=None, **kwargs):
        if "return_value" in kwargs:
            value = lambda *args, **kw: kwargs["return_value"]
        return mock_patch.dict(namespace, {name.split(".")[-1]: value})

    namespace["patch"] = patch
    load_definitions(Path(__file__).with_name("test_dspark_checkpoint.py"))
    result = unittest.TextTestRunner(verbosity=2).run(
        unittest.defaultTestLoader.loadTestsFromModule(module)
    )
    print(
        "ISOLATED CPU CHECKS ONLY: server imports, distributed initialization and GPU execution were not tested."
    )
    return int(not result.wasSuccessful())


if __name__ == "__main__":
    raise SystemExit(main())
