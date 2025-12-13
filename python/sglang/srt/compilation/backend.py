# Adapted from https://github.com/vllm-project/vllm/blob/v0.10.0/vllm/compilation/backend.py


import ast
import dataclasses
import logging
import os
import pprint
import time
from collections.abc import Sequence
from contextlib import contextmanager
from typing import Any, Callable, Optional

import torch
import torch.fx as fx
from torch._dispatch.python import enable_python_dispatcher

from sglang.srt.compilation.compilation_config import CompilationConfig
from sglang.srt.compilation.compilation_counter import compilation_counter
from sglang.srt.compilation.compiler_interface import EagerAdapter, InductorAdaptor
from sglang.srt.compilation.cuda_piecewise_backend import CUDAPiecewiseBackend
from sglang.srt.compilation.npu_piecewise_backend import NPUPiecewiseBackend
from sglang.srt.compilation.pass_manager import PostGradPassManager
from sglang.srt.utils.common import is_npu, rank0_log

logger = logging.getLogger(__name__)


SPLIT_OPS = [
    "sglang.unified_attention_with_output",
    "sglang.gdn_with_output",
]


def add_split_ops(ops):
    SPLIT_OPS.extend(ops)


def make_compiler(config: CompilationConfig):
    if config.compiler == "eager":
        return EagerAdapter()
    elif config.compiler == "inductor":
        return InductorAdaptor()
    else:
        raise ValueError(f"Unknown compiler: {config.compiler}")


def make_backend(
    graph: fx.GraphModule,
    compile_config: CompilationConfig,
    inductor_config: dict[str, Any],
    graph_pool: Any,
    piecewise_compile_index: int,
    total_piecewise_compiles: int,
    sym_shape_indices: list[int],
    compiled_graph_for_general_shape: Callable,
    sglang_backend,
):

    backend_cls = CUDAPiecewiseBackend if not is_npu() else NPUPiecewiseBackend
    return backend_cls(
        graph,
        compile_config,
        inductor_config,
        graph_pool,
        piecewise_compile_index,
        total_piecewise_compiles,
        sym_shape_indices,
        compiled_graph_for_general_shape,
        sglang_backend,
    )


class CompilerManager:
    def __init__(
        self,
        config: CompilationConfig,
    ):
        self.cache = dict()
        self.is_cache_updated = False
        self.compiler = make_compiler(config)

    def compute_hash(self):
        return self.compiler.compute_hash()

    def initialize_cache(
        self, cache_dir: str, disable_cache: bool = False, prefix: str = ""
    ):
        self.disable_cache = disable_cache
        self.cache_dir = cache_dir
        self.cache_file_path = os.path.join(cache_dir, "sglang_compile_cache.py")

        if not disable_cache and os.path.exists(self.cache_file_path):
            with open(self.cache_file_path) as f:
                self.cache = ast.literal_eval(f.read())

        self.compiler.initialize_cache(
            cache_dir=cache_dir, disable_cache=disable_cache, prefix=prefix
        )

    def save_to_file(self):
        if self.disable_cache or not self.is_cache_updated:
            return
        printer = pprint.PrettyPrinter(indent=4)
        data = printer.pformat(self.cache)
        with open(self.cache_file_path, "w") as f:
            f.write(data)

    def load(
        self,
        graph: fx.GraphModule,
        example_inputs: list[Any],
        graph_index: int,
        runtime_shape: Optional[int] = None,
    ) -> Optional[Callable]:
        handle = self.cache[(runtime_shape, graph_index, self.compiler.name)]
        compiled_graph = self.compiler.load(
            handle, graph, example_inputs, graph_index, runtime_shape
        )
        if runtime_shape is None:
            logger.debug(
                "Directly load the %s-th graph for dynamic shape from %s via "
                "handle %s",
                graph_index,
                self.compiler.name,
                handle,
            )
        else:
            logger.debug(
                "Directly load the %s-th graph for shape %s from %s via " "handle %s",
                graph_index,
                str(runtime_shape),
                self.compiler.name,
                handle,
            )
        return compiled_graph

    def compile(
        self,
        graph: fx.GraphModule,
        example_inputs,
        inductor_config: dict[str, Any],
        graph_index: int = 0,
        num_graphs: int = 1,
        runtime_shape: Optional[int] = None,
    ) -> Any:
        if graph_index == 0:
            # before compiling the first graph, record the start time
            global compilation_start_time
            compilation_start_time = time.time()

        compilation_counter.num_backend_compilations += 1

        compiled_graph = None

        # TODO(Yuwei): support cache loading

        # no compiler cached the graph, or the cache is disabled,
        # we need to compile it
        if isinstance(self.compiler, InductorAdaptor):
            maybe_key = None
        else:
            maybe_key = f"artifact_shape_{runtime_shape}_subgraph_{graph_index}"
        compiled_graph, handle = self.compiler.compile(
            graph, example_inputs, inductor_config, runtime_shape, maybe_key
        )

        assert compiled_graph is not None, "Failed to compile the graph"

        # store the artifact in the cache
        if handle is not None:
            self.cache[(runtime_shape, graph_index, self.compiler.name)] = handle
            compilation_counter.num_cache_entries_updated += 1
            self.is_cache_updated = True
            if graph_index == 0:
                # adds some info logging for the first graph
                if runtime_shape is None:
                    logger.info("Cache the graph for dynamic shape for later use")
                else:
                    logger.info(
                        "Cache the graph of shape %s for later use", str(runtime_shape)
                    )
            if runtime_shape is None:
                logger.debug(
                    "Store the %s-th graph for dynamic shape from %s via " "handle %s",
                    graph_index,
                    self.compiler.name,
                    handle,
                )
            else:
                logger.debug(
                    "Store the %s-th graph for shape %s from %s via handle %s",
                    graph_index,
                    str(runtime_shape),
                    self.compiler.name,
                    handle,
                )

        # after compiling the last graph, record the end time
        if graph_index == num_graphs - 1:
            now = time.time()
            elapsed = now - compilation_start_time
            if runtime_shape is None:
                logger.info("Compiling a graph for dynamic shape takes %.2f s", elapsed)
            else:
                logger.info(
                    "Compiling a graph for shape %s takes %.2f s",
                    runtime_shape,
                    elapsed,
                )

        return compiled_graph


@dataclasses.dataclass
class SplitItem:
    submod_name: str
    graph_id: int
    is_splitting_graph: bool
    graph: fx.GraphModule


def split_graph(
    graph: fx.GraphModule, ops: list[str]
) -> tuple[fx.GraphModule, list[SplitItem]]:
    # split graph by ops
    subgraph_id = 0
    node_to_subgraph_id = {}
    split_op_graphs = []
    for node in graph.graph.nodes:
        if node.op in ("output", "placeholder"):
            continue
        if node.op == "call_function" and str(node.target) in ops:
            subgraph_id += 1
            node_to_subgraph_id[node] = subgraph_id
            split_op_graphs.append(subgraph_id)
            subgraph_id += 1
        else:
            node_to_subgraph_id[node] = subgraph_id

    # `keep_original_order` is important!
    # otherwise pytorch might reorder the nodes and
    # the semantics of the graph will change when we
    # have mutations in the graph
    split_gm = torch.fx.passes.split_module.split_module(
        graph, None, lambda node: node_to_subgraph_id[node], keep_original_order=True
    )

    outputs = []

    names = [name for (name, module) in split_gm.named_modules()]

    for name in names:
        if "." in name or name == "":
            # recursive child module or the root module
            continue

        module = getattr(split_gm, name)

        graph_id = int(name.replace("submod_", ""))
        outputs.append(SplitItem(name, graph_id, (graph_id in split_op_graphs), module))

    # sort by intetger graph_id, rather than string name
    outputs.sort(key=lambda x: x.graph_id)

    return split_gm, outputs


# we share the global graph pool among all the backends
global_graph_pool = None

compilation_start_time = 0.0


class PiecewiseCompileInterpreter(torch.fx.Interpreter):
    def __init__(
        self,
        module: torch.fx.GraphModule,
        compile_submod_names: list[str],
        inductor_config: dict[str, Any],
        graph_pool,
        compile_config: CompilationConfig,
        sglang_backend: "SGLangBackend",
    ):
        super().__init__(module)
        from torch._guards import detect_fake_mode

        self.fake_mode = detect_fake_mode()
        self.compile_submod_names = compile_submod_names
        self.graph_pool = graph_pool
        self.sglang_backend = sglang_backend
        # When True, it annoyingly dumps the torch.fx.Graph on errors.
        self.extra_traceback = False
        self.inductor_config = inductor_config
        self.compile_config = compile_config

    def run(self, *args):
        fake_args = [
            self.fake_mode.from_tensor(t) if isinstance(t, torch.Tensor) else t
            for t in args
        ]
        with self.fake_mode, enable_python_dispatcher():
            return super().run(*fake_args)

    def call_module(
        self,
        target: torch.fx.node.Target,
        args: tuple[torch.fx.node.Argument, ...],
        kwargs: dict[str, Any],
    ) -> Any:
        assert isinstance(target, str)
        output = super().call_module(target, args, kwargs)

        if target in self.compile_submod_names:
            index = self.compile_submod_names.index(target)
            submod = self.fetch_attr(target)
            sym_shape_indices = [
                i for i, x in enumerate(args) if isinstance(x, torch.SymInt)
            ]
            global compilation_start_time
            compiled_graph_for_dynamic_shape = (
                self.sglang_backend.compiler_manager.compile(
                    submod,
                    args,
                    self.inductor_config,
                    graph_index=index,
                    num_graphs=len(self.compile_submod_names),
                    runtime_shape=None,
                )
            )

            self.module.__dict__[target] = make_backend(
                submod,
                self.compile_config,
                self.inductor_config,
                self.graph_pool,
                index,
                len(self.compile_submod_names),
                sym_shape_indices,
                compiled_graph_for_dynamic_shape,
                self.sglang_backend,
            )

            compilation_counter.num_piecewise_capturable_graphs_seen += 1

        return output


model_tag: str = "backbone"


@contextmanager
def set_model_tag(tag: str):
    """Context manager to set the model tag."""
    global model_tag
    assert (
        tag != model_tag
    ), f"Model tag {tag} is the same as the current tag {model_tag}."
    old_tag = model_tag
    model_tag = tag
    try:
        yield
    finally:
        model_tag = old_tag


class SGLangBackend:

    graph_pool: Any
    _called: bool = False
    # the graph we compiled
    graph: fx.GraphModule
    # the stiching graph module for all the piecewise graphs
    split_gm: fx.GraphModule
    piecewise_graphs: list[SplitItem]
    returned_callable: Callable
    # Inductor passes to run on the graph pre-defunctionalization
    post_grad_passes: Sequence[Callable]
    sym_tensor_indices: list[int]
    input_buffers: list[torch.Tensor]
    compiler_manager: CompilerManager

    def __init__(
        self,
        config: CompilationConfig,
        graph_pool: Any,
    ):
        rank0_log(f"Initializing SGLangBackend")
        assert graph_pool is not None
        self.graph_pool = graph_pool

        self.post_grad_pass_manager = PostGradPassManager()
        self.sym_tensor_indices = []
        self.input_buffers = []

        self.compiler_manager = CompilerManager(config)
        self.inductor_config = {
            "enable_auto_functionalized_v2": False,
        }
        self.compile_config = config

    def configure_post_pass(self):
        self.post_grad_pass_manager.configure()
        self.inductor_config["post_grad_custom_post_pass"] = self.post_grad_pass_manager

    def __call__(self, graph: fx.GraphModule, example_inputs) -> Callable:
        rank0_log(f"SGLangBackend __call__")
        base_cache_dir = os.path.expanduser(
            os.getenv("SGLANG_CACHE_DIR", "~/.cache/sglang/")
        )

        cache_hash = self.compiler_manager.compute_hash()
        cache_dir = os.path.join(
            base_cache_dir,
            "torch_compile_cache",
            cache_hash,
        )

        os.makedirs(cache_dir, exist_ok=True)
        rank = 0
        dp_rank = 0
        local_cache_dir = os.path.join(cache_dir, f"rank_{rank}_{dp_rank}", model_tag)
        os.makedirs(local_cache_dir, exist_ok=True)
        self.compiler_manager.initialize_cache(
            local_cache_dir, disable_cache=False, prefix=""
        )
        compilation_counter.num_graphs_seen += 1

        assert not self._called, "SGLangBackend can only be called once"

        self.graph = graph
        self.configure_post_pass()

        self.split_gm, self.piecewise_graphs = split_graph(
            graph,
            SPLIT_OPS,
        )
        from torch._dynamo.utils import lazy_format_graph_code

        # depyf will hook lazy_format_graph_code and dump the graph
        # for debugging, no need to print the graph here
        lazy_format_graph_code("before split", self.graph)
        lazy_format_graph_code("after split", self.split_gm)

        compilation_counter.num_piecewise_graphs_seen += len(self.piecewise_graphs)

        submod_names_to_compile = [
            item.submod_name
            for item in self.piecewise_graphs
            if not item.is_splitting_graph
        ]

        PiecewiseCompileInterpreter(
            self.split_gm,
            submod_names_to_compile,
            self.inductor_config,
            self.graph_pool,
            self.compile_config,
            self,
        ).run(*example_inputs)

        rank = torch.distributed.get_rank()

        if rank == 0:
            graph_path = os.path.join(
                local_cache_dir, f"computation_graph_{time.time()}.py"
            )
            if not os.path.exists(graph_path):
                # code adapted from https://github.com/thuml/depyf/blob/dab831108a752d1facc00acdd6d4243891845c37/depyf/explain/patched_lazy_format_graph_code.py#L30 # noqa
                # use `print_readable` because it can include submodules
                src = (
                    "from __future__ import annotations\nimport torch\n"
                    + self.split_gm.print_readable(print_output=False)
                )
                src = src.replace("<lambda>", "GraphModule")
                with open(graph_path, "w") as f:
                    f.write(src)

                rank0_log(f"Computation graph saved to {graph_path}")

        self._called = True
        return self.split_gm
