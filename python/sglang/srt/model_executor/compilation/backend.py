import torch
import ast
import dataclasses
import os
import pprint
import time
from collections.abc import Sequence
from contextlib import contextmanager
from typing import Any, Callable, Optional
import logging

import torch
import torch.fx as fx
from torch._dispatch.python import enable_python_dispatcher
from .compiler_interface import InductorAdaptor
from .compilation_counter import compilation_counter
from .pass_manager import PostGradPassManager
from .cuda_piecewise_backend import CUDAPiecewiseBackend

logger = logging.getLogger(__name__)

def make_compiler():
    return InductorAdaptor()

class CompilerManager:
    def __init__(self,):
        self.cache = dict()
        self.is_cache_updated = False
        self.compiler = make_compiler()
    
    def compute_hash(self):
        return self.compiler.compute_hash()

    def initialize_cache(self,
                         cache_dir: str,
                         disable_cache: bool = False,
                         prefix: str = ""):
        self.disable_cache = disable_cache
        self.cache_dir = cache_dir
        self.cache_file_path = os.path.join(cache_dir, "sglang_compile_cache.py")

        if not disable_cache and os.path.exists(self.cache_file_path):
            with open(self.cache_file_path) as f:
                self.cache = ast.literal_eval(f.read())

        self.compiler.initialize_cache(cache_dir=cache_dir,
                                       disable_cache=disable_cache,
                                       prefix=prefix)

    def save_to_file(self):
        if self.disable_cache or not self.is_cache_updated:
            return
        printer = pprint.PrettyPrinter(indent=4)
        data = printer.pformat(self.cache)
        with open(self.cache_file_path, "w") as f:
            f.write(data)

    def load(self,
             graph: fx.GraphModule,
             example_inputs: list[Any],
             graph_index: int,
             runtime_shape: Optional[int] = None) -> Optional[Callable]:
        handle = self.cache[(runtime_shape, graph_index, self.compiler.name)]
        compiled_graph = self.compiler.load(handle, graph, example_inputs,
                                            graph_index, runtime_shape)
        if runtime_shape is None:
            logger.debug(
                "Directly load the %s-th graph for dynamic shape from %s via "
                "handle %s", graph_index, self.compiler.name, handle)
        else:
            logger.debug(
                "Directly load the %s-th graph for shape %s from %s via "
                "handle %s", graph_index, str(runtime_shape),
                self.compiler.name, handle)
        return compiled_graph

    def compile(self,
                graph: fx.GraphModule,
                example_inputs,
                inductor_config: dict[str, Any],
                graph_index: int = 0,
                num_graphs: int = 1,
                runtime_shape: Optional[int] = None) -> Any:
        if graph_index == 0:
            # before compiling the first graph, record the start time
            global compilation_start_time
            compilation_start_time = time.time()

        compilation_counter.num_backend_compilations += 1

        compiled_graph = None

        # try to load from the cache
        # compiled_graph = self.load(graph, example_inputs, graph_index,
        #                            runtime_shape)
        # if compiled_graph is not None:
        #     if graph_index == num_graphs - 1:
        #         # after loading the last graph for this shape, record the time.
        #         # there can be multiple graphs due to piecewise compilation.
        #         now = time.time()
        #         elapsed = now - compilation_start_time
        #         if runtime_shape is None:
        #             logger.info(
        #                 "Directly load the compiled graph(s) for dynamic shape "
        #                 "from the cache, took %.3f s", elapsed)
        #         else:
        #             logger.info(
        #                 "Directly load the compiled graph(s) for shape %s "
        #                 "from the cache, took %.3f s", str(runtime_shape),
        #                 elapsed)
        #     return compiled_graph

        # no compiler cached the graph, or the cache is disabled,
        # we need to compile it
        if isinstance(self.compiler, InductorAdaptor):
            # Let compile_fx generate a key for us
            maybe_key = None
        else:
            maybe_key = \
                f"artifact_shape_{runtime_shape}_subgraph_{graph_index}"
        compiled_graph, handle = self.compiler.compile(
            graph, example_inputs, inductor_config,runtime_shape,
            maybe_key)

        assert compiled_graph is not None, "Failed to compile the graph"

        # store the artifact in the cache
        if handle is not None:
            self.cache[(runtime_shape, graph_index,
                        self.compiler.name)] = handle
            compilation_counter.num_cache_entries_updated += 1
            self.is_cache_updated = True
            if graph_index == 0:
                # adds some info logging for the first graph
                if runtime_shape is None:
                    logger.info(
                        "Cache the graph for dynamic shape for later use")
                else:
                    logger.info("Cache the graph of shape %s for later use",
                                str(runtime_shape))
            if runtime_shape is None:
                logger.debug(
                    "Store the %s-th graph for dynamic shape from %s via "
                    "handle %s", graph_index, self.compiler.name, handle)
            else:
                logger.debug(
                    "Store the %s-th graph for shape %s from %s via handle %s",
                    graph_index, str(runtime_shape), self.compiler.name,
                    handle)

        # after compiling the last graph, record the end time
        if graph_index == num_graphs - 1:
            now = time.time()
            elapsed = now - compilation_start_time
            if runtime_shape is None:
                logger.info("Compiling a graph for dynamic shape takes %.2f s",
                            elapsed)
            else:
                logger.info("Compiling a graph for shape %s takes %.2f s",
                            runtime_shape, elapsed)

        return compiled_graph

@dataclasses.dataclass
class SplitItem:
    submod_name: str
    graph_id: int
    is_splitting_graph: bool
    graph: fx.GraphModule

def split_graph(graph: fx.GraphModule,
                ops: list[str]) -> tuple[fx.GraphModule, list[SplitItem]]:
    # split graph by ops
    subgraph_id = 0
    node_to_subgraph_id = {}
    split_op_graphs = []
    for node in graph.graph.nodes:
        if node.op in ("output", "placeholder"):
            continue
        if node.op == 'call_function' and str(node.target) in ops:
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
        graph,
        None,
        lambda node: node_to_subgraph_id[node],
        keep_original_order=True)

    outputs = []

    names = [name for (name, module) in split_gm.named_modules()]

    for name in names:
        if "." in name or name == "":
            # recursive child module or the root module
            continue

        module = getattr(split_gm, name)

        graph_id = int(name.replace("submod_", ""))
        outputs.append(
            SplitItem(name, graph_id, (graph_id in split_op_graphs), module))

    # sort by intetger graph_id, rather than string name
    outputs.sort(key=lambda x: x.graph_id)

    return split_gm, outputs


# we share the global graph pool among all the backends
global_graph_pool = None

compilation_start_time = 0.0

class PiecewiseCompileInterpreter(torch.fx.Interpreter):
    def __init__(self, module: torch.fx.GraphModule,
                 compile_submod_names: list[str],
                 inductor_config: dict[str, Any],
                 graph_pool, sglang_backend: "SGLangBackend"):
        super().__init__(module)
        from torch._guards import detect_fake_mode
        self.fake_mode = detect_fake_mode()
        self.compile_submod_names = compile_submod_names
        self.graph_pool = graph_pool
        self.sglang_backend = sglang_backend
        # When True, it annoyingly dumps the torch.fx.Graph on errors.
        self.extra_traceback = False
        self.inductor_config = inductor_config

    def run(self, *args):
        fake_args = [
            self.fake_mode.from_tensor(t) if isinstance(t, torch.Tensor) else t
            for t in args
        ]
        with self.fake_mode, enable_python_dispatcher():
            return super().run(*fake_args)

    def call_module(self, target: torch.fx.node.Target,
                    args: tuple[torch.fx.node.Argument,
                                ...], kwargs: dict[str, Any]) -> Any:
        assert isinstance(target, str)
        output = super().call_module(target, args, kwargs)

        if target in self.compile_submod_names:
            index = self.compile_submod_names.index(target)
            submod = self.fetch_attr(target)
            print(f"args length: {len(args)}")
            sym_shape_indices = [
                i for i, x in enumerate(args) if isinstance(x, torch.SymInt)
            ]
            global compilation_start_time
            compiled_graph_for_dynamic_shape = self.sglang_backend.\
                compiler_manager.compile(
                submod,
                args,
                self.inductor_config,
                graph_index=index,
                num_graphs=len(self.compile_submod_names),
                runtime_shape=None)

            self.module.__dict__[target] = CUDAPiecewiseBackend(
                submod, self.inductor_config, self.graph_pool, index,
                len(self.compile_submod_names), sym_shape_indices,
                compiled_graph_for_dynamic_shape, self.sglang_backend)

            compilation_counter.num_piecewise_capturable_graphs_seen += 1

        return output

model_tag: str = "backbone"

@contextmanager
def set_model_tag(tag: str):
    """Context manager to set the model tag."""
    global model_tag
    assert tag != model_tag, \
        f"Model tag {tag} is the same as the current tag {model_tag}."
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
    
    def __init__(self,):
        global global_graph_pool
        if global_graph_pool is None:
            global_graph_pool = torch.cuda.graph_pool_handle()
        self.graph_pool = global_graph_pool

        self.post_grad_pass_manager = PostGradPassManager()
        self.sym_tensor_indices = []
        self.input_buffers = []

        self.compiler_manager = CompilerManager()
        self.inductor_config = {
            "enable_auto_functionalized_v2": False,
        }

    def configure_post_pass(self):
        self.post_grad_pass_manager.configure()
        self.inductor_config["post_grad_custom_post_pass"] = self.post_grad_pass_manager

    def __call__(self, graph: fx.GraphModule, example_inputs) -> Callable:
        cache_dir = os.path.join(
            "/home/ubuntu/.cache/sglang/",
            "torch_compile_cache",
            "08329392",
        )

        print(f"example_inputs[0]: {example_inputs[0]}")
        print(f"len(example_inputs): {len(example_inputs)}")
        os.makedirs(cache_dir, exist_ok=True)
        rank = 0
        dp_rank = 0
        local_cache_dir = os.path.join(cache_dir, f"rank_{rank}_{dp_rank}",
                                       model_tag)
        os.makedirs(local_cache_dir, exist_ok=True)
        self.compiler_manager.initialize_cache(local_cache_dir, disable_cache=False, prefix="")
        compilation_counter.num_graphs_seen += 1

        assert not self._called, "SGLangBackend can only be called once"

        self.graph = graph
        self.configure_post_pass()

        self.split_gm, self.piecewise_graphs = split_graph(
            graph, ["unified_attention_with_output"])
        

        from torch._dynamo.utils import lazy_format_graph_code

        # depyf will hook lazy_format_graph_code and dump the graph
        # for debugging, no need to print the graph here
        lazy_format_graph_code("before split", self.graph)
        lazy_format_graph_code("after split", self.split_gm)

        compilation_counter.num_piecewise_graphs_seen += len(
            self.piecewise_graphs)
        
        submod_names_to_compile = [
            item.submod_name for item in self.piecewise_graphs
            if not item.is_splitting_graph
        ]

        PiecewiseCompileInterpreter(self.split_gm, submod_names_to_compile,
                                    self.inductor_config,
                                    self.graph_pool, self).run(*example_inputs)
        
        graph_path = os.path.join(local_cache_dir, "computation_graph.py")
        if not os.path.exists(graph_path):
            # code adapted from https://github.com/thuml/depyf/blob/dab831108a752d1facc00acdd6d4243891845c37/depyf/explain/patched_lazy_format_graph_code.py#L30 # noqa
            # use `print_readable` because it can include submodules
            src = "from __future__ import annotations\nimport torch\n" + \
                self.split_gm.print_readable(print_output=False)
            src = src.replace("<lambda>", "GraphModule")
            with open(graph_path, "w") as f:
                f.write(src)

            logger.debug("Computation graph saved to %s", graph_path)
        
        self._called = True
        return self.split_gm

        if not self.compilation_config.use_cudagraph or \
            not self.compilation_config.cudagraph_copy_inputs:
            return self.split_gm

        from torch._guards import detect_fake_mode
        fake_mode = detect_fake_mode()
        fake_args = [
            fake_mode.from_tensor(t) if isinstance(t, torch.Tensor) else t
            for t in example_inputs
        ]
        
        from torch.fx.experimental.symbolic_shapes import is_symbolic
        self.sym_tensor_indices = [
            i for i, x in enumerate(fake_args)
            if isinstance(x, torch._subclasses.fake_tensor.FakeTensor) and \
                any(is_symbolic(d) for d in x.size())
        ]
        
        self.input_buffers = [
            example_inputs[x].clone() for x in self.sym_tensor_indices
        ]
        
        def copy_and_call(*args):
            list_args = list(args)
            for i, index in enumerate(self.sym_tensor_indices):
                runtime_tensor = list_args[index]
                runtime_shape = runtime_tensor.shape[0]
                static_tensor = self.input_buffers[i][:runtime_shape]
                
        return copy_and_call