import multiprocessing
from concurrent.futures import ThreadPoolExecutor
from queue import Queue
from typing import List, Union, Optional

from sglang.global_config import global_config
from sglang.lang.interpreter import ProgramState, StreamExecutor, cache_program
from sglang.lang.ir import (
    SglArgument,
    SglConstantText,
    SglExpr,
    SglSamplingParams,
    SglVariable,
)


def compile_func(function, backend):
    tracer = function.trace(backend=backend)
    compiler = CompiledFunction(tracer, function)
    return compiler


class CompiledFunction:
    def __init__(self, tracer, function):
        self.function = function

        self.last_node = CompGraphNode(tracer.last_node)
        self.expr_to_node = {}
        self.build_graph(tracer)
        self.topological_sort()

    def build_graph(self, tracer):
        self.nodes = [self.last_node]
        self.expr_to_node[tracer.last_node] = self.nodes[-1]

        rename_pid = {}

        visited = set([tracer.last_node])
        head = 0
        while head < len(self.nodes):
            cur_node = self.nodes[head]

            # add prev node
            prev_node = cur_node.expr.prev_node
            if prev_node is not None:
                if prev_node not in visited:
                    visited.add(prev_node)
                    self.nodes.append(CompGraphNode(prev_node))
                    self.expr_to_node[prev_node] = self.nodes[-1]
                cur_node.prev_node = self.expr_to_node[prev_node]
                self.expr_to_node[prev_node].add_next_node(cur_node)

            # add source node
            if isinstance(cur_node.expr, SglVariable):
                if cur_node.expr.name in tracer.variables:
                    source = tracer.variables[cur_node.expr.name].source
                else:
                    source = cur_node.expr.source
                if source not in visited:
                    visited.add(source)
                    self.nodes.append(CompGraphNode(source))
                    self.expr_to_node[source] = self.nodes[-1]
                cur_node.source_node = self.expr_to_node[source]
                self.expr_to_node[source].add_next_node(cur_node)
            head += 1

            # rename pid
            if cur_node.expr.pid not in rename_pid:
                rename_pid[cur_node.expr.pid] = len(rename_pid)
            cur_node.expr.pid = rename_pid[cur_node.expr.pid]

    def topological_sort(self):
        prevd = {}
        cand = Queue()
        for x in self.nodes:
            prevd[x] = (x.prev_node is not None) + (x.source_node is not None)
            if prevd[x] == 0:
                cand.put(x)
        new_list = []
        while cand.qsize() > 0:
            head = cand.get()
            new_list.append(head)
            for x in head.next_nodes:
                prevd[x] -= 1
                if prevd[x] == 0:
                    cand.put(x)
        self.nodes = new_list

    def print_graph(
        self,
    ):
        for node in self.nodes:
            print(node)

    def run_internal(
        self,
        backend,
        kwargs,
        default_sampling_para,
    ):
        stream_executor_ids = set([x.expr.pid for x in self.nodes])
        stream_executors = {}
        for x in stream_executor_ids:
            arguments = kwargs if x == self.last_node.expr.pid else {}
            stream_executors[x] = StreamExecutor(
                backend, arguments, default_sampling_para, None, False
            )
        for node in self.nodes:
            se_id = node.expr.pid
            expr = node.expr
            if isinstance(expr, SglVariable):
                # Make a copy for SglVariable
                expr = SglVariable(expr.name, expr.source)
                expr.source_stream_executor = stream_executors[
                    node.source_node.expr.pid
                ]
            elif isinstance(expr, SglArgument):
                # Substitute SglArgument
                expr = kwargs[expr.name]
            stream_executors[se_id].submit(expr)
        for stream_executor in stream_executors.values():
            stream_executor.end()
        return ProgramState(stream_executors[self.last_node.expr.pid])

    def run(
        self,
        *,
        max_new_tokens: int = 128,
        stop: Union[str, List[str]] = (),
        temperature: float = 1.0,
        top_p: float = 1.0,
        top_k: int = -1,
        min_p: float = 0.0,
        frequency_penalty: float = 0.0,
        presence_penalty: float = 0.0,
        dry_multiplier: float = 0.0,
        dry_base: float = 0.0,
        dry_allowed_length: int = 2,
        dry_penalty_last_n: int = 0,
        dry_sequence_breakers: Optional[List[str]] = [],
        backend=None,
        **kwargs,
    ):
        backend = backend or global_config.default_backend

        kwargs.update(self.function.bind_arguments)

        default_sampling_para = SglSamplingParams(
            max_new_tokens=max_new_tokens,
            stop=stop,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            min_p=min_p,
            frequency_penalty=frequency_penalty,
            presence_penalty=presence_penalty,
            dry_multiplier=dry_multiplier,
            dry_base=dry_base,
            dry_allowed_length=dry_allowed_length,
            dry_penalty_last_n=dry_penalty_last_n,
            dry_sequence_breakers=dry_sequence_breakers,
        )

        return self.run_internal(backend, kwargs, default_sampling_para)

    def run_batch(
        self,
        batch_kwargs,
        *,
        max_new_tokens: int = 128,
        stop: Union[str, List[str]] = (),
        temperature: float = 1.0,
        top_p: float = 1.0,
        top_k: int = -1,
        min_p: float = 0.0,
        frequency_penalty: float = 0.0,
        presence_penalty: float = 0.0,
        dry_multiplier: float = 0.0,
        dry_base: float = 0.0,
        dry_allowed_length: int = 2,
        dry_penalty_last_n: int = 0,
        dry_sequence_breakers: Optional[List[str]] = [],
        backend=None,
        num_threads: Union[str, int] = "auto",
    ):
        assert isinstance(batch_kwargs, (list, tuple))
        if len(batch_kwargs) == 0:
            return []
        assert isinstance(batch_kwargs[0], dict)

        backend = backend or global_config.default_backend

        default_sampling_para = SglSamplingParams(
            max_new_tokens=max_new_tokens,
            stop=stop,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            min_p=min_p,
            frequency_penalty=frequency_penalty,
            presence_penalty=presence_penalty,
            dry_multiplier=dry_multiplier,
            dry_base=dry_base,
            dry_allowed_length=dry_allowed_length,
            dry_penalty_last_n=dry_penalty_last_n,
            dry_sequence_breakers=dry_sequence_breakers,
        )

        # Extract prefix by tracing and cache it
        if len(batch_kwargs) > 1:
            cache_program(self.function, backend)

        # Run all programs
        if num_threads == "auto":
            num_threads = multiprocessing.cpu_count()
        num_threads = min(num_threads, len(batch_kwargs))

        if num_threads == 1:
            rets = []
            for arguments in batch_kwargs:
                rets.append(
                    self.run_internal(backend, arguments, default_sampling_para)
                )
        else:
            with ThreadPoolExecutor(num_threads) as executor:
                futures = []
                for arguments in batch_kwargs:
                    futures.append(
                        executor.submit(
                            self.run_internal, backend, arguments, default_sampling_para
                        )
                    )
                rets = [f.result() for f in futures]
            rets[-1].sync()

        return rets


class CompGraphNode:
    def __init__(
        self, expr: SglExpr, prev_node=None, next_nodes=None, source_node=None
    ):
        self.expr = expr
        self.next_nodes = next_nodes or []
        self.prev_node = prev_node
        self.source_node = source_node

    def add_next_node(self, other):
        self.next_nodes.append(other)

    def __repr__(self):
        re = f"stream {self.expr.pid:2d}: "
        re += f"%{self.expr.node_id} = "
        if self.prev_node is not None:
            re += f"%{self.prev_node.expr.node_id} + "
        re += repr(self.expr)
        return re
