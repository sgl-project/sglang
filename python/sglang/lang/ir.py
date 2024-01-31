"""The intermediate representation."""

import dataclasses
import inspect
import warnings
from typing import List, Optional, Union

from sglang.global_config import global_config

REGEX_INT = r"[-+]?[0-9]+"
REGEX_FLOAT = r"[-+]?[0-9]*\.?[0-9]+"
REGEX_BOOL = r"(True|False)"
REGEX_STRING = r"\"[\w\d\s]*\""  # bugs with regex r"\".*\"" in interegular pkg


@dataclasses.dataclass
class SglSamplingParams:
    max_new_tokens: int = 16
    stop: Union[str, List[str]] = ()
    temperature: float = 1.0
    top_p: float = 1.0
    top_k: int = -1  # -1 means disable
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0
    ignore_eos: bool = False

    # for constrained generation, not included in to_xxx_kwargs
    dtype: Optional[str] = None
    regex: Optional[str] = None

    def clone(self):
        return SglSamplingParams(
            self.max_new_tokens,
            self.stop,
            self.temperature,
            self.top_p,
            self.top_k,
            self.frequency_penalty,
            self.presence_penalty,
        )

    def to_openai_kwargs(self):
        # OpenAI does not support top_k, so we drop it here
        if self.regex is not None:
            warnings.warn("Regular expression is not supported in the OpenAI backend.")
        return {
            "max_tokens": self.max_new_tokens,
            "stop": self.stop or None,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "frequency_penalty": self.frequency_penalty,
            "presence_penalty": self.presence_penalty,
        }

    def to_vertexai_kwargs(self):
        if self.regex is not None:
            warnings.warn(
                "Regular expression is not supported in the VertexAI backend."
            )
        return {
            "candidate_count": 1,
            "max_output_tokens": self.max_new_tokens,
            "stop_sequences": self.stop,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "top_k": self.top_k if self.top_k > 0 else None,
        }

    def to_anthropic_kwargs(self):
        # Anthropic does not support frequency_penalty or presence_penalty, so we drop it here
        if self.regex is not None:
            warnings.warn(
                "Regular expression is not supported in the Anthropic backend."
            )
        return {
            "max_tokens_to_sample": self.max_new_tokens,
            "stop_sequences": self.stop
            if isinstance(self.stop, (list, tuple))
            else [self.stop],
            "temperature": self.temperature,
            "top_p": self.top_p,
            "top_k": self.top_k,
        }

    def to_srt_kwargs(self):
        return {
            "max_new_tokens": self.max_new_tokens,
            "stop": self.stop,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "top_k": self.top_k,
            "frequency_penalty": self.frequency_penalty,
            "presence_penalty": self.presence_penalty,
            "ignore_eos": self.ignore_eos,
            "regex": self.regex,
        }


class SglFunction:
    def __init__(self, func, api_num_spec_tokens=None, bind_arguments=None):
        self.func = func
        self.api_num_spec_tokens = api_num_spec_tokens
        self.bind_arguments = bind_arguments or {}
        self.pin_prefix_rid = None

        # Parse arguments
        argspec = inspect.getfullargspec(func)
        assert argspec.args[0] == "s", 'The first argument must be "s"'
        self.arg_names = argspec.args[1:]

    def bind(self, **kwargs):
        assert all(key in self.arg_names for key in kwargs)

        new_bind_dict = {**self.bind_arguments, **kwargs}
        return SglFunction(self.func, bind_arguments=new_bind_dict)

    def run(
        self,
        *args,
        max_new_tokens: int = 16,
        stop: Union[str, List[str]] = (),
        temperature: float = 1.0,
        top_p: float = 1.0,
        top_k: int = -1,
        frequency_penalty: float = 0.0,
        presence_penalty: float = 0.0,
        ignore_eos: bool = False,
        stream: bool = False,
        backend=None,
        **kwargs,
    ):
        from sglang.lang.interpreter import run_program

        default_sampling_para = SglSamplingParams(
            max_new_tokens=max_new_tokens,
            stop=stop,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            frequency_penalty=frequency_penalty,
            presence_penalty=presence_penalty,
            ignore_eos=ignore_eos,
        )
        backend = backend or global_config.default_backend
        return run_program(self, backend, args, kwargs, default_sampling_para, stream)

    def run_batch(
        self,
        batch_kwargs,
        *,
        max_new_tokens: int = 16,
        stop: Union[str, List[str]] = (),
        temperature: float = 1.0,
        top_p: float = 1.0,
        top_k: int = -1,
        frequency_penalty: float = 0.0,
        presence_penalty: float = 0.0,
        ignore_eos: bool = False,
        backend=None,
        num_threads: Union[str, int] = "auto",
        progress_bar: bool = False,
    ):
        from sglang.lang.interpreter import run_program_batch

        assert isinstance(batch_kwargs, (list, tuple))
        if len(batch_kwargs) == 0:
            return []
        assert isinstance(batch_kwargs[0], dict)

        default_sampling_para = SglSamplingParams(
            max_new_tokens=max_new_tokens,
            stop=stop,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            frequency_penalty=frequency_penalty,
            presence_penalty=presence_penalty,
            ignore_eos=ignore_eos,
        )
        backend = backend or global_config.default_backend
        return run_program_batch(
            self,
            backend,
            batch_kwargs,
            default_sampling_para,
            num_threads,
            progress_bar,
        )

    def trace(self, *, backend=None, **kwargs):
        from sglang.lang.tracer import trace_program

        backend = backend or global_config.default_backend
        return trace_program(self, kwargs, backend)

    def pin(self, backend=None):
        from sglang.lang.interpreter import pin_program

        backend = backend or global_config.default_backend
        return pin_program(self, backend)

    def unpin(self, backend=None):
        from sglang.lang.interpreter import unpin_program

        backend = backend or global_config.default_backend
        return unpin_program(self, backend)

    def compile(self, *, backend=None):
        from sglang.lang.compiler import compile_func

        return compile_func(self, backend)

    def __call__(self, *args, **kwargs):
        from sglang.lang.tracer import TracingScope

        tracing_scope = TracingScope.get_current_scope()
        if tracing_scope is None:
            return self.run(*args, **kwargs)
        else:
            kwargs["backend"] = tracing_scope.tracer_state.backend
            return self.trace(*args, **kwargs)


class SglExpr:
    node_ct = 0

    def __init__(self):
        self.node_id = SglExpr.node_ct
        self.prev_node = None
        self.pid = None
        SglExpr.node_ct += 1

    def __add__(self, other):
        if isinstance(other, str):
            other = SglConstantText(other)
        assert isinstance(other, SglExpr)

        return self.concatenate_ir(self, other)

    def __radd__(self, other):
        if isinstance(other, str):
            other = SglConstantText(other)
        assert isinstance(other, SglExpr), f"{other}"

        return self.concatenate_ir(other, self)

    def concatenate_ir(self, a, b):
        if isinstance(a, SglExprList):
            if isinstance(b, SglExprList):
                return SglExprList(a.expr_list + b.expr_list)
            else:
                return SglExprList(a.expr_list + [b])
        elif isinstance(b, SglExprList):
            return SglExprList([a] + b.expr_list)

        return SglExprList([a, b])

    def print_graph_dfs(self):
        ret = [""]
        visited = set()

        def dfs_print(x):
            if x is None or x in visited:
                return
            visited.add(x)

            # Print dependency
            if x.prev_node is not None:
                dfs_print(x.prev_node)

            if isinstance(x, SglExprList):
                for y in x.expr_list:
                    dfs_print(y)
            # elif isinstance(x, SglRole):
            #    dfs_print(x.expr)
            elif isinstance(x, SglVariable):
                dfs_print(x.source)

            # Print the node itself
            if isinstance(x, (SglFork, SglGetForkItem)):
                ret[0] += f"%{x.node_id} = {x}\n"
            else:
                if x.prev_node is not None:
                    ret[0] += (
                        f"%{x.node_id} = %{x.prev_node.node_id} + " + str(x) + "\n"
                    )
                else:
                    ret[0] += f"%{x.node_id} = " + str(x) + "\n"

        dfs_print(self)
        return ret[0]


class SglExprList(SglExpr):
    def __init__(self, expr_list: List[SglExpr]):
        super().__init__()
        self.expr_list = expr_list

    def __repr__(self):
        return f"ExprList({self.expr_list})"


class SglArgument(SglExpr):
    def __init__(self, name: str, value: str):
        super().__init__()
        self.name = name
        self.value = value

    def __repr__(self):
        return f"Argument(name={self.name}, value={repr(self.value)})"

    def __len__(self):
        return len(self.value)

    def __getitem__(self, i):
        return self.value[i]

    def __int__(self):
        return self.value

    def __bool__(self):
        return self.value

    def __format__(self, *args):
        raise TypeError(
            "Cannot put argument inside a f-string. "
            "This is not compatible with the tracer. "
        )


class SglImage(SglExpr):
    def __init__(self, path):
        self.path = path

    def __repr__(self) -> str:
        return f"SglImage({self.path})"


class SglGen(SglExpr):
    def __init__(
        self,
        name,
        max_new_tokens,
        stop,
        temperature,
        top_p,
        top_k,
        frequency_penalty,
        presence_penalty,
        ignore_eos,
        dtype,
        regex,
    ):
        super().__init__()
        self.name = name
        self.sampling_params = SglSamplingParams(
            max_new_tokens=max_new_tokens,
            stop=stop,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            frequency_penalty=frequency_penalty,
            presence_penalty=presence_penalty,
            ignore_eos=ignore_eos,
            dtype=dtype,
            regex=regex,
        )

    def __repr__(self):
        return f"Gen('{self.name}')"


class SglConstantText(SglExpr):
    def __init__(self, value):
        super().__init__()
        self.value = value

    def __repr__(self):
        return f"Constant({repr(self.value)})"


class SglRoleBegin(SglExpr):
    def __init__(self, role):
        super().__init__()
        self.role = role

    def __repr__(self):
        return f"RoleBegin({self.role})"


class SglRoleEnd(SglExpr):
    def __init__(self, role):
        super().__init__()
        self.role = role

    def __repr__(self):
        return f"RoleEnd({self.role})"


class SglSelect(SglExpr):
    def __init__(self, name, choices, temperature):
        super().__init__()
        self.name = name
        self.choices = choices
        self.temperature = temperature

    def __repr__(self):
        return f"Select({self.name}, choices={self.choices})"


class SglFork(SglExpr):
    def __init__(self, number, position_ids_offset=None):
        super().__init__()
        self.number = number
        self.position_ids_offset = position_ids_offset

    def __repr__(self):
        return (
            f"Fork(%{self.prev_node.node_id}, number={self.number}, "
            f"position_ids_offset={self.position_ids_offset})"
        )


class SglGetForkItem(SglExpr):
    def __init__(self, index):
        super().__init__()
        self.index = index

    def __repr__(self):
        return f"GetForkItem(%{self.prev_node.node_id}, index={self.index})"


class SglVariable(SglExpr):
    def __init__(self, name, source):
        super().__init__()
        self.name = name
        self.source = source

    def __repr__(self):
        return f"Variable('{self.name}', source=%{self.source.node_id})"


class SglVarScopeBegin(SglExpr):
    def __init__(self, name):
        super().__init__()
        self.name = name

    def __repr__(self):
        return f"VarScopeBegin('{self.name}')"


class SglVarScopeEnd(SglExpr):
    def __init__(self, name):
        super().__init__()
        self.name = name

    def __repr__(self):
        return f"VarScopeEnd('{self.name}')"


class SglConcateAndAppend(SglExpr):
    def __init__(self, states):
        super().__init__()
        self.states = states

    def __repr__(self):
        return f"ConcatenateAndAppend('{self.states}')"


class SglCommitLazy(SglExpr):
    def __init__(self):
        super().__init__()

    def __repr__(self):
        return f"CommitLazy()"
