"""Tracing a program."""

import uuid
from typing import Any, Callable, Dict, List, Optional, Union

from sglang.backend.base_backend import BaseBackend
from sglang.global_config import global_config
from sglang.lang.interpreter import ProgramState, ProgramStateGroup
from sglang.lang.ir import (
    SglArgument,
    SglCommitLazy,
    SglConcateAndAppend,
    SglConstantText,
    SglExpr,
    SglExprList,
    SglFork,
    SglFunction,
    SglGen,
    SglGetForkItem,
    SglRoleBegin,
    SglRoleEnd,
    SglSelect,
    SglVariable,
    SglVarScopeBegin,
    SglVarScopeEnd,
)


class StopTracing(Exception):
    pass


def extract_prefix_by_tracing(program, backend):
    # Create dummy arguments
    dummy_arguments = {name: SglArgument(name, None) for name in program.arg_names}
    arguments = dummy_arguments
    arguments.update(program.bind_arguments)

    # Trace
    tracer = TracerProgramState(backend, arguments, only_trace_prefix=True)
    try:
        with TracingScope(tracer):
            tracer.ret_value = program.func(tracer, **arguments)
    except (StopTracing, TypeError, AttributeError):
        # Some exceptions may not be catched
        pass

    # Run and cache prefix
    prefix = ""
    for expr in tracer.flatten_nodes():
        if isinstance(expr, SglConstantText):
            prefix += expr.value
        else:
            break
    return prefix


def trace_program(program, arguments, backend):
    # Create dummy backend
    if backend is None:
        backend = BaseBackend()

    # Create dummy arguments
    dummy_arguments = {
        name: SglArgument(name, None)
        for name in program.arg_names
        if name not in arguments
    }
    arguments.update(dummy_arguments)
    arguments.update(program.bind_arguments)

    # Trace
    tracer = TracerProgramState(backend, arguments, only_trace_prefix=False)
    with TracingScope(tracer):
        tracer.ret_value = program.func(tracer, **arguments)
    return tracer


class TracerProgramState(ProgramState):
    def __init__(self, backend, arguments, only_trace_prefix):
        self.pid = uuid.uuid4().hex
        self.backend = backend
        self.arguments: Dict[str, Any] = arguments
        self.only_trace_prefix = only_trace_prefix

        if hasattr(backend, "endpoint"):
            self.backend = backend.endpoint

        self.nodes = []
        self.last_node = None
        self.variables = {}
        self.ret_value = None

        # For completion

        # For chat
        self.messages_ = []
        self.cur_role = None
        self.chat_template = self.backend.get_chat_template()

        # For multi states
        self.child_states = []

        cur_scope = TracingScope.get_current_scope()
        if cur_scope is not None:
            cur_scope.add_child_state(self)

    ##################################
    ########### Public API ###########
    ##################################

    def fork(self, size: int = 1, position_ids_offset: Optional[List[int]] = None):
        assert size >= 1

        if self.only_trace_prefix:
            raise StopTracing()

        fork_node = SglFork(size)
        fork_node.prev_node = self.last_node

        states = [
            TracerProgramState(self.backend, self.arguments, self.only_trace_prefix)
            for _ in range(size)
        ]

        for i in range(size):
            node = SglGetForkItem(i)
            node.prev_node = fork_node
            states[i].last_node = node
            states[i].variables = dict(self.variables)
            states[i].messages_ = list(self.messages_)
            states[i].cur_role = self.cur_role
            states[i].chat_template = self.chat_template

        state_group = ProgramStateGroup(states, self)

        return state_group

    ##################################
    ########## Internal API ##########
    ##################################

    def _append_node(self, other: SglExpr):
        self.nodes.append(other)
        other.prev_node = self.last_node
        self.last_node = other

    def _execute(self, other: SglExpr):
        if isinstance(other, str):
            other = SglConstantText(other)

        other.pid = self.pid

        if isinstance(other, SglConstantText):
            self._execute_fill(other)
        elif isinstance(other, SglGen):
            self._execute_gen(other)
        elif isinstance(other, SglSelect):
            self._execute_select(other)
        elif isinstance(other, SglExprList):
            for x in other.expr_list:
                self._execute(x)
        elif isinstance(other, SglRoleBegin):
            self._execute_role_begin(other)
        elif isinstance(other, SglRoleEnd):
            self._execute_role_end(other)
        elif isinstance(other, SglVarScopeBegin):
            self._execute_var_scope_begin(other)
        elif isinstance(other, SglVarScopeEnd):
            self._execute_var_scope_end(other)
        else:
            if self.only_trace_prefix:
                raise StopTracing()
            else:
                self._append_node(other)

        return self

    def __iadd__(self, other):
        self._execute(other)
        return self

    def _execute_fill(self, expr: SglConstantText):
        if isinstance(expr, str):
            expr = SglConstantText(expr)
        self._append_node(expr)

    def _execute_gen(self, expr: SglGen):
        name = expr.name if expr.name is not None else "gen_" + str(len(self.variables))
        new_node = SglVariable(name, source=expr)
        self.variables[name] = new_node
        self._append_node(expr)

    def _execute_select(self, expr: SglSelect):
        name = (
            expr.name if expr.name is not None else "select_" + str(len(self.variables))
        )
        new_node = SglVariable(name, source=expr)
        self.variables[name] = new_node
        self._append_node(expr)

    def _execute_role_begin(self, expr: SglRoleBegin):
        assert self.cur_role is None, "Nested roles are not allowed."

        if len(self.messages_) == 0 and expr.role != "system":
            # Insert default system message
            default_system = self.chat_template.default_system_prompt
            if default_system:
                self._execute_role_begin(SglRoleBegin("system"))
                self._execute_fill(default_system)
                self._execute_role_end(SglRoleEnd("system"))

        self.cur_role = expr.role

        prefix, suffix = self.chat_template.get_prefix_and_suffix(
            expr.role, self.messages_
        )

        self._execute_fill(prefix)

    def _execute_role_end(self, expr: SglRoleEnd):
        prefix, suffix = self.chat_template.get_prefix_and_suffix(
            expr.role, self.messages_
        )

        self._execute_fill(suffix)

        self.messages_.append({"role": expr.role, "content": ""})

        self.cur_role = None

    def _execute_var_scope_end(self, expr: SglVarScopeEnd):
        new_node = SglVariable(name, source=self.last_node)
        self.variables[name] = new_node

    def get_var(self, name):
        ret = self.arguments.get(name, None)
        if ret is not None:
            return ret

        v = self.variables[name]
        return SglVariable(v.name, v.source)

    def flatten_nodes(self):
        def traverse(cur):
            if isinstance(cur, SglExprList):
                for child in cur.expr_list:
                    traverse(child)
            else:
                ret.append(cur)

        ret = []
        for x in self.nodes:
            traverse(x)
        return ret

    def __del__(self):
        pass


class TracingScope:
    cur_scope = None

    def __init__(self, tracer_state: TracerProgramState):
        self.tracer_state = tracer_state
        self.last_scope = TracingScope.cur_scope

    def __enter__(self):
        TracingScope.cur_scope = self
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        TracingScope.cur_scope = self.last_scope

    @staticmethod
    def get_current_scope():
        return TracingScope.cur_scope

    def add_child_state(self, state: TracerProgramState):
        cur_scope = self
        while cur_scope != None:
            cur_scope.tracer_state.child_states.append(state)
            cur_scope = cur_scope.last_scope
