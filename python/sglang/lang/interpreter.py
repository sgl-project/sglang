"""The interpreter that executes SGL programs"""

import asyncio
import contextvars
import copy
import multiprocessing
import queue
import threading
import uuid
import warnings
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from typing import Any, Callable, Dict, List, Optional

import tqdm

from sglang.global_config import global_config
from sglang.lang.ir import (
    SglCommitLazy,
    SglConcateAndAppend,
    SglConstantText,
    SglExpr,
    SglExprList,
    SglGen,
    SglImage,
    SglRoleBegin,
    SglRoleEnd,
    SglSelect,
    SglVariable,
    SglVarScopeBegin,
    SglVarScopeEnd,
    SglVideo,
)
from sglang.utils import (
    encode_image_base64,
    encode_video_base64,
    get_exception_traceback,
)


def run_internal(state, program, func_args, func_kwargs, sync):
    try:
        state.ret_value = program.func(state, *func_args, **func_kwargs)
    except Exception as e:
        raise e
    finally:
        state.stream_executor.end()

    if sync:
        state.stream_executor.sync()

    if global_config.verbosity >= 2:
        print(state.text())


def run_program(
    program,
    backend,
    func_args,
    func_kwargs,
    default_sampling_para,
    stream,
    sync=False,
    use_thread=True,
):
    if hasattr(backend, "endpoint"):
        backend = backend.endpoint
    assert backend is not None, "Please specify a backend"
    func_kwargs.update(program.bind_arguments)
    stream_executor = StreamExecutor(
        backend,
        func_kwargs,
        default_sampling_para,
        chat_template=None,
        stream=stream,
        num_api_spec_tokens=program.num_api_spec_tokens,
        use_thread=use_thread,
    )
    state = ProgramState(stream_executor)

    if stream:
        t = threading.Thread(
            target=run_internal, args=(state, program, func_args, func_kwargs, sync)
        )
        t.start()
        return state
    else:
        run_internal(state, program, func_args, func_kwargs, sync)
        return state


def run_program_batch(
    program,
    backend,
    batch_arguments,
    default_sampling_para,
    num_threads,
    progress_bar,
    generator_style=False,
):
    if hasattr(backend, "endpoint"):
        backend = backend.endpoint

    # Pre-cache the common prefix for a batch. The prefix is extracted by tracing the program.
    if global_config.enable_precache_with_tracing and len(batch_arguments) > 1:
        cache_program(program, backend)

    # Run all programs
    if num_threads == "auto":
        num_threads = max(96, multiprocessing.cpu_count() * 16)
    num_threads = min(num_threads, len(batch_arguments))

    if generator_style:
        return _run_program_batch_generator(
            program,
            backend,
            batch_arguments,
            default_sampling_para,
            num_threads,
            progress_bar,
        )

    # Original code path when generator_style=False
    if num_threads == 1:
        rets = []
        if progress_bar:
            for arguments in tqdm.tqdm(batch_arguments):
                rets.append(
                    run_program(
                        program,
                        backend,
                        (),
                        arguments,
                        default_sampling_para,
                        False,
                        True,
                    )
                )
        else:
            for arguments in batch_arguments:
                rets.append(
                    run_program(
                        program,
                        backend,
                        (),
                        arguments,
                        default_sampling_para,
                        False,
                        True,
                    )
                )
    else:
        if progress_bar:
            pbar = tqdm.tqdm(total=len(batch_arguments))

        with ThreadPoolExecutor(num_threads) as executor:
            futures = []
            for arguments in batch_arguments:
                futures.append(
                    executor.submit(
                        run_program,
                        program,
                        backend,
                        (),
                        arguments,
                        default_sampling_para,
                        False,
                        True,
                    )
                )
                if progress_bar:
                    futures[-1].add_done_callback(lambda _: pbar.update())

            rets = [f.result() for f in futures]
        rets[-1].sync()

        if progress_bar:
            pbar.close()

    return rets


def _run_program_batch_generator(
    program,
    backend,
    batch_arguments,
    default_sampling_para,
    num_threads,
    progress_bar,
):
    """Helper function that yields results one by one using chunking to avoid overwhelming ThreadPoolExecutor."""
    if num_threads == 1:
        iterator = tqdm.tqdm(batch_arguments) if progress_bar else batch_arguments
        for arguments in iterator:
            yield run_program(
                program,
                backend,
                (),
                arguments,
                default_sampling_para,
                False,
                True,
            )
    else:
        pbar = tqdm.tqdm(total=len(batch_arguments)) if progress_bar else None

        # Process in chunks to avoid overwhelming ThreadPoolExecutor
        # Otherwise, ThreadPoolExecutor.submit will block after adding certain number of tasks
        # so we will never reach "yield" until all tasks are done
        chunk_size = 200

        with ThreadPoolExecutor(num_threads) as executor:
            for chunk_start in range(0, len(batch_arguments), chunk_size):
                chunk_end = min(chunk_start + chunk_size, len(batch_arguments))
                chunk_futures = []

                # Submit chunk of tasks
                for i in range(chunk_start, chunk_end):
                    future = executor.submit(
                        run_program,
                        program,
                        backend,
                        (),
                        batch_arguments[i],
                        default_sampling_para,
                        False,
                        True,
                    )
                    if pbar:
                        future.add_done_callback(lambda _: pbar.update())
                    chunk_futures.append(future)

                # Yield results from this chunk as they complete
                for future in chunk_futures:
                    yield future.result()

        if pbar:
            pbar.close()


def cache_program(program, backend):
    from sglang.lang.tracer import extract_prefix_by_tracing

    prefix = extract_prefix_by_tracing(program, backend)
    if prefix and len(prefix) > 64:
        backend.cache_prefix(prefix)


class StreamExecutor:
    """A stream executor that executes SGL expressions in a background thread."""

    def __init__(
        self,
        backend,
        arguments,
        default_sampling_para,
        chat_template,
        stream,
        num_api_spec_tokens=None,
        use_thread=True,
    ):
        from sglang.lang.backend.base_backend import BaseBackend

        self.sid = uuid.uuid4().hex
        self.backend: BaseBackend = backend
        self.arguments: Dict[str, Any] = arguments
        self.default_sampling_para = default_sampling_para
        self.stream = stream

        self.variables = {}  # Dict[name: str -> value: str]
        self.variable_event = {}  # Dict[name: str -> event: threading.Event]
        self.meta_info = {}  # Dict[name: str -> info: str]
        self.is_finished = False
        self.error_ = None

        # For completion
        self.text_ = ""  # The full text

        # For chat
        self.messages_ = []  # The messages in the OpenAI API format
        self.chat_template = chat_template or self.backend.get_chat_template()
        self.cur_role = None
        self.cur_role_begin_pos = None

        # For vision
        self.images_ = []
        self.cur_images = []

        # For fork/join
        self.fork_start_text_pos = None

        # For speculative execution
        self.num_api_spec_tokens = num_api_spec_tokens
        self.speculated_text = ""

        # Worker thread
        self.use_thread = use_thread
        if self.use_thread:
            self.queue = queue.Queue()

            def _run_worker_in_context():
                self._thread_worker_func()

            self.worker = threading.Thread(
                target=contextvars.copy_context().run, args=(_run_worker_in_context,)
            )
            self.worker.start()

        # For streaming
        if stream:
            self.stream_text_event = threading.Event()
            self.stream_var_event = {}
        else:
            self.stream_text_event = None
            self.stream_var_event = None

    def submit(self, expr: SglExpr):
        self._init_var_event(expr)

        if self.use_thread:
            self.queue.put(expr)
        else:
            self._execute(expr)

    def sync(self):
        if self.use_thread:
            self.queue.join()

    def get_var(self, name):
        if name in self.variable_event:
            self.variable_event[name].wait()
        return self.variables[name]

    def set_var(self, name, value):
        self.variables[name] = value

    def get_meta_info(self, name, timeout=None):
        if name in self.variable_event:
            got = self.variable_event[name].wait(timeout)
            if not got:
                raise TimeoutError(f"Timeout while waiting for event '{name}'")
        ret = self.meta_info.get(name, None)
        return ret

    def fork(
        self,
        size: int = 1,
        position_ids_offset: Optional[List[int]] = None,
    ):
        if size > 1 and str(self.text_):
            self.submit(SglCommitLazy())

        self.sync()
        size = int(size)

        exes = [
            StreamExecutor(
                self.backend,
                self.arguments,
                self.default_sampling_para,
                self.chat_template,
                self.stream,
            )
            for _ in range(size)
        ]
        for i in range(size):
            exes[i].variables = dict(self.variables)
            exes[i].text_ = str(self.text_)
            exes[i].messages_ = list(self.messages_)
            exes[i].cur_role = self.cur_role
            exes[i].cur_role_begin_pos = self.cur_role_begin_pos
            exes[i].fork_start_text_pos = len(self.text_)
            exes[i].images_ = list(self.images_)

            # TODO(ying): handle API speculative execution

        return exes

    def text(self):
        self.sync()
        return self.text_

    def messages(self):
        self.sync()
        return self.messages_

    def error(self):
        self.sync()
        return self.error_

    def end(self):
        if self.use_thread:
            if self.worker.is_alive():
                self.queue.put(None)
        self.backend.end_program(self)

    def _thread_worker_func(self):
        error = None

        while True:
            expr = self.queue.get()
            if expr is None:
                self.queue.task_done()
                break

            try:
                self._execute(expr)
            except Exception as e:
                warnings.warn(f"Error in stream_executor: {get_exception_traceback()}")
                error = e
                break
            self.queue.task_done()
            if self.stream_text_event:
                self.stream_text_event.set()

        # Clean the queue and events
        if error is not None:
            try:
                while True:
                    self.queue.task_done()
                    self.queue.get_nowait()
            except queue.Empty:
                pass
            for name in self.variable_event:
                self.variable_event[name].set()
            if self.stream_var_event:
                for name in self.stream_var_event:
                    self.stream_var_event[name].set()
            self.error_ = error

        if self.stream_text_event:
            self.stream_text_event.set()

        self.is_finished = True

    def _execute(self, other):
        if isinstance(other, str):
            other = SglConstantText(other)

        assert isinstance(other, SglExpr), f"{other}"

        if isinstance(other, SglConstantText):
            self._execute_fill(other.value)
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
        elif isinstance(other, SglImage):
            self._execute_image(other)
        elif isinstance(other, SglVideo):
            self._execute_video(other)
        elif isinstance(other, SglVariable):
            self._execute_variable(other)
        elif isinstance(other, SglVarScopeBegin):
            self._execute_var_scope_begin(other)
        elif isinstance(other, SglVarScopeEnd):
            self._execute_var_scope_end(other)
        elif isinstance(other, SglCommitLazy):
            self._execute_commit_lazy_operations(other)
        elif isinstance(other, SglConcateAndAppend):
            if (
                global_config.enable_parallel_encoding
                and self.backend.support_concate_and_append
            ):
                self._execute_concatenate_and_append_kv_cache(other)
            else:
                self._execute_concatenate_and_append_text(other)
        else:
            raise ValueError(f"Unknown type: {type(other)}")

    def _execute_fill(self, value: str, prefix=False):
        value = str(value)

        if (
            self.cur_role == "assistant"
            and self.num_api_spec_tokens is not None
            and self.backend.is_chat_model
            and not prefix
        ):
            self.backend.spec_fill(value)
            return

        if self.speculated_text.startswith(value):
            self.speculated_text = self.speculated_text[len(value) :]
        else:
            self.speculated_text = ""

        self.text_ += value

    def _execute_image(self, expr: SglImage):
        path = expr.path

        base64_data = encode_image_base64(path)

        self.images_.append((path, base64_data))
        self.cur_images.append((path, base64_data))
        self.text_ += self.chat_template.image_token

    def _execute_video(self, expr: SglVideo):
        path = expr.path
        num_frames = expr.num_frames

        base64_data = encode_video_base64(path, num_frames)

        self.images_.append((path, base64_data))
        self.cur_images.append((path, base64_data))
        self.text_ += self.chat_template.image_token

    def _spec_gen(self, sampling_params):
        stop = sampling_params.stop
        max_new_tokens = sampling_params.max_new_tokens
        meta_info = {}

        def regen():
            nonlocal meta_info

            sampling_params.max_new_tokens = max(
                sampling_params.max_new_tokens, self.num_api_spec_tokens
            )
            sampling_params.stop = None
            self.speculated_text, meta_info = self.backend.generate(
                self, sampling_params=sampling_params
            )

        def find_stop():
            if isinstance(stop, str):
                return self.speculated_text.find(stop)
            elif isinstance(stop, (tuple, list)):
                pos = -1
                for stop_str in stop:
                    stop_pos = self.speculated_text.find(stop_str)
                    if stop_pos != -1 and (pos == -1 or stop_pos < pos):
                        pos = stop_pos
                return pos
            else:
                raise Exception("Wrong type of stop in sampling parameters.")

        if stop is None:
            if len(self.speculated_text) < max_new_tokens:
                regen()
            comp = self.speculated_text[:max_new_tokens]
            self.speculated_text = self.speculated_text[max_new_tokens:]
        elif isinstance(stop, (str, list, tuple)):
            if self.speculated_text == "":
                regen()
            stop_pos = find_stop()
            if stop_pos == -1:
                stop_pos = min(
                    sampling_params.max_new_tokens,
                    len(self.speculated_text),
                )
            comp = self.speculated_text[:stop_pos]
            self.speculated_text = self.speculated_text[stop_pos:]
        else:
            raise ValueError("Wrong type of stop in sampling parameters.")

        return comp, meta_info

    def _execute_gen(self, expr: SglGen):
        sampling_params = self._resolve_sampling_params(expr.sampling_params)
        name = expr.name

        if not self.stream:
            if self.num_api_spec_tokens is None:
                comp, meta_info = self.backend.generate(
                    self,
                    sampling_params=sampling_params,
                )
            else:
                if self.backend.is_chat_model:
                    # Speculative execution on models with only chat interface.
                    # Store the calls into a temporary list.
                    # They will be lazily executed later.
                    comp, meta_info = self.backend.generate(
                        self,
                        sampling_params=sampling_params,
                        spec_var_name=name,
                    )
                    return

                else:  # Speculative execution on models with completion interface
                    comp, meta_info = self._spec_gen(sampling_params)

            self.text_ += comp

            self.variables[name] = comp
            self.meta_info[name] = meta_info
            self.variable_event[name].set()
        else:
            assert (
                self.num_api_spec_tokens is None
            ), "stream is not supported with api speculative execution"
            generator = self.backend.generate_stream(
                self, sampling_params=sampling_params
            )

            self.variables[name] = ""
            self.stream_var_event[name].set()

            for comp, meta_info in generator:
                self.text_ += comp
                self.variables[name] += comp
                self.meta_info[name] = meta_info
                self.stream_var_event[name].set()
                self.stream_text_event.set()

            self.variable_event[name].set()
            self.stream_var_event[name].set()

    def _execute_select(self, expr: SglSelect):
        choices_decision = self.backend.select(
            self, expr.choices, expr.temperature, expr.choices_method
        )
        if expr.name is not None:
            name = expr.name
            self.variables[name] = choices_decision.decision
            self.meta_info[name] = choices_decision.meta_info
            self.variable_event[name].set()
            if self.stream_var_event:
                self.stream_var_event[name].set()
        self.text_ += choices_decision.decision

    def _execute_variable(self, expr: SglVariable):
        src_executor = expr.source_stream_executor
        value = src_executor.get_var(expr.name)
        self._execute_fill(value)

    def _execute_role_begin(self, expr: SglRoleBegin):
        assert self.cur_role is None, "Nested roles are not allowed."

        if len(self.messages_) == 0 and expr.role != "system":
            # Insert the default system message
            default_system = self.chat_template.default_system_prompt
            if default_system:
                self._execute_role_begin(SglRoleBegin("system"))
                self._execute_fill(default_system)
                self._execute_role_end(SglRoleEnd("system"))

        self.cur_role = expr.role

        prefix, _ = self.chat_template.get_prefix_and_suffix(expr.role, self.messages_)

        self._execute_fill(prefix, prefix=True)
        self.cur_role_begin_pos = len(self.text_)

    def _execute_role_end(self, expr: SglRoleEnd):
        if (
            self.cur_role == "assistant"
            and self.num_api_spec_tokens is not None
            and self.backend.is_chat_model
        ):
            # Execute the stored lazy generation calls
            self.backend.role_end_generate(self)
        self.cur_role = None

        new_text = self.text_[self.cur_role_begin_pos :].lstrip()

        _, suffix = self.chat_template.get_prefix_and_suffix(expr.role, self.messages_)
        self._execute_fill(suffix)

        if self.cur_images:
            # OpenAI vision API format
            last_msg = {
                "role": expr.role,
                "content": [{"type": "text", "text": new_text}],
            }
            for image_path, image_base64_data in self.cur_images:
                last_msg["content"].append(
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{image_base64_data}"
                        },
                    }
                )
            self.messages_.append(last_msg)
            self.cur_images = []
        else:
            # OpenAI chat API format
            self.messages_.append({"role": expr.role, "content": new_text})

    def _execute_var_scope_begin(self, expr: SglVarScopeBegin):
        self.variables[expr.name] = int(len(self.text_))

    def _execute_var_scope_end(self, expr: SglVarScopeEnd):
        self.variables[expr.name] = self.text_[self.variables[expr.name] :]
        self.variable_event[expr.name].set()

    def _execute_commit_lazy_operations(self, expr: SglCommitLazy):
        self.backend.commit_lazy_operations(self)

    def _execute_concatenate_and_append_text(self, expr: SglConcateAndAppend):
        new_text = ""
        for s in expr.states:
            exe = s.stream_executor
            exe.sync()
            new_text += exe.text_[exe.fork_start_text_pos :]

        self._execute_fill(new_text)

    def _execute_concatenate_and_append_kv_cache(self, expr: SglConcateAndAppend):
        self_len = len(self.text_)

        for i, s in enumerate(expr.states):
            exe = s.stream_executor
            exe.submit(SglCommitLazy())

        for i, s in enumerate(expr.states):
            exe = s.stream_executor
            exe.sync()
            assert exe.fork_start_text_pos == self_len
            self.text_ += exe.text_[exe.fork_start_text_pos :]

        src_rids = [state.stream_executor.sid for state in expr.states]
        self.backend.concatenate_and_append(src_rids, self.sid)

    def _init_var_event(self, expr):
        if isinstance(expr, (SglGen, SglSelect, SglVarScopeBegin)):
            self.variable_event[expr.name] = threading.Event()
            if self.stream:
                self.stream_var_event[expr.name] = threading.Event()
        elif isinstance(expr, SglExprList):
            for e in expr.expr_list:
                self._init_var_event(e)

    def _resolve_sampling_params(self, sampling_params):
        """
        Construct sampling param based on default + override values

        The default values of sampling are populated in `default_sampling_para` via sgl.function.run(...sampling_args)
        , and `sampling_params` contains the override values from sgl.gen().

        Here we use default_sampling_para as the base and override the values if they exist in `sampling_params`.
        It also extends the stop tokens based on the chat template.
        """

        # deepcopy is required because the dict has lists inside
        clone = copy.deepcopy(self.default_sampling_para)

        for item in [
            "max_new_tokens",
            "min_new_tokens",
            "stop",
            "stop_token_ids",
            "temperature",
            "top_p",
            "top_k",
            "min_p",
            "frequency_penalty",
            "presence_penalty",
            "ignore_eos",
            "return_logprob",
            "logprob_start_len",
            "top_logprobs_num",
            "return_text_in_logprobs",
            "dtype",
            "regex",
            "json_schema",
        ]:
            value = getattr(sampling_params, item, None)
            if value is not None:
                setattr(clone, item, value)

        if self.chat_template.stop_str:
            if clone.stop == ():
                clone.stop = []
            elif isinstance(clone.stop, str):
                clone.stop = [clone.stop]
            clone.stop += self.chat_template.stop_str

        return clone

    def __del__(self):
        self.end()


class ProgramState:
    """The state of an SGL program."""

    def __init__(self, stream_executor: StreamExecutor):
        self.stream_executor = stream_executor

    def _role_common(self, name: str, expr: Optional[SglExpr] = None):
        if expr is not None:
            role_expr = SglExprList([SglRoleBegin(name), expr, SglRoleEnd(name)])
            self.stream_executor.submit(role_expr)
            return role_expr
        else:

            @contextmanager
            def role_scope():
                self.stream_executor.submit(SglRoleBegin(name))
                yield
                self.stream_executor.submit(SglRoleEnd(name))

            return role_scope()

    def system(self, expr: Optional[SglExpr] = None):
        return self._role_common("system", expr)

    def user(self, expr: Optional[SglExpr] = None):
        return self._role_common("user", expr)

    def assistant(self, expr: Optional[SglExpr] = None):
        return self._role_common("assistant", expr)

    @contextmanager
    def var_scope(self, name: str):
        self.stream_executor.submit(SglVarScopeBegin(name))
        yield
        self.stream_executor.submit(SglVarScopeEnd(name))

    def fork(
        self,
        size: int = 1,
        position_ids_offset: Optional[List[int]] = None,
    ):
        stream_executors = self.stream_executor.fork(size, position_ids_offset)
        states = [ProgramState(x) for x in stream_executors]
        state_group = ProgramStateGroup(states, self)
        return state_group

    @contextmanager
    def copy(self, position_ids_offset: Optional[List[int]] = None):
        state_group = self.fork(1, position_ids_offset)
        try:
            yield state_group[0]
        finally:
            state_group.join()

    def text(self):
        return self.stream_executor.text()

    def messages(self):
        return self.stream_executor.messages()

    def sync(self):
        return self.stream_executor.sync()

    def error(self):
        return self.stream_executor.error()

    def text_iter(self, var_name: Optional[str] = None):
        if self.stream_executor.stream:
            prev = 0
            if var_name is None:
                event = self.stream_executor.stream_text_event
                while True:
                    event.wait()
                    event.clear()
                    out = str(self.stream_executor.text_[prev:])
                    prev += len(out)
                    if out:
                        yield out
                    if self.stream_executor.is_finished:
                        break
            else:
                event = None
                while not event:
                    if var_name in self.stream_executor.stream_var_event:
                        event = self.stream_executor.stream_var_event[var_name]
                    if self.stream_executor.is_finished:
                        yield ""
                        return

                while True:
                    event.wait()
                    event.clear()
                    out = str(self.stream_executor.variables[var_name][prev:])
                    prev += len(out)
                    if out:
                        yield out
                    if self.stream_executor.variable_event[var_name].is_set():
                        break
        else:
            if var_name is None:
                yield self.text()
            else:
                yield self.get_var(var_name)

    async def text_async_iter(
        self, var_name: Optional[str] = None, return_meta_data: bool = False
    ):
        loop = asyncio.get_running_loop()

        if self.stream_executor.stream:
            prev = 0
            if var_name is None:
                event = self.stream_executor.stream_text_event
                while True:
                    await loop.run_in_executor(None, event.wait)
                    event.clear()
                    out = str(self.stream_executor.text_[prev:])
                    prev += len(out)
                    if out:
                        yield out
                    if self.stream_executor.is_finished:
                        break
            else:
                event = None
                while not event:
                    if var_name in self.stream_executor.stream_var_event:
                        event = self.stream_executor.stream_var_event[var_name]
                    if self.stream_executor.is_finished:
                        yield ""
                        return

                while True:
                    await loop.run_in_executor(None, event.wait)
                    event.clear()
                    out = str(self.stream_executor.variables[var_name][prev:])
                    prev += len(out)
                    if out:
                        if return_meta_data:
                            yield out, self.stream_executor.meta_info[var_name]
                        else:
                            yield out
                    if self.stream_executor.variable_event[var_name].is_set():
                        break
        else:
            if var_name is None:
                yield self.text()
            else:
                yield self.get_var(var_name)

    def get_var(self, name):
        return self.stream_executor.get_var(name)

    def set_var(self, name, value):
        return self.stream_executor.set_var(name, value)

    def get_meta_info(self, name):
        return self.stream_executor.get_meta_info(name)

    def __iadd__(self, other):
        if other is None:
            raise ValueError("Tried to append None to state.")
        self.stream_executor.submit(other)
        return self

    def __getitem__(self, name):
        return self.get_var(name)

    def __setitem__(self, name, value):
        self.set_var(name, value)

    def __contains__(self, name):
        return name in self.stream_executor.variables

    def __del__(self):
        self.stream_executor.end()

    def __repr__(self) -> str:
        return f"ProgramState({self.text()})"


class ProgramStateGroup:
    def __init__(
        self, states: List[ProgramState], src_state: Optional[ProgramState] = None
    ):
        self.states = states
        self.src_state = src_state

    def join(self, mode: str = "gather_variable"):
        if mode == "gather_variable":
            # Copy variables back
            src_vars = self.src_state.stream_executor.variables
            src_var_set = set(src_vars.keys())
            for child_state in self.states:
                child_state.stream_executor.sync()
                child_vars = child_state.stream_executor.variables
                new_vars = set(child_vars.keys()) - src_var_set

                for k in new_vars:
                    if k in src_vars:
                        src_vars[k].append(child_vars[k])
                    else:
                        src_vars[k] = [child_vars[k]]
        elif mode == "concate_and_append":
            # Concatenate and append KV cache
            self.src_state += SglConcateAndAppend(self.states)
            # Need a sync here. Otherwise, `states` can be deleted.
            self.src_state.stream_executor.sync()
        else:
            raise ValueError(f"Invalid join mode: {mode}")

        for s in self.states:
            s.stream_executor.end()

    def __getitem__(self, i: int):
        return self.states[i]

    def __setitem__(self, i: int, value):
        assert self.states[i] == value

    def __iadd__(self, other):
        if isinstance(other, Callable):
            # lambda function
            for i in range(len(self.states)):
                self.states[i] += other(i)
        elif isinstance(other, SglExpr):
            for i in range(len(self.states)):
                self.states[i] += other
        elif isinstance(other, (list, tuple)):
            for i in range(len(self.states)):
                self.states[i] += other[i]
        else:
            raise ValueError(f"Invalid value: {other}")

        return self
