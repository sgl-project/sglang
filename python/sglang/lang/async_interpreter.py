"""The async interpreter that executes SGL programs with asyncio primitives"""

import asyncio
import copy
import uuid
import warnings
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator, Dict, Optional

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
    SglSeparateReasoning,
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


async def async_run_internal(state, program, func_args, func_kwargs, sync):
    try:
        state.ret_value = await program.func(state, *func_args, **func_kwargs)
    except Exception as e:
        raise e
    finally:
        await state.async_executor.end()

    if sync:
        await state.async_executor.sync()

    if global_config.verbosity >= 2:
        print(await state.text())


async def async_run_program(
    program,
    backend,
    func_args,
    func_kwargs,
    default_sampling_para,
    stream,
    sync=False,
):
    """Async version of run_program"""
    if hasattr(backend, "endpoint"):
        backend = backend.endpoint
    assert backend is not None, "Please specify an async backend"
    func_kwargs.update(program.bind_arguments)

    async_executor = AsyncStreamExecutor(
        backend,
        func_kwargs,
        default_sampling_para,
        chat_template=None,
        stream=stream,
        num_api_spec_tokens=program.num_api_spec_tokens,
    )
    state = AsyncProgramState(async_executor)

    if stream:
        # Start worker task in background
        asyncio.create_task(
            async_run_internal(state, program, func_args, func_kwargs, sync)
        )
        return state
    else:
        await async_run_internal(state, program, func_args, func_kwargs, sync)
        return state


async def async_run_program_batch(
    program,
    backend,
    batch_arguments,
    default_sampling_para,
    max_concurrency=10,
    progress_bar=False,
):
    """True concurrent batch execution with semaphore-based concurrency control

    Args:
        program: The AsyncSglFunction to execute
        backend: The async backend to use
        batch_arguments: List of argument dicts for each execution
        default_sampling_para: Default sampling parameters
        max_concurrency: Maximum number of concurrent executions
        progress_bar: Whether to show a progress bar (requires tqdm)
    """
    if hasattr(backend, "endpoint"):
        backend = backend.endpoint

    semaphore = asyncio.Semaphore(max_concurrency)

    async def run_single(arguments):
        async with semaphore:
            return await async_run_program(
                program, backend, (), arguments, default_sampling_para, stream=False
            )

    task_list = [asyncio.create_task(run_single(args)) for args in batch_arguments]

    try:
        if progress_bar:
            try:
                from tqdm.asyncio import tqdm
                results = await tqdm.gather(*task_list, desc="Processing batch")
            except ImportError:
                warnings.warn(
                    "tqdm is not installed. Progress bar disabled. Install with: pip install tqdm"
                )
                results = await asyncio.gather(*task_list)
        else:
            results = await asyncio.gather(*task_list)
        return results
    except Exception as e:
        # Cancel remaining tasks on error
        for task in task_list:
            if not task.done():
                task.cancel()

        await asyncio.gather(*task_list, return_exceptions=True)
        raise


async def async_run_program_batch_streaming(
    program,
    backend,
    batch_arguments,
    default_sampling_para,
    max_concurrency=10,
):
    """Generator-style batch execution that yields results as they complete

    This allows processing results as soon as they're ready, without waiting
    for the entire batch to complete.

    Args:
        program: The AsyncSglFunction to execute
        backend: The async backend to use
        batch_arguments: List of argument dicts for each execution
        default_sampling_para: Default sampling parameters
        max_concurrency: Maximum number of concurrent executions

    Yields:
        Tuple[int, AsyncProgramState]: (index, result) pairs as tasks complete

    Example:
        async for idx, result in program.run_batch_streaming(batch_args, backend=backend):
            answer = await result.get_var("answer")
            print(f"Result {idx}: {answer}")
    """
    if hasattr(backend, "endpoint"):
        backend = backend.endpoint

    semaphore = asyncio.Semaphore(max_concurrency)

    async def run_single_with_index(idx, arguments):
        async with semaphore:
            result = await async_run_program(
                program, backend, (), arguments, default_sampling_para, stream=False
            )
            return idx, result

    tasks = [
        asyncio.create_task(run_single_with_index(i, args))
        for i, args in enumerate(batch_arguments)
    ]

    try:
        for coro in asyncio.as_completed(tasks):
            idx, result = await coro
            yield idx, result
    except Exception as e:
        # Cancel remaining tasks on error
        for task in tasks:
            if not task.done():
                task.cancel()

        await asyncio.gather(*tasks, return_exceptions=True)
        raise


async def async_cache_program(program, backend):
    """Async version of cache_program"""
    from sglang.lang.tracer import async_extract_prefix_by_tracing

    prefix = await async_extract_prefix_by_tracing(program, backend)
    if prefix and len(prefix) > 64:
        await backend.cache_prefix(prefix)


class AsyncStreamExecutor:
    """An async stream executor that executes SGL expressions using asyncio primitives"""

    def __init__(
        self,
        backend,
        arguments,
        default_sampling_para,
        chat_template,
        stream,
        num_api_spec_tokens=None,
    ):
        self.sid = uuid.uuid4().hex
        self.backend = backend
        self.arguments: Dict[str, Any] = arguments
        self.default_sampling_para = default_sampling_para
        self.stream = stream

        self.variables = {}
        self.variable_event = {}  # Dict[name: str -> event: asyncio.Event]
        self.meta_info = {}
        self.is_finished = False
        self.error_ = None

        # For completion
        self.text_ = ""

        # For chat
        self.messages_ = []
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

        # Async worker using asyncio.Queue (unbounded)
        self.queue = asyncio.Queue()
        self.worker_task = asyncio.create_task(self._async_worker_func())

        # For streaming
        if stream:
            self.stream_text_event = asyncio.Event()
            self.stream_var_event = {}
        else:
            self.stream_text_event = None
            self.stream_var_event = None

    async def submit(self, expr: SglExpr):
        """Submit expression to async queue"""
        self._init_var_event(expr)
        await self.queue.put(expr)

    async def sync(self):
        """Wait for all tasks to complete"""
        await self.queue.join()

    async def get_var(self, name):
        """Get variable value (waits if not ready)"""
        if name in self.variable_event:
            await self.variable_event[name].wait()
        return self.variables[name]

    def set_var(self, name, value):
        self.variables[name] = value

    async def get_meta_info(self, name, timeout=None):
        if name in self.variable_event:
            try:
                if timeout:
                    await asyncio.wait_for(
                        self.variable_event[name].wait(), timeout=timeout
                    )
                else:
                    await self.variable_event[name].wait()
            except asyncio.TimeoutError:
                raise TimeoutError(f"Timeout while waiting for event '{name}'")
        ret = self.meta_info.get(name, None)
        return ret

    async def text(self):
        await self.sync()
        return self.text_

    async def messages(self):
        await self.sync()
        return self.messages_

    async def error(self):
        await self.sync()
        return self.error_

    async def end(self):
        """End the async executor"""
        if self.worker_task and not self.worker_task.done():
            await self.queue.put(None)
        if hasattr(self.backend, "end_program"):
            await self.backend.end_program(self)

    async def _async_worker_func(self):
        """Async worker loop"""
        error = None

        while True:
            expr = await self.queue.get()
            if expr is None:
                self.queue.task_done()
                break

            try:
                await self._execute_async(expr)
            except Exception as e:
                warnings.warn(f"Error in async_worker: {get_exception_traceback()}")
                error = e
                self.queue.task_done()
                break

            self.queue.task_done()
            if self.stream_text_event:
                self.stream_text_event.set()

        # Clean the queue and events
        if error is not None:
            # Drain remaining queue items
            while not self.queue.empty():
                try:
                    self.queue.get_nowait()
                    self.queue.task_done()
                except asyncio.QueueEmpty:
                    break

            # Notify all waiters
            for event in self.variable_event.values():
                event.set()
            if self.stream_var_event:
                for event in self.stream_var_event.values():
                    event.set()
            self.error_ = error

        if self.stream_text_event:
            self.stream_text_event.set()

        self.is_finished = True

    async def _execute_async(self, other):
        """Execute an expression asynchronously"""
        if isinstance(other, str):
            other = SglConstantText(other)

        assert isinstance(other, SglExpr), f"{other}"

        if isinstance(other, SglConstantText):
            self._execute_fill(other.value)
        elif isinstance(other, SglGen):
            await self._execute_gen_async(other)
        elif isinstance(other, SglSelect):
            await self._execute_select_async(other)
        elif isinstance(other, SglExprList):
            for x in other.expr_list:
                await self._execute_async(x)
        elif isinstance(other, SglRoleBegin):
            self._execute_role_begin(other)
        elif isinstance(other, SglRoleEnd):
            await self._execute_role_end_async(other)
        elif isinstance(other, SglImage):
            self._execute_image(other)
        elif isinstance(other, SglVideo):
            self._execute_video(other)
        elif isinstance(other, SglVariable):
            await self._execute_variable_async(other)
        elif isinstance(other, SglVarScopeBegin):
            self._execute_var_scope_begin(other)
        elif isinstance(other, SglVarScopeEnd):
            self._execute_var_scope_end(other)
        elif isinstance(other, SglCommitLazy):
            await self._execute_commit_lazy_operations_async(other)
        elif isinstance(other, SglConcateAndAppend):
            if (
                global_config.enable_parallel_encoding
                and self.backend.support_concate_and_append
            ):
                await self._execute_concatenate_and_append_kv_cache_async(other)
            else:
                await self._execute_concatenate_and_append_text_async(other)
        elif isinstance(other, SglSeparateReasoning):
            await self._execute_separate_reasoning_async(other)
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
            if hasattr(self.backend, "spec_fill"):
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

    async def _execute_gen_async(self, expr: SglGen):
        sampling_params = self._resolve_sampling_params(expr.sampling_params)
        name = expr.name

        if not self.stream:
            # Non-streaming generation
            comp, meta_info = await self.backend.generate(
                self, sampling_params=sampling_params
            )

            if isinstance(comp, list):
                self.text_ += comp[0]
            else:
                assert isinstance(comp, str)
                self.text_ += comp

            self.variables[name] = comp
            self.meta_info[name] = meta_info
            self.variable_event[name].set()
        else:
            # Streaming generation
            self.variables[name] = ""
            self.stream_var_event[name].set()

            async for comp, meta_info in self.backend.generate_stream(
                self, sampling_params=sampling_params
            ):
                self.text_ += comp
                self.variables[name] += comp
                self.meta_info[name] = meta_info
                self.stream_var_event[name].set()
                self.stream_text_event.set()

            self.variable_event[name].set()
            self.stream_var_event[name].set()

    async def _execute_select_async(self, expr: SglSelect):
        choices_decision = await self.backend.select(
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

    async def _execute_variable_async(self, expr: SglVariable):
        src_executor = expr.source_stream_executor
        value = await src_executor.get_var(expr.name)
        self._execute_fill(value)

    def _execute_role_begin(self, expr: SglRoleBegin):
        assert self.cur_role is None, "Nested roles are not allowed."

        if len(self.messages_) == 0 and expr.role != "system":
            default_system = self.chat_template.default_system_prompt
            if default_system:
                self._execute_role_begin(SglRoleBegin("system"))
                self._execute_fill(default_system)
                asyncio.create_task(self._execute_role_end_async(SglRoleEnd("system")))

        self.cur_role = expr.role
        prefix, _ = self.chat_template.get_prefix_and_suffix(expr.role, self.messages_)
        self._execute_fill(prefix, prefix=True)
        self.cur_role_begin_pos = len(self.text_)

    async def _execute_role_end_async(self, expr: SglRoleEnd):
        if (
            self.cur_role == "assistant"
            and self.num_api_spec_tokens is not None
            and self.backend.is_chat_model
        ):
            if hasattr(self.backend, "role_end_generate"):
                await self.backend.role_end_generate(self)

        self.cur_role = None
        new_text = self.text_[self.cur_role_begin_pos :].lstrip()
        _, suffix = self.chat_template.get_prefix_and_suffix(expr.role, self.messages_)
        self._execute_fill(suffix)

        if self.cur_images:
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
            self.messages_.append({"role": expr.role, "content": new_text})

    def _execute_var_scope_begin(self, expr: SglVarScopeBegin):
        self.variables[expr.name] = int(len(self.text_))

    def _execute_var_scope_end(self, expr: SglVarScopeEnd):
        self.variables[expr.name] = self.text_[self.variables[expr.name] :]
        self.variable_event[expr.name].set()

    async def _execute_commit_lazy_operations_async(self, expr: SglCommitLazy):
        if hasattr(self.backend, "commit_lazy_operations"):
            await self.backend.commit_lazy_operations(self)

    async def _execute_concatenate_and_append_text_async(
        self, expr: SglConcateAndAppend
    ):
        new_text = ""
        for s in expr.states:
            exe = s.async_executor
            await exe.sync()
            new_text += exe.text_[exe.fork_start_text_pos :]
        self._execute_fill(new_text)

    async def _execute_concatenate_and_append_kv_cache_async(
        self, expr: SglConcateAndAppend
    ):
        self_len = len(self.text_)

        for s in expr.states:
            exe = s.async_executor
            await exe.submit(SglCommitLazy())

        for s in expr.states:
            exe = s.async_executor
            await exe.sync()
            assert exe.fork_start_text_pos == self_len
            self.text_ += exe.text_[exe.fork_start_text_pos :]

        src_rids = [state.async_executor.sid for state in expr.states]
        if hasattr(self.backend, "concatenate_and_append"):
            await self.backend.concatenate_and_append(src_rids, self.sid)

    async def _execute_separate_reasoning_async(self, expr: SglSeparateReasoning):
        if self.stream:
            return

        if (
            self.cur_role == "assistant"
            and self.num_api_spec_tokens is not None
            and self.backend.is_chat_model
        ):
            if hasattr(self.backend, "role_end_generate"):
                await self.backend.role_end_generate(self)

        from sglang.srt.parser.reasoning_parser import ReasoningParser

        reasoning_parser = ReasoningParser(expr.model_type)
        other = expr.expr
        if not other:
            return
        elif isinstance(other, SglGen) or isinstance(other, SglSelect):
            cur_text = await self.get_var(other.name)
            reasoning, normal_text = reasoning_parser.parse_non_stream(cur_text)
            reasoning_name = expr.process_name_for_reasoning(other.name)
            self.set_var(other.name, normal_text)
            self.set_var(reasoning_name, reasoning)
            self.variable_event[reasoning_name].set()
            self.text_ = self.text_[: self.cur_role_begin_pos] + normal_text
        elif isinstance(other, SglExprList):
            for x in other.expr_list:
                await self._execute_separate_reasoning_async(
                    SglSeparateReasoning(expr.model_type, x)
                )

    def _init_var_event(self, expr):
        """Initialize asyncio.Event objects for expressions that produce variables"""
        if isinstance(
            expr, (SglGen, SglSelect, SglVarScopeBegin, SglSeparateReasoning)
        ):
            self.variable_event[expr.name] = asyncio.Event()
            if self.stream:
                self.stream_var_event[expr.name] = asyncio.Event()
        elif isinstance(expr, SglExprList):
            for e in expr.expr_list:
                self._init_var_event(e)

    def _resolve_sampling_params(self, sampling_params):
        """Construct sampling param based on default + override values"""
        clone = copy.deepcopy(self.default_sampling_para)

        for item in [
            "max_new_tokens",
            "min_new_tokens",
            "n",
            "stop",
            "stop_token_ids",
            "stop_regex",
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


class AsyncProgramState:
    """The state of an async SGL program"""

    def __init__(self, async_executor: AsyncStreamExecutor):
        self.async_executor = async_executor

    def _role_common(self, name: str, expr: Optional[SglExpr] = None):
        if expr is not None:
            role_expr = SglExprList([SglRoleBegin(name), expr, SglRoleEnd(name)])
            asyncio.create_task(self.async_executor.submit(role_expr))
            return role_expr
        else:

            @asynccontextmanager
            async def role_scope():
                await self.async_executor.submit(SglRoleBegin(name))
                yield
                await self.async_executor.submit(SglRoleEnd(name))

            return role_scope()

    def system(self, expr: Optional[SglExpr] = None):
        return self._role_common("system", expr)

    def user(self, expr: Optional[SglExpr] = None):
        return self._role_common("user", expr)

    def assistant(self, expr: Optional[SglExpr] = None):
        return self._role_common("assistant", expr)

    @asynccontextmanager
    async def var_scope(self, name: str):
        await self.async_executor.submit(SglVarScopeBegin(name))
        yield
        await self.async_executor.submit(SglVarScopeEnd(name))

    async def text(self) -> str:
        """Get complete text"""
        await self.async_executor.sync()
        if self.async_executor.error_:
            raise self.async_executor.error_
        return self.async_executor.text_

    async def messages(self):
        """Get messages"""
        await self.async_executor.sync()
        if self.async_executor.error_:
            raise self.async_executor.error_
        return self.async_executor.messages_

    async def sync(self):
        """Sync the executor"""
        return await self.async_executor.sync()

    async def error(self):
        """Get error if any"""
        return await self.async_executor.error()

    async def get_var(self, name: str) -> Any:
        """Get variable value (waits if not ready)"""
        if name in self.async_executor.variable_event:
            await self.async_executor.variable_event[name].wait()
        if self.async_executor.error_:
            raise self.async_executor.error_
        return self.async_executor.variables[name]

    def set_var(self, name, value):
        return self.async_executor.set_var(name, value)

    async def get_meta_info(self, name):
        return await self.async_executor.get_meta_info(name)

    async def text_async_iter(
        self, var_name: Optional[str] = None
    ) -> AsyncIterator[str]:
        """True async iterator for streaming"""
        if not self.async_executor.stream:
            yield await self.text()
            return

        prev = 0
        if var_name is None:
            event = self.async_executor.stream_text_event
            while True:
                await event.wait()
                event.clear()

                if self.async_executor.error_:
                    raise self.async_executor.error_

                out = str(self.async_executor.text_[prev:])
                prev += len(out)
                if out:
                    yield out
                if self.async_executor.is_finished:
                    break
        else:
            event = None
            while not event:
                if var_name in self.async_executor.stream_var_event:
                    event = self.async_executor.stream_var_event[var_name]
                if self.async_executor.is_finished:
                    yield ""
                    return
                # avoid busy waiting, wait for event
                await asyncio.sleep(0)

            while True:
                await event.wait()
                event.clear()

                if self.async_executor.error_:
                    raise self.async_executor.error_

                out = str(self.async_executor.variables[var_name][prev:])
                prev += len(out)
                if out:
                    yield out
                if self.async_executor.variable_event[var_name].is_set():
                    break

    def __iadd__(self, other):
        """Submit expression (synchronous operation)"""
        if other is None:
            raise ValueError("Tried to append None to state.")

        # Convert to SglExpr if needed
        if isinstance(other, str):
            other = SglConstantText(other)

        # Initialize variable events (must be done before queuing)
        self.async_executor._init_var_event(other)

        # Synchronous submission to unbounded queue
        self.async_executor.queue.put_nowait(other)
        return self

    def __getitem__(self, name):
        # Note: This returns a coroutine that must be awaited
        return self.get_var(name)

    def __setitem__(self, name, value):
        self.set_var(name, value)

    def __contains__(self, name):
        return name in self.async_executor.variables
