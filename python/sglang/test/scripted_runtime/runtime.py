"""ScriptedRuntime: generator-driven scheduler harness.

Lives inside each scheduler subprocess. The driver rank
(``pp_rank == tp_rank == attn_cp_rank == 0``) advances a caller-provided
generator one step per scheduler event-loop iteration (one
``recv_requests`` call). Non-driver ranks join the cross-rank cpu
broadcast that carries the script's done / error state so every rank
exits together when the script finishes.
"""

from __future__ import annotations

import importlib
import logging
import traceback
from array import array
from typing import TYPE_CHECKING, Callable, Generator, List, Optional, Tuple

from sglang.srt.managers.io_struct import TokenizedGenerateReqInput
from sglang.srt.sampling.sampling_params import SamplingParams
from sglang.srt.utils.common import broadcast_pyobj
from sglang.test.scripted_runtime.req_handle import ReqHandle, ReqStatus
from sglang.test.scripted_runtime.tokenizer_recv_proxy import TokenizerRecvProxy

if TYPE_CHECKING:
    from sglang.srt.managers.scheduler import Scheduler

logger = logging.getLogger(__name__)


class ScriptedRuntimeFinished(Exception):
    """Raised on every rank when the script generator finishes.

    ``ok=True``: normal return / ``StopIteration``.
    ``ok=False``: generator raised; ``exc_traceback`` carries the text.
    Caught by ``run_scheduler_process`` to exit cleanly without
    SIGQUITing the parent.
    """

    def __init__(self, *, ok: bool, exc_traceback: Optional[str] = None) -> None:
        self.ok = ok
        self.exc_traceback = exc_traceback
        super().__init__(f"ScriptedRuntime finished (ok={ok})")


def _resolve_fn(qualified: str) -> Callable:
    """Resolve ``"module.path:qualname"`` to the function object.

    The leaf must be importable across processes — no lambdas / closures.
    """
    module_name, sep, fn_name = qualified.partition(":")
    if not sep or not module_name or not fn_name:
        raise ValueError(
            f"scripted_runtime_fn_path must be 'module.path:function_name', "
            f"got {qualified!r}"
        )
    obj = importlib.import_module(module_name)
    for part in fn_name.split("."):
        obj = getattr(obj, part)
    if not callable(obj):
        raise TypeError(f"resolved object is not callable: {qualified!r} -> {obj!r}")
    return obj


class ScriptedRuntime:
    """Generator-driven harness installed in every scheduler subprocess.

    Constructed by ``Scheduler.__init__`` when ``scripted_runtime_fn_path``
    is set. On the driver rank, instantiates the script generator and
    advances it one step per ``_yield_to_script`` call (invoked by
    ``SchedulerRequestReceiver.recv_requests`` every event-loop iter).
    When the generator finishes, every rank raises
    ``ScriptedRuntimeFinished`` so all subprocesses exit together.
    """

    def __init__(
        self,
        *,
        scheduler: "Scheduler",
        script_fn_path: str,
        tokenizer_recv_proxy: TokenizerRecvProxy,
    ) -> None:
        self._scheduler = scheduler
        self._tokenizer_recv_proxy = tokenizer_recv_proxy
        self._is_driver = (
            scheduler.ps.pp_rank == 0
            and scheduler.ps.tp_rank == 0
            and scheduler.ps.attn_cp_rank == 0
        )
        self._script_fn_path = script_fn_path

        if self._is_driver:
            script_fn = _resolve_fn(script_fn_path)
            generator = script_fn(self)
            if not hasattr(generator, "__next__"):
                raise TypeError(
                    f"scripted_runtime function {script_fn_path!r} must be a "
                    f"generator (use 'yield' inside it); got {type(generator).__name__}"
                )
            self._generator: Optional[Generator] = generator
        else:
            self._generator = None

        self._req_handles: dict[str, ReqHandle] = {}
        self._req_counter = 0

    # ============================================================
    # Public API: called from test scripts on the driver rank.
    # ============================================================

    def start_req(
        self,
        *,
        prompt_len: int,
        max_new_tokens: int = 8,
    ) -> ReqHandle:
        """Inject a synthetic request into the scheduler's input queue.

        Visible to the scheduler on the next ``yield`` (next
        ``recv_requests`` iteration).
        """
        assert self._is_driver, "start_req is only callable from the driver rank"
        rid = f"scripted-{self._req_counter}"
        self._req_counter += 1
        req = self._build_tokenized_req(
            rid=rid,
            prompt_len=prompt_len,
            max_new_tokens=max_new_tokens,
        )
        self._tokenizer_recv_proxy.inject(req)
        handle = ReqHandle(rid=rid, runtime=self)
        self._req_handles[rid] = handle
        return handle

    # ============================================================
    # Lookups used by ReqHandle (driver-rank-local view).
    # ============================================================

    def _lookup_req_status(self, rid: str) -> ReqStatus:
        if any(getattr(r, "rid", None) == rid for r in self._scheduler.waiting_queue):
            return "waiting"
        running_batch = self._scheduler.running_batch
        if running_batch is not None and any(
            getattr(r, "rid", None) == rid for r in running_batch.reqs
        ):
            return "running"
        return "unknown"

    # ============================================================
    # Internal: invoked by SchedulerRequestReceiver at every iter.
    # ============================================================

    def _yield_to_script(self) -> None:
        """Advance the generator one step (driver only) and broadcast
        completion state. Raises :class:`ScriptedRuntimeFinished` on
        every rank when the script finishes or raises.
        """
        if self._is_driver:
            payload: List = list(self._advance_generator())
        else:
            # ``broadcast_pyobj`` ignores the value on non-source ranks.
            payload = []

        payload = broadcast_pyobj(
            data=payload,
            rank=self._scheduler.world_group.rank,
            dist_group=self._scheduler.world_group.cpu_group,
            src=0,
        )
        done, exc_tb = payload[0], payload[1]
        if done:
            raise ScriptedRuntimeFinished(
                ok=(exc_tb is None),
                exc_traceback=exc_tb,
            )

    def _advance_generator(self) -> Tuple[bool, Optional[str]]:
        try:
            next(self._generator)
            return (False, None)
        except StopIteration:
            return (True, None)
        except BaseException:  # noqa: BLE001 — capture every kind of failure
            return (True, traceback.format_exc())

    # ============================================================
    # Helpers
    # ============================================================

    @staticmethod
    def _build_tokenized_req(
        *,
        rid: str,
        prompt_len: int,
        max_new_tokens: int,
    ) -> TokenizedGenerateReqInput:
        # Token id 1 is BOS for most tokenizers; any valid token works
        # since the harness does not validate decode quality.
        input_ids = array("i", [1] * prompt_len)
        sampling_params = SamplingParams(max_new_tokens=max_new_tokens)
        return TokenizedGenerateReqInput(
            rid=rid,
            input_text="",
            input_ids=input_ids,
            mm_inputs=None,
            sampling_params=sampling_params,
            return_logprob=False,
            logprob_start_len=0,
            top_logprobs_num=0,
            token_ids_logprob=[],
            stream=False,
        )
