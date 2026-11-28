# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the SGLang project

"""One tune table and the communicators it opened.

``GroupCoordinator`` keeps two ``TunedAllReduce`` instances, eager and graph
capture. Either is None when that table was not given. Opening and dispatch
live here so ``parallel_state`` only chooses the instance and, when
``apply_tuned`` returns None, continues with the built-in all-reduce order.
"""

from __future__ import annotations

import importlib
import inspect
import logging
import os
import re
from typing import (
    TYPE_CHECKING,
    Callable,
    Dict,
    List,
    Literal,
    NamedTuple,
    Optional,
    Protocol,
    Tuple,
    Union,
    cast,
)

import torch
from torch.distributed import ProcessGroup

from sglang.srt.distributed.device_communicators.allreduce_tune_table import (
    AllreduceTuneTable,
)
from sglang.srt.distributed.device_communicators.pynccl import PyNcclCommunicator
from sglang.srt.distributed.device_communicators.torch_symm_mem import (
    TorchSymmMemCommunicator,
)
from sglang.srt.environ import envs
from sglang.srt.model_executor.runner_backend_utils.tc_piecewise_cuda_graph import (
    is_in_tc_piecewise_cuda_graph,
)

if TYPE_CHECKING:
    # Not imported at run time: custom_all_reduce -> custom_all_reduce_utils
    # imports parallel_state, which imports this module.
    from sglang.srt.distributed.device_communicators.custom_all_reduce import (
        CustomAllreduce,
    )

logger = logging.getLogger(__name__)

# Method prefix for a kernel the tune table chose. The rest is the full class
# name. ``_all_reduce_out_place`` does not dispatch on that name: it calls
# ``TunedAllReduce.apply_tuned``, which reads the table again.
_TUNED_METHOD_PREFIX = "tuned:"

_QR_QUANT_ENVS = (
    "ROCM_QUICK_REDUCE_QUANTIZATION",
    "AITER_QUICK_REDUCE_QUANTIZATION",
)

_QUANT_CLASS = re.compile(r"QuickAllReduce(Int4|INT4|INT8|INT6|INT3|FP8|FP)$")

# None until the first lookup. False when AITER's class cannot be imported.
# Otherwise the CustomAllreduce class itself.
_AITER_CUSTOM_AR: Union[None, bool, type] = None


class ShareKey(NamedTuple):
    """Identity of a communicator two tables may share.

    ``qr`` compares the codec. ``fly`` compares the schedule. ``one`` is the
    class name alone. Unused fields stay at their defaults, so two keys are
    equal only when the fields that kind uses are equal.
    """

    kind: Literal["qr", "fly", "one"]
    quant: Optional[str] = None
    cast: Optional[bool] = None
    algorithm: str = ""


class _TuneGroup(Protocol):
    """The GroupCoordinator fields this module reads. Avoids the import cycle."""

    cpu_group: ProcessGroup
    device: torch.device
    rank_in_group: int
    world_size: int
    pynccl_comm: Optional[PyNcclCommunicator]
    torch_symm_mem_comm: Optional[TorchSymmMemCommunicator]


class _FlyComm(Protocol):
    """FlyDSL ``allreduce(inp, out)``, including the quant-suffixed aliases."""

    min_bytes: int

    def is_beneficial(self, nbytes: int) -> bool: ...
    def allreduce(self, inp: torch.Tensor, out: torch.Tensor) -> None: ...
    def preload(self) -> None: ...
    def compile(self, inp: torch.Tensor, out: torch.Tensor) -> None: ...


class _QuickComm(Protocol):
    """HIP QuickReduce."""

    def should_quick_allreduce(self, tensor: torch.Tensor) -> bool: ...
    def quick_all_reduce(self, tensor: torch.Tensor) -> torch.Tensor: ...


class _CustomComm(Protocol):
    """``custom_all_reduce`` on SGLang, AITER, or custom all-reduce v2."""

    def should_custom_ar(self, tensor: torch.Tensor) -> bool: ...
    def custom_all_reduce(self, tensor: torch.Tensor) -> Optional[torch.Tensor]: ...


class _AiterCustomComm(_CustomComm, Protocol):
    """AITER's CustomAllreduce, whose past-cap call is ``all_reduce``."""

    world_size: int
    max_size: int
    fully_connected: bool
    _IS_CAPTURING: bool
    enable_register_for_capturing: bool

    def all_reduce(
        self, tensor: torch.Tensor, registered_input: bool = False
    ) -> torch.Tensor: ...


class _SglangCustomImpl(Protocol):
    """SGLang CustomAllreduce past ``should_custom_ar``."""

    _IS_CAPTURING: bool
    tms_cudagraph: bool

    def _all_reduce_impl(
        self, tensor: torch.Tensor, registered: bool = False
    ) -> torch.Tensor: ...


# One opened communicator. The collective's callback narrows this union.
Comm = Union[
    PyNcclCommunicator,
    TorchSymmMemCommunicator,
    "CustomAllreduce",
    _FlyComm,
    _QuickComm,
    _CustomComm,
]

# Values ``_constructor_kwargs`` passes into a communicator constructor.
_CtorArg = Union[ProcessGroup, torch.device, str, int, bool]

# group is the GroupCoordinator. Communicators are whatever class the table named.
FitsFn = Callable[[_TuneGroup, Comm, torch.Tensor], bool]
CallFn = Callable[[_TuneGroup, Comm, torch.Tensor], Optional[torch.Tensor]]


def _flydsl_quick_allreduce_fits(
    group: _TuneGroup, comm: _FlyComm, tensor: torch.Tensor
) -> bool:
    """Whether AITER FlyDSL INT4 QuickReduce can run this tensor.

    The class itself refuses anything under ``min_bytes``. bf16, a contiguous
    layout, and 16-byte alignment are the same limits the tuner applies.
    """
    if tensor.dtype != torch.bfloat16 or not tensor.is_contiguous():
        return False
    nbytes = tensor.numel() * tensor.element_size()
    if nbytes % 16 != 0 or tensor.data_ptr() % 16 != 0:
        return False
    if hasattr(comm, "is_beneficial"):
        return bool(comm.is_beneficial(nbytes))
    return nbytes >= int(getattr(comm, "min_bytes", 0))


# AITER's CustomAllreduce, or None when that package has no such class.
def _load_aiter_custom_class() -> Optional[type]:
    global _AITER_CUSTOM_AR
    if _AITER_CUSTOM_AR is None:
        try:
            imported = __import__(
                "aiter.dist.device_communicators.custom_all_reduce",
                fromlist=["CustomAllreduce"],
            )
        except ImportError:
            imported = None
        # A present package can still lack the class. A missing name has to
        # miss instead of raising.
        found = None if imported is None else getattr(imported, "CustomAllreduce", None)
        _AITER_CUSTOM_AR = found if isinstance(found, type) else False
    cls = _AITER_CUSTOM_AR
    return cls if isinstance(cls, type) else None


# Drop a communicator this table constructed and then refused.
def _close_comm(comm: Comm) -> None:
    close = getattr(comm, "close", None)
    if callable(close):
        close()


# The fits/call stored on a communicator this table opened.
def _tuned_spec(comm: Comm) -> Optional[_BoundCall]:
    bound = getattr(comm, "_sglang_tune_spec", None)
    return bound if isinstance(bound, _BoundCall) else None


# FlyDSL schedule from the table: mesh, unless the table says ring.
def _fly_algorithm(table: AllreduceTuneTable) -> str:
    algorithm = getattr(table, "fly_algorithm", None) or "mesh"
    if algorithm not in ("mesh", "ring"):
        return "mesh"
    return algorithm


class _TableSpec:
    """One class named by the tune table, and the collective that row timed."""

    # Split ``module.Class`` into the module and the class name.
    def __init__(self, class_name: str, method: str) -> None:
        module, _, name = class_name.rpartition(".")
        self.class_name: str = class_name
        self.module: str = module
        self.name: str = name
        self.method: str = method


# Run FlyDSL ``allreduce(inp, out)`` and return the output.
def _call_tuned_flydsl(
    group: _TuneGroup, comm: _FlyComm, tensor: torch.Tensor
) -> torch.Tensor:
    out = torch.empty_like(tensor)
    comm.allreduce(tensor, out)
    return out


# Byte size past a custom all-reduce cap, or None when the tensor cannot launch.
def _past_cap_nbytes(tensor: torch.Tensor) -> Optional[int]:
    from sglang.srt.distributed.device_communicators.custom_all_reduce_utils import (
        is_weak_contiguous,
    )

    nbytes = tensor.numel() * tensor.element_size()
    if nbytes % 16 != 0 or not is_weak_contiguous(tensor):
        return None
    return nbytes


# True or False from the class check. None when that check raised.
def _should_custom_ar(comm: _CustomComm, tensor: torch.Tensor) -> Optional[bool]:
    try:
        return bool(comm.should_custom_ar(tensor))
    except Exception:
        return None


# SGLang CustomAllreduce, including sizes up to its whole pool.
def _fits_sglang_custom(
    group: _TuneGroup, comm: CustomAllreduce, tensor: torch.Tensor
) -> bool:
    accepted = _should_custom_ar(comm, tensor)
    if accepted is None:
        return False
    if accepted:
        return True
    # A tune-table row also covers sizes up to the next row, so a size the
    # tuner never measured can land here. The lift stops at this class's pool.
    nbytes = _past_cap_nbytes(tensor)
    if nbytes is None or tensor.dtype not in comm._SUPPORTED_DTYPES:
        return False
    topology_ok = bool(comm.use_amd_deterministic_impl or comm.full_nvlink)
    return topology_ok and nbytes <= int(comm.max_size)


def _call_sglang_custom(
    group: _TuneGroup, comm: CustomAllreduce, tensor: torch.Tensor
) -> Optional[torch.Tensor]:
    """Inside the cap, ``custom_all_reduce``. Past it, ``_all_reduce_impl``."""
    if comm.should_custom_ar(tensor):
        return comm.custom_all_reduce(tensor)
    sglang_comm = cast(_SglangCustomImpl, comm)
    if sglang_comm._IS_CAPTURING:
        if torch.cuda.is_current_stream_capturing():
            return sglang_comm._all_reduce_impl(
                tensor, registered=not sglang_comm.tms_cudagraph
            )
        if is_in_tc_piecewise_cuda_graph():
            return sglang_comm._all_reduce_impl(tensor, registered=False)
        return torch.zeros_like(tensor)
    return sglang_comm._all_reduce_impl(tensor, registered=False)


# AITER CustomAllreduce. Past the cap the kernel is two-stage, so half the pool.
def _fits_aiter_custom(
    group: _TuneGroup, comm: _AiterCustomComm, tensor: torch.Tensor
) -> bool:
    accepted = _should_custom_ar(comm, tensor)
    if accepted is None:
        return False
    if accepted:
        return True
    nbytes = _past_cap_nbytes(tensor)
    if nbytes is None:
        return False
    topology_ok = comm.world_size == 2 or comm.fully_connected
    return topology_ok and nbytes <= int(comm.max_size) // 2


def _call_aiter_custom(
    group: _TuneGroup, comm: _AiterCustomComm, tensor: torch.Tensor
) -> Optional[torch.Tensor]:
    """Inside the cap, ``custom_all_reduce``. Past it, ``all_reduce``."""
    if comm.should_custom_ar(tensor):
        return comm.custom_all_reduce(tensor)
    return TunedAllReduce._aiter_custom_ar_past_cap(comm, tensor)


# Custom all-reduce v2 has no wider buffer than ``should_custom_ar``.
def _fits_v2(group: _TuneGroup, comm: _CustomComm, tensor: torch.Tensor) -> bool:
    return _should_custom_ar(comm, tensor) is True


def _call_v2(
    group: _TuneGroup, comm: _CustomComm, tensor: torch.Tensor
) -> Optional[torch.Tensor]:
    return comm.custom_all_reduce(tensor)


# Torch symmetric memory accepts the tensor when its own check does.
def _fits_symm(
    group: _TuneGroup, comm: TorchSymmMemCommunicator, tensor: torch.Tensor
) -> bool:
    return bool(comm.should_torch_symm_mem_allreduce(tensor))


# ``all_reduce`` returns None when the call wrote nothing. None would make
# the caller run a second reduction, so the input stands in for that result.
def _call_symm(
    group: _TuneGroup, comm: TorchSymmMemCommunicator, tensor: torch.Tensor
) -> torch.Tensor:
    out = comm.all_reduce(tensor)
    return tensor if out is None else out


# HIP QuickReduce accepts the tensor when its own check does.
def _fits_quick(group: _TuneGroup, comm: _QuickComm, tensor: torch.Tensor) -> bool:
    return bool(comm.should_quick_allreduce(tensor))


# HIP QuickReduce returns the reduced tensor.
def _call_quick(group: _TuneGroup, comm: _QuickComm, tensor: torch.Tensor) -> torch.Tensor:
    return comm.quick_all_reduce(tensor)


# PyNccl outplace all-reduce has no separate fits check.
def _fits_pynccl(group: _TuneGroup, comm: PyNcclCommunicator, tensor: torch.Tensor) -> bool:
    return True


# Run PyNccl ``outplace_all_reduce`` with the communicator enabled.
def _call_pynccl(
    group: _TuneGroup, comm: PyNcclCommunicator, tensor: torch.Tensor
) -> torch.Tensor:
    with comm.change_state(enable=True):
        out = comm.outplace_all_reduce(tensor)
    return tensor if out is None else out


class _BoundCall:
    """fits/call for the collective a tune-table row recorded."""

    # Store the fits check, the call, and whether capture must enter it.
    def __init__(self, fits: FitsFn, call: CallFn, captures: bool) -> None:
        self.fits: FitsFn = fits
        self.call: CallFn = call
        self.captures: bool = captures


# A communicator GroupCoordinator already created. Only these two are reused.
GroupComm = Union[PyNcclCommunicator, TorchSymmMemCommunicator]

_QR_CAST_ENV = "ROCM_QUICK_REDUCE_CAST_BF16_TO_FP16"


# Import the class named by one tune-table row, or None when it cannot be imported.
def _import_class(spec: _TableSpec) -> Optional[type]:
    qualname = spec.class_name
    try:
        try:
            cls = getattr(importlib.import_module(spec.module), spec.name)
        except Exception:
            cls = None
        if cls is None:
            # FlyDSL's QuickAllReduceInt4 alias lives on aiter.ops.flydsl,
            # not in the submodule the scan sometimes records.
            for module_name in (
                "aiter.ops.flydsl",
                "aiter.ops.flydsl.quick_allreduce",
            ):
                try:
                    module = importlib.import_module(module_name)
                except ImportError:
                    continue
                cls = getattr(module, spec.name, None)
                if cls is not None:
                    break
            if cls is None:
                raise ImportError(f"aiter.ops.flydsl has no {spec.name}")
        if not isinstance(cls, type):
            raise TypeError(f"{qualname} is not a class")
        return cls
    except Exception as e:
        logger.warning("[AR] Tune table names %s, which did not import: %s.", qualname, e)
        return None


# The group's PyNccl or symmetric-memory object when it is the class the row named.
def _group_comm(group: _TuneGroup, cls: type) -> Optional[GroupComm]:
    pynccl = group.pynccl_comm
    if isinstance(pynccl, cls):
        return pynccl
    symm = group.torch_symm_mem_comm
    if isinstance(symm, cls):
        return symm
    return None


# ``__init__`` parameters, including bases reached through ``**kwargs``.
def _init_parameters(cls: type) -> Dict[str, inspect.Parameter]:
    owner = next(
        (klass for klass in cls.__mro__ if "__init__" in klass.__dict__),
        cls,
    )
    merged: Dict[str, inspect.Parameter] = {}
    accepts_var_keyword = False
    for klass in owner.__mro__:
        init = klass.__dict__.get("__init__")
        if init is None:
            continue
        try:
            parameters = inspect.signature(init).parameters
        except (TypeError, ValueError):
            continue
        var_keyword = any(
            param.kind is inspect.Parameter.VAR_KEYWORD for param in parameters.values()
        )
        if klass is owner:
            merged.update(parameters)
            accepts_var_keyword = var_keyword
            if not var_keyword:
                break
            continue
        if not accepts_var_keyword:
            break
        for name, param in parameters.items():
            if name in merged or param.kind in (
                inspect.Parameter.VAR_POSITIONAL,
                inspect.Parameter.VAR_KEYWORD,
            ):
                continue
            merged[name] = param
        if not var_keyword:
            break
    return merged


# Fill the constructor from the group and the table's FlyDSL schedule.
def _constructor_kwargs(cls: type, group: _TuneGroup, algorithm: str) -> Dict[str, _CtorArg]:
    kwargs: Dict[str, _CtorArg] = {}
    for name, param in _init_parameters(cls).items():
        if name == "self" or param.kind in (
            inspect.Parameter.VAR_POSITIONAL,
            inspect.Parameter.VAR_KEYWORD,
        ):
            continue
        if name in ("group", "cpu_group"):
            kwargs[name] = group.cpu_group
        elif name == "algorithm":
            kwargs[name] = algorithm
        elif name == "device":
            kwargs[name] = group.device
        elif name == "rank":
            kwargs[name] = group.rank_in_group
        elif name == "world_size":
            kwargs[name] = group.world_size
        elif name == "enable_register_for_capturing":
            kwargs[name] = not envs.SGLANG_MEMORY_SAVER_CUDA_GRAPH.get()
        elif param.default is inspect.Parameter.empty:
            raise TypeError(f"{cls.__name__} requires {name}")
    return kwargs


# Put the QuickReduce codec env back after a construct attempt.
def _restore_quick_env(
    previous: Optional[Dict[str, Optional[str]]],
    previous_cast: Optional[str],
    cast_bf16_to_fp16: Optional[bool],
) -> None:
    if previous is not None:
        for name, value in previous.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value
    if previous_cast is not None or (previous is not None and cast_bf16_to_fp16 is not None):
        if previous_cast is None:
            os.environ.pop(_QR_CAST_ENV, None)
        elif previous is not None:
            os.environ[_QR_CAST_ENV] = previous_cast


# Construct the class. None when QuickReduce has no codec or the constructor fails.
def _construct_comm(
    cls: type,
    group: _TuneGroup,
    table: AllreduceTuneTable,
    method: str,
    quant: bool,
    qualname: str,
) -> Optional[Comm]:
    previous: Optional[Dict[str, Optional[str]]] = None
    previous_cast: Optional[str] = None
    try:
        if method == "quick_all_reduce":
            if not table.qr_quant:
                logger.warning(
                    "[AR] Tune table %s names %s but records no qr_quant.",
                    table.source,
                    qualname,
                )
                return None
            previous_cast = os.environ.get(_QR_CAST_ENV)
            if table.qr_cast_bf16_to_fp16 is not None:
                os.environ[_QR_CAST_ENV] = "1" if table.qr_cast_bf16_to_fp16 else "0"
            previous = {name: os.environ.get(name) for name in _QR_QUANT_ENVS}
            for name in _QR_QUANT_ENVS:
                os.environ[name] = table.qr_quant
        comm: Comm = cls(**_constructor_kwargs(cls, group, _fly_algorithm(table)))
        # Compile a FlyDSL communicator before the first call or a capture.
        if method == "allreduce" or quant:
            fly = cast(_FlyComm, comm)
            if hasattr(fly, "preload"):
                fly.preload()
            elif hasattr(fly, "compile"):
                warmup = torch.empty(
                    1024 * 1024 // 2,
                    dtype=torch.bfloat16,
                    device=group.device,
                )
                fly.compile(warmup, torch.empty_like(warmup))
                torch.cuda.synchronize(group.device)
        return comm
    except Exception as e:
        logger.warning("Setup %s for the tune table failed with %s.", qualname, e)
        return None
    finally:
        _restore_quick_env(previous, previous_cast, table.qr_cast_bf16_to_fp16)


# fits/call for this class, or None when this server has none for it.
# ``all_reduce`` and ``custom_all_reduce`` are different collectives on
# different classes, so the class selects the call. The row's method only
# says which collective was timed.
def _bound_call(cls: type, class_name: str, captures: bool) -> Optional[_BoundCall]:
    # Imported here, after parallel_state has finished loading (see TYPE_CHECKING above).
    from sglang.srt.distributed.device_communicators.custom_all_reduce import (
        CustomAllreduce,
    )

    if ".flydsl." in class_name or _QUANT_CLASS.search(cls.__name__):
        return _BoundCall(
            cast(FitsFn, _flydsl_quick_allreduce_fits),
            cast(CallFn, _call_tuned_flydsl),
            False,
        )
    if class_name.endswith(".QuickAllReduce"):
        return _BoundCall(cast(FitsFn, _fits_quick), cast(CallFn, _call_quick), False)
    if issubclass(cls, PyNcclCommunicator):
        return _BoundCall(cast(FitsFn, _fits_pynccl), cast(CallFn, _call_pynccl), captures)
    if issubclass(cls, TorchSymmMemCommunicator):
        return _BoundCall(cast(FitsFn, _fits_symm), cast(CallFn, _call_symm), captures)
    if issubclass(cls, CustomAllreduce):
        return _BoundCall(
            cast(FitsFn, _fits_sglang_custom),
            cast(CallFn, _call_sglang_custom),
            captures,
        )
    if class_name.endswith(".CustomAllReduceV2"):
        return _BoundCall(cast(FitsFn, _fits_v2), cast(CallFn, _call_v2), captures)
    aiter_cls = _load_aiter_custom_class()
    if aiter_cls is not None and issubclass(cls, aiter_cls):
        return _BoundCall(
            cast(FitsFn, _fits_aiter_custom),
            cast(CallFn, _call_aiter_custom),
            captures,
        )
    return None


class TunedAllReduce:
    """One tune table and the communicators it opened.

    ``GroupCoordinator`` holds two of these, eager and graph capture, and
    either is None when that table was not given. A class name present in
    both tables shares one communicator (``opened``), so graph capture
    registers that object once. ``custom_all_reduce`` then hands the kernel
    the captured input instead of copying it into a staging buffer.
    QuickAllReduce is one exception: its class name does not include the
    codec, so a second table measured at another ``qr_quant`` opens its own
    instance. ``QuickAllReduceInt4`` and ``FlyQuickAllReduce`` are the other:
    a second table measured with another ``fly_algorithm`` opens its own.
    """

    # Open one communicator for each class the table names.
    def __init__(
        self,
        group: _TuneGroup,
        table: AllreduceTuneTable,
        opened: Dict[str, Optional[Comm]],
        opened_qr: Dict[str, ShareKey],
    ) -> None:
        self.group: _TuneGroup = group
        self.table: AllreduceTuneTable = table
        self.specs: List[_TableSpec] = [
            _TableSpec(class_name, table.method_for(class_name))
            for class_name in table.class_names()
        ]
        self.comms: Dict[str, Comm] = {}
        self._fallback_logged: bool = False
        for spec in self.specs:
            class_name = spec.class_name
            # HIP QuickReduce's class name omits the codec. FlyDSL's class
            # name omits the schedule. Everything else is one object per class.
            key: ShareKey
            if class_name.endswith(".QuickAllReduce"):
                key = ShareKey("qr", table.qr_quant, table.qr_cast_bf16_to_fp16)
            elif ".flydsl." in class_name:
                key = ShareKey("fly", algorithm=_fly_algorithm(table))
            else:
                key = ShareKey("one")
            reuse = class_name in opened and opened_qr.get(class_name) == key
            if not reuse:
                opened[class_name] = self._open_class(spec)
                opened_qr[class_name] = key

            shared = opened[class_name]
            if shared is not None:
                self.comms[class_name] = shared

    # Import, reuse, or construct the class this row timed. None when it cannot open.
    def _open_class(self, spec: _TableSpec) -> Optional[Comm]:
        qualname = spec.class_name
        method = spec.method
        if not spec.module or not spec.name:
            logger.warning("[AR] Tune table names %s, which is not a class.", qualname)
            return None
        if not method:
            logger.warning(
                "[AR] Tune table %s names %s but records no method. Re-run the "
                "tuner so the row carries the collective it timed.",
                self.table.source,
                qualname,
            )
            return None
        cls = _import_class(spec)
        if cls is None:
            return None
        quant = bool(_QUANT_CLASS.search(spec.name))
        existing: Optional[GroupComm] = _group_comm(self.group, cls)
        reused = False
        comm: Comm
        if existing is not None and method == "outplace_all_reduce":
            if not getattr(existing, "available", False):
                logger.warning("[AR] Tune table names %s, which opened disabled.", qualname)
                return None
            comm = existing
            reused = True
        elif existing is not None and not getattr(existing, "disabled", True):
            comm = existing
            reused = True
        else:
            built = _construct_comm(cls, self.group, self.table, method, quant, qualname)
            if built is None:
                return None
            comm = built
        if method == "outplace_all_reduce":
            ok = bool(getattr(comm, "available", False))
        else:
            ok = not getattr(comm, "disabled", False)
        if not ok:
            if not reused:
                _close_comm(comm)
            logger.warning("[AR] Tune table names %s, which opened disabled.", qualname)
            return None
        # The class selects the collective. ``method`` only records which
        # one the row timed, so a class whose rows disagree was not opened.
        captures = callable(getattr(comm, "capture", None))
        bound = _bound_call(cls, qualname, captures)
        if bound is None:
            if not reused:
                _close_comm(comm)
            logger.warning(
                "[AR] Tune table names %s, which has no tuned call.", qualname
            )
            return None
        setattr(comm, "_sglang_tune_spec", bound)
        return comm

    def custom_ar_comms(self) -> List[Comm]:
        """Custom all-reduce objects this table opened.

        These are the ones ``graph_capture`` must enter. QuickReduce and
        FlyDSL are not among them: their IPC workspace is collected in init.
        """
        comms: List[Comm] = []
        for comm in self.comms.values():
            spec = _tuned_spec(comm)
            if spec is not None and spec.captures:
                comms.append(comm)
        return comms

    # Communicator and spec for this tensor, or None when it cannot run.
    def _choose(self, input_: torch.Tensor) -> Optional[Tuple[Comm, _BoundCall]]:
        nbytes = input_.numel() * input_.element_size()
        class_name = self.table.class_for(nbytes, str(input_.dtype))
        comm = self.comms.get(class_name)
        if comm is not None:
            spec = _tuned_spec(comm)
            if spec is not None and spec.fits(self.group, comm, input_):
                return comm, spec
        if not self._fallback_logged:
            logger.info(
                "[AR] Tune table %s chose %s for %d bytes, which this group "
                "cannot run. Using the built-in all-reduce order for that size.",
                self.table.source,
                class_name,
                nbytes,
            )
            self._fallback_logged = True
        return None

    # ``"tuned:<class name>"`` when this table can run the tensor.
    def select_method(self, input_: torch.Tensor) -> Optional[str]:
        chosen = self._choose(input_)
        if chosen is None:
            return None
        comm, _spec = chosen
        for class_name, stored in self.comms.items():
            if stored is comm:
                return _TUNED_METHOD_PREFIX + class_name
        return None

    def apply_tuned(self, input_: torch.Tensor) -> Optional[torch.Tensor]:
        """Run the communicator this table chose.

        Returns None when this table has no communicator that can run
        ``input_``. The caller then uses the built-in all-reduce order.
        """
        chosen = self._choose(input_)
        if chosen is None:
            return None
        comm, spec = chosen
        return spec.call(self.group, comm, input_)

    # Run AITER's ``all_reduce`` past ``should_custom_ar``, including capture.
    @staticmethod
    def _aiter_custom_ar_past_cap(
        comm: _AiterCustomComm, input_: torch.Tensor
    ) -> torch.Tensor:
        if comm._IS_CAPTURING:
            if torch.cuda.is_current_stream_capturing():
                return comm.all_reduce(
                    input_, registered_input=comm.enable_register_for_capturing
                )
            return torch.zeros_like(input_)
        return comm.all_reduce(input_, registered_input=False)
