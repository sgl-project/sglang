from __future__ import annotations

import weakref
from abc import ABC, abstractmethod
from enum import Enum
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Optional,
    OrderedDict,
    Protocol,
    Tuple,
    TypeGuard,
    Union,
    runtime_checkable,
)

import torch

if TYPE_CHECKING:
    from sglang.srt.batch_overlap.single_batch_overlap import CombineOverlapArgs
    from sglang.srt.layers.moe.token_dispatcher import (
        DeepEPLLCombineInput,
        DeepEPLLDispatchOutput,
        DeepEPNormalCombineInput,
        DeepEPNormalDispatchOutput,
        StandardCombineInput,
        StandardDispatchOutput,
    )
    from sglang.srt.layers.moe.topk import TopKOutput


# ------------------------------ Dispatcher Hook -------------------------------------


class _RemovableDispatcherHandle:

    next_id = 0  # Global counter for unique IDs

    def __init__(self, hooks_dict: OrderedDict):
        self.id = _RemovableDispatcherHandle.next_id
        _RemovableDispatcherHandle.next_id += 1
        self.weak_hooks_dict = weakref.ref(hooks_dict)

    def remove(self):
        hooks_dict = self.weak_hooks_dict()
        if hooks_dict is not None and self.id in hooks_dict:
            del hooks_dict[self.id]


class DispatcherBaseHooks:

    def __init__(self):
        self.hook_dict = OrderedDict[int, Callable]()

    def register_hook(self, hook_fun: Callable) -> _RemovableDispatcherHandle:
        handle = _RemovableDispatcherHandle(self.hook_dict)
        self.hook_dict[handle.id] = hook_fun
        return handle

    def __call__(self, *args, **kwargs) -> Optional[Any]:
        raise NotImplementedError("This method should be overridden by subclasses")


class _PreDispatchHooks(DispatcherBaseHooks):

    def __call__(
        self,
        dispatcher: BaseDispatcher,
        hidden_states: torch.Tensor,
        topk_output: TopKOutput,
    ) -> Optional[Tuple[torch.Tensor, TopKOutput]]:
        for hook_fun in self.hook_dict.values():
            hook_output = hook_fun(dispatcher, hidden_states, topk_output)
            if hook_output is not None:
                hidden_states, topk_output = hook_output
        return hidden_states, topk_output


class _PostDispatchHooks(DispatcherBaseHooks):

    def __call__(
        self, dispatcher: BaseDispatcher, dispatch_output: DispatchOutput
    ) -> Optional[DispatchOutput]:
        for hook_fun in self.hook_dict.values():
            hook_output = hook_fun(dispatcher, dispatch_output)
            if hook_output is not None:
                dispatch_output = hook_output
        return dispatch_output


class _PreCombineHooks(DispatcherBaseHooks):

    def __call__(
        self, dispatcher: BaseDispatcher, combine_input: CombineInput
    ) -> Optional[CombineInput]:
        for hook_fun in self.hook_dict.values():
            hook_output = hook_fun(dispatcher, combine_input)
            if hook_output is not None:
                combine_input = hook_output
        return combine_input


class _PostCombineHooks(DispatcherBaseHooks):

    def __call__(
        self, dispatcher: BaseDispatcher, hidden_states: torch.Tensor
    ) -> Optional[torch.Tensor]:
        for hook_fun in self.hook_dict.values():
            hook_output = hook_fun(dispatcher, hidden_states)
            if hook_output is not None:
                hidden_states = hook_output
        return hidden_states


# ------------------------------ Dispatch Output -------------------------------------


class DispatchOutputChecker:

    @staticmethod
    def format_is_standard(
        dispatch_output: DispatchOutput,
    ) -> TypeGuard[StandardDispatchOutput]:
        return dispatch_output.format.is_standard()

    @staticmethod
    def format_is_triton_kernels(
        dispatch_output: DispatchOutput,
    ) -> TypeGuard[StandardDispatchOutput]:
        return dispatch_output.format.is_standard()

    @staticmethod
    def format_is_deepep_normal(
        dispatch_output: DispatchOutput,
    ) -> TypeGuard[DeepEPNormalDispatchOutput]:
        return dispatch_output.format.is_deepep_normal()

    @staticmethod
    def format_is_deepep_ll(
        dispatch_output: DispatchOutput,
    ) -> TypeGuard[DeepEPLLDispatchOutput]:
        return dispatch_output.format.is_deepep_ll()

    @staticmethod
    def format_is_deepep(
        dispatch_output: DispatchOutput,
    ) -> TypeGuard[Union[DeepEPNormalDispatchOutput, DeepEPLLDispatchOutput]]:
        return dispatch_output.format.is_deepep()


class DispatchOutputFormat(Enum):

    STANDARD = "standard"
    DEEPEP_NORMAL = "deepep_normal"
    DEEPEP_LL = "deepep_ll"

    def is_standard(self) -> bool:
        return self == DispatchOutputFormat.STANDARD

    def is_deepep_normal(self) -> bool:
        return self == DispatchOutputFormat.DEEPEP_NORMAL

    def is_deepep_ll(self) -> bool:
        return self == DispatchOutputFormat.DEEPEP_LL

    def is_deepep(self) -> bool:
        return self in [
            DispatchOutputFormat.DEEPEP_NORMAL,
            DispatchOutputFormat.DEEPEP_LL,
        ]


@runtime_checkable
class DispatchOutput(Protocol):
    """Protocol for dispatch outputs in different formats."""

    hidden_states: torch.Tensor

    @property
    def format(self) -> DispatchOutputFormat: ...


# ------------------------------ Combine Input -------------------------------------


class CombineInputChecker:
    @staticmethod
    def format_is_standard(
        combine_input: CombineInput,
    ) -> TypeGuard[StandardCombineInput]:
        return combine_input.format == CombineInputFormat.STANDARD

    @staticmethod
    def format_is_deepep_normal(
        combine_input: CombineInput,
    ) -> TypeGuard[DeepEPNormalCombineInput]:
        return combine_input.format == CombineInputFormat.DEEPEP_NORMAL

    @staticmethod
    def format_is_deepep_ll(
        combine_input: CombineInput,
    ) -> TypeGuard[DeepEPLLCombineInput]:
        return combine_input.format == CombineInputFormat.DEEPEP_LL

    @staticmethod
    def format_is_deepep(
        combine_input: CombineInput,
    ) -> TypeGuard[Union[DeepEPNormalCombineInput, DeepEPLLCombineInput]]:
        return combine_input.format in [
            CombineInputFormat.DEEPEP_NORMAL,
            CombineInputFormat.DEEPEP_LL,
        ]


class CombineInputFormat(Enum):
    STANDARD = "standard"
    DEEPEP_NORMAL = "deepep_normal"
    DEEPEP_LL = "deepep_ll"


@runtime_checkable
class CombineInput(Protocol):
    """Protocol for combine inputs in different formats."""

    # TODO: add hidden_states to the protocol

    @property
    def format(self) -> CombineInputFormat: ...


# ------------------------------ Base Dispatcher -------------------------------------


class BaseDispatcherConfig(ABC):
    """Base class for dispatcher configs."""

    pass


class BaseDispatcher(ABC):
    """Base class for dispatchers."""

    def __init__(self):
        self.quant_config: Optional[dict] = None

        # Overlap args
        self.overlap_args: Optional[CombineOverlapArgs] = None
        self.meta_overlap_args: Optional[dict] = None

        # Hooks
        self._pre_dispatch_hooks: Optional[_PreDispatchHooks] = None
        self._post_dispatch_hooks: Optional[_PostDispatchHooks] = None
        self._pre_combine_hooks: Optional[_PreCombineHooks] = None
        self._post_combine_hooks: Optional[_PostCombineHooks] = None
        self._original_dispatch_func: Optional[Callable] = None
        self._original_combine_func: Optional[Callable] = None

    @abstractmethod
    def dispatch(
        self, hidden_states: torch.Tensor, topk_output: TopKOutput
    ) -> DispatchOutput:
        pass

    def _dispatch_with_hook(
        self, hidden_states: torch.Tensor, topk_output: TopKOutput
    ) -> DispatchOutput:
        if self._pre_dispatch_hooks is not None:
            hidden_states, topk_output = self._pre_dispatch_hooks(
                self, hidden_states, topk_output
            )
        dispatch_output = self._original_dispatch_func(
            hidden_states=hidden_states, topk_output=topk_output
        )
        if self._post_dispatch_hooks is not None:
            dispatch_output = self._post_dispatch_hooks(self, dispatch_output)
        return dispatch_output

    def _override_dispatch_func(self) -> None:
        if self._original_dispatch_func is None:
            self._original_dispatch_func = self.dispatch
            self.dispatch = self._dispatch_with_hook

    @abstractmethod
    def combine(self, combine_input: CombineInput) -> torch.Tensor:
        pass

    def _combine_with_hook(self, combine_input: CombineInput) -> torch.Tensor:
        if self._pre_combine_hooks is not None:
            combine_input = self._pre_combine_hooks(self, combine_input)
        hidden_states = self._original_combine_func(combine_input=combine_input)
        if self._post_combine_hooks is not None:
            hidden_states = self._post_combine_hooks(self, hidden_states)
        return hidden_states

    def _override_combine_func(self) -> None:
        if self._original_combine_func is None:
            self._original_combine_func = self.combine
            self.combine = self._combine_with_hook

    def register_pre_dispatch_hook(
        self,
        hook: Callable[
            [BaseDispatcher, torch.Tensor, TopKOutput],
            Optional[Tuple[torch.Tensor, TopKOutput]],
        ],
    ) -> _RemovableDispatcherHandle:
        if self._pre_dispatch_hooks is None:
            self._pre_dispatch_hooks = _PreDispatchHooks()
            self._override_dispatch_func()
        handle = self._pre_dispatch_hooks.register_hook(hook)
        return handle

    def register_post_dispatch_hook(
        self, hook: Callable[[BaseDispatcher, DispatchOutput], Optional[DispatchOutput]]
    ) -> _RemovableDispatcherHandle:
        if self._post_dispatch_hooks is None:
            self._post_dispatch_hooks = _PostDispatchHooks()
            self._override_dispatch_func()
        handle = self._post_dispatch_hooks.register_hook(hook)
        return handle

    def register_pre_combine_hook(
        self, hook: Callable[[BaseDispatcher, CombineInput], Optional[CombineInput]]
    ) -> _RemovableDispatcherHandle:
        if self._pre_combine_hooks is None:
            self._pre_combine_hooks = _PreCombineHooks()
            self._override_combine_func()
        handle = self._pre_combine_hooks.register_hook(hook)
        return handle

    def register_post_combine_hook(
        self, hook: Callable[[BaseDispatcher, torch.Tensor], Optional[torch.Tensor]]
    ) -> _RemovableDispatcherHandle:
        if self._post_combine_hooks is None:
            self._post_combine_hooks = _PostCombineHooks()
            self._override_combine_func()
        handle = self._post_combine_hooks.register_hook(hook)
        return handle

    def set_quant_config(self, quant_config: dict) -> None:
        self.quant_config = quant_config

    def set_overlap_args(
        self, combine_overlap_args: CombineOverlapArgs, meta_overlap_args: dict
    ) -> None:
        self.overlap_args = combine_overlap_args
        self.meta_overlap_args = meta_overlap_args

    def clear_overlap_args(self) -> None:
        self.overlap_args = None
        self.meta_overlap_args = None
