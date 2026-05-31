import contextlib
import enum
import functools
from typing import Any, ContextManager, List, Optional

import torch


# NOTE: Use concat-style all-gather here.
# Stack-style all-gather has compatibility issues with `torch.compile`.
# See https://github.com/pytorch/pytorch/issues/138795.
def allocate_all_gather(input_: torch.Tensor, world_size: int) -> torch.Tensor:
    input_shape = input_.shape
    return torch.empty(
        (world_size * input_shape[0],) + input_shape[1:],
        dtype=input_.dtype,
        device=input_.device,
    )


def allocate_reduce_scatter(input_: torch.Tensor, world_size: int) -> torch.Tensor:
    input_shape = input_.shape
    assert input_shape[0] % world_size == 0
    return torch.empty(
        (input_shape[0] // world_size,) + input_shape[1:],
        dtype=input_.dtype,
        device=input_.device,
    )


class AllReduceMode(enum.Enum):
    BOTH = "both"
    INPLACE = "inplace"
    OUTPLACE = "outplace"

    def can_inplace(self) -> bool:
        return self is not AllReduceMode.OUTPLACE

    def can_outplace(self) -> bool:
        return self is not AllReduceMode.INPLACE

    def supports(self, inplace: Optional[bool]) -> bool:
        """Whether this mode can serve a request with the given `inplace` flag
        (`None` means the caller accepts either)."""
        if inplace is None:
            return True
        return self.can_inplace() if inplace else self.can_outplace()


class BaseCommunicator:
    name: str  # should be set by subclass

    def __init__(self, world_size: int, disabled: bool = False):
        self.world_size = world_size
        self._disabled = disabled  # NOTE: must use `change_state` to modify

    # Helper functions

    def assert_inplace(self, op: str, inplace: Optional[bool]):
        if inplace is False:
            raise ValueError(f"{self.name} does not allow out-of-place {op} now")

    def assert_outplace(self, op: str, inplace: Optional[bool]):
        if inplace is True:
            raise ValueError(f"{self.name} does not allow in-place {op} now")

    @staticmethod
    def validate(f):
        """
        Guard a public communicator method against calls while the communicator is
        disabled.
        """

        @functools.wraps(f)
        def wrapper(self, *args, **kwargs):
            if self.disabled:
                raise RuntimeError(f"{self.name} is disabled")
            return f(self, *args, **kwargs)

        return wrapper

    def allocate_all_gather(self, input_: torch.Tensor) -> torch.Tensor:
        return allocate_all_gather(input_, self.world_size)

    def allocate_reduce_scatter(self, input_: torch.Tensor) -> torch.Tensor:
        return allocate_reduce_scatter(input_, self.world_size)

    # Public API

    @property
    def disabled(self) -> bool:
        """
        Whether this communicator is currently disabled.

        Public methods on this interface should not be called while the
        communicator is disabled. Subclasses may override this property to add
        derived enablement conditions on top of `_disabled`.

        Do not modify `self._disabled` directly outside this class. Use
        `change_state()` instead.
        """
        return self._disabled

    @contextlib.contextmanager
    def change_state(self, enable: bool):
        """
        Temporarily enable or disable the communicator within a context.

        :param enable: Whether the communicator should be enabled in the
            wrapped block.
        """
        old_value = self._disabled
        self._disabled = not enable
        try:
            yield
        finally:
            self._disabled = old_value

    def graph_capture_context(self) -> Optional[ContextManager[Any]]:
        """
        Return a context manager for graph capture, if the communicator needs
        special handling during capture.

        Returning `None` means no extra handling is required.
        """
        return None

    def should_use_custom_op(self) -> bool:
        """
        Whether this communicator should use `register_custom_op` for `torch.compile`
        compatibility.
        If `False`, this means either:
        1. This backend doesn't support `torch.compile`
        2. This implementation is `torch.compile` friendly
        """
        return False

    def get_all_reduce_mode(self, input_: torch.Tensor) -> Optional[AllReduceMode]:
        """
        Report the preferred all-reduce mode for `input_`.

        :param input_: Input tensor for the all-reduce.
        :return:
            - `AllReduceMode.INPLACE` if in-place all-reduce is preferred.
            - `AllReduceMode.OUTPLACE` if out-of-place all-reduce is preferred.
            - `AllReduceMode.BOTH` if both modes are fine.
            - `None` if the communicator cannot run all-reduce on `input_` (e.g.,
              due to unsupported dtype, shape or alignment).

        This is orthogonal to `self.disabled`, which covers broader reasons why
        the communicator is unavailable.
        """
        raise NotImplementedError()

    def all_reduce(
        self,
        input_: torch.Tensor,
        *,
        inplace: Optional[bool] = None,
    ) -> torch.Tensor:
        """
        Run all-reduce on `input_`.

        Preconditions:
        1. `self.get_all_reduce_mode(input_)` must not return `None`.
        2. `inplace=True` requires `self.get_all_reduce_mode(input_).can_inplace()`.
        3. `inplace=False` requires `self.get_all_reduce_mode(input_).can_outplace()`.
        4. `self.disabled` must be `False`.

        :param input_: Input tensor for the all-reduce.
        :param inplace: Whether the operation should be in-place. If `None`, the
            communicator may choose its preferred mode. If specified, it must be
            consistent with `get_all_reduce_mode(input_)`.
        :return: The reduced tensor. If the operation is in-place, this must be
            `input_` itself.
        """
        raise NotImplementedError()

    def all_gather_into_tensor(
        self,
        input_: torch.Tensor,
        *,
        out: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Run concat-style all-gather on `input_`.

        :param input_: Input tensor for all-gather.
        :param out: Optional preallocated output tensor. If omitted, the
            communicator may allocate and return one.
        :return: The gathered tensor. If `out` is provided, this must be `out`.
        """
        raise NotImplementedError()

    def all_gather(
        self,
        input_: torch.Tensor,
        *,
        out_list: Optional[List[torch.Tensor]] = None,
    ) -> List[torch.Tensor]:
        """
        Run list-style all-gather on `input_`.

        :param input_: Input tensor for all-gather.
        :param out_list: Optional preallocated output list. If provided, the
            communicator should fill and return it.
        :return: The gathered tensor list. If `out_list` is provided, this must
            be `out_list`.
        """
        raise NotImplementedError()

    def reduce_scatter_tensor(
        self,
        input_: torch.Tensor,
        *,
        out: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Run tensor-style reduce-scatter on `input_`.

        :param input_: Input tensor for reduce-scatter.
        :param out: Optional preallocated output tensor. If omitted, the
            communicator may allocate and return one.
        :return: The reduced shard. If `out` is provided, this must be `out`.
        """
        raise NotImplementedError()

    def reduce_scatter(
        self,
        input_list: List[torch.Tensor],
        *,
        out: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Run list-style reduce-scatter on `input_list`.

        :param input_list: Input tensor list for reduce-scatter.
        :param out: Optional preallocated output tensor. If omitted, the
            communicator may allocate and return one.
        :return: The reduced shard. If `out` is provided, this must be `out`.
        """
        raise NotImplementedError()

    def gather(
        self,
        input_: torch.Tensor,
        dst: int,
        *,
        dim: int = 0,
    ) -> Optional[torch.Tensor]:
        """
        Gather `input_` to the destination rank.

        :param input_: Input tensor for gather.
        :param dst: Destination rank within the communicator.
        :param dim: Concatenation dimension in the returned tensor on the
            destination rank.
        :return: The gathered tensor on the destination rank, otherwise `None`.
        """
        raise NotImplementedError()
