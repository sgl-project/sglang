import dataclasses
from typing import Callable, Dict, List, Tuple, Union, overload

import torch
import triton

from .utils import is_using_profiling_tools


class empty_suppress:
    def __enter__(self):
        return self

    def __exit__(self, *_):
        pass


@triton.jit
def profiler_range_start_marker_kernel():
    pass


def _run_profiler_range_start_marker_kernel():
    profiler_range_start_marker_kernel[(1,)]()


@dataclasses.dataclass
class BenchKinetoRawResult:
    """
    A struct holding the result of `bench_kineto`
    """

    is_using_nsys: bool
    num_tests: int
    time_ranges: Dict[str, List[Tuple[float, float]]]

    def _get_matched_kernel_name(
        self,
        name_substr: str,
        allow_no_match: bool = False,
        allow_multiple_match: bool = False,
    ) -> List[str]:
        matched_names = [
            name for name in self.time_ranges.keys() if name_substr in name
        ]
        if not allow_no_match and len(matched_names) == 0:
            all_kernel_names_str = "\n  - " + "\n  - ".join(self.time_ranges.keys())
            raise ValueError(
                f"Error: No kernel name matched for substring {name_substr}.\nAvailable kernels are: {all_kernel_names_str}"
            )
        if not allow_multiple_match and len(matched_names) > 1:
            raise ValueError(
                f"Error: Multiple kernel matched for substring {name_substr}: {', '.join(matched_names)}"
            )
        return matched_names

    def get_kernel_names(self) -> List[str]:
        return list(self.time_ranges.keys())

    def get_kernel_times(
        self,
        kernel_names_substr: List[str],
        allow_indivisible_run_count: bool = False,
        allow_missing: bool = False,
        allow_multiple_match: bool = False,
        return_avg_individual_run: bool = False,
    ) -> List[float]:
        """
        Get the average each-run time usage of each kernel provided in `kernel_names`

        If return_avg_individual_run is False, return sum(time) / num_tests, else return sum(time) / len(time)
        If is_using_profiling_tools (which is conflict with bench_kineto), return a series of 1 seconds
        """
        if is_using_profiling_tools():
            return [1 for _ in range(len(kernel_names_substr))]

        result = []
        for substr in kernel_names_substr:
            matched_names = self._get_matched_kernel_name(
                substr,
                allow_no_match=allow_missing,
                allow_multiple_match=allow_multiple_match,
            )
            if len(matched_names) == 0:
                assert allow_missing
                result.append(0)
            else:
                time_usage_sum = 0
                run_cnt_sum = 0
                for matched_name in matched_names:
                    run_cnt = len(self.time_ranges[matched_name])
                    if (
                        not allow_indivisible_run_count
                        and run_cnt % self.num_tests != 0
                    ):
                        raise RuntimeError(
                            f"Error: the number of runs for kernel {matched_name} ({run_cnt}) is indivisible by `num_tests` ({self.num_tests})"
                        )
                    time_usage_sum += sum(
                        [end - start for (start, end) in self.time_ranges[matched_name]]
                    )
                    run_cnt_sum += run_cnt
                denominator = (
                    run_cnt_sum if return_avg_individual_run else self.num_tests
                )
                result.append(time_usage_sum / denominator)
        return result

    def get_kernel_time(self, kernel_name_substr: str) -> float:
        return self.get_kernel_times([kernel_name_substr])[0]

    def get_e2e_time(
        self, start_kernel_name_substr: str, end_kenrel_name_substr: str
    ) -> float:
        """
        Get the end-to-end time usage for a sequence of kernels
        defined as "last kernel end time" - "first kernel start time"
        If is_using_profiling_tools (which is conflict with bench_kineto), return 1 second
        """
        if is_using_profiling_tools():
            return 1

        start_kernel_name = self._get_matched_kernel_name(start_kernel_name_substr)[0]
        end_kernel_name = self._get_matched_kernel_name(end_kenrel_name_substr)[0]
        num_start_kernels = len(self.time_ranges[start_kernel_name])
        num_end_kernels = len(self.time_ranges[end_kernel_name])
        if num_start_kernels % self.num_tests != 0:
            raise RuntimeError(
                f"Error: the number of runs for kernel {start_kernel_name} ({num_start_kernels}) is indivisible by `num_tests` ({self.num_tests})"
            )
        if num_end_kernels % self.num_tests != 0:
            raise RuntimeError(
                f"Error: the number of runs for kernel {end_kernel_name} ({num_end_kernels}) is indivisible by `num_tests` ({self.num_tests})"
            )
        time_spans = []
        for i in range(self.num_tests):
            end_time = self.time_ranges[end_kernel_name][
                (i + 1) * (num_end_kernels // self.num_tests) - 1
            ][1]
            start_time = self.time_ranges[start_kernel_name][
                i * (num_start_kernels // self.num_tests)
            ][0]
            time_spans.append((start_time, end_time))
        result = sum([end - start for (start, end) in time_spans]) / self.num_tests
        return result


def bench_kineto(
    fn: Callable, num_tests: int = 30, flush_l2: bool = True
) -> BenchKinetoRawResult:
    """
    Run `fn` for `num_tests` times under `bench_kineto` (CUPTI), and returns a BenchKinetoRawResult
    """
    using_nsys = is_using_profiling_tools()

    # By default, flush L2 with an excessive 8GB memset to give the GPU some (literal) chill time without full idle
    flush_l2_size = int(8e9 // 4)

    schedule = (
        torch.profiler.schedule(wait=0, warmup=1, active=1, repeat=1)
        if not using_nsys
        else None
    )
    profiler = (
        torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CUDA], schedule=schedule
        )
        if not using_nsys
        else empty_suppress()
    )
    with profiler:
        for i in range(2):
            if i == 1 and not using_nsys:
                _run_profiler_range_start_marker_kernel()  # This marks the start of the profiling range
            for _ in range(num_tests):
                if flush_l2:
                    torch.empty(flush_l2_size, dtype=torch.int, device="cuda").zero_()
                enable_nvtx_range = i == 1 and _ == num_tests - 1
                if enable_nvtx_range:
                    torch.cuda.nvtx.range_push("profile_target")
                fn()
                if enable_nvtx_range:
                    torch.cuda.nvtx.range_pop()
            if not using_nsys:
                if i == 0:
                    torch.cuda.synchronize()
                profiler.step()

    if using_nsys:
        return BenchKinetoRawResult(True, num_tests, {})

    from torch.autograd.profiler_util import (  # pylint: disable=import-outside-toplevel
        EventList,
        FunctionEvent,
    )

    events: EventList = profiler.events()  # type: ignore

    # Filter out all events that are not function events
    events: List[FunctionEvent] = [
        event for event in events if isinstance(event, FunctionEvent)
    ]

    # Filter out all events before the range marker
    for idx, event in enumerate(events):
        if event.name == "profiler_range_start_marker_kernel":
            events = events[idx + 1 :]
            break
    else:
        raise RuntimeError("Could not find profiler range start marker kernel event")

    # Get time ranges of each kernel
    kernel_times = {}
    for event in events:
        kernel_name = event.name
        if kernel_name not in kernel_times:
            kernel_times[kernel_name] = []
        kernel_times[kernel_name].append(
            (event.time_range.start / 1e6, event.time_range.end / 1e6)
        )

    return BenchKinetoRawResult(False, num_tests, kernel_times)


@overload
def bench_by_cuda_events(
    kernels: List[Callable], num_warmups_each: int, num_runs_each: int
) -> List[float]: ...


@overload
def bench_by_cuda_events(
    kernels: Callable, num_warmups_each: int, num_runs_each: int
) -> float: ...


def bench_by_cuda_events(
    kernels: Union[List[Callable], Callable], num_warmups_each: int, num_runs_each: int
) -> Union[List[float], float]:
    buf_for_l2_clear = torch.empty(int(256e6 // 4), dtype=torch.int32, device="cuda")

    is_kernel_single_callable = isinstance(kernels, Callable)
    if is_kernel_single_callable:
        kernels = [kernels]

    torch.cuda.synchronize()
    for i in range(num_warmups_each):
        for kernel in kernels:
            kernel()
            if i == 0:
                # Ensure the first run is successful
                try:
                    torch.cuda.synchronize()
                except Exception as e:
                    print(f"Kernel {kernel.__name__} failed on warmup run {i}: {e}")
                    return []

    start_events = [
        [torch.cuda.Event(enable_timing=True) for _ in range(num_runs_each)]
        for _ in kernels
    ]
    end_events = [
        [torch.cuda.Event(enable_timing=True) for _ in range(num_runs_each)]
        for _ in kernels
    ]
    for i in range(num_runs_each):
        for j, kernel in enumerate(kernels):
            buf_for_l2_clear.random_()
            if i == num_runs_each - 1:
                torch.cuda.nvtx.range_push("profile_target")
            start_events[j][i].record()
            kernel()
            end_events[j][i].record()
            if i == num_runs_each - 1:
                torch.cuda.nvtx.range_pop()

    torch.cuda.synchronize()
    time_usages = [
        sum(
            [
                start_events[j][i].elapsed_time(end_events[j][i]) * 1e-3
                for i in range(num_runs_each)
            ]
        )
        / num_runs_each
        for j in range(len(kernels))
    ]
    if is_kernel_single_callable:
        time_usages = time_usages[0]
    return time_usages
