import timeit
from typing import Any, Callable, List, Tuple, Type

from sglang.test.ci.ci_register import register_amd_ci
from sglang.utils import TypeBasedDispatcher

register_amd_ci(est_time=10, suite="stage-b-test-small-1-gpu-amd")


class TypeBasedDispatcherList:
    def __init__(self, mapping: List[Tuple[Type, Callable]]):
        self._mapping = mapping
        self._fallback_fn = None

    def add_fallback_fn(self, fallback_fn: Callable):
        self._fallback_fn = fallback_fn

    def __iadd__(self, other: "TypeBasedDispatcher"):
        self._mapping.extend(other._mapping)
        return self

    def __call__(self, obj: Any):
        for ty, fn in self._mapping:
            if isinstance(obj, ty):
                return fn(obj)

        if self._fallback_fn is not None:
            return self._fallback_fn(obj)
        raise ValueError(f"Invalid object: {obj}")


def create_test_mapping(num_types=30):
    types = [type(f"RequestType{i}", (), {}) for i in range(num_types)]

    def create_handler(i):
        def handler(req):
            return f"handler{i}"

        return handler

    handlers = [create_handler(i) for i in range(num_types)]

    return list(zip(types, handlers))


def test_inheritance():
    print("\n" + "=" * 60)
    print("test for inheritance")
    print("=" * 60)

    class BaseRequest:
        pass

    def base_handler(req):
        return "base_handler"

    class DerivedRequest(BaseRequest):
        pass

    mapping = [(BaseRequest, base_handler)]
    dict_dispatcher = TypeBasedDispatcher(mapping)

    derived_obj = DerivedRequest()
    expected = "base_handler"

    # This test will fail with the current implementation, but pass with the suggested MRO-based fix
    result_dict = dict_dispatcher(derived_obj)
    assert result_dict == expected, f"Expected '{expected}', but got '{result_dict}'"
    print("Pass: dict dispatcher handles inheritance.")


def benchmark_with_inheritance():
    """Performance test with inheritance scenarios"""
    print("\nBenchmarking with inheritance scenarios...")

    # Create type hierarchy with inheritance relationships
    class BaseType:
        pass

    class ChildType1(BaseType):
        pass

    class ChildType2(BaseType):
        pass

    class GrandChildType(ChildType1):
        pass

    class UnrelatedType:
        pass

    def base_handler(obj):
        return "handled"

    mapping = [(BaseType, base_handler)]
    dispatcher = TypeBasedDispatcher(mapping)

    test_cases = [
        BaseType(),
        ChildType1(),
        ChildType2(),
        GrandChildType(),
        UnrelatedType(),
    ]

    # Test first call (includes MRO lookup)
    first_call_times = []
    for case in test_cases:
        if not isinstance(case, UnrelatedType):
            time_taken = timeit.timeit(lambda: dispatcher(case), number=1000)
            first_call_times.append(time_taken)

    # Test subsequent calls (using cache)
    cached_call_times = []
    for case in test_cases:
        if not isinstance(case, UnrelatedType):
            time_taken = timeit.timeit(lambda: dispatcher(case), number=1000)
            cached_call_times.append(time_taken)

    print(
        f"First call (with MRO lookup): {sum(first_call_times)/len(first_call_times):.6f}s avg"
    )
    print(f"Cached call: {sum(cached_call_times)/len(cached_call_times):.6f}s avg")
    print(f"Caching improvement: {sum(first_call_times)/sum(cached_call_times):.2f}x")


def benchmark_dispatchers():
    mapping = create_test_mapping(30)
    list_dispatcher = TypeBasedDispatcherList(mapping)
    dist_dispatcher = TypeBasedDispatcher(mapping)

    test_cases = []
    for _, (ty, _) in enumerate(mapping):
        test_cases.append(ty())

    test_scenarios = [
        ("the first", [test_cases[0]] * 1000),
        ("the middle", [test_cases[len(test_cases) // 2]] * 1000),
        ("the last", [test_cases[-1]] * 1000),
        ("the random", test_cases * 1000),
    ]

    print("=" * 60)
    print("TypeBasedDispatcher benchmark test")
    print("=" * 60)

    for scenario_name, cases in test_scenarios:
        print(f"\ntest scenario: {scenario_name}")
        print(f"\ntest numbers: {len(cases)}")

        list_time = timeit.timeit(
            lambda: [list_dispatcher(case) for case in cases], number=10
        )

        dict_time = timeit.timeit(
            lambda: [dist_dispatcher(case) for case in cases], number=10
        )

        print(f"for list: {list_time:.4f} s")
        print(f"for dict: {dict_time:.4f} s")
        print(f"improvement: {list_time/dict_time:.2f} x")
        print(f"time reduce: {(1-dict_time/list_time) * 100:.1f} %")


def test_memory_usage():
    import sys

    mapping = create_test_mapping(30)
    list_dispatcher = TypeBasedDispatcherList(mapping)
    dict_dispatcher = TypeBasedDispatcher(mapping)

    print("\n" + "=" * 60)
    print("compare memory used:")
    print("=" * 60)

    list_size = sys.getsizeof(list_dispatcher._mapping)
    dict_size = sys.getsizeof(dict_dispatcher._mapping)

    print(f"memory used by list version: {list_size} bytes")
    print(f"memory used by dict version: {dict_size} bytes")
    print(f"compare memory used by the two version: {dict_size - list_size} bytes")


def test_edge_case():
    """test for edge case"""
    print("\n" + "=" * 60)
    print("test for edge case")
    print("=" * 60)

    mapping = create_test_mapping(30)
    list_dispatcher = TypeBasedDispatcherList(mapping)
    dict_dispatcher = TypeBasedDispatcher(mapping)

    test_obj = mapping[0][0]()
    result1 = list_dispatcher(test_obj)
    result2 = dict_dispatcher(test_obj)

    assert result1 == result2
    print("Pass for normal test")

    class UnkownType:
        pass

    try:
        list_dispatcher(UnkownType())
        print("exception was thrown from list version as expected")
    except ValueError:
        print("exception thrown from list version was processed...")

    try:
        dict_dispatcher(UnkownType())
        print("exception was thrown from dict version as expected")
    except ValueError:
        print("exception thrown from dict version was processed...")


def simulate_real_workload():
    """simulate real workload"""

    print("\n" + "=" * 60)
    print("simulate real workload")
    print("=" * 60)

    mapping = create_test_mapping(30)

    request_distribution = {
        0: 0.2,
        5: 0.3,
        10: 0.1,
        15: 0.15,
    }

    list_dispatcher = TypeBasedDispatcherList(mapping)
    dict_dispatcher = TypeBasedDispatcher(mapping)

    test_requests = []
    for idx, prob in request_distribution.items():
        count = int(1000 * prob)
        test_requests.extend([mapping[idx][0]()] * count)

    remaining = 1000 - len(test_requests)
    for i in range(remaining):
        test_requests.append(mapping[i % len(mapping)][0]())

    list_time = timeit.timeit(
        lambda: [list_dispatcher(req) for req in test_requests], number=100
    )

    dict_time = timeit.timeit(
        lambda: [dict_dispatcher(req) for req in test_requests], number=100
    )

    print(f"list version: {list_time:.4f} s")
    print(f"dict version: {dict_time:.4f} s")
    print(f"improvement: {list_time/dict_time:.2f} x")


if __name__ == "__main__":
    benchmark_dispatchers()
    test_memory_usage()
    test_edge_case()
    simulate_real_workload()
    test_inheritance()
    benchmark_with_inheritance()
