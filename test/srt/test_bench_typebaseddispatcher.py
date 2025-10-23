import timeit
from typing import List, Tuple, Type, Callable, Any
from sglang.srt.utils import TypeBasedDispatcher

class TypeBasedDispatcherList:
    '''origin implementation of TypeBasedDispatcher'''

    def __init__(self, mapping: List[Tuple[Type, Callable]]):
        self._mapping = mapping
        self._fallback_fn = None

    def __call__(self, obj: Any):
        for ty, fn in self._mapping:
            if isinstance(obj, ty):
                return fn(obj)
        raise ValueError(f"Invalid object: {obj}")

class RequestType0: pass
class RequestType1: pass
class RequestType2: pass
class RequestType3: pass
class RequestType4: pass
class RequestType5: pass
class RequestType6: pass
class RequestType7: pass
class RequestType8: pass
class RequestType9: pass
class RequestType10: pass
class RequestType11: pass
class RequestType12: pass
class RequestType13: pass
class RequestType14: pass
class RequestType15: pass
class RequestType16: pass
class RequestType17: pass
class RequestType18: pass
class RequestType19: pass
class RequestType20: pass
class RequestType21: pass
class RequestType22: pass
class RequestType23: pass
class RequestType24: pass
class RequestType25: pass
class RequestType26: pass
class RequestType27: pass
class RequestType28: pass
class RequestType29: pass

def handler0(req): return "handler0"
def handler1(req): return "handler1"
def handler2(req): return "handler2"
def handler3(req): return "handler3"
def handler4(req): return "handler4"
def handler5(req): return "handler5"
def handler6(req): return "handler6"
def handler7(req): return "handler7"
def handler8(req): return "handler8"
def handler9(req): return "handler9"
def handler10(req): return "handler10"
def handler11(req): return "handler11"
def handler12(req): return "handler12"
def handler13(req): return "handler13"
def handler14(req): return "handler14"
def handler15(req): return "handler15"
def handler16(req): return "handler16"
def handler17(req): return "handler17"
def handler18(req): return "handler18"
def handler19(req): return "handler19"
def handler20(req): return "handler20"
def handler21(req): return "handler21"
def handler22(req): return "handler22"
def handler23(req): return "handler23"
def handler24(req): return "handler24"
def handler25(req): return "handler25"
def handler26(req): return "handler26"
def handler27(req): return "handler27"
def handler28(req): return "handler28"
def handler29(req): return "handler29"


def create_test_mapping(num_types=30):

    types = [type(f"RequestType{i}", (), {}) for i in range(num_types)]
    handlers = [getattr(__import__(__name__), f"handler{i}") for i in range(num_types)]

    return list(zip(types, handlers))

def benchmark_dispatchers():
    mapping = create_test_mapping(30)
    list_dispatcher = TypeBasedDispatcherList(mapping)
    dist_dispatcher = TypeBasedDispatcher(mapping)

    test_cases = []
    for _, (ty, _) in enumerate(mapping):
        test_cases.append(ty())

    test_scenarios = [
        ("the first", [test_cases[0]] * 1000),
        ("the middle", [test_cases[len(test_cases)//2]] * 1000),
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
            lambda: [list_dispatcher(case) for case in cases],
            number=10
        )

        dict_time = timeit.timeit(
            lambda: [dist_dispatcher(case) for case in cases],
            number=10
        )

        print(f"for list: {list_time:.4f} s")
        print(f"for dict: {dict_time:.4f} s")
        print(f"improvment: {list_time/dict_time:.2f}x")
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

def test_egde_case():
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

    class UnkownType: pass

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

    from collections import Counter
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
        test_requests.append(mapping[ i  % len(mapping)][0]())

    list_time = timeit.timeit(
        lambda: [list_dispatcher(req) for req in test_requests],
        number=100
    )

    dict_time = timeit.timeit(
            lambda: [dict_dispatcher(req) for req in test_requests],
            number=100
    )

    print(f"list version: {list_time:.4f} s")
    print(f"dict version: {dict_time:.4f} s")
    print(f"improvment: {list_time/dict_time:.2f} x")
if __name__ == "__main__":
    benchmark_dispatchers()
    test_memory_usage()
    test_egde_case()
    simulate_real_workload()



