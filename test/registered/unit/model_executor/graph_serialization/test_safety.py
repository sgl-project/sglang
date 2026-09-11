"""CPU tests for ``graph_serialization.safety``: ``SafetyPolicy`` kernel and
node-type verdicts, the PhiloxCudaState heuristic, ``CoverageReport`` and the
``SelfCheck`` / ``ShadowVerifier`` stubs.
"""

import struct
import sys

import msgspec
import pytest

from sglang.srt.model_executor.graph_serialization.format import GraphVerdict
from sglang.srt.model_executor.graph_serialization.plan import VerifyMode
from sglang.srt.model_executor.graph_serialization.safety import (
    PHILOX_STATE_NBYTES,
    REFUSED_KERNEL_PREFIXES,
    REFUSED_NODE_TYPES,
    CoverageReport,
    SafetyPolicy,
    SelfCheck,
    ShadowVerifier,
    canonical_node_type,
    is_func_fallback_name,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


# --- kernel verdicts -----------------------------------------------------------


@pytest.mark.parametrize(
    "name, prefix",
    [
        ("ncclDevKernel_AllReduce_Sum_bf16_RING_LL", "ncclDevKernel"),
        ("ncclKernel_SendRecv_RING_SIMPLE_Sum_int8_t", "ncclKernel"),
        ("nvshmemi_transfer_rma_p", "nvshmemi_"),
        ("cross_device_reduce_1stage", "cross_device_reduce_"),
    ],
)
def test_kernel_verdict_refuses_process_local_handle_kernels(name, prefix):
    reason = SafetyPolicy().kernel_verdict(name)

    assert reason is not None
    assert name in reason and prefix in reason
    assert prefix in REFUSED_KERNEL_PREFIXES


def test_kernel_verdict_refuses_func_handle_fallback_name():
    reason = SafetyPolicy().kernel_verdict("func:1234")

    assert reason is not None
    assert "func:1234" in reason
    assert "process-specific" in reason


@pytest.mark.parametrize(
    "name",
    [
        "nvjet_tst_128x64_64x8_1x2_v_bz_TNN",
        "_ZN2at6native29vectorized_elementwise_kernelILi4E",
        "flashinfer::BatchDecodeWithPagedKVCacheKernel",
        "func:abc",  # not the numeric fallback shape
        "funcy_kernel",
        "",
    ],
)
def test_kernel_verdict_accepts_ordinary_names(name):
    assert SafetyPolicy().kernel_verdict(name) is None


def test_is_func_fallback_name():
    assert is_func_fallback_name("func:0")
    assert is_func_fallback_name("func:139877498112000")
    assert not is_func_fallback_name("func:")
    assert not is_func_fallback_name("func:1x")
    assert not is_func_fallback_name("kernel_func:12")


def test_custom_refused_prefixes_replace_the_defaults_but_keep_func_fallback():
    policy = SafetyPolicy(refused_prefixes=("my_private_",))

    assert policy.refused_prefixes == ("my_private_",)
    assert policy.kernel_verdict("my_private_kernel") is not None
    assert policy.kernel_verdict("ncclKernel_AllReduce") is None
    assert policy.kernel_verdict("func:7") is not None


# --- node type verdicts ----------------------------------------------------------


@pytest.mark.parametrize("node_type", REFUSED_NODE_TYPES)
def test_node_type_verdict_refuses_each_listed_type(node_type):
    reason = SafetyPolicy().node_type_verdict(node_type)

    assert reason is not None
    assert node_type in reason


@pytest.mark.parametrize(
    "node_type",
    ["kernel", "memcpy", "memset", "event_record", "event_wait", "empty"],
)
def test_node_type_verdict_accepts_serializable_types(node_type):
    assert SafetyPolicy().node_type_verdict(node_type) is None


@pytest.mark.parametrize(
    "enum_name, canonical",
    [
        ("CU_GRAPH_NODE_TYPE_GRAPH", "child_graph"),
        ("CU_GRAPH_NODE_TYPE_HOST", "host"),
        ("CU_GRAPH_NODE_TYPE_MEM_ALLOC", "mem_alloc"),
        ("CU_GRAPH_NODE_TYPE_MEM_FREE", "mem_free"),
        ("CU_GRAPH_NODE_TYPE_CONDITIONAL", "conditional"),
        ("CU_GRAPH_NODE_TYPE_BATCH_MEM_OP", "batch_memop"),
    ],
)
def test_node_type_verdict_understands_driver_enum_spellings(enum_name, canonical):
    assert canonical_node_type(enum_name) == canonical
    assert SafetyPolicy().node_type_verdict(enum_name) is not None


def test_driver_enum_spelling_of_serializable_types_is_accepted():
    for enum_name in (
        "CU_GRAPH_NODE_TYPE_KERNEL",
        "CU_GRAPH_NODE_TYPE_MEMCPY",
        "CU_GRAPH_NODE_TYPE_EVENT_RECORD",
    ):
        assert SafetyPolicy().node_type_verdict(enum_name) is None


# --- graph verdict fold and strict mode ------------------------------------------


def test_graph_verdict_folds_reasons():
    policy = SafetyPolicy()

    assert policy.graph_verdict([]) == (GraphVerdict.SERIALIZABLE, "")
    assert policy.graph_verdict(["", "first", "second"]) == (
        GraphVerdict.NEEDS_RECAPTURE,
        "first",
    )


def test_check_fallback_raises_only_in_strict_mode():
    SafetyPolicy(strict=False).check_fallback("shape 8 recaptured")

    with pytest.raises(RuntimeError, match="strict cuda-graph cache mode.*shape 8"):
        SafetyPolicy(strict=True).check_fallback("shape 8 recaptured")


# --- PhiloxCudaState heuristic ---------------------------------------------------

SEED_PTR = 0x7F00_0000_0200
OFFSET_PTR = 0x7F00_0000_0400


def philox_state(*, seed=SEED_PTR, offset=OFFSET_PTR, intragraph=0x10, captured=1):
    state = struct.pack("<QQIB3x", seed, offset, intragraph, captured)
    assert len(state) == PHILOX_STATE_NBYTES
    return state


def test_philox_heuristic_detects_captured_form():
    assert SafetyPolicy().looks_like_philox_state(philox_state())


def test_philox_heuristic_detects_state_embedded_in_a_larger_struct():
    raw = struct.pack("<QQ", SEED_PTR - 0x1000, 4096) + philox_state() + b"\0" * 16
    assert SafetyPolicy().looks_like_philox_state(raw)


def test_philox_heuristic_ignores_uncaptured_form():
    # captured_ == 0: the kernel baked the seed and offset values themselves.
    assert not SafetyPolicy().looks_like_philox_state(philox_state(captured=0))


@pytest.mark.parametrize(
    "raw",
    [
        b"",
        b"\0" * 64,
        struct.pack("<QQII", SEED_PTR, OFFSET_PTR, 5, 7),  # two ptrs, two ints
        struct.pack("<QQIB3x", 0, OFFSET_PTR, 1, 1),  # null seed payload
        struct.pack("<QQIB3x", SEED_PTR, SEED_PTR, 1, 1),  # same word twice
        struct.pack("<QQIB3x", 0x8, 0x10, 1, 1),  # tiny scalars
        struct.pack("<QQIBBH", SEED_PTR, OFFSET_PTR, 1, 1, 1, 0),  # non-zero padding
        struct.pack("<QQIB3x", SEED_PTR + 1, OFFSET_PTR, 1, 1),  # unaligned word
    ],
)
def test_philox_heuristic_rejects_lookalikes(raw):
    assert not SafetyPolicy().looks_like_philox_state(raw)


def test_philox_heuristic_errs_toward_refusing_aligned_low_words():
    # Two aligned in-range words followed by the captured marker cannot be told
    # apart from a seed / offset pointer pair by bytes alone; the heuristic
    # refuses (one recaptured shape) rather than risk a silent RNG replay bug.
    raw = struct.pack("<QQIB3x", 0x400000, 0x400008, 1, 1)
    assert SafetyPolicy().looks_like_philox_state(raw)


def test_philox_heuristic_checks_only_aligned_windows():
    # The marker pattern shifted by 4 bytes is not a by-value struct field.
    raw = b"\0" * 4 + philox_state() + b"\0" * 4
    assert not SafetyPolicy().looks_like_philox_state(raw)


# --- CoverageReport ---------------------------------------------------------------


def test_coverage_report_defaults_and_msgpack_round_trip():
    empty = CoverageReport()
    assert empty.graphs_by_verdict == {} and empty.harvest_forwards == 0

    report = CoverageReport(
        graphs_by_verdict={"serializable": 3, "needs_recapture": 1},
        reasons={"size=8": "kernel 'ncclKernel_x' matches refused prefix"},
        slots_by_kind={"weight": 400, "pool": 12},
        kernels_by_provider={"elf_section": 200, "live_harvest": 4},
        harvest_forwards=2,
    )
    decoded = msgspec.msgpack.decode(
        msgspec.msgpack.encode(report), type=CoverageReport
    )
    assert decoded == report
    with pytest.raises(AttributeError):
        report.harvest_forwards = 3  # frozen


# --- SelfCheck / ShadowVerifier ------------------------------------------------------


def test_self_check_is_a_stub():
    check = SelfCheck(codec=object(), resolver=object())
    with pytest.raises(NotImplementedError, match="SelfCheck.check.*section 6.11"):
        check.check(object(), live_graph=None, live_output=None, events=None)


@pytest.mark.parametrize(
    "mode, expected",
    [
        (VerifyMode.NONE, ()),
        ("none", ()),
        (VerifyMode.SHADOW_ONE, ("size=64",)),
        ("shadow-one", ("size=64",)),
        (VerifyMode.SHADOW_ALL, ("size=64", "size=8", "size=1")),
        ("shadow-all", ("size=64", "size=8", "size=1")),
    ],
)
def test_shadow_verifier_selects_shapes_per_mode(mode, expected):
    verifier = ShadowVerifier(mode)
    assert isinstance(verifier.mode, VerifyMode)
    assert verifier.select_shapes(["size=64", "size=8", "size=1"]) == expected


def test_shadow_verifier_shadow_one_with_no_shapes_selects_nothing():
    assert ShadowVerifier("shadow-one").select_shapes([]) == ()
    assert ShadowVerifier().mode is VerifyMode.NONE


def test_shadow_verifier_rejects_unknown_mode():
    with pytest.raises(ValueError):
        ShadowVerifier("sometimes")


def test_shadow_verifier_diff_is_a_stub():
    with pytest.raises(NotImplementedError, match="ShadowVerifier.diff.*6.11"):
        ShadowVerifier("shadow-all").diff(object(), object())


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
