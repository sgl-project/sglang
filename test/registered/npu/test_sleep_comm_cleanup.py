"""Stub-based unit tests for the NPU sleep-mode HCCL comm cleanup.

These tests run on any host (no torch / no NPU required): fake ``torch``,
``torch_npu`` and ``sglang`` dependency modules are injected into
``sys.modules`` before the real target modules under test are imported from
the repository.
"""

from __future__ import annotations

import os
import sys
import types
import unittest
import weakref
from types import SimpleNamespace

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
PY = os.path.join(REPO, "python")

# ---------------------------------------------------------------------------
# Shared state: fakes record observable side effects; feature flag is mutable.
# ---------------------------------------------------------------------------
EVENTS: list[str] = []
NEW_GROUP_CALLS: list[dict] = []
DESTROYED_PGS: list[str] = []
_DEEPEP_BUFFERS: dict = {}
_FEATURE_FLAGS = SimpleNamespace(enable_sleep_comm_cleanup=True)


def _reset_events():
    EVENTS.clear()
    NEW_GROUP_CALLS.clear()
    DESTROYED_PGS.clear()


# ---------------------------------------------------------------------------
# Fake torch / torch_npu
# ---------------------------------------------------------------------------
class _FakeTensor:
    def __init__(self):
        self.data = {}

    def __setitem__(self, key, value):
        self.data[key] = value

    def __getitem__(self, key):
        return _FakeTensor()


class _Device:
    def __init__(self, spec):
        self.spec = spec

    def __repr__(self):
        return f"device({self.spec})"

    def __eq__(self, other):
        return isinstance(other, _Device) and self.spec == other.spec


class _FakeProcessGroup:
    _counter = 0

    def __init__(self, desc):
        self.desc = desc
        _FakeProcessGroup._counter += 1
        self.seq = _FakeProcessGroup._counter


class _Stream:
    pass


def _build_fake_torch() -> types.ModuleType:
    torch = types.ModuleType("torch")

    dist = types.ModuleType("torch.distributed")

    class ReduceOp:
        SUM = 2

    class Backend:
        NCCL = "nccl"
        GLOO = "gloo"

    def new_group(ranks=None, backend=None, pg_options=None, timeout=None, group_desc=None):
        NEW_GROUP_CALLS.append(
            {
                "ranks": ranks,
                "backend": backend,
                "pg_options": pg_options,
                "group_desc": group_desc,
            }
        )
        return _FakeProcessGroup(group_desc)

    def destroy_process_group(group=None):
        desc = getattr(group, "desc", None) if group is not None else "default"
        DESTROYED_PGS.append(desc)
        EVENTS.append(f"destroy_pg:{desc}")

    dist.ReduceOp = ReduceOp
    dist.Backend = Backend
    dist.ProcessGroup = _FakeProcessGroup
    dist.Work = _FakeProcessGroup
    dist.new_group = new_group
    dist.destroy_process_group = destroy_process_group
    dist.get_rank = lambda: 0
    dist.get_world_size = lambda group=None: 2
    dist.barrier = lambda group=None: EVENTS.append("barrier")
    dist.all_reduce = lambda *a, **k: None
    dist.send = lambda *a, **k: None
    dist.recv = lambda *a, **k: None
    torch.distributed = dist

    torch.Tensor = _FakeTensor
    torch.device = _Device
    torch.int32 = "int32"
    torch.zeros = lambda *a, **k: _FakeTensor()
    torch.ones = lambda *a, **k: _FakeTensor()
    torch.get_device_module = lambda device=None: SimpleNamespace(
        Stream=_Stream,
        synchronize=lambda: EVENTS.append("device.synchronize"),
    )
    torch.npu = SimpleNamespace(
        synchronize=lambda: EVENTS.append("npu.synchronize"),
    )

    class _InferenceMode:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

    torch.inference_mode = lambda *a, **k: _InferenceMode()
    return torch


def _build_fake_torch_npu() -> types.ModuleType:
    mod = types.ModuleType("torch_npu")

    class _HCCLOptions:
        def __init__(self):
            self.hccl_config = None

    mod._C = SimpleNamespace(
        _distributed_c10d=SimpleNamespace(
            ProcessGroupHCCL=SimpleNamespace(Options=_HCCLOptions)
        )
    )
    return mod


# ---------------------------------------------------------------------------
# Fake sglang dependency modules
# ---------------------------------------------------------------------------
def _pkg(name):
    mod = types.ModuleType(name)
    mod.__path__ = [os.path.join(PY, *name.split("."))]
    sys.modules[name] = mod
    parent_name, _, attr = name.rpartition(".")
    if parent_name:
        setattr(sys.modules[parent_name], attr, mod)
    return mod


def _leaf(name, **attrs):
    mod = types.ModuleType(name)
    for k, v in attrs.items():
        setattr(mod, k, v)
    sys.modules[name] = mod
    parent_name, _, attr = name.rpartition(".")
    if parent_name and parent_name in sys.modules:
        setattr(sys.modules[parent_name], attr, mod)
    return mod


def _leaf_with_getattr(name):
    mod = types.ModuleType(name)
    mod.__getattr__ = lambda attr: type(attr, (), {})
    sys.modules[name] = mod
    parent_name, _, attr = name.rpartition(".")
    if parent_name and parent_name in sys.modules:
        setattr(sys.modules[parent_name], attr, mod)
    return mod


class _FakeComm:
    """Stand-in for NpuCommunicator / PyNcclCommunicator / etc."""

    def __init__(self, group=None, **kwargs):
        self.group = group
        self.destroyed = False
        EVENTS.append(f"create_comm:{type(self).__name__}")

    def destroy(self):
        self.destroyed = True
        EVENTS.append(f"destroy_comm:{type(self).__name__}")


class _FakeMessageQueue:
    @classmethod
    def create_from_process_group(cls, *a, **k):
        EVENTS.append("create_comm:MessageQueue")
        return cls()


def _identity_decorator_factory(*args, **kwargs):
    def deco(fn):
        return fn

    return deco


def _install_sglang_stubs():
    for name in [
        "sglang",
        "sglang.srt",
        "sglang.srt.distributed",
        "sglang.srt.distributed.device_communicators",
        "sglang.srt.managers",
        "sglang.srt.managers.scheduler_components",
        "sglang.srt.layers",
        "sglang.srt.layers.moe",
        "sglang.srt.layers.moe.token_dispatcher",
        "sglang.srt.hardware_backend",
        "sglang.srt.hardware_backend.npu",
        "sglang.srt.model_executor",
        "sglang.srt.model_executor.runner_backend_utils",
        "sglang.srt.eplb",
        "sglang.srt.compilation",
        "sglang.srt.platforms",
        "sglang.srt.disaggregation",
        "sglang.srt.arg_groups",
    ]:
        _pkg(name)

    _leaf("sglang.srt.platforms")
    _leaf(
        "sglang.srt.platforms.device_mixin",
        _DEVICE_TO_DISTRIBUTED_BACKEND={},
    )
    _leaf(
        "sglang.srt.compilation.compilation_config",
        register_split_op=_identity_decorator_factory,
    )
    _leaf(
        "sglang.srt.distributed.utils",
        set_global_tcp_store=lambda *a, **k: None,
    )

    _envs = SimpleNamespace(
        SGLANG_ONE_VISIBLE_DEVICE_PER_PROCESS=SimpleNamespace(get=lambda: False),
        SGLANG_ZBAL_LOCAL_MEM_SIZE=SimpleNamespace(get=lambda: 0),
        SGLANG_IN_DEEPGEMM_PRECOMPILE_STAGE=SimpleNamespace(get=lambda: False),
    )
    _leaf("sglang.srt.environ", envs=_envs)

    _leaf(
        "sglang.srt.model_executor.runner_backend_utils.tc_piecewise_cuda_graph",
        is_in_tc_piecewise_cuda_graph=lambda: False,
    )

    def _get_resources():
        return SimpleNamespace(buffers=_DEEPEP_BUFFERS)

    def _get_exec():
        return SimpleNamespace(features=_FEATURE_FLAGS)

    def _get_model():
        return SimpleNamespace(weight_cache_mode="off")

    _leaf(
        "sglang.srt.runtime_context",
        derive_parallel_widths=lambda *a, **k: None,
        get_global_dwdp_manager=lambda: None,
        set_global_dwdp_manager=lambda *a, **k: None,
        get_parallel=lambda *a, **k: None,
        get_resources=_get_resources,
        get_exec=_get_exec,
        get_model=_get_model,
    )

    _utils = _leaf("sglang.srt.utils")
    for name in (
        "is_cpu", "is_cuda", "is_cuda_alike", "is_gfx95_supported",
        "is_hip", "is_musa", "is_shm_available", "is_xpu",
        "is_blackwell", "is_flashinfer_available",
    ):
        setattr(_utils, name, (lambda: False))
    _utils.is_npu = lambda: True
    _utils.get_int_env_var = lambda key, default=0: default
    _utils.get_bool_env_var = lambda *a, **k: False
    _utils.get_current_device_stream_fast = lambda: None
    _utils.get_cuda_version = lambda: (0, 0)
    _utils.load_json_config = lambda *a, **k: {}

    _leaf(
        "sglang.srt.utils.custom_op",
        register_custom_op=_identity_decorator_factory,
    )
    _leaf("sglang.srt.utils.network", get_local_ip_auto=lambda: "127.0.0.1")
    _leaf("sglang.srt.utils.stale_shm_cleanup", make_shm_name=lambda name: name)

    # device communicator leaves (consumed by GroupCoordinator._init_communicators)
    _leaf(
        "sglang.srt.distributed.device_communicators.custom_all_reduce",
        dispatch_custom_allreduce=lambda **k: _FakeComm,
    )
    _leaf(
        "sglang.srt.distributed.device_communicators.pymscclpp",
        PyMscclppCommunicator=_FakeComm,
    )
    _leaf(
        "sglang.srt.distributed.device_communicators.pynccl",
        PyNcclCommunicator=_FakeComm,
    )
    _leaf(
        "sglang.srt.distributed.device_communicators.pynccl_allocator",
        debug_check_symmetric_mempool=lambda: None,
        is_symmetric_memory_enabled=lambda: False,
        use_symmetric_memory=False,
    )
    _leaf(
        "sglang.srt.distributed.device_communicators.torch_symm_mem",
        TorchSymmMemCommunicator=_FakeComm,
    )
    _leaf(
        "sglang.srt.distributed.device_communicators.shm_broadcast",
        MessageQueue=_FakeMessageQueue,
    )
    for hw in ("hpu", "npu", "xpu"):
        _leaf(
            f"sglang.srt.distributed.device_communicators.{hw}_communicator",
            **{
                f"{hw.capitalize()}Communicator": type(
                    f"{hw.capitalize()}Communicator", (_FakeComm,), {}
                )
            },
        )

    _leaf(
        "sglang.srt.layers.dp_attention",
        is_allocation_symmetric=lambda: False,
        get_is_extend_in_batch=lambda: False,
    )

    # token_dispatcher base classes for deepep.py
    class _BaseDispatcher:
        pass

    class _BaseDispatcherConfig:
        pass

    class _DispatcherBaseHooks:
        hook_dict = {}

    _leaf(
        "sglang.srt.layers.moe.token_dispatcher.base",
        BaseDispatcher=_BaseDispatcher,
        BaseDispatcherConfig=_BaseDispatcherConfig,
        CombineInput=type,  # isinstance(<namedtuple class>, type) is True
        CombineInputFormat=SimpleNamespace(DEEPEP_NORMAL=1, DEEPEP_LL=2),
        DispatcherBaseHooks=_DispatcherBaseHooks,
        DispatchOutput=type,
        DispatchOutputFormat=SimpleNamespace(DEEPEP_NORMAL=1, DEEPEP_LL=2),
    )
    _leaf("sglang.srt.layers.moe.topk", TopKOutput=type)
    _leaf(
        "sglang.srt.layers.moe.utils",
        DeepEPMode=SimpleNamespace(
            NORMAL=SimpleNamespace(
                enable_normal=lambda: True, enable_low_latency=lambda: False
            ),
            LOW_LATENCY=SimpleNamespace(
                enable_normal=lambda: False, enable_low_latency=lambda: True
            ),
            AUTO=SimpleNamespace(
                enable_normal=lambda: True, enable_low_latency=lambda: True
            ),
        ),
        DispatcherOutputDtype=SimpleNamespace(AUTO="auto"),
        get_deepep_config=lambda: None,
        get_deepep_output_dtype=lambda: "auto",
        is_tbo_enabled=lambda: False,
    )
    _leaf("sglang.srt.layers.deep_gemm_wrapper")
    _leaf(
        "sglang.srt.eplb.expert_distribution",
        get_global_expert_distribution_recorder=lambda: None,
    )

    # exec_.py deps (config field namespace)
    class _A:
        def __class_getitem__(cls, item):
            return cls

    class _Arg:
        def __init__(self, *args, **kwargs):
            pass

    class _Derived:
        def __init__(self, *args, **kwargs):
            pass

    _leaf(
        "sglang.srt.arg_groups.arg_utils",
        A=_A,
        Arg=_Arg,
        Derived=_Derived,
    )
    _leaf(
        "sglang.srt.arg_groups.choices",
        ATTENTION_BACKEND_CHOICES=[],
        FP4_GEMM_RUNNER_BACKEND_CHOICES=[],
        FP8_GEMM_RUNNER_BACKEND_CHOICES=[],
        GRAMMAR_BACKEND_CHOICES=[],
        LINEAR_ATTN_KERNEL_BACKEND_CHOICES=[],
        MOE_RUNNER_BACKEND_CHOICES=[],
        RL_ON_POLICY_TARGET_CHOICES=[],
    )

    class _CudaGraphBackend:
        ALL = ["full", "breakable", "tc_piecewise", "disabled"]

    _leaf(
        "sglang.srt.model_executor.cuda_graph_config",
        Backend=_CudaGraphBackend,
        CudaGraphConfig=type("CudaGraphConfig", (), {}),
        parse_cuda_graph_config_arg=lambda value: value,
    )

    # weight_updater deps
    GPU_MEMORY_ALL_TYPES = ["kv_cache", "weights", "cuda_graph"]
    _leaf(
        "sglang.srt.constants",
        GPU_MEMORY_ALL_TYPES=GPU_MEMORY_ALL_TYPES,
        GPU_MEMORY_TYPE_KV_CACHE="kv_cache",
        GPU_MEMORY_TYPE_WEIGHTS="weights",
        GPU_MEMORY_TYPE_CUDA_GRAPH="cuda_graph",
    )
    _leaf(
        "sglang.srt.disaggregation.utils",
        DisaggregationMode=SimpleNamespace(DECODE="decode", PREFILL="prefill", NULL="null"),
    )
    _leaf_with_getattr("sglang.srt.managers.io_struct")


# ---------------------------------------------------------------------------
# Install all stubs, then import the real modules under test.
# ---------------------------------------------------------------------------
sys.modules["torch"] = _build_fake_torch()
sys.modules["torch.distributed"] = sys.modules["torch"].distributed
sys.modules["torch_npu"] = _build_fake_torch_npu()

try:
    import msgspec  # noqa: F401
except ImportError:
    _msgspec = types.ModuleType("msgspec")

    class _Struct:
        def __init_subclass__(cls, **kwargs):
            super().__init_subclass__(**kwargs)
            defaults = {}
            for base in reversed(cls.__mro__[1:]):
                defaults.update(getattr(base, "__defaults__", {}))
            for key, value in cls.__dict__.items():
                if not key.startswith("_") and not callable(value):
                    defaults[key] = value
            cls.__defaults__ = defaults

        def __init__(self, **kwargs):
            for name, default in type(self).__defaults__.items():
                setattr(self, name, kwargs.pop(name, default))
            if kwargs:
                raise TypeError(f"unexpected fields: {sorted(kwargs)}")

    _msgspec.Struct = _Struct
    _msgspec.__path__ = []
    _structs = types.ModuleType("msgspec.structs")
    _structs.fields = lambda source: []
    _msgspec.structs = _structs
    sys.modules["msgspec"] = _msgspec
    sys.modules["msgspec.structs"] = _structs

_install_sglang_stubs()

from sglang.srt.distributed import parallel_state  # noqa: E402
from sglang.srt.distributed.parallel_state import GroupCoordinator  # noqa: E402
from sglang.srt.hardware_backend.npu import sleep_comm_manager  # noqa: E402
from sglang.srt.layers.moe.token_dispatcher import deepep  # noqa: E402
from sglang.srt.managers.scheduler_components import (  # noqa: E402
    weight_updater,
)


def _make_group(group_name="moe_ep"):
    return GroupCoordinator(
        group_ranks=[[0, 1]],
        local_rank=0,
        torch_distributed_backend="hccl",
        use_pynccl=True,
        use_pymscclpp=True,
        use_custom_allreduce=True,
        use_torch_symm_mem_all_reduce=True,
        use_hpu_communicator=True,
        use_xpu_communicator=True,
        use_npu_communicator=True,
        use_message_queue_broadcaster=True,
        group_name=group_name,
    )


class TestGroupCoordinatorReleaseRestore(unittest.TestCase):
    def setUp(self):
        _reset_events()
        os.environ.pop("HCCL_BUFFSIZE", None)
        os.environ.pop("DEEPEP_HCCL_BUFFSIZE", None)
        parallel_state._groups.clear()
        self.group = _make_group()

    def test_init_creates_device_and_cpu_groups(self):
        backends = [c["backend"] for c in NEW_GROUP_CALLS]
        self.assertEqual(backends, ["hccl", "gloo"])
        dev_call = NEW_GROUP_CALLS[0]
        self.assertEqual(dev_call["group_desc"], "moe_ep:device")
        self.assertEqual(
            dev_call["pg_options"].hccl_config["hccl_buffer_size"], 200
        )

    def test_pg_options_skipped_for_non_moe_groups(self):
        parallel_state._groups.clear()
        _reset_events()
        _make_group(group_name="tp")
        self.assertIsNone(NEW_GROUP_CALLS[0]["pg_options"])

    def test_release_order_and_cpu_group_kept(self):
        cpu_group = self.group.cpu_group
        released = self.group.release_device_comm()
        self.assertTrue(released)

        comm_events = [e for e in EVENTS if e.startswith("destroy_comm:")]
        pg_events = [e for e in EVENTS if e.startswith("destroy_pg:")]
        self.assertGreater(len(comm_events), 0)
        # hard ordering: communicators before the process group
        self.assertLess(
            max(EVENTS.index(e) for e in comm_events),
            min(EVENTS.index(e) for e in pg_events),
        )
        # exactly one process group destroyed, and it is not the gloo group
        self.assertEqual(len(pg_events), 1)
        self.assertIn(":device", pg_events[0])
        self.assertNotIn(cpu_group.desc, DESTROYED_PGS)
        self.assertIsNone(self.group.device_group)
        self.assertIs(self.group.cpu_group, cpu_group)

        # idempotent: second release reports nothing
        EVENTS.clear()
        self.assertFalse(self.group.release_device_comm())
        self.assertEqual(EVENTS, [])

    def test_restore_rebuilds_device_only(self):
        cpu_group = self.group.cpu_group
        self.group.release_device_comm()
        EVENTS.clear()

        restored = self.group.restore_device_comm()
        self.assertTrue(restored)
        self.assertIsNotNone(self.group.device_group)
        self.assertIs(self.group.cpu_group, cpu_group)

        new_dev_calls = [c for c in NEW_GROUP_CALLS[2:] if c["backend"] == "hccl"]
        new_cpu_calls = [c for c in NEW_GROUP_CALLS[2:] if c["backend"] == "gloo"]
        self.assertEqual(len(new_dev_calls), 1)
        self.assertEqual(
            new_dev_calls[0]["pg_options"].hccl_config["hccl_buffer_size"], 200
        )
        self.assertEqual(len(new_cpu_calls), 0)
        # communicators recreated on the new device group
        self.assertIsNotNone(self.group.npu_communicator)
        self.assertIs(self.group.npu_communicator.group, self.group.device_group)

        # idempotent: second restore reports nothing
        self.assertFalse(self.group.restore_device_comm())


class TestSleepCommManager(unittest.TestCase):
    def setUp(self):
        parallel_state._groups.clear()
        self.group = _make_group()
        _reset_events()  # drop group-construction events

    def test_release_synchronizes_and_dedups(self):
        # same group referenced under an extra name must be destroyed once
        parallel_state._groups["dup"] = weakref.ref(self.group)

        count = sleep_comm_manager.release_device_comms()
        self.assertEqual(count, 1)
        self.assertEqual(EVENTS[0], "npu.synchronize")
        self.assertEqual(
            len([e for e in EVENTS if e.startswith("destroy_pg:")]), 1
        )

        # idempotent across the registry
        self.assertEqual(sleep_comm_manager.release_device_comms(), 0)

    def test_restore_rebuilds_and_invalidates_deepep_buffer(self):
        self.assertEqual(sleep_comm_manager.release_device_comms(), 1)
        EVENTS.clear()
        state = deepep.DeepEPBuffer._state()
        state.buffer = object()
        parallel_state._groups["dup"] = weakref.ref(self.group)

        count = sleep_comm_manager.restore_device_comms()
        self.assertEqual(count, 1)
        self.assertIsNone(state.buffer)
        self.assertIsNotNone(self.group.device_group)

        self.assertEqual(sleep_comm_manager.restore_device_comms(), 0)


class TestDeepEPBufferInvalidate(unittest.TestCase):
    def test_invalidate_drops_cached_buffer(self):
        state = deepep.DeepEPBuffer._state()
        sentinel = object()
        state.buffer = sentinel
        self.assertIs(deepep.DeepEPBuffer._state().buffer, sentinel)
        deepep.DeepEPBuffer.invalidate_buffer()
        self.assertIsNone(deepep.DeepEPBuffer._state().buffer)
        # invalidate on an empty cache is a no-op
        deepep.DeepEPBuffer.invalidate_buffer()
        self.assertIsNone(deepep.DeepEPBuffer._state().buffer)


class _RecordingMemorySaver:
    def __init__(self):
        self.paused = []
        self.resumed = []

    def pause(self, tag):
        self.paused.append(tag)
        EVENTS.append(f"pause:{tag}")

    def resume(self, tag):
        self.resumed.append(tag)
        EVENTS.append(f"resume:{tag}")


class _RecordingGraphBackend:
    def cleanup(self):
        EVENTS.append("graph_cleanup")


def _make_weight_updater(saver):
    model_runner = SimpleNamespace(
        model=SimpleNamespace(named_buffers=lambda: iter([])),
        decode_cuda_graph_runner=SimpleNamespace(backend=_RecordingGraphBackend()),
        prefill_cuda_graph_runner=SimpleNamespace(backend=_RecordingGraphBackend()),
    )
    tp_worker = SimpleNamespace(model_runner=model_runner)
    return weight_updater.SchedulerWeightUpdaterManager(
        tp_worker=tp_worker,
        draft_worker=None,
        tp_cpu_group=object(),
        memory_saver_adapter=saver,
        flush_cache=lambda **k: True,
        is_fully_idle=lambda: True,
        scheduler=None,
    )


class TestWeightUpdaterHooks(unittest.TestCase):
    def setUp(self):
        parallel_state._groups.clear()
        _DEEPEP_BUFFERS.clear()
        _FEATURE_FLAGS.enable_sleep_comm_cleanup = True
        self.group = _make_group()
        _reset_events()  # drop group-construction events
        self.saver = _RecordingMemorySaver()
        self.mgr = _make_weight_updater(self.saver)

    @staticmethod
    def _req(tags=None):
        return SimpleNamespace(tags=tags)

    def test_release_hook_order(self):
        self.mgr.release_memory_occupation(self._req())

        self.assertEqual(EVENTS[0], "pause:kv_cache")
        self.assertIn("pause:weights", EVENTS)
        idx_graph_pause = EVENTS.index("pause:cuda_graph")
        idx_sync = EVENTS.index("npu.synchronize")
        idx_cleanup = EVENTS.index("graph_cleanup")
        idx_dev_sync = EVENTS.index("device.synchronize")
        # sleep: graphs paused -> comms destroyed -> graph caches dropped
        self.assertLess(idx_graph_pause, idx_sync)
        self.assertLess(idx_sync, idx_cleanup)
        self.assertLess(idx_cleanup, idx_dev_sync)
        # comms actually destroyed
        self.assertIsNone(self.group.device_group)

    def test_resume_hook_order(self):
        self.mgr.release_memory_occupation(self._req())
        EVENTS.clear()

        self.mgr.resume_memory_occupation(self._req())
        idx_create = next(
            i for i, e in enumerate(EVENTS) if e.startswith("create_comm:")
        )
        idx_graph_resume = EVENTS.index("resume:cuda_graph")
        # wakeup: comms restored before graphs resume
        self.assertLess(idx_create, idx_graph_resume)
        self.assertIsNotNone(self.group.device_group)

    def test_switch_off_is_noop(self):
        _FEATURE_FLAGS.enable_sleep_comm_cleanup = False
        self.mgr.release_memory_occupation(self._req())
        self.assertNotIn("npu.synchronize", EVENTS)
        self.assertNotIn("graph_cleanup", EVENTS)
        self.assertIsNotNone(self.group.device_group)

        self.mgr.resume_memory_occupation(self._req())
        self.assertIn("resume:cuda_graph", EVENTS)

    def test_partial_tags_switch_off_keeps_comms(self):
        _FEATURE_FLAGS.enable_sleep_comm_cleanup = False
        self.mgr.release_memory_occupation(self._req(["kv_cache"]))
        self.assertIsNotNone(self.group.device_group)


class TestExecFeatureFlag(unittest.TestCase):
    def test_enable_sleep_comm_cleanup_field(self):
        from sglang.srt.arg_groups.fields import exec_ as exec_fields

        features = exec_fields.ExecFeatures()
        self.assertFalse(features.enable_sleep_comm_cleanup)
        features = exec_fields.ExecFeatures(enable_sleep_comm_cleanup=True)
        self.assertTrue(features.enable_sleep_comm_cleanup)


if __name__ == "__main__":
    unittest.main(verbosity=2)
