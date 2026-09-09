"""A deliberately narrow external nccl.ep test double backed by CUDA kernels.

Only one rank, two experts, K=2, routes [0,1]/[1,0], and a valid prefix are
supported. This is not a NCCL EP reference implementation. Its purpose is to
observe Python resource calls while real CUDA Graph replay bypasses Python.
"""

import importlib
import sys
import weakref
from contextlib import contextmanager
from dataclasses import dataclass
from types import ModuleType, SimpleNamespace
from typing import Annotated

import torch


@dataclass
class Tensor:
    value: torch.Tensor


class FakeLibrary:
    def __init__(self):
        self.events = []
        self.groups = []
        self.updates = []
        self.core = ModuleType("nccl.core")
        self.core.Communicator = SimpleNamespace
        self.ep = ModuleType("nccl.ep")
        self.ep.Tensor = Tensor
        self.ep.Algorithm = SimpleNamespace(LOW_LATENCY="ll")
        self.ep.Layout = SimpleNamespace(EXPERT_MAJOR="expert_major")
        # Match the external class boundary so EpAudit can wrap real and fake
        # bindings in exactly the same way while preserving each implementation.
        library = self

        class Group(FakeGroup):
            @classmethod
            def create(cls, comm, config):
                return library.create_group(comm, config)

        self.ep.Group = Group
        self.ep.Handle = FakeHandle
        for name in (
            "GroupConfig",
            "HandleConfig",
            "DispatchInputs",
            "DispatchOutputs",
            "DispatchConfig",
            "LayoutInfo",
            "CombineInputs",
            "CombineOutputs",
            "CombineConfig",
        ):
            setattr(self.ep, name, SimpleNamespace)

    def create_group(self, comm, config):
        assert not torch.cuda.is_current_stream_capturing(), "Group creation in capture"
        assert config.num_experts == 2, "Fake EP supports one rank/two experts only"
        group = self.ep.Group(self, config)
        self.groups.append(group)
        self.events.append("group_create")
        return group


class FakeGroup:
    def __init__(self, library, config):
        self.library, self.config = library, config
        self.handles = []
        self.destroyed = False

    def create_handle(self, *, layout, topk_idx, config, stream):
        assert (
            not torch.cuda.is_current_stream_capturing()
        ), "Handle creation in capture"
        assert not self.destroyed
        handle = FakeHandle(self, topk_idx.value)
        # Multiple handles are legal at this API boundary. The production
        # graph owner chooses one to control shared communication state.
        self.handles.append(handle)
        self.library.events.append("handle_create")
        return handle

    def destroy(self):
        assert (
            not torch.cuda.is_current_stream_capturing()
        ), "Group destruction in capture"
        assert not self.destroyed
        assert all(
            handle.destroyed for handle in self.handles
        ), "Live EP handles remain"
        self.destroyed = True
        self.library.events.append("group_destroy")


class FakeHandle:
    def __init__(self, group, ids):
        self.group = group
        self.ids = ids
        self.base = ids.data_ptr()
        self.capacity = group.config.max_dispatch_tokens_per_rank
        self.pending = None
        self.destroyed = False

    def update(self, topk_idx, *, stream):
        assert not torch.cuda.is_current_stream_capturing(), "Handle.update in capture"
        assert not self.destroyed and self.pending is None
        assert topk_idx.value.data_ptr() == self.base, "Routing base address changed"
        assert len(topk_idx.value) <= self.capacity
        self.ids = topk_idx.value
        self.group.library.updates.append((self.base, tuple(self.ids.shape)))
        self.group.library.events.append("handle_update")

    def dispatch(self, inputs, outputs, *, layout_info, config, stream):
        assert not self.destroyed and self.pending is None
        assert config.send_only == 1
        x, received = inputs.tokens.value, outputs.tokens.value
        assert x.shape[0] == self.ids.shape[0]
        if not torch.cuda.is_current_stream_capturing():
            # Do not silently apply this toy layout to unsupported fixtures.
            routes = self.ids.cpu()
            valid = (routes >= 0).any(dim=1)
            n = int(valid.sum())
            assert valid.tolist() == [True] * n + [False] * (len(routes) - n)
            assert torch.equal(
                routes[:n].sort(dim=1).values, torch.tensor([0, 1]).expand(n, 2)
            )
            assert (routes[n:] == -1).all()
        received[:, : len(x)].copy_(x.unsqueeze(0).expand(2, -1, -1))
        count = (self.ids >= 0).any(dim=1).sum(dtype=torch.int32)
        layout_info.expert_counters.value.copy_(count.expand(2))
        # The native send_only continuation borrows tensor descriptors until
        # complete(). Weak references expose premature wrapper destruction.
        self.pending_descriptors = [
            weakref.ref(value)
            for bundle in (inputs, outputs, layout_info)
            for value in vars(bundle).values()
            if isinstance(value, Tensor)
        ]
        self.pending = "dispatch"
        self.group.library.events.append("dispatch")

    def combine(self, inputs, outputs, *, config, stream):
        assert not self.destroyed and self.pending is None
        assert config.send_only == 1
        experts = inputs.tokens.value
        weights = outputs.topk_weights.value
        combined = outputs.tokens.value
        row = torch.arange(len(self.ids), device=self.ids.device)
        result = torch.zeros_like(combined, dtype=torch.float32)
        for route in range(2):
            expert = self.ids[:, route]
            value = experts[expert.clamp_min(0), row].float()
            contribution = value * weights[:, route, None]
            result.add_(torch.where(expert[:, None] >= 0, contribution, 0))
        combined.copy_(result)
        self.pending_descriptors = [
            weakref.ref(value)
            for bundle in (inputs, outputs)
            for value in vars(bundle).values()
            if isinstance(value, Tensor)
        ]
        self.pending = "combine"
        self.group.library.events.append("combine")

    def complete(self, *, config, stream):
        assert not self.destroyed and self.pending is not None
        assert all(
            ref() is not None for ref in self.pending_descriptors
        ), "EP tensor descriptor released before complete"
        self.pending_descriptors = []
        self.pending = None
        self.group.library.events.append("complete")

    def destroy(self):
        assert (
            not torch.cuda.is_current_stream_capturing()
        ), "Handle destruction in capture"
        assert not self.destroyed and self.pending is None
        self.destroyed = True
        self.group.library.events.append("handle_destroy")


@contextmanager
def replace_ep_modules(modules):
    # patch.dict(sys.modules) restores the entire dictionary and removes other
    # modules imported during a test. Re-importing Torch operator registries
    # after that rollback can register the same native namespace twice.
    missing = object()
    previous = {key: sys.modules.get(key, missing) for key in modules}
    sys.modules.update(modules)
    try:
        yield
    finally:
        for key, value in previous.items():
            if value is missing:
                sys.modules.pop(key, None)
            else:
                sys.modules[key] = value


@contextmanager
def dispatcher_environment(*, capacity=32):
    """Replace only the external EP modules, retain actual SGLang operations."""
    from sglang.srt.arg_groups.arg_utils import NS
    from sglang.srt.layers.moe.moe_runner.base import MoeRunnerConfig
    from sglang.srt.layers.moe.utils import MoeA2ABackend, NcclEpMode
    from sglang.srt.runtime_context import get_context, get_flags, get_resources

    @dataclass
    class LabConfig:
        enable_deterministic_inference: Annotated[bool, NS("exec.deterministic")] = (
            False
        )

    library = FakeLibrary()
    root = ModuleType("nccl")
    root.__path__ = []
    root.core, root.ep = library.core, library.ep
    coordinator = SimpleNamespace(
        world_size=1,
        rank=0,
        rank_in_group=0,
        device=torch.device("cuda", torch.cuda.current_device()),
        pynccl_comm=SimpleNamespace(available=True, comm=SimpleNamespace(value=32774)),
        barrier=lambda: None,
    )
    context = get_context()
    with context.preserve_config(), get_flags().moe.override(
        a2a_backend=MoeA2ABackend.NCCL_EP,
        nccl_ep_mode=NcclEpMode.LOW_LATENCY,
        nccl_ep_num_max_dispatch_tokens_per_rank=capacity,
    ), replace_ep_modules(
        {"nccl": root, "nccl.core": library.core, "nccl.ep": library.ep}
    ):
        context.set_server_args(LabConfig())
        module = importlib.import_module(
            "sglang.srt.layers.moe.token_dispatcher.nccl_ep"
        )
        importlib.reload(module)

        def dispatcher(*, layer_id):
            return module.NcclEpDispatcher(
                MoeRunnerConfig(
                    num_experts=2,
                    num_local_experts=2,
                    hidden_size=2048,
                    top_k=2,
                    params_dtype=torch.bfloat16,
                    layer_id=layer_id,
                ),
                coordinator,
            )

        library.dispatcher = dispatcher
        library.coordinator = coordinator
        try:
            yield library
        finally:
            torch.cuda.synchronize()
            module.NcclEpBuffer.destroy()
            get_resources().buffers.pop("nccl_ep_state", None)
            # Restore the import cache while the module remains importable.
            importlib.reload(module)
    assert all(group.destroyed for group in library.groups), "Unclosed fake EP group"
