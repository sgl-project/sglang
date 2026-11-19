"""Which NVIDIA runtime children engage on each deployment shape, from structure alone.

Every child is resolved with the runtime master on and every child unset, on the
server topology of each deployment shape: the PD decode, PD prefill and aggregated
shapes of the Qwen3.5 GB300 megakernel layouts, plus other shapes that set the master.
Where a child reads a runtime fact instead of a server argument (hosts per TP group, a
user-set env, the platform), the shape states it. Each child names itself in the INFO
line of its decision, so a server log can be checked against the same table.

Most resolvers call the child's own decision helper. Where that decision is inline in
a builder or a loop, the resolver models it instead, and a change there does not change
this table: the exchange hosts of the shared-memory exchanges (_exchange_hosts) and what
the megakernel plugin binds (the gathered verify sampler).
"""

import os
import unittest
from contextlib import ExitStack, contextmanager
from types import SimpleNamespace
from typing import Callable, Dict, FrozenSet
from unittest.mock import patch

import msgspec

from sglang.srt.arg_groups import overrides
from sglang.srt.disaggregation import decode_freed_slots, pp_early_finish
from sglang.srt.disaggregation.pp_consensus_store import PPConsensusStore
from sglang.srt.environ import envs, nv_mega_runtime_switches
from sglang.srt.managers import scheduler_pp_mixin
from sglang.srt.managers.scheduler_components import dp_sync_exchange, shm_tag_exchange
from sglang.srt.models.qwen3_5 import Qwen3_5MoeForConditionalGeneration
from sglang.srt.runtime_context import get_parallel
from sglang.srt.server_args import ServerArgs
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, published_topology

register_cpu_ci(est_time=60, suite="base-a-test-cpu")

_RUNTIME_MASTER = "SGLANG_ENABLE_NV_MEGA_RUNTIME_OPTIMIZATION"
# The user-set envs a shape may carry; everything else is unset.
_USER_ENVS = (
    "SGLANG_DISAGG_STAGING_BUFFER",
    "SGLANG_ENABLE_METRICS_DEVICE_TIMER",
    "SGLANG_PP_COMM_OVERLAP",
)


class Shape(msgspec.Struct, frozen=True):
    """A deployment shape: the server arguments a role starts with, and the runtime
    facts its children read."""

    role: str  # "decode", "prefill" or "aggregated"
    args: Dict[str, object]
    hosts: int = 1  # hosts the TP group spans
    megakernel: bool = False  # --enable-nvidia-megakernel with a built plugin
    target_class: str = ""  # the target model's class name
    envs: Dict[str, str] = {}


_EAGLE_CHAIN = dict(
    speculative_algorithm="NEXTN",
    speculative_num_steps=6,
    speculative_eagle_topk=1,
    speculative_num_draft_tokens=7,
)
_QWEN35_DECODE = dict(
    disaggregation_mode="decode",
    attention_backend="trtllm_mha",
    moe_runner_backend="flashinfer_trtllm",
    disable_radix_cache=True,
    **_EAGLE_CHAIN,
)
_QWEN35_PREFILL = dict(
    disaggregation_mode="prefill",
    attention_backend="trtllm_mha",
    enable_hierarchical_cache=True,
    **_EAGLE_CHAIN,
)


def _qwen35(**kwargs) -> Shape:
    return Shape(target_class=Qwen3_5MoeForConditionalGeneration.__name__, **kwargs)


SHAPES = {
    "decode-dep4": _qwen35(
        role="decode",
        args=dict(
            _QWEN35_DECODE,
            tp_size=4,
            attn_dp_size=4,
            ep_size=4,
            moe_a2a_backend="flashinfer_nvlink_one_sided",
        ),
        megakernel=True,
        envs={"SGLANG_ENABLE_METRICS_DEVICE_TIMER": "1"},
    ),
    "decode-tep8": _qwen35(
        role="decode",
        args=dict(_QWEN35_DECODE, tp_size=8, nnodes=2, ep_size=4),
        hosts=2,
        megakernel=True,
        envs={
            "SGLANG_DISAGG_STAGING_BUFFER": "1",
            "SGLANG_ENABLE_METRICS_DEVICE_TIMER": "1",
        },
    ),
    "decode-tp8-one-request": _qwen35(
        role="decode",
        args=dict(_QWEN35_DECODE, tp_size=8, nnodes=2, max_running_requests=1),
        hosts=2,
        megakernel=True,
        envs={"SGLANG_DISAGG_STAGING_BUFFER": "1"},
    ),
    "prefill-tp1-pp2": _qwen35(
        role="prefill",
        args=dict(
            _QWEN35_PREFILL,
            tp_size=1,
            pp_size=2,
            disable_overlap_schedule=True,
            hicache_write_policy="write_back",
            moe_runner_backend="flashinfer_cutedsl",
        ),
        megakernel=True,
        envs={
            "SGLANG_DISAGG_STAGING_BUFFER": "1",
            "SGLANG_ENABLE_METRICS_DEVICE_TIMER": "1",
            "SGLANG_PP_COMM_OVERLAP": "1",
        },
    ),
    # Other shapes that set the master (Qwen3.5 GB300, Qwen3.8-Flash-Next,
    # DeepSeek-V4.1-Flash); arguments not listed are at their defaults.
    "aggregated-tp4": _qwen35(
        role="aggregated",
        args=dict(
            attention_backend="trtllm_mha",
            tp_size=4,
            enable_hierarchical_cache=True,
            **_EAGLE_CHAIN,
        ),
        megakernel=True,
    ),
    "aggregated-tp8": _qwen35(
        role="aggregated",
        args=dict(
            attention_backend="trtllm_mha",
            tp_size=8,
            nnodes=2,
            enable_hierarchical_cache=True,
            **_EAGLE_CHAIN,
        ),
        hosts=2,
        megakernel=True,
    ),
    "decode-tp2": _qwen35(
        role="decode", args=dict(_QWEN35_DECODE, tp_size=2), megakernel=True
    ),
    "decode-tp8-ep4": _qwen35(
        role="decode",
        args=dict(_QWEN35_DECODE, tp_size=8, nnodes=2, ep_size=4),
        hosts=2,
        megakernel=True,
    ),
    "decode-dep8": _qwen35(
        role="decode",
        args=dict(
            _QWEN35_DECODE,
            tp_size=8,
            nnodes=2,
            attn_dp_size=8,
            ep_size=8,
            moe_a2a_backend="flashinfer_nvlink_one_sided",
        ),
        hosts=2,
        megakernel=True,
    ),
    "decode-dep16": _qwen35(
        role="decode",
        args=dict(
            _QWEN35_DECODE,
            tp_size=16,
            nnodes=4,
            attn_dp_size=16,
            ep_size=16,
            moe_a2a_backend="flashinfer_nvlink_one_sided",
        ),
        hosts=4,
    ),
    "prefill-tp4": _qwen35(role="prefill", args=dict(_QWEN35_PREFILL, tp_size=4)),
    "prefill-tp1-pp4": _qwen35(
        role="prefill", args=dict(_QWEN35_PREFILL, tp_size=1, pp_size=4)
    ),
    "qwen38-aggregated-tp1": Shape(
        role="aggregated",
        target_class="Qwen4ExpForConditionalGeneration",
        args=dict(
            speculative_algorithm="NEXTN",
            speculative_num_steps=3,
            speculative_eagle_topk=1,
            speculative_num_draft_tokens=4,
        ),
    ),
    "qwen38-aggregated-tp4-ep4": Shape(
        role="aggregated",
        target_class="Qwen4ExpForConditionalGeneration",
        args=dict(
            tp_size=4,
            ep_size=4,
            speculative_algorithm="NEXTN",
            speculative_num_steps=3,
            speculative_eagle_topk=1,
            speculative_num_draft_tokens=4,
        ),
    ),
    "dsv41-aggregated-tp4": Shape(
        role="aggregated",
        args=dict(
            tp_size=4,
            attention_backend="flashmla",
            speculative_algorithm="DSPARK",
            speculative_num_draft_tokens=5,
        ),
        megakernel=True,
    ),
}


# Runtime facts the children read besides the server arguments.
def _exchange_hosts(shape: Shape) -> int:
    """Hosts of a shared-memory exchange over the shape's TP group, as the stock
    get_shm_exchange builds it for the PD decode kinds (one host, or two with the
    multi-host child); 0 when the stock collective runs."""
    if not envs.SGLANG_OPT_SCHEDULER_SHM_SYNC.get():
        return 0
    if shape.hosts > 1 and not envs.SGLANG_OPT_SCHEDULER_SHM_SYNC_MULTIHOST.get():
        return 0
    return shape.hosts if shape.hosts <= shm_tag_exchange.PD_DECODE_MAX_HOSTS else 0


def _overlap_loop(shape: Shape) -> bool:
    return not shape.args.get("disable_overlap_schedule", False)


def _pd_decode_overlap_loop(shape: Shape) -> bool:
    return (
        shape.role == "decode" and _overlap_loop(shape) and get_parallel().pp_size == 1
    )


def _group_of(size: int):
    return patch.object(
        shm_tag_exchange.dist, "get_world_size", side_effect=lambda group=None: size
    )


def _dp_rows(shape: Shape) -> bool:
    with _group_of(get_parallel().tp_size):
        declined = dp_sync_exchange._decline_reason(group=object())
    return declined is None and _exchange_hosts(shape) > 0


def _freed_slots(shape: Shape) -> bool:
    return (
        _pd_decode_overlap_loop(shape) and decode_freed_slots.freed_slot_admission_on()
    )


def _pp_prefill_loop(shape: Shape) -> bool:
    return shape.role == "prefill" and get_parallel().pp_size > 1


def _pp_early_finish(shape: Shape) -> bool:
    if not (_pp_prefill_loop(shape) and envs.SGLANG_OPT_PP_PREFILL_EARLY_FINISH.get()):
        return False
    kv_manager = SimpleNamespace(
        request_status=PPConsensusStore.__new__(PPConsensusStore)
    )
    output_streamer = SimpleNamespace(
        rust_server=None,
        send_to_detokenizer=SimpleNamespace(enable_thread_safety=lambda: None),
    )
    with patch.object(pp_early_finish.EarlyFinishThread, "start"):
        tracker = pp_early_finish.start_pp_prefill_early_finish(
            kv_manager=kv_manager, transfer_queue=[], output_streamer=output_streamer
        )
    return tracker is not None


def _pp_early_relay(shape: Shape) -> bool:
    return (
        _pp_prefill_loop(shape) and scheduler_pp_mixin._pp_prefill_early_relay_enabled()
    )


def _pp_proxy_recv_after_forward(shape: Shape) -> bool:
    # PP rank 0 receives no proxy tensors: a later stage decides.
    comm_overlap = os.environ.get("SGLANG_PP_COMM_OVERLAP") == "1"
    return (
        _pp_prefill_loop(shape)
        and envs.SGLANG_OPT_PP_PROXY_RECV_AFTER_FORWARD.get()
        and scheduler_pp_mixin._pp_prefill_proxy_recv_after_forward_engages(
            comm_overlap=comm_overlap
        )
    )


RESOLVERS: Dict[str, Callable[[Shape], bool]] = {
    "SGLANG_OPT_SCHEDULER_SHM_SYNC_DP_ROWS": _dp_rows,
    "SGLANG_OPT_PD_DECODE_ADMIT_FREED_SLOTS": _freed_slots,
    "SGLANG_OPT_PP_PREFILL_EARLY_FINISH": _pp_early_finish,
    "SGLANG_OPT_PP_EARLY_RELAY": _pp_early_relay,
    "SGLANG_OPT_PP_PROXY_RECV_AFTER_FORWARD": _pp_proxy_recv_after_forward,
}


@contextmanager
def _deployed(shape: Shape, *, master: bool = True):
    """The shape's server topology published, the runtime master set, its user envs
    set and every other runtime child unset."""
    environ = {
        k: v
        for k, v in os.environ.items()
        if k not in _USER_ENVS and not k.startswith("SGLANG_OPT_")
    }
    environ.update(shape.envs, **{_RUNTIME_MASTER: "1" if master else "0"})
    args = dict(shape.args, enable_nvidia_megakernel=shape.megakernel)
    # The one argument default a child reads that resolution fills in.
    args.update(
        overrides._tp_lm_head_all_to_all_default(ServerArgs(model_path="x", **args))
    )
    with ExitStack() as stack:
        stack.enter_context(patch.dict(os.environ, environ, clear=True))
        stack.enter_context(published_topology(**args))
        yield


# Untabled: main's own children, the Mooncake KV-send child (module not imported here)
# and the draft provider attention child (the plugin's slot plan decides where).
_UNTABLED = frozenset(
    (
        "SGLANG_DSPARK_DEFER_CONTEXT_KV",
        "SGLANG_DSPARK_FUSED_ACCEPT_TAIL",
        "SGLANG_ENABLE_EAGLE_POST_VERIFY_CUDA_GRAPH",
        "SGLANG_ENABLE_GDN_PREFILL_GRAPH_EXTEND",
        "SGLANG_OPT_CUDA_GRAPH_FILL_FAST_PATH",
        "SGLANG_OPT_CUTEDSL_EP_DEFERRED_FINALIZE",
        "SGLANG_OPT_DSV41_BLOCKWISE_REJECTION",
        "SGLANG_OPT_DSV41_DSPARK_DRAFT_PREP_FORK",
        "SGLANG_OPT_DSV41_DSPARK_MID_FOLD",
        "SGLANG_OPT_DSV41_DSPARK_ONE_LAUNCH_COMMIT_STAGE",
        "SGLANG_OPT_DSV41_DSPARK_POST_FOLD",
        "SGLANG_OPT_DSV41_DSPARK_PREPARE_INPUTS_ONCE",
        "SGLANG_OPT_DSV41_DSPARK_SKIP_PREFILLED_INPUTS",
        "SGLANG_OPT_DSV41_POST_FORWARD_CUDA_GRAPH",
        "SGLANG_OPT_DSV41_SAMPLING_ACCEPT_GRAPH",
        "SGLANG_OPT_DSV41_TARGET_METADATA_EARLY",
        "SGLANG_OPT_DSV41_TARGET_VERIFY_METADATA_GLUE",
        "SGLANG_OPT_MXFP4_DEFERRED_FINALIZED_SHAPE",
        "SGLANG_OPT_SCHEDULER_SHM_SYNC",
        "SGLANG_OPT_SCHEDULER_SHM_SYNC_LEADER_TCP",
        "SGLANG_OPT_SCHEDULER_SHM_SYNC_MULTIHOST",
        "SGLANG_OPT_SKIP_GLUED_REPLAY_VIEW",
        "SGLANG_OPT_SMALL_FIELD_COPY_VECTORIZED",
        "SGLANG_OPT_SPEC_COMPACT_VERIFY_SAMPLING",
        "SGLANG_OPT_SPEC_PACKED_VERIFY_BCAST",
        "SGLANG_OPT_SPEC_POST_VERIFY_ONE_LAUNCH_LOAD",
        "SGLANG_OPT_SPEC_RELAY_IN_PLACE",
        "SGLANG_OPT_SPEC_VERIFY_REUSE_STAGED_INPUTS",
        "SGLANG_OPT_TRTLLM_MHA_CHAIN_VERIFY",
        "SGLANG_OPT_TRTLLM_MHA_DRAFT_PROVIDER_ATTENTION",
    )
)


def engaged(shape: Shape, *, master: bool = True) -> FrozenSet[str]:
    with _deployed(shape, master=master):
        return frozenset(name for name, on in RESOLVERS.items() if on(shape))


def _children(*names: str) -> FrozenSet[str]:
    return frozenset(f"SGLANG_OPT_{name}" for name in names)


# Every PD decode loop with the overlap scheduler.
_PD_DECODE = _children("PD_DECODE_ADMIT_FREED_SLOTS")
# A PD decode without attention DP.
_PD_DECODE_TP = _children()
_TWO_HOSTS = _children()
# Main's deferred KV destination publish runs only without the staging buffer.
_NO_STAGING = _children()
# A PD decode with attention DP; the exchange rows on at most two hosts, the
# post-verify graphs with a bound gathered sampler.
_PD_DECODE_DP = _children()
_DP_EXCHANGE = _children("SCHEDULER_SHM_SYNC_DP_ROWS")
_DP_GRAPHS = _children()
# A top-k-1 EAGLE chain outside a PD prefill and attention DP: the generic paths,
# then those of the TRT-LLM MHA backend, the vocab-shard verify and the model trait.
_CHAIN = _children()
_TRTLLM_CHAIN = _children()
_QWEN35_CHAIN = _children()
_QWEN35_TP_SPEC = _CHAIN | _TRTLLM_CHAIN | _QWEN35_CHAIN
# A PD prefill with pipeline parallelism; the proxy receive with
# SGLANG_PP_COMM_OVERLAP, the state victims with HiCache write_back.
_PP_PREFILL = _children("PP_EARLY_RELAY", "PP_PREFILL_EARLY_FINISH")

EXPECTED = {
    "decode-dep4": _PD_DECODE | _NO_STAGING | _PD_DECODE_DP | _DP_EXCHANGE | _DP_GRAPHS,
    "decode-tep8": _PD_DECODE
    | _PD_DECODE_TP
    | _TWO_HOSTS
    | _QWEN35_TP_SPEC
    | _children(),
    "decode-tp8-one-request": _PD_DECODE | _PD_DECODE_TP | _TWO_HOSTS | _QWEN35_TP_SPEC,
    "prefill-tp1-pp2": _PP_PREFILL | _children("PP_PROXY_RECV_AFTER_FORWARD"),
    "aggregated-tp4": _QWEN35_TP_SPEC,
    "aggregated-tp8": _QWEN35_TP_SPEC,
    "decode-tp2": _PD_DECODE | _PD_DECODE_TP | _NO_STAGING | _QWEN35_TP_SPEC,
    "decode-tp8-ep4": _PD_DECODE
    | _PD_DECODE_TP
    | _TWO_HOSTS
    | _NO_STAGING
    | _QWEN35_TP_SPEC,
    "decode-dep8": _PD_DECODE | _NO_STAGING | _PD_DECODE_DP | _DP_EXCHANGE | _DP_GRAPHS,
    # Four hosts: no shared-memory exchange; no megakernel: no gathered sampler.
    "decode-dep16": _PD_DECODE | _NO_STAGING | _PD_DECODE_DP,
    "prefill-tp4": frozenset(),
    "prefill-tp1-pp4": _PP_PREFILL,
    # Not in the vocab-shard verify allowlist, no declared trait, no TRT-LLM MHA.
    "qwen38-aggregated-tp1": _CHAIN,
    "qwen38-aggregated-tp4-ep4": _CHAIN,
    "dsv41-aggregated-tp4": frozenset(),
}


class TestRuntimeChildEngagement(CustomTestCase):
    def test_each_shape_engages_its_children(self):
        """Each child's structural checks give this table: a change to a check a
        resolver calls, to the structure it reads, or to a shape's arguments changes
        this table."""
        self.assertEqual(set(EXPECTED), set(SHAPES))
        for name, shape in SHAPES.items():
            with self.subTest(shape=name):
                got = engaged(shape)
                self.assertEqual(
                    got,
                    EXPECTED[name],
                    f"unexpected: {sorted(got - EXPECTED[name])}, "
                    f"missing: {sorted(EXPECTED[name] - got)}",
                )

    def test_every_child_engages_somewhere(self):
        """A child no shape engages would be inert on every shape that sets the
        master."""
        self.assertEqual(set().union(*EXPECTED.values()), set(RESOLVERS))

    def test_every_child_of_the_master_has_a_resolver(self):
        """A new child of the master needs a resolver and table rows, or an entry in
        _UNTABLED."""
        children = {field.name for field in nv_mega_runtime_switches()[1:]}
        self.assertEqual(set(RESOLVERS) | _UNTABLED, children)
        self.assertFalse(set(RESOLVERS) & _UNTABLED)

    def test_nothing_engages_with_the_master_off(self):
        for name, shape in SHAPES.items():
            with self.subTest(shape=name):
                self.assertEqual(engaged(shape, master=False), frozenset())


if __name__ == "__main__":
    unittest.main()
