"""CPU contracts of the monokernel registry, checks, routing, coverage and copies."""

import inspect
import unittest
from functools import partial
from types import SimpleNamespace
from unittest.mock import patch

import torch
from msgspec.structs import replace

from sglang.kernels.plugin_kernels.amd import spec as registry
from sglang.kernels.plugin_kernels.amd.spec import (
    DeviceLimits,
    Feature,
    MonoSpec,
    Numerics,
    NumericsKind,
    Requirements,
    ShapeKey,
    StepMetadata,
)
from sglang.srt.environ import envs
from sglang.srt.kernel_plugins.amd import binding
from sglang.srt.kernel_plugins.amd.binding import (
    MAX_DEADLINE_S,
    covered_capture_sizes,
    deadline_s,
    features_in_use,
    monokernel_disable_reasons,
)
from sglang.srt.model_executor.cuda_graph_config import (
    Phase,
    default_cuda_graph_config,
    with_phase,
)
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.runtime_context import publish, reset_context
from sglang.srt.server_args import ServerArgs
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, stage="base-a", runner_config="cpu")

ARCH = "FakeMonoForCausalLM"
BACKEND = registry.BACKEND
WIDTHS = (1, 2, 4, 6, 12)


class FakeBuilt:
    """Record routed steps to test state isolation."""

    def __init__(self):
        self.steps = []

    def forward(self, plan, input_ids, positions, forward_batch, input_embeds=None):
        plan.state["layers"] = plan.state.get("layers", 0) + 1
        self.steps.append((plan.key.rows, dict(plan.state)))
        return "mono"


def _load_tensor(copies, model, name, tensor):
    if name.endswith("shared.weight"):
        copies["shared expert", name] = tensor.clone()


SPEC = MonoSpec(
    name="fake",
    devices={
        "gfx950": DeviceLimits(widths=WIDTHS, tp_sizes=(2, 4), resident_blocks=256)
    },
    features=frozenset({Feature.SPECULATIVE_DECODING}),
    requires=Requirements(
        server_args={"kv_cache_dtype": frozenset({"fp8_e4m3"})},
        hf_config={"model_type": "fake"},
        env={"SGLANG_DSPARK_EMBED_IN_GRAPH": True},
    ),
    numerics=Numerics(kind=NumericsKind.BITWISE),
    config_disable_reasons=lambda cfg, hf: (
        ["context_length too long"] if cfg.context_length == 7 else []
    ),
    model_disable_reasons=lambda model, hf, copies: [],
    build=lambda model, hf, ctx: FakeBuilt(),
    api_version=registry.MONOKERNEL_API_VERSION,
    load_tensor=_load_tensor,
)
OLD_SPEC = replace(SPEC, api_version=registry.MONOKERNEL_API_VERSION - 1)
# a DSpark draft at 2 draft tokens with anchor sampling: 1-token blocks
ONE_TOKEN_DRAFT = replace(SPEC, tokens_per_request=lambda model: 1)
# an EAGLE draft that serves its own draft decode and draft extend steps
EAGLE_DRAFT = replace(
    SPEC,
    step_modes=lambda model: [("decode", 1), ("draft_extend", 4)],
    unread_metadata=frozenset(
        {StepMetadata.DSA_REPLAY_PLAN, StepMetadata.DSA_REPLAY_PAGE_TABLE}
    ),
)
# one whose kernels cannot serve this draft
IDLE_DRAFT = replace(SPEC, step_modes=lambda model: [])
# one capped at 4 rows at TP4, whose wider steps run the model's forward
CAPPED = replace(
    SPEC,
    devices={
        "gfx950": DeviceLimits(
            widths=WIDTHS, tp_sizes=(2, 4), resident_blocks=256, max_rows={4: 4}
        )
    },
    unread_metadata=frozenset({StepMetadata.DSA_REPLAY_PLAN}),
)


def server_args(**kw):
    defaults = {
        "model_path": "dummy",
        "tp_size": 4,
        "kv_cache_dtype": "fp8_e4m3",
        "enable_amd_monokernel": True,
    }
    return ServerArgs(**{**defaults, **kw})


def reasons_for(cfg, arch=ARCH, isa="gfx950", model_type="fake"):
    model_config = SimpleNamespace(
        hf_config=SimpleNamespace(architectures=[arch], model_type=model_type),
        dtype="bfloat16",
    )
    return monokernel_disable_reasons(cfg=cfg, model_config=model_config, isa=isa)


def batch(mode, bs, rows):
    return SimpleNamespace(
        forward_mode=ForwardMode[mode.upper()],
        batch_size=bs,
        input_ids=None if rows is None else torch.zeros(rows, dtype=torch.int64),
    )


def install(*, archs, lookups=None, is_load_observed=False):
    """install_monokernel_forward on an Lm, as rank 0 of two whose gathered lookups
    are lookups, or as the only rank."""
    lm = Lm()
    if is_load_observed:
        lm._mono_copies = {}  # what monokernel_weights_iterator leaves on the model
    gather = {"return_value": lookups} if lookups else {"side_effect": lambda x: [x]}
    gpu = SimpleNamespace(multi_processor_count=256)
    with (
        patch.object(binding, "_backend", return_value=BACKEND),
        patch.object(binding, "device_isa", return_value="gfx950"),
        patch.object(binding, "_gather_tp", **gather),
        patch.object(torch.cuda, "current_device", return_value=0),
        patch.object(torch.cuda, "get_device_properties", return_value=gpu),
        patch.object(torch.cuda, "memory_allocated", return_value=0),
    ):
        binding.install_monokernel_forward(
            model=lm,
            hf_config=SimpleNamespace(architectures=archs),
            capture_tail_hooks=[],
        )
    return lm


def runner(*, capture_bs):
    """A decode CUDA graph runner, as the capture tail hook sees it."""
    return SimpleNamespace(capture_bs=capture_bs)


class DraftDecodeRunner(SimpleNamespace):
    pass


class DraftExtendRunner(SimpleNamespace):
    pass


class Lm(torch.nn.Module):
    def forward(self, input_ids, positions, forward_batch, input_embeds=None):
        return "normal"


def switched(mode="decode", tokens=1, spec=SPEC):
    """Build a fake model whose steps of this mode and tokens per request route."""
    lm, built = Lm(), FakeBuilt()
    served = binding._served_keys(mode=mode, widths=WIDTHS, tokens=tokens)
    binding._install_forward(model=lm, built=built, served=served, spec=spec)
    return lm, built


class TestMonokernelFramework(CustomTestCase):
    def setUp(self):
        registry_patch = patch.dict(registry._REGISTRY, clear=True)
        registry_patch.start()
        self.addCleanup(registry_patch.stop)
        registry.register(backend=BACKEND, architecture=ARCH, path=f"{__name__}:SPEC")

    def test_find_spec_refuses_with_a_reason(self):
        register = partial(registry.register, backend=BACKEND)
        register(architecture="Broken", path="no_such_module:SPEC")
        register(architecture="Old", path=f"{__name__}:OLD_SPEC")
        find = partial(registry.find_spec, backend=BACKEND)
        self.assertIs(find(architectures=[ARCH], isa="gfx950")[0], SPEC)
        version = registry.MONOKERNEL_API_VERSION
        api = f"API version {version - 1}; this SGLang requires version {version}"
        for archs, isa, why in (
            ([ARCH], "gfx942", "has no spec"),
            (["Other"], "gfx950", "has no spec"),
            (["Broken"], "gfx950", "does not import: ModuleNotFoundError"),
            (["Old"], "gfx950", api),
        ):
            with self.subTest(archs=archs, isa=isa):
                spec, reason = find(architectures=archs, isa=isa)
                self.assertIsNone(spec)
                self.assertIn(why, reason)
        spec, reason = registry.find_spec(
            backend="nope", architectures=[ARCH], isa="gfx950"
        )
        self.assertIsNone(spec)
        self.assertIn("no monokernel backend 'nope' is registered", reason)
        self.assertIn(repr(BACKEND), reason)  # the registered ones are listed

    def test_a_backend_has_one_spec_an_architecture(self):
        # Repeated registration must not silently replace another plugin's spec.
        registry.register(backend=BACKEND, architecture=ARCH, path=f"{__name__}:SPEC")
        with self.assertRaises(ValueError):
            registry.register(backend=BACKEND, architecture=ARCH, path="other:SPEC")
        self.assertIs(
            registry.find_spec(backend=BACKEND, architectures=[ARCH], isa="gfx950")[0],
            SPEC,
        )

    def test_features_in_use_are_refused_unless_accepted(self):
        # every probe is off at SGLang's defaults: one stuck on refuses every server
        self.assertEqual(features_in_use(ServerArgs(model_path="dummy")), set())
        _, reasons = reasons_for(server_args(pp_size=2, speculative_algorithm="DSPARK"))
        self.assertEqual(len(reasons), 1, reasons)  # SPEC accepts speculative decoding
        self.assertIn(Feature.PIPELINE_PARALLEL.value, reasons[0])

    def test_every_refusal_reason_is_listed(self):
        with envs.SGLANG_DSPARK_EMBED_IN_GRAPH.override(False):
            spec, reasons = reasons_for(
                server_args(
                    tp_size=8,
                    kv_cache_dtype="auto",
                    enable_eplb=True,
                    context_length=7,
                    watchdog_timeout=6,
                ),
                model_type="other",
            )
        self.assertIs(spec, SPEC)
        for want in (
            "tp_size 8",
            "kv_cache_dtype 'auto'",
            "config.model_type 'other'",
            "SGLANG_DSPARK_EMBED_IN_GRAPH False",
            "EPLB",
            "context_length",
            "--watchdog-timeout",
        ):
            self.assertTrue(any(want in r for r in reasons), (want, reasons))
        self.assertEqual(reasons_for(server_args())[1], [])
        # a spec whose requirements state every check it needs has no config hook
        bare = replace(SPEC, config_disable_reasons=None)
        with patch.object(binding, "find_spec", return_value=(bare, "")):
            self.assertEqual(reasons_for(server_args(context_length=7))[1], [])
        self.assertIsNone(reasons_for(server_args(), arch="Other")[0])
        self.assertEqual(
            reasons_for(server_args(), isa=None), (None, ["no GPU is visible"])
        )

    def test_exact_modes_take_only_bitwise_specs(self):
        modes = {"enable_deterministic_inference": True, "rl_on_policy_target": "fsdp"}
        self.assertEqual(reasons_for(server_args(**modes))[1], [])  # SPEC is BITWISE
        tolerance = replace(
            SPEC,
            numerics=Numerics(kind=NumericsKind.TOLERANCE, bounds={"k1.q.rel": 5e-3}),
        )
        with patch.object(binding, "find_spec", return_value=(tolerance, "")):
            _, reasons = reasons_for(server_args(**modes))
        for flag in ("--enable-deterministic-inference", "--rl-on-policy-target"):
            self.assertTrue(any(flag in r for r in reasons), (flag, reasons))

    def test_capture_sizes_are_the_widths_up_to_the_largest(self):
        # decode: rows are requests
        sizes = covered_capture_sizes(
            widths=WIDTHS, tokens_per_request=1, sizes=[1, 2, 3, 4, 8, 16, 32]
        )
        self.assertEqual(sizes, [1, 2, 4, 6, 12, 16, 32])
        # verify at 6 tokens a request: widths 6 and 12 are 1 and 2 requests
        sizes = covered_capture_sizes(
            widths=WIDTHS, tokens_per_request=6, sizes=[1, 2, 3, 4, 8]
        )
        self.assertEqual(sizes, [1, 2, 3, 4, 8])

    def test_the_switch_routes_by_graph_key(self):
        lm, _ = switched()
        # SGLang's graph runners choose what to pass from the forward's parameters
        self.assertEqual(inspect.signature(lm.forward), inspect.signature(Lm().forward))
        x = torch.zeros(1)
        self.assertEqual(lm(x, None, batch("decode", 2, 2)), "mono")
        verify, _ = switched(mode="target_verify", tokens=6)
        self.assertEqual(
            verify(x, None, forward_batch=batch("target_verify", 2, 12)), "mono"
        )
        # a speculative server's target runs verify steps only
        self.assertEqual(verify(x, None, batch("decode", 6, 6)), "normal")
        for normal in (
            batch("decode", 3, 3),  # a width not built
            batch("decode", 64, 64),  # over the largest width
            batch("target_verify", 2, 7),  # uneven rows a request
            batch("extend", 2, 9),
        ):
            self.assertEqual(lm(x, None, normal), "normal")

    def test_each_forward_has_its_own_plan_state(self):
        # Reusing a plan would leak layer state into the next forward.
        lm, built = switched()
        for _ in range(2):
            lm(torch.zeros(1), None, batch("decode", 2, 2))
        self.assertEqual(built.steps, [(2, {"layers": 1}), (2, {"layers": 1})])

    def test_only_routed_steps_skip_the_metadata_a_spec_never_reads(self):
        schedule = StepMetadata.LOW_RATIO_DECODE_SCHEDULE
        lm, _ = switched(spec=replace(SPEC, unread_metadata=frozenset({schedule})))
        self.assertTrue(binding.is_metadata_unread(lm, batch("decode", 2, 2), schedule))
        # a width not built runs the standard forward, which reads it
        self.assertFalse(
            binding.is_metadata_unread(lm, batch("decode", 3, 3), schedule)
        )
        reads_all, _ = switched()
        without_monokernels = Lm()
        for model in (reads_all, without_monokernels):
            self.assertFalse(
                binding.is_metadata_unread(model, batch("decode", 2, 2), schedule)
            )

    def test_routed_graphs_are_captured_without_attention_variants(self):
        from sglang.srt.model_executor.runner.decode_cuda_graph_runner import (
            DecodeCudaGraphRunner as Runner,
        )

        class Variants:
            capture_labels = ("dense", "sparse")

            def select(self, forward_batch):
                # stands for the host read of the batch's longest request
                return "selected"

        variant = StepMetadata.ATTENTION_GRAPH_VARIANT
        unread = replace(SPEC, unread_metadata=frozenset({variant}))

        def runner(model, *, mode="decode", tokens=1, sizes=None):
            r = SimpleNamespace(
                attention_graph_variants=Variants(),
                attention_variant_sizes=sizes,
                model_runner=SimpleNamespace(model=model),
                capture_forward_mode=ForwardMode[mode.upper()],
                capture_bs=[1, 2, 3, 4, 8],
                captured_req_width=tokens,
            )
            r._attention_varies_at = partial(Runner._attention_varies_at, r)
            Runner._capture_monokernel_steps_once(r)
            return r

        def resolve(r, bs):
            step = batch(r.capture_forward_mode.name.lower(), bs, bs)
            return Runner._resolve_attention_variant(r, step, bs)

        # the widths built (1, 2, 4) are captured once and replay without the read;
        # every other size captures and selects a variant
        routed = runner(switched(spec=unread)[0])
        self.assertEqual(routed.attention_variant_sizes, frozenset({3, 8}))
        self.assertEqual([resolve(routed, bs) for bs in (2, 3)], [None, "selected"])
        for r in (
            runner(switched()[0]),  # a spec that reads the variant
            runner(Lm()),  # no monokernels
            # a verify graph, while the forward routes decode steps
            runner(switched(spec=unread)[0], mode="target_verify", tokens=1),
        ):
            self.assertIsNone(r.attention_variant_sizes)
            self.assertEqual(resolve(r, 2), "selected")
        # the sizes another plugin set are kept
        kept = runner(switched(spec=unread)[0], sizes=frozenset({8}))
        self.assertEqual(kept.attention_variant_sizes, frozenset({8}))

    def test_the_weights_iterator_passes_weights_through_and_keeps_copies(self):
        iterator = binding.monokernel_weights_iterator
        model, other = torch.nn.Module(), torch.nn.Module()
        model.config = SimpleNamespace(architectures=[ARCH])
        other.config = SimpleNamespace(architectures=["Other"])
        weights = [
            ("layers.0.shared.weight", torch.ones(16, dtype=torch.uint8)),
            ("layers.0.other.weight", torch.ones(16)),
        ]
        with (
            patch.object(binding, "_backend", return_value=BACKEND),
            patch.object(binding, "device_isa", return_value="gfx950"),
        ):
            watched = list(iterator(model=model, weights=iter(weights)))
            self.assertIs(iterator(model=other, weights=weights), weights)
        # Observing a weight must preserve its identity and checkpoint order.
        self.assertEqual(
            [(n, id(t)) for n, t in watched], [(n, id(t)) for n, t in weights]
        )
        self.assertEqual(
            list(model.__dict__["_mono_copies"]),
            [("shared expert", "layers.0.shared.weight")],
        )
        with patch.object(binding, "_backend", return_value=None):
            self.assertIs(iterator(model=model, weights=weights), weights)

    def test_a_load_path_that_skips_the_observer_is_refused(self):
        # the layered, dummy and remote loaders never call monokernel_weights_iterator:
        # the spec would build without its weight copies
        with self.assertRaisesRegex(ValueError, "monokernel_weights_iterator"):
            install(archs=[ARCH])

    def test_a_one_token_draft_block_is_a_verify_step(self):
        # the step mode follows speculation: a mode guessed from 1 token a request
        # served decode keys, and the draft's verify steps never routed
        registry.register(
            backend=BACKEND, architecture="Draft", path=f"{__name__}:ONE_TOKEN_DRAFT"
        )
        publish(
            ServerArgs(
                model_path="dummy",
                enable_amd_monokernel=True,
                speculative_algorithm="DSPARK",
                speculative_num_draft_tokens=2,
                disable_cuda_graph=True,
            ),
            role="test",
        )
        self.addCleanup(reset_context)
        lm = install(archs=["Draft"], is_load_observed=True)
        self.assertEqual(lm(torch.zeros(1), None, batch("target_verify", 2, 2)), "mono")

    def test_every_rank_takes_the_same_lookup_branch(self):
        # every rank refuses together: one returning alone would hang its peers
        ours = (True, True, "", (SPEC.name, SPEC.api_version))
        with self.assertRaisesRegex(ValueError, "rank 1: broken"):
            install(archs=[ARCH], lookups=[ours, (False, True, "broken", None)])
        # ranks that load different specs (another plugin version) must not pair up
        other = (True, True, "", ("fake_v2", SPEC.api_version))
        with self.assertRaisesRegex(ValueError, "the ranks load different specs"):
            install(archs=[ARCH], lookups=[ours, other])
        # a model the backend registers no spec for keeps its forward on every rank
        lm = install(archs=["Other"], lookups=[(False, False, "none", None)] * 2)
        self.assertNotIn("forward", vars(lm))

    def test_the_decode_attention_backend_is_checked(self):
        # the steps a spec serves run on the decode backend, not the base one
        dsv4 = replace(
            SPEC,
            requires=replace(SPEC.requires, attention_backends=frozenset({"dsv4"})),
        )
        with patch.object(binding, "find_spec", return_value=(dsv4, "")):
            _, refused = reasons_for(
                server_args(attention_backend="dsv4", decode_attention_backend="triton")
            )
            _, accepted = reasons_for(
                server_args(attention_backend="triton", decode_attention_backend="dsv4")
            )
        self.assertTrue(any("decode attention backend 'triton'" in r for r in refused))
        self.assertEqual(accepted, [])
        # a draft runner may override it: the switch reads the backend each step runs
        # on before the plugin's first forward, with or without CUDA graphs
        # (the runner's stamp, the configured decode backend, accepted)
        for stamped, configured, ok in (
            ("triton", "dsv4", False),  # the runner's own backend wins
            (None, "dsv4", True),  # an unstamped runner runs the configured one
            ("dsv4", "triton", True),
        ):
            lm, built = switched(spec=dsv4)
            backend = SimpleNamespace(decode_attention_backend_str=stamped)
            with (
                self.subTest(stamped=stamped, configured=configured),
                patch.object(binding, "get_attn_backend", return_value=backend),
                patch.object(
                    binding, "attention_backends", return_value=("x", configured)
                ),
            ):
                step = batch("decode", 2, 2)
                if ok:
                    self.assertEqual(lm(torch.zeros(1), None, step), "mono")
                else:
                    with self.assertRaisesRegex(ValueError, "this runner's is"):
                        lm(torch.zeros(1), None, step)
                    self.assertEqual(built.steps, [])  # the plugin never ran

    def test_the_captured_graphs_decide_coverage(self):
        # the runner may capture fewer sizes than configured (alignment, the request
        # pool): configured 4 and 8 became a captured 2, which the spec does not serve
        served = frozenset(
            ShapeKey(forward_mode="decode", batch_size=bs, tokens_per_request=1)
            for bs in (4, 8)
        )
        check = binding._CaptureCheck(spec=SPEC, isa="gfx950", served=served)
        capturing = patch.object(torch.cuda, "is_current_stream_capturing")
        with capturing as is_capturing:
            is_capturing.return_value = False  # a warmup run is not a capture
            check(runner(capture_bs=[2]), None, batch("decode", 2, 2), 2)
            is_capturing.return_value = True
            with self.assertRaisesRegex(ValueError, "none of the 1 graphs"):
                check(runner(capture_bs=[2]), None, batch("decode", 2, 2), 2)
        check = binding._CaptureCheck(spec=SPEC, isa="gfx950", served=served)
        with capturing:
            for bs in (2, 4):  # the check waits for every captured size
                check(runner(capture_bs=[2, 4]), None, batch("decode", bs, bs), bs)
        self.assertEqual(check.done, {"SimpleNamespace"})
        # a step pads up to a captured graph that holds it: only a served key past
        # the largest graph of its mode runs eagerly
        captured = check.captured["SimpleNamespace"].values()
        eager = binding._eager_keys(served=served, captured=captured)
        self.assertEqual([k.batch_size for k in eager], [8])

    def test_each_graph_runner_of_a_draft_is_checked(self):
        # a draft model's draft decode and draft extend runners share its hooks;
        # each is checked once all of its own graphs are in
        served = frozenset(
            {
                ShapeKey(forward_mode="decode", batch_size=1, tokens_per_request=1),
                ShapeKey(
                    forward_mode="draft_extend", batch_size=1, tokens_per_request=4
                ),
            }
        )
        check = binding._CaptureCheck(spec=SPEC, isa="gfx950", served=served)
        decode, extend = (
            DraftDecodeRunner(capture_bs=[1, 2]),
            DraftExtendRunner(capture_bs=[1]),
        )
        with patch.object(torch.cuda, "is_current_stream_capturing", return_value=True):
            check(decode, None, batch("decode", 1, 1), 1)
            check(extend, None, batch("draft_extend_v2", 1, 4), 4)
            self.assertEqual(check.done, {"DraftExtendRunner"})
            check(decode, None, batch("decode", 2, 2), 2)
        self.assertEqual(check.done, {"DraftDecodeRunner", "DraftExtendRunner"})

    def test_a_draft_spec_serves_its_own_step_modes(self):
        registry.register(
            backend=BACKEND, architecture="Draft", path=f"{__name__}:EAGLE_DRAFT"
        )
        publish(
            ServerArgs(
                model_path="dummy",
                enable_amd_monokernel=True,
                speculative_algorithm="EAGLE",
                speculative_num_steps=3,
                speculative_num_draft_tokens=4,
                disable_cuda_graph=True,
            ),
            role="test",
        )
        self.addCleanup(reset_context)
        lm = install(archs=["Draft"], is_load_observed=True)
        x = torch.zeros(1)
        self.assertEqual(lm(x, None, batch("draft_extend_v2", 1, 4)), "mono")
        self.assertEqual(lm(x, None, batch("decode", 2, 2)), "mono")
        # the target's verify step, which a draft never runs, is not its key
        self.assertEqual(lm(x, None, batch("target_verify", 1, 4)), "normal")
        # a draft decode plans its metadata before its steps set input_ids
        replay = batch("decode", 2, None)
        plan = StepMetadata.DSA_REPLAY_PLAN
        self.assertTrue(binding.is_metadata_unread(lm, replay, plan))
        self.assertTrue(
            binding.is_metadata_unread(lm, batch("draft_extend_v2", 3, 12), plan)
        )
        self.assertFalse(
            binding.is_metadata_unread(lm, batch("target_verify", 1, 4), plan)
        )

    def test_a_draft_spec_with_no_step_keeps_the_normal_forward(self):
        registry.register(
            backend=BACKEND, architecture="Draft", path=f"{__name__}:IDLE_DRAFT"
        )
        publish(
            ServerArgs(
                model_path="dummy", enable_amd_monokernel=True, disable_cuda_graph=True
            ),
            role="test",
        )
        self.addCleanup(reset_context)
        lm = install(archs=["Draft"], is_load_observed=True)
        self.assertNotIn("forward", vars(lm))

    def test_a_spec_caps_its_rows_at_a_tp_size(self):
        registry.register(
            backend=BACKEND, architecture="Capped", path=f"{__name__}:CAPPED"
        )
        self.addCleanup(reset_context)
        plan, x = StepMetadata.DSA_REPLAY_PLAN, torch.zeros(1)
        for tp_size, wide in ((4, "normal"), (2, "mono")):
            with self.subTest(tp_size=tp_size):
                publish(
                    ServerArgs(
                        model_path="dummy",
                        tp_size=tp_size,
                        enable_amd_monokernel=True,
                        disable_cuda_graph=True,
                    ),
                    role="test",
                )
                lm = install(archs=["Capped"], is_load_observed=True)
                self.assertEqual(lm(x, None, batch("decode", 4, 4)), "mono")
                self.assertTrue(
                    binding.is_metadata_unread(lm, batch("decode", 4, 4), plan)
                )
                # past the cap the step runs the model's forward, which reads the
                # backend's metadata, so the backend builds it
                self.assertEqual(lm(x, None, batch("decode", 6, 6)), wide)
                self.assertEqual(
                    binding.is_metadata_unread(lm, batch("decode", 6, 6), plan),
                    wide == "mono",
                )

    def test_the_dsa_replay_builds_only_the_metadata_its_forward_reads(self):
        from sglang.srt.layers.attention.dsa_backend import DeepseekSparseAttnBackend

        page, bs = 4, 2
        req_to_token = torch.randperm(8 * 64).reshape(8, 64)
        requests = torch.tensor([5, 2])

        def replay(*, unread, mode="decode", rows=bs, page_size=page):
            tokens = rows // bs
            spec = replace(SPEC, unread_metadata=frozenset(unread))
            lm, _ = switched(mode=mode, tokens=tokens, spec=spec)
            if mode == "decode" and tokens > 1:
                lm, _ = switched(spec=spec)  # serves 1-token decode steps only
            backend = DeepseekSparseAttnBackend.__new__(DeepseekSparseAttnBackend)
            captured = SimpleNamespace(
                page_table_1=torch.zeros(4, 64, dtype=torch.int64),
                real_page_table=torch.zeros(4, 64 // page_size, dtype=torch.int64),
            )
            if page_size == 1:
                captured.real_page_table = captured.page_table_1
            backend.__dict__.update(
                model_runner=SimpleNamespace(model=lm),
                decode_cuda_graph_metadata={bs: captured},
                req_to_token=req_to_token,
                physical_page_size=page_size,
                forward_metadata=None,
            )
            step = SimpleNamespace(
                forward_mode=ForwardMode[mode.upper()],
                batch_size=bs,
                input_ids=torch.zeros(rows, dtype=torch.int64),
                req_pool_indices=requests,
                seq_lens=torch.full((bs,), 64),
                seq_lens_cpu=torch.full((bs,), 64),
                spec_info=None,
            )
            with patch.object(backend, "_apply_cuda_graph_metadata") as full:
                backend.init_forward_metadata_out_graph(step)
            return backend, captured, full.called

        plan = StepMetadata.DSA_REPLAY_PLAN
        table = StepMetadata.DSA_REPLAY_PAGE_TABLE
        # nothing read: the captured metadata as is
        backend, captured, full = replay(unread={plan, table})
        self.assertFalse(full)
        self.assertIs(backend.forward_metadata, captured)
        self.assertFalse(captured.real_page_table.any())
        # only the page table read: SGLang's real page table, req_to_token at each
        # page's first token, divided by the page size
        for page_size in (page, 1):
            backend, captured, full = replay(unread={plan}, page_size=page_size)
            self.assertFalse(full)
            self.assertIs(backend.forward_metadata, captured)
            want = req_to_token[requests][:, ::page_size] // page_size
            self.assertTrue(torch.equal(captured.real_page_table[:bs], want))
            # negative control: the requests in another order
            other = req_to_token[requests.flip(0)][:, ::page_size] // page_size
            self.assertFalse(torch.equal(captured.real_page_table[:bs], other))
        # every other step builds it all: the plan read, a width the forward does not
        # serve, or a page-table-only replay outside decode
        for unread, mode, rows in (
            ({table}, "decode", bs),
            ({plan, table}, "decode", 2 * bs),
            ({plan}, "target_verify", 2 * bs),
        ):
            with self.subTest(unread=unread, mode=mode, rows=rows):
                self.assertTrue(replay(unread=unread, mode=mode, rows=rows)[2])

    def test_the_runtime_reads_the_published_config(self):
        graphs = with_phase(default_cuda_graph_config(), Phase.DECODE, bs=[1, 2, 8])
        publish(
            ServerArgs(
                model_path="dummy",
                enable_amd_monokernel=True,
                speculative_algorithm="DSPARK",
                speculative_num_draft_tokens=6,
                cuda_graph_config=graphs,
            ),
            role="test",
        )
        self.addCleanup(reset_context)
        self.assertEqual(binding._backend(), BACKEND)
        mode, tokens = binding._server_step()
        self.assertEqual((mode, tokens), ("target_verify", 6))
        self.assertEqual(
            [
                (k.forward_mode, k.batch_size, k.rows)
                for k in binding._capture_keys(mode=mode, tokens=tokens)
            ],
            [
                ("target_verify", 1, 6),
                ("target_verify", 2, 12),
                ("target_verify", 8, 48),
            ],
        )

    def test_deadlines_stay_below_the_watchdog(self):
        self.assertEqual(deadline_s(None), MAX_DEADLINE_S)
        self.assertEqual(deadline_s(3600), MAX_DEADLINE_S)
        self.assertEqual(deadline_s(30), 15.0)


if __name__ == "__main__":
    unittest.main()
