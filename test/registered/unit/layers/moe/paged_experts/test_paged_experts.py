import contextlib
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.layers.moe.paged_experts import PagedExpertsMoEMethod, formats, sizing
from sglang.srt.layers.moe.paged_experts.formats import UnquantizedFormat, after_loading
from sglang.srt.layers.moe.paged_experts.method import (
    _cap_decode_capture_batch_sizes,
    check_paged_experts_compat,
    make_for_layer,
)
from sglang.srt.layers.moe.paged_experts.residency import (
    DeviceResidency,
    LRUPolicy,
    plan_waves,
)
from sglang.srt.layers.moe.paged_experts.runners import (
    MARLIN,
    TRITON,
    MarlinContract,
    TritonContract,
    contract_for,
)
from sglang.srt.layers.moe.paged_experts.store import HostExpertStore
from sglang.srt.layers.moe.token_dispatcher.standard import StandardDispatchOutput
from sglang.srt.layers.moe.topk import StandardTopKOutput
from sglang.srt.model_executor.runner_backend_utils.breakable_cuda_graph import (
    breakable_cuda_graph,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

E, K, H, TOP_K = 16, 4, 8, 2


class _LinearMoE:
    """Stand-in for the fused-MoE method in no-combine mode: the [tokens, top_k, hidden]
    entries w[t,j] * x[t] @ w13[id] @ w2[id], left for the caller to sum; expert id -1 gives
    zero entries."""

    runner = SimpleNamespace(config=SimpleNamespace(num_experts=E, num_local_experts=K))

    def __init__(self):
        self.rows = 0

    def apply(self, layer, dispatch_output):
        topk = dispatch_output.topk_output
        ids, w = topk.topk_ids.long(), topk.topk_weights
        valid = ids >= 0
        self.rows += int(valid.sum())
        ids = ids.clamp(min=0)
        y = torch.einsum(
            "th,tjhi->tji", dispatch_output.hidden_states, layer.w13_weight[ids]
        )
        y = torch.einsum("tji,tjio->tjo", y, layer.w2_weight[ids])
        return SimpleNamespace(hidden_states=(w * valid).unsqueeze(-1) * y)

    def create_moe_runner(self, layer, moe_runner_config):
        pass

    def process_weights_after_loading(self, layer):
        pass


class _CpuStore(HostExpertStore):
    def _alloc(self, shape, dtype):
        return torch.empty(shape, dtype=dtype)


def _sum_top_k(self, entries, out, routed_scaling_factor):
    out.copy_(entries.sum(1) * routed_scaling_factor)


class _CompactingContract(MarlinContract):
    _sum_top_k = _sum_top_k


class _SkippingContract(TritonContract):
    _sum_top_k = _sum_top_k


CONTRACTS = (_CompactingContract(), _SkippingContract())


def _reference(full, dispatch_output):
    return _LinearMoE().apply(full, dispatch_output).hidden_states.sum(1)


def _step(num_tokens, padded=False):
    """A random routing step; ``padded`` marks the first token's entries as padding (-1)."""
    ids = torch.stack([torch.randperm(E)[:TOP_K] for _ in range(num_tokens)]).int()
    if padded:
        ids[0] = -1
    weights = torch.rand(num_tokens, TOP_K, dtype=torch.float32)
    x = torch.randn(num_tokens, H, dtype=torch.float32)
    return StandardDispatchOutput(
        hidden_states=x,
        hidden_states_scale=None,
        topk_output=StandardTopKOutput(
            topk_weights=weights, topk_ids=ids, router_logits=None
        ),
    )


_METHOD = "sglang.srt.layers.moe.paged_experts.method."


def _create(method, layer, contract):
    """Create the base method's expert tensors and runner as FusedMoE does, on a CPU store,
    with ``contract`` for the runner."""
    with (
        patch(_METHOD + "HostExpertStore", _CpuStore),
        patch(_METHOD + "runner_config", side_effect=lambda config, num_slots: config),
        patch(_METHOD + "contract_for", return_value=contract),
    ):
        method.create_weights(
            layer=layer,
            num_experts=E,
            hidden_size=H,
            intermediate_size_per_partition=H,
            params_dtype=torch.float32,
        )
        method.create_moe_runner(layer, SimpleNamespace(routed_scaling_factor=None))


def _load(method, layer):
    with patch.object(formats, "HostExpertStore", _CpuStore):
        method.process_weights_after_loading(layer)


def _paged_layer(
    contract=CONTRACTS[0],
    extra_params=(),
    global_params=(),
    freed=(),
    load=True,
    num_resident=K,
    auto_size=False,
    top_k=None,
    seed=0,
):
    """A paged method over random experts, its layer created through the base method on a CPU
    store; ``load`` fills the store with the experts and runs the post-load step. ``auto_size``
    builds it with K = ``num_resident`` slots that sizing then grows, as for
    --paged-experts-num-resident auto; ``top_k`` sets the layer's router top-k."""
    torch.manual_seed(seed)
    full = {
        name: torch.randn(E, H, H)
        for name in ("w13_weight", "w2_weight", *extra_params)
    }

    class _Base(_LinearMoE):
        def create_weights(self, layer, num_experts, **kwargs):
            for name in ("w13_weight", "w2_weight", *extra_params, *global_params):
                param = torch.nn.Parameter(
                    torch.full((num_experts, H, H), 0.5), requires_grad=False
                )
                param.quant_method = "block"
                if name in global_params:
                    param._sglang_require_global_experts = True
                layer.register_parameter(name, param)

    expert_format = UnquantizedFormat()
    expert_format.freed = freed
    method = PagedExpertsMoEMethod(
        base_method=_Base(),
        expert_format=expert_format,
        num_experts=E,
        num_resident=num_resident,
        auto_size=auto_size,
    )
    layer = torch.nn.Module()
    layer.weight_load_method = None
    layer.top_k = top_k
    layer.quant_method = method  # as FusedMoE holds it
    _create(method, layer, contract)
    if load:
        for name, t in full.items():
            method.store.host[name].copy_(t)
        _load(method, layer)
    return method, layer, SimpleNamespace(**full)


class TestPagedForward(CustomTestCase):
    def test_matches_unpaged(self):
        """Consecutive steps (which reuse and evict residents), in one wave (1, 2 tokens) or
        several (64 tokens), with and without padded entries: each output equals the unpaged
        layer, the slots hold the resident experts in every per-expert tensor (here also a
        bias the kernel ignores), and no wave computes the entries of another (without
        compaction, every wave would compute all of them)."""
        for contract in CONTRACTS:
            for num_tokens, padded in ((1, False), (2, False), (64, False), (3, True)):
                with self.subTest(
                    contract=type(contract).__name__,
                    num_tokens=num_tokens,
                    padded=padded,
                ):
                    method, layer, full = _paged_layer(
                        contract, extra_params=("w2_weight_bias",)
                    )
                    for _ in range(8):
                        d = _step(num_tokens, padded=padded)
                        method.base_method.rows = 0
                        out = method.apply(layer=layer, dispatch_output=d).hidden_states
                        torch.testing.assert_close(out, _reference(full, d))
                        for name, host in method.store.host.items():
                            for expert, slot in method.policy.resident.items():
                                torch.testing.assert_close(
                                    getattr(layer, name)[slot], host[expert]
                                )
                        self.assertLess(
                            method.base_method.rows, 2 * d.topk_output.topk_ids.numel()
                        )

    def test_trims_a_runner_padded_hidden_dimension(self):
        """A runner that pads the hidden dimension (e.g. MXFP4 Marlin) returns wider entries;
        the output keeps the input's width, in one wave (1 token) or several (64)."""
        for contract in CONTRACTS:
            method, layer, full = _paged_layer(contract)
            apply = method.base_method.apply
            method.base_method.apply = lambda layer, dispatch_output: SimpleNamespace(
                hidden_states=torch.nn.functional.pad(
                    apply(layer, dispatch_output).hidden_states, (0, 3), value=7.0
                )
            )
            for num_tokens in (1, 64):
                d = _step(num_tokens)
                out = method.apply(layer=layer, dispatch_output=d).hidden_states
                torch.testing.assert_close(out, _reference(full, d))

    def test_triton_waves_find_a_runner_the_scheme_owns(self):
        """A method that delegates to a per-layer scheme (e.g. compressed-tensors FP8)
        keeps its triton runner on the scheme, not on itself."""
        method, layer, full = _paged_layer(CONTRACTS[1])
        layer.scheme = SimpleNamespace(runner=method.base_method.runner)
        method.base_method.runner = None
        for num_tokens in (1, 64):
            d = _step(num_tokens)
            out = method.apply(layer=layer, dispatch_output=d).hidden_states
            torch.testing.assert_close(out, _reference(full, d))

    def test_capture_skips_the_waves_and_replay_runs_them(self):
        """Breakable capture runs the break once, before any captured kernel has run, on
        routing ids nothing has computed: the waves are skipped then, and replay runs them
        on the ids the graph produced."""
        method, layer, full = _paged_layer()
        d = _step(4)
        ids = torch.full_like(d.topk_output.topk_ids, 2**31 - 1)  # not computed yet
        out = torch.empty_like(d.hidden_states)
        resident = dict(method.policy.resident)
        capture = SimpleNamespace(
            _end_current_segment=lambda: None,
            cuda_graph=SimpleNamespace(_break_fns=[]),
            _begin_new_segment=lambda: None,
        )
        device = SimpleNamespace(current_stream=lambda: SimpleNamespace(stream_id=0))
        token = breakable_cuda_graph._current_capture_var.set(capture)
        try:
            with patch.object(
                breakable_cuda_graph, "get_device_module", return_value=device
            ):
                method._run_eager_into(
                    layer,
                    d.hidden_states,
                    d.topk_output.topk_weights,
                    ids,
                    None,
                    out,
                )
        finally:
            breakable_cuda_graph._current_capture_var.reset(token)
        self.assertEqual(method.base_method.rows, 0)
        self.assertEqual(dict(method.policy.resident), resident)

        ids.copy_(d.topk_output.topk_ids)
        (replay,) = capture.cuda_graph._break_fns
        replay()
        torch.testing.assert_close(out, _reference(full, d))


class TestLoading(CustomTestCase):
    def test_rejects_post_load_weight_transform(self):
        """Without a repack, host rows are paged verbatim: the post-load step must leave the
        expert tensors as loaded."""
        method, layer, _ = _paged_layer(TritonContract(), load=False)
        method.base_method.process_weights_after_loading = lambda layer: (
            layer.w2_weight.data.mul_(2)
        )
        with self.assertRaisesRegex(RuntimeError, "transforms w2_weight"):
            _load(method, layer)

    def test_host_loader_follows_base_method_and_param_tags(self):
        """Host rows are loaded as natively: the layer's loader follows the wrapped method's
        layout, and sees the parameter's tags (a scale's ``quant_method`` decides its
        layout)."""
        method, layer, _ = _paged_layer(load=False)
        self.assertIs(layer.weight_load_method, method.base_method)
        self.assertEqual(method.store.host["w2_weight"].shape, (E, H, H))
        method.store.host["w2_weight"].zero_()
        seen = {}

        def impl(self, param, loaded_weight, weight_name, shard_id, expert_id):
            seen["tag"] = param.quant_method
            param.data[expert_id].copy_(loaded_weight)

        layer._weight_loader_impl = impl.__get__(layer)
        layer.w2_weight.weight_loader(
            layer.w2_weight, torch.ones(H, H), "experts.w2_weight", "w2", 9
        )

        self.assertEqual(seen, {"tag": "block"})
        self.assertTrue(
            torch.equal(method.store.host["w2_weight"][9], torch.ones(H, H))
        )
        self.assertFalse(method.store.host["w2_weight"][8].any())

    def test_host_store_starts_from_the_created_values(self):
        """What a loader leaves unwritten (e.g. padding a kernel adds) holds the values the
        base method created, as on the GPU, not uninitialized memory."""
        method, _, _ = _paged_layer(load=False)
        for host in method.store.host.values():
            self.assertTrue((host == 0.5).all())

    def test_stacked_checkpoint_goes_to_the_native_loader_as_called(self):
        """A checkpoint tensor holding every expert (e.g. gpt-oss) goes, as called, to the
        layer's native loader (fused for bf16, padded for MXFP4) on the whole host store; the
        padding it leaves keeps the value the base method created."""
        method, layer, _ = _paged_layer(load=False)
        calls = []

        def native(param, loaded_weight, weight_name, shard_id, *rest, **kwargs):
            calls.append((tuple(param.shape), rest, kwargs))
            param.data[..., : loaded_weight.shape[-1]].copy_(loaded_weight)

        method.native_loader = native
        stacked = torch.randn(E, H, H - 1)  # narrower, as into a padded parameter
        loader = layer.w2_weight.weight_loader
        loader(layer.w2_weight, stacked, "experts.w2_weight", "w2")
        loader(
            layer.w2_weight, stacked, weight_name="w2", shard_id=None, expert_id=None
        )
        self.assertEqual(
            calls, [((E, H, H), (), {}), ((E, H, H), (), {"expert_id": None})]
        )
        self.assertTrue(torch.equal(method.store.host["w2_weight"][..., :-1], stacked))
        self.assertTrue((method.store.host["w2_weight"][..., -1] == 0.5).all())

    def test_rejects_online_weight_updates(self):
        from sglang.srt.model_executor.model_runner_components.weight_updater import (
            _unsupported_derived_weight_cache_error,
        )

        _, layer, _ = _paged_layer(load=False)
        model = torch.nn.Sequential(layer)
        self.assertIn(
            "--enable-paged-experts", _unsupported_derived_weight_cache_error(model)
        )

    def test_finds_the_per_expert_tensors_as_eplb_does(self):
        """Every tensor with one row per expert is loaded into the host store, except those
        sized for all experts, which stay whole on the GPU, and those the format frees."""
        method, layer, _ = _paged_layer(
            extra_params=("w13_weight_scale", "w13_blockscale_swizzled"),
            global_params=("w13_input_scale",),
            freed=("w13_blockscale_swizzled",),
            load=False,
        )
        self.assertEqual(
            method.checkpoint_params, ("w13_weight", "w2_weight", "w13_weight_scale")
        )
        self.assertEqual(set(method.store.host), set(method.checkpoint_params))
        self.assertIsNone(layer.w13_blockscale_swizzled)

    def _staged(self, layer, num_experts):
        staged = _CpuStore(
            layer=layer, names=("w13_weight",), num_experts=num_experts, pin=False
        )
        staged.host["w13_weight"].copy_(
            torch.arange(num_experts * 6.0).reshape(num_experts, 6)
        )
        return staged

    def test_repack_runs_per_chunk_and_keeps_its_output(self):
        """A post-load step that repacks each expert into a new tensor runs over every expert
        of the host store, K at a time, each chunk from the layer's state before the first (a
        step that skips once it has run still repacks every chunk); the tensors it leaves per
        expert are paged, and the slots hold experts 0..K-1 repacked."""

        class _Repacking:
            def process_weights_after_loading(self, layer):
                if getattr(layer, "converted", False):  # e.g. is_marlin_converted
                    return
                layer.converted = True
                raw = layer.w13_weight.data
                packed = (raw * 10 + 1).reshape(raw.shape[0], 2, -1)
                del layer.w13_weight
                layer.register_parameter(
                    "w13_qweight", torch.nn.Parameter(packed, requires_grad=False)
                )

        layer = torch.nn.Module()
        layer.register_parameter(
            "w13_weight", torch.nn.Parameter(torch.zeros(K, 6), requires_grad=False)
        )
        staged = self._staged(layer, E + 1)  # a partial last chunk
        with patch.object(formats, "HostExpertStore", _CpuStore):
            packed = after_loading(
                layer, _Repacking(), staged, num_slots=K, repacks=True, unpaged=()
            )

        expected = (staged.host["w13_weight"] * 10 + 1).reshape(E + 1, 2, 3)
        self.assertEqual(
            list(packed.host), ["w13_qweight"]
        )  # not the deleted w13_weight
        self.assertTrue(torch.equal(packed.host["w13_qweight"], expected))
        self.assertTrue(torch.equal(layer.w13_qweight.data, expected[:K]))

    def test_a_short_last_chunk_is_padded_with_experts(self):
        """With E % K != 0, the rows past the last chunk hold a real expert's weights and scales,
        not what the base method created (torch.empty): NVFP4A16's post-load step checks that
        every row's gate and up share one global scale, and passes on a valid checkpoint."""
        from sglang.srt.layers.quantization.compressed_tensors.schemes.compressed_tensors_w4a16_nvfp4_moe import (
            CompressedTensorsW4A16Nvfp4MoE,
        )

        scheme = CompressedTensorsW4A16Nvfp4MoE()
        layer = torch.nn.Module()
        layer.moe_runner_config = SimpleNamespace(is_gated=True)
        scheme.create_weights(
            layer,
            num_experts=K,
            hidden_size=128,
            intermediate_size_per_partition=64,
            params_dtype=torch.bfloat16,
        )
        for param in layer.parameters():
            if param.is_floating_point():
                param.data.fill_(float("nan"))  # what torch.empty may hold
        names = [name for name, _ in layer.named_parameters()]
        staged = _CpuStore(layer=layer, names=names, num_experts=E + 1, pin=False)
        for host in staged.host.values():  # a last chunk of 1 expert
            if host.is_floating_point():
                host.copy_(torch.rand(host.shape).add(0.5).to(host.dtype))
            else:
                host.copy_(torch.randint(0, 256, host.shape, dtype=host.dtype))
        global_scale = staged.host["w13_weight_global_scale"]
        global_scale[:, 1] = global_scale[:, 0]  # gate and up share it, as exported
        with (
            patch.object(formats, "HostExpertStore", _CpuStore),
            patch(
                "sglang.srt.layers.quantization.marlin_utils_fp4."
                "prepare_moe_nvfp4_layer_for_marlin"
            ),
        ):
            packed = after_loading(
                layer, scheme, staged, num_slots=K, repacks=True, unpaged=()
            )
        self.assertTrue(
            torch.equal(packed.host["w13_weight_scale_2"], 1 / global_scale[:, 0])
        )
        self.assertTrue(
            torch.equal(packed.host["w13_weight"], staged.host["w13_weight_packed"])
        )

    def test_rejects_a_repack_that_mixes_experts(self):
        """A post-load step that depends on the other experts of its chunk (here, their
        mean) cannot run K at a time."""

        class _Mixing:
            def process_weights_after_loading(self, layer):
                raw = layer.w13_weight.data
                raw.sub_(raw.mean(0))

        layer = torch.nn.Module()
        layer.register_parameter(
            "w13_weight", torch.nn.Parameter(torch.zeros(K, 6), requires_grad=False)
        )
        staged = self._staged(layer, E)
        with (
            patch.object(formats, "HostExpertStore", _CpuStore),
            self.assertRaisesRegex(RuntimeError, "each expert on its own"),
        ):
            after_loading(
                layer, _Mixing(), staged, num_slots=K, repacks=True, unpaged=()
            )


class TestResidency(CustomTestCase):
    def test_lru_evicts_least_recently_used(self):
        policy = LRUPolicy(4)  # slots start with experts 0..3
        wave = policy.place([0, 5])  # 0 stays; 5 evicts 1, the least recently used
        self.assertEqual(wave.loads, [(5, 1)])
        self.assertEqual(wave.slots, [0, 1])
        wave = policy.place([6, 7])  # evicts 2 and 3; 0 and 5 stay
        self.assertEqual(sorted(wave.loads), [(6, 2), (7, 3)])
        self.assertEqual(sorted(policy.resident), [0, 5, 6, 7])

    def test_resident_wave_loads_nothing(self):
        self.assertEqual(LRUPolicy(4).place([3, 1]).loads, [])

    def test_waves_serve_residents_before_evicting_them(self):
        policy = LRUPolicy(2)
        policy.place([2, 3])  # residents {2, 3}
        waves = plan_waves(policy=policy, distinct=[0, 1, 2, 3])
        self.assertEqual([w.experts for w in waves], [[2, 3], [0, 1]])
        self.assertEqual(sum(len(w.loads) for w in waves), 2)

    def test_waves_cover_every_expert_once(self):
        for distinct, sizes in (([], [0]), ([2, 9], [2]), (list(range(10)), [4, 4, 2])):
            with self.subTest(distinct=distinct):
                waves = plan_waves(policy=LRUPolicy(4), distinct=distinct)
                self.assertEqual([len(w.experts) for w in waves], sizes)
                self.assertEqual([e for w in waves for e in w.experts], distinct)
                for w in waves:  # within a wave, every expert has its own slot
                    self.assertEqual(len(set(w.slots)), len(w.slots))

    def test_device_state_round_trips_through_the_policy(self):
        """Host-planned steps between graph replays continue from the device's LRU state and
        leave theirs for the next replay."""
        state = DeviceResidency(num_experts=E, num_slots=4, device="cpu")
        state.slot_expert.copy_(torch.tensor([9, 2, 7, 5], dtype=torch.int32))
        state.slot_lastuse.copy_(torch.tensor([6, 3, 8, 3], dtype=torch.int32))
        state.step.fill_(8)
        policy = LRUPolicy(4)

        counts = torch.zeros(E + 1, dtype=torch.int32)
        counts[11] = 3
        self.assertEqual(state.load_into(policy, counts), counts.tolist())
        self.assertEqual(
            list(policy.resident.items()), [(2, 1), (5, 3), (9, 0), (7, 2)]
        )
        policy.place([11])  # evicts 2, the least recently used
        state.store_from(policy)

        self.assertEqual(state.slot_expert.tolist(), [9, 11, 7, 5])
        self.assertEqual(state.slot_lastuse.tolist(), [10, 12, 11, 9])
        self.assertEqual(int(state.step), 12)
        self.assertEqual(state.expert_slot[[9, 11, 7, 5, 2]].tolist(), [0, 1, 2, 3, -1])


class TestDecodeCaptureCap(CustomTestCase):
    def _capped(self, bs, max_bs, full=True):
        decode = SimpleNamespace(bs=list(bs), max_bs=max(bs))
        exec_ctx = SimpleNamespace(
            graph=SimpleNamespace(cuda_graph_config=SimpleNamespace(decode=decode))
        )
        with (
            patch(_METHOD + "get_exec", return_value=exec_ctx),
            patch(_METHOD + "check_cuda_graph_backend", return_value=full),
        ):
            _cap_decode_capture_batch_sizes(max_bs)
        return decode.bs, decode.max_bs

    def test_only_shrinks_the_configured_sizes(self):
        self.assertEqual(self._capped([1, 2, 4], 16), ([1, 2, 4], 4))
        self.assertEqual(self._capped([1, 2, 4, 8], 3), ([1, 2, 3], 3))
        self.assertEqual(self._capped([1, 2, 4], 0), ([1], 1))  # at least one request

    def test_only_full_decode_graphs_are_capped(self):
        """Breakable or disabled decode graphs break out for host-planned waves: their sizes
        stay. Control: the same call under a full decode graph caps them."""
        self.assertEqual(self._capped([1, 2, 4, 8], 3, full=False), ([1, 2, 4, 8], 8))
        self.assertEqual(self._capped([1, 2, 4, 8], 3, full=True), ([1, 2, 3], 3))

    def test_an_explicit_k_caps_at_construction_and_auto_does_not(self):
        """An explicit K caps the capture sizes when the layer is built; auto (None, or the
        string a Python caller passes) waits for the sized K, building with K = top_k."""
        layer = SimpleNamespace(num_local_experts=E, top_k=TOP_K)
        for requested, caps_expected, k in (
            (K, [K // TOP_K], K),
            (None, [], TOP_K),
            ("auto", [], TOP_K),
        ):
            with self.subTest(requested=requested):
                caps = []
                moe = SimpleNamespace(paged_experts_num_resident=requested)
                with (
                    patch(_METHOD + "get_exec", return_value=SimpleNamespace(moe=moe)),
                    patch(_METHOD + "format_for", return_value=UnquantizedFormat()),
                    patch(_METHOD + "check_paged_experts_compat"),
                    patch(_METHOD + "cuda_graph_fully_disabled", return_value=False),
                    patch(
                        _METHOD + "_cap_decode_capture_batch_sizes",
                        side_effect=caps.append,
                    ),
                ):
                    method = make_for_layer(layer, _LinearMoE())
                self.assertEqual(caps, caps_expected)
                self.assertEqual(method.num_resident, k)
                self.assertEqual(method.auto_size, requested != K)


class TestFormats(CustomTestCase):
    def test_every_format_resolves_its_method(self):
        """Each format imports the method it handles (a moved class fails here, not at a
        user's startup) and claims no other."""
        for fmt in formats.FORMATS:
            with self.subTest(fmt.__name__):
                self.assertFalse(fmt.supports(object(), None))

    def test_contract_follows_the_runner(self):
        from sglang.srt.layers.moe.utils import MoeRunnerBackend

        def runner(backend):
            return SimpleNamespace(runner_backend=backend)

        self.assertIs(contract_for(runner(MoeRunnerBackend.TRITON)), TRITON)
        self.assertIs(contract_for(runner(MoeRunnerBackend.MARLIN)), MARLIN)
        # These sum the top-k outputs inside the kernel, so waves cannot be exact.
        for backend in (
            MoeRunnerBackend.FLASHINFER_CUTLASS,
            MoeRunnerBackend.TRITON_KERNELS,
        ):
            with self.assertRaisesRegex(RuntimeError, backend.value):
                contract_for(runner(backend))
        with self.assertRaisesRegex(RuntimeError, "triton or Marlin"):
            contract_for(None)


class TestCompat(CustomTestCase):
    def _check(self, num_resident=K, speculative_algorithm=None, **flags):
        method = "sglang.srt.layers.moe.paged_experts.method."
        moe = SimpleNamespace(
            enable_eplb=False,
            paged_experts_num_resident=num_resident,
            enable_fused_moe_sum_all_reduce=False,
        )
        features = SimpleNamespace(
            enable_memory_saver=flags.get("enable_memory_saver", False)
        )
        kernel = SimpleNamespace(
            enable_nvidia_megakernel=flags.get("enable_nvidia_megakernel", False)
        )
        lora = SimpleNamespace(enable_lora=flags.get("enable_lora", False))
        layer = SimpleNamespace(top_k=TOP_K, num_fused_shared_experts=0)
        with (
            patch(
                method + "get_parallel",
                return_value=SimpleNamespace(
                    tp_size=1, ep_size=1, pp_size=1, dp_size=1
                ),
            ),
            patch(
                method + "get_exec",
                return_value=SimpleNamespace(moe=moe, features=features, kernel=kernel),
            ),
            patch(method + "get_lora", return_value=lora),
            patch(
                method + "get_spec",
                return_value=SimpleNamespace(
                    speculative_algorithm=speculative_algorithm
                ),
            ),
            patch(
                method + "get_model", return_value=SimpleNamespace(load_format="auto")
            ),
            patch(method + "check_cuda_graph_backend", return_value=True),
        ):
            check_paged_experts_compat(layer=layer, expert_format=object())

    def test_accepts_a_supported_config(self):
        self._check()

    def test_rejects_speculative_decoding(self):
        with self.assertRaisesRegex(RuntimeError, "speculative decoding"):
            self._check(speculative_algorithm="EAGLE")

    def test_num_resident_auto_is_accepted(self):
        self._check(num_resident=None)  # auto: sized when the KV cache is configured
        self._check(num_resident="auto")  # as a Python caller passes it

    def test_rejects_num_resident_below_top_k(self):
        with self.assertRaisesRegex(RuntimeError, "at least the router's top-k"):
            self._check(num_resident=TOP_K - 1)

    def test_auto_refuses_what_binds_or_moves_the_slot_tables(self):
        for flag in ("enable_memory_saver", "enable_lora", "enable_nvidia_megakernel"):
            with self.subTest(flag=flag):
                for auto in (None, "auto"):
                    with self.assertRaisesRegex(
                        RuntimeError, "explicit --paged-experts"
                    ):
                        self._check(num_resident=auto, **{flag: True})
                self._check(num_resident=K, **{flag: True})  # an explicit K is fine


class TestSizing(CustomTestCase):
    _SIZING = "sglang.srt.layers.moe.paged_experts.sizing."

    def _schedule(self, **over):
        return SimpleNamespace(
            **{
                "max_total_tokens": None,
                "max_running_requests": None,
                "max_mamba_cache_size": None,
                "mamba_full_memory_ratio": 0.9,
                "page_size": 1,
                **over,
            }
        )

    @contextlib.contextmanager
    def _patched(self, context_length=None, **schedule):
        """sizing's view of the server arguments; yields the overrides it makes."""
        overrides = []
        context = SimpleNamespace(override=lambda source, **f: overrides.append(f))
        with (
            patch(
                self._SIZING + "get_schedule", return_value=self._schedule(**schedule)
            ),
            patch(
                self._SIZING + "get_model",
                return_value=SimpleNamespace(context_length=context_length),
            ),
            patch(self._SIZING + "get_context", return_value=context),
        ):
            yield overrides

    def _floor(self, model_context, context_length=None, **schedule):
        with self._patched(context_length, **schedule):
            return sizing.kv_floor(model_context)

    def test_kv_floor_promises_one_request_and_aims_for_the_rest(self):
        f = self._floor(262144, max_total_tokens=10240, max_running_requests=4)
        self.assertEqual((f.tokens, f.minimum, f.hard), (10240, 10240, True))
        f = self._floor(8192, context_length=8192, max_running_requests=4)
        self.assertEqual((f.tokens, f.minimum, f.hard), (8192 * 4, 8192, True))
        f = self._floor(262144)  # nothing set: one request of the model's context, soft
        self.assertEqual((f.tokens, f.minimum, f.hard), (262144, 262144, False))
        f = self._floor(4096, max_running_requests=2)
        self.assertEqual((f.tokens, f.minimum, f.hard), (4096 * 2, 4096, False))

    def test_kv_floor_holds_a_full_request_in_whole_pages(self):
        """The KV cache rounds down to whole pages, so a request of 10001 tokens needs 10016
        at page size 16 (control: at page size 1, 10001)."""
        f = self._floor(
            10001, context_length=10001, max_running_requests=2, page_size=16
        )
        self.assertEqual((f.tokens, f.minimum), (10016 * 2, 10016))
        f = self._floor(10001, context_length=10001, max_running_requests=2)
        self.assertEqual((f.tokens, f.minimum), (10001 * 2, 10001))

    def test_slots_grow_in_whole_rounds_until_a_layer_holds_every_expert(self):
        grow = sizing.ExpertSlotSizing._grow
        self.assertEqual(grow([2, 2], [32, 32], [100, 100], 2050), [12, 12])
        # The small layer stops at E; the rest goes to the big one alone.
        self.assertEqual(grow([2, 2], [8, 32], [100, 100], 3000), [8, 26])
        self.assertEqual(grow([2, 2], [32, 32], [100, 100], 199), [2, 2])

    def _sized(
        self,
        budget_slots,
        kv_tokens=10240,
        top_ks=(TOP_K, TOP_K),
        context_length=None,
        refuse=False,
        **schedule,
    ):
        """Two layers built with K = top_k, sized through ``ExpertSlotSizing`` against a static
        budget of ``budget_slots`` slots per layer (each layer's slot bytes). Returns K per
        layer, the layers, the decode caps applied and the overrides made."""
        built = [
            _paged_layer(seed=seed, num_resident=top_k, top_k=top_k, auto_size=True)
            for seed, top_k in enumerate(top_ks)
        ]
        layers = [layer for _, layer, _ in built]
        row = sizing._bytes_per_slot(built[0][0])
        kvc = self._kvc()
        caps = []
        with (
            self._patched(context_length, **schedule) as overrides,
            patch(
                self._SIZING + "static_budget_bytes",
                return_value=budget_slots * row * len(layers),
            ),
            patch(
                _METHOD + "runner_config", side_effect=lambda config, num_slots: config
            ),
            patch(_METHOD + "runner_of", return_value=_LinearMoE.runner),
            patch(_METHOD + "_cap_decode_capture_batch_sizes", side_effect=caps.append),
        ):
            model = torch.nn.ModuleList(layers)
            slots = sizing.ExpertSlotSizing.for_model(model, kvc, 0)
            # ``refuse``: the allocator refuses the slab (only around ``size``).
            refused = torch.OutOfMemoryError("refused")
            with (
                patch(self._SIZING + "torch.empty", side_effect=refused)
                if refuse
                else (contextlib.nullcontext())
            ):
                ks = slots.size(kv_tokens)
        return ks, built, caps, overrides

    @staticmethod
    def _kvc():
        return SimpleNamespace(
            mambaish_config=None,
            post_capture_kv_active=False,
            model_config=SimpleNamespace(context_len=4096),
        )

    def test_layers_with_an_explicit_k_are_not_sized(self):
        """Layers built with an explicit K leave the KV cache alone: no sizing, no
        --max-total-tokens override. Control: an auto layer gets both."""
        for auto_size, sized in ((False, False), (True, True)):
            with self.subTest(auto_size=auto_size):
                _, layer, _ = _paged_layer(
                    num_resident=TOP_K, top_k=TOP_K, auto_size=auto_size
                )
                with (
                    self._patched() as overrides,
                    patch(self._SIZING + "static_budget_bytes", return_value=0),
                ):
                    slots = sizing.ExpertSlotSizing.for_model(
                        torch.nn.ModuleList([layer]), self._kvc(), 0
                    )
                self.assertEqual(slots is not None, sized)
                self.assertEqual(bool(overrides), sized)

    def test_the_kv_target_is_applied_as_max_total_tokens_unless_set(self):
        _, _, _, overrides = self._sized(budget_slots=4, max_running_requests=2)
        self.assertEqual(overrides, [{"max_total_tokens": 4096 * 2}])
        _, _, _, overrides = self._sized(budget_slots=4, max_total_tokens=10240)
        self.assertEqual(overrides, [])

    def test_slots_take_the_measured_remainder_and_serve_the_unpaged_forward(self):
        """K = 4 per layer from a budget of 4 slots each; every token routes to experts 2 and
        3, which only the grown slots hold. The negative control corrupts grown slot 3."""
        ks, built, caps, _ = self._sized(budget_slots=4)
        self.assertEqual(ks, [K, K])
        self.assertEqual(
            caps, [K // TOP_K]
        )  # the decode capture cap, once, at the final K
        storage = {
            layer.w13_weight.untyped_storage().data_ptr() for _, layer, _ in built
        }
        self.assertEqual(len(storage), 1)  # every slot table lives in one allocation
        method, layer, full = built[0]
        paged = []
        page_in = method.store.page_in
        method.store.page_in = lambda layer, loads: (
            paged.extend(loads),
            page_in(layer, loads),
        )
        for corrupt in (False, True):
            if corrupt:
                layer.w13_weight.data[3] += 1.0
            step = _step(5)
            step.topk_output.topk_ids.copy_(torch.tensor([2, 3]).int().expand(5, TOP_K))
            out = method.apply(layer, step).hidden_states
            same = torch.allclose(out, _reference(full, step), atol=1e-4)
            self.assertEqual(same, not corrupt)
            self.assertEqual(paged, [])  # served from the grown slots: nothing paged in

    def test_the_decode_cap_holds_for_every_layer(self):
        """Layers with top_k 2 and 4 grow to K = 7 and 9: the cap is the smallest K // top_k
        of any layer (9 // 4 = 2), not the smallest K over the smallest top_k (7 // 2 = 3)."""
        ks, _, caps, _ = self._sized(budget_slots=8, top_ks=(2, 4))
        self.assertEqual(ks, [7, 9])
        self.assertEqual(caps, [2])

    def test_a_refused_allocation_fails_naming_the_flags(self):
        with self.assertRaisesRegex(
            RuntimeError, "lower --max-total-tokens, --context-length or --max-running"
        ):
            self._sized(budget_slots=4, refuse=True)

    def test_a_kv_floor_the_user_set_that_does_not_fit_fails_naming_its_source(self):
        with self.assertRaisesRegex(
            RuntimeError, "fewer than the 20000 from --max-total-tokens"
        ):
            self._sized(budget_slots=4, kv_tokens=8000, max_total_tokens=20000)
        with self.assertRaisesRegex(
            RuntimeError, r"fewer than one request of 4096 \(--context-length"
        ):
            self._sized(budget_slots=4, kv_tokens=1000, context_length=4096)

    def test_a_model_context_that_does_not_fit_sizes_and_warns(self):
        """The model's own context is soft: a KV cache shorter than it is a warning naming
        --context-length, and the slots are still sized. Control: one that fits, no warning."""
        with self.assertLogs(sizing.logger, level="WARNING") as logs:
            ks, _, _, _ = self._sized(budget_slots=4, kv_tokens=1000)
        self.assertEqual(ks, [K, K])
        self.assertTrue(any("--context-length" in line for line in logs.output))
        with self.assertNoLogs(sizing.logger, level="WARNING"):
            self._sized(budget_slots=4, kv_tokens=4096)

    def test_the_log_names_k_the_kv_cache_its_aim_and_how_to_pin(self):
        with self.assertLogs(sizing.logger, level="INFO") as logs:
            self._sized(budget_slots=4, max_total_tokens=10240)
        line = logs.output[0]
        for part in (
            f"K={K} of {E} experts",
            "KV cache 10240 tokens (aim 10240: --max-total-tokens)",
            f"To pin: --paged-experts-num-resident {K}",
        ):
            self.assertIn(part, line)

    def test_more_requests_are_a_target_one_request_the_promise(self):
        """--context-length x --max-running-requests is aimed for; a KV cache that holds one
        full-length request but not all of them sizes, logs how many fit, and does not fail."""
        with self.assertLogs(sizing.logger, level="INFO") as logs:
            self._sized(
                budget_slots=4,
                kv_tokens=6000,
                context_length=4096,
                max_running_requests=4,
            )
        self.assertTrue(
            any("1 request(s) of 4096 at once" in line for line in logs.output)
        )

    def test_auto_refuses_post_capture_kv_sizing(self):
        built = [_paged_layer(num_resident=TOP_K, top_k=TOP_K, auto_size=True)]
        kvc = SimpleNamespace(post_capture_kv_active=True)
        with self.assertRaisesRegex(
            RuntimeError, "SGLANG_ENABLE_POST_CAPTURE_KV_SIZING"
        ):
            sizing.ExpertSlotSizing.for_model(
                torch.nn.ModuleList([built[0][1]]), kvc, 0
            )

    def _mamba_slots(self, budget, context_length=None, kv_bytes=1, **schedule):
        """Mamba slots of 1000 B, 2 per request; ``kv_bytes`` per KV token, rounded down to
        whole pages like the KV cache."""
        page = schedule.get("page_size", 1)
        kvc = SimpleNamespace(
            mambaish_config=SimpleNamespace(
                mamba2_cache_params=SimpleNamespace(mamba_cache_per_req=1000)
            ),
            extra_mamba_cache_bytes_per_req=0,
            _calculate_mamba_ratio=lambda: 2,
            model_config=SimpleNamespace(context_len=context_length or 4096),
        )
        with (
            self._patched(context_length, **schedule) as overrides,
            patch(
                self._SIZING + "_kv_tokens",
                side_effect=lambda kvc, nbytes: (
                    max(nbytes, 0) // kv_bytes // page * page
                ),
            ),
        ):
            sizing.ExpertSlotSizing([], kvc, 0, budget)
        return [
            o["max_mamba_cache_size"] for o in overrides if "max_mamba_cache_size" in o
        ]

    def _sglang_mamba(self, rest_bytes, per_slot, **schedule):
        """SGLang's own Mamba accounting (``KVCacheConfigurator._handle_max_mamba_cache``) on
        a configurator of one GPU without speculative decoding: the bytes it deducts and the
        Mamba pool size it settles on."""
        from sglang.srt.mem_cache import kv_cache_configurator as kvc_mod

        sched = self._schedule(**schedule)
        context = SimpleNamespace(
            override=lambda source, **f: [setattr(sched, k, v) for k, v in f.items()]
        )
        fake = SimpleNamespace(
            mambaish_config=SimpleNamespace(
                mamba2_cache_params=SimpleNamespace(
                    layers=[0, 1], mamba_cache_per_req=per_slot
                )
            ),
            pp_size=1,
            fused_entry_bytes=lambda name: None,
            spec_algorithm=SimpleNamespace(is_none=lambda: True),
            hybrid_gdn_config=None,
            hybrid_kda_config=None,
            extra_mamba_cache_bytes_per_req=0,
            attn_dp_size=1,
        )
        exec_ = SimpleNamespace(
            mamba=SimpleNamespace(enable_linear_replayssm_spec=False)
        )
        with (
            patch.object(kvc_mod, "get_schedule", return_value=sched),
            patch.object(kvc_mod, "get_context", return_value=context),
            patch.object(kvc_mod, "get_exec", return_value=exec_),
            patch.object(
                kvc_mod,
                "get_memory",
                return_value=SimpleNamespace(disable_radix_cache=False),
            ),
        ):
            rest = kvc_mod.KVCacheConfigurator._handle_max_mamba_cache(
                fake, rest_bytes / sizing._GIB
            )
        return round(
            (rest_bytes / sizing._GIB - rest) * sizing._GIB
        ), sched.max_mamba_cache_size

    def test_mamba_charge_and_share_match_sglang(self):
        """Our Mamba charge and share are SGLang's own: with --max-mamba-cache-size set (as
        auto sets it) SGLang deducts exactly ``_mamba_charge``, and with it unset its ratio
        branch settles on ``_mamba_share``. Control: a charge without the padding slot does
        not match."""
        per_slot = 1_000_000
        for slots in (1, 3, 16):
            charged, _ = self._sglang_mamba(
                10 * sizing._GIB, per_slot, max_mamba_cache_size=slots
            )
            self.assertEqual(charged, sizing._mamba_charge(slots, per_slot))
            self.assertNotEqual(charged, slots * per_slot)
        for budget, ratio in (
            (10 * sizing._GIB, 0.9),
            (3 * sizing._GIB, 9),
            (sizing._GIB, 0.5),
        ):
            _, size = self._sglang_mamba(
                budget, per_slot, mamba_full_memory_ratio=ratio
            )
            self.assertEqual(size, sizing._mamba_share(budget, ratio, per_slot))

    def test_mamba_state_holds_the_requests_the_kv_cache_aims_for(self):
        """2 state slots per request: one request by default, the --max-running-requests set
        (8: 16 slots), capped at SGLang's share (0.9 / 1.9 of 19,000 B = 9,000 B, less the
        pool's padding slot: 8 slots of 1000 B), at least one request; nothing when
        --max-mamba-cache-size is set."""
        self.assertEqual(self._mamba_slots(10**6), [2])
        self.assertEqual(self._mamba_slots(10**6, max_running_requests=8), [16])
        self.assertEqual(self._mamba_slots(19_000, max_running_requests=8), [8])
        self.assertEqual(self._mamba_slots(100, max_running_requests=8), [2])
        self.assertEqual(self._mamba_slots(10**6, max_mamba_cache_size=5), [])

    def test_one_full_request_comes_before_more_mamba_state(self):
        """Near the limit, the Mamba state grows only as far as one full-length request still
        fits next to it; more requests queue. The pool holds one padding slot beyond its size.
        20,000 B, a share of 17 slots (ratio 9), 8 requests of 4096 tokens at 2 B: 16 slots
        (17 with the padding) would leave 3,000 B, short of the 8,192 one request needs, so
        10 (5 requests; 11 charged) leave 9,000 B. Control: in the old order, 16 slots leave
        the KV cache short of one request, which startup rejects."""
        args = dict(context_length=4096, kv_bytes=2, max_running_requests=8)
        args["mamba_full_memory_ratio"] = 9
        self.assertEqual(self._mamba_slots(20_000, **args), [10])
        self.assertLess((20_000 - 17 * 1000) // 2, 4096)
        # Room for all 8 requests: no change. No room even for one: one request's state,
        # and startup names the context that does not fit.
        self.assertEqual(self._mamba_slots(10**6, **args), [16])
        self.assertEqual(self._mamba_slots(5_000, **args), [2])
        # --max-total-tokens is the minimum when set. 2000 tokens: 16 slots fit only without
        # the padding slot (4,000 B left vs 3,000 B with it), so 14.
        args = dict(kv_bytes=2, max_running_requests=8, mamba_full_memory_ratio=9)
        self.assertEqual(self._mamba_slots(20_000, max_total_tokens=2000, **args), [14])
        self.assertEqual(self._mamba_slots(20_000, max_total_tokens=5000, **args), [8])

    def test_the_log_says_how_many_requests_run_and_how_to_run_more(self):
        """Each running request needs its own Mamba state, so the log says how many run at a
        time; without --max-running-requests it names that flag, with it it says how many of
        them queue. Control: all requests set fit, no hint."""
        self.assertEqual(
            sizing._mamba_text(3, 3, None),
            "Mamba state for 1 request(s) at a time (3 slot(s)); "
            "set --max-running-requests to run more",
        )
        self.assertEqual(
            sizing._mamba_text(9, 2, 8),
            "Mamba state for 4 request(s) at a time (9 slot(s)) "
            "of --max-running-requests 8; the rest queue",
        )
        self.assertEqual(
            sizing._mamba_text(16, 2, 8),
            "Mamba state for 8 request(s) at a time (16 slot(s))",
        )

    def test_kv_minimum_by_size_rule_in_whole_pages(self):
        """A --max-total-tokens that is no page multiple: the KV cache can hold at most
        9,984 of 10,000 tokens at page size 64, which size() accepts, so the Mamba state
        keeps all 8 requests when 9,984 fit (26,990 B less 17 slots of 1000 B = 9,990 B).
        Control: "fewer tokens than asked" would call 9,984 short and drop one request."""
        args = dict(max_running_requests=8, mamba_full_memory_ratio=9, page_size=64)
        self.assertEqual(
            self._mamba_slots(26_990, max_total_tokens=10_000, **args), [16]
        )
        self.assertLess(9_984, 10_000)
        self.assertGreater(9_984 + 64, 10_000)


if __name__ == "__main__":
    unittest.main()
