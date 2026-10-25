import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.layers.moe.paged_experts import PagedExpertsMoEMethod, formats
from sglang.srt.layers.moe.paged_experts.formats import UnquantizedFormat, after_loading
from sglang.srt.layers.moe.paged_experts.method import (
    _cap_decode_capture_batch_sizes,
    check_paged_experts_compat,
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
    contract=CONTRACTS[0], extra_params=(), global_params=(), freed=(), load=True
):
    """A paged method over random experts, its layer created through the base method on a CPU
    store; ``load`` fills the store with the experts and runs the post-load step."""
    torch.manual_seed(0)
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
        base_method=_Base(), expert_format=expert_format, num_experts=E, num_resident=K
    )
    layer = torch.nn.Module()
    layer.weight_load_method = None
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
            _barrier_fn=None,
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

        state.load_into(policy)
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
    def _capped(self, bs, max_bs):
        decode = SimpleNamespace(bs=list(bs), max_bs=max(bs))
        exec_ctx = SimpleNamespace(
            graph=SimpleNamespace(cuda_graph_config=SimpleNamespace(decode=decode))
        )
        with patch(
            "sglang.srt.layers.moe.paged_experts.method.get_exec", return_value=exec_ctx
        ):
            _cap_decode_capture_batch_sizes(max_bs)
        return decode.bs, decode.max_bs

    def test_only_shrinks_the_configured_sizes(self):
        self.assertEqual(self._capped([1, 2, 4], 16), ([1, 2, 4], 4))
        self.assertEqual(self._capped([1, 2, 4, 8], 3), ([1, 2, 3], 3))


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
    def _check(self, num_resident=K, speculative_algorithm=None):
        method = "sglang.srt.layers.moe.paged_experts.method."
        moe = SimpleNamespace(
            enable_eplb=False,
            paged_experts_num_resident=num_resident,
            enable_fused_moe_sum_all_reduce=False,
        )
        layer = SimpleNamespace(top_k=TOP_K, num_fused_shared_experts=0)
        with (
            patch(
                method + "get_parallel",
                return_value=SimpleNamespace(
                    tp_size=1, ep_size=1, pp_size=1, dp_size=1
                ),
            ),
            patch(method + "get_exec", return_value=SimpleNamespace(moe=moe)),
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

    def test_requires_num_resident(self):
        with self.assertRaisesRegex(RuntimeError, "--paged-experts-num-resident"):
            self._check(num_resident=None)


if __name__ == "__main__":
    unittest.main()
