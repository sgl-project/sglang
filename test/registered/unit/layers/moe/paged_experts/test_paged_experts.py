import contextlib
import errno
import fcntl
import gc
import mmap
import os
import tempfile
import threading
import unittest
import weakref
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from sglang.srt.layers.moe.paged_experts import (
    PagedExpertsMoEMethod,
    disk_tier,
    expert_cache,
    formats,
    runners,
    sizing,
)
from sglang.srt.layers.moe.paged_experts import store as store_module
from sglang.srt.layers.moe.paged_experts.disk_tier import TieredExpertStore
from sglang.srt.layers.moe.paged_experts.expert_cache import (
    ExpertCache,
    ExpertReader,
    _sampler,
    read_many,
)
from sglang.srt.layers.moe.paged_experts.formats import (
    PostLoad,
    UnquantizedFormat,
    build_store,
)
from sglang.srt.layers.moe.paged_experts.method import (
    _cap_decode_capture_batch_sizes,
    _loads_mapped_files,
    _slot_table_bytes,
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
    RunnerContract,
    TritonContract,
    contract_for,
)
from sglang.srt.layers.moe.paged_experts.store import (
    ExpertLoads,
    HostExpertStore,
    same_bytes,
)
from sglang.srt.layers.moe.token_dispatcher.standard import StandardDispatchOutput
from sglang.srt.layers.moe.topk import StandardTopKOutput
from sglang.srt.model_executor.runner.base_cuda_graph_runner import freeze_gc
from sglang.srt.model_executor.runner_backend_utils.breakable_cuda_graph import (
    breakable_cuda_graph,
)
from sglang.test.ci.ci_register import register_cpu_ci, register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=40, suite="base-a-test-cpu")
# The captured cases (disk reads waited for or replayed inside a CUDA graph) need a GPU.
register_cuda_ci(est_time=60, stage="base-b-kernel-unit", runner_config="1-gpu-small")


@contextlib.contextmanager
def _captured(graph, **kwargs):
    """A capture taken the way the server's graph runners take one (``freeze_gc``: the cyclic gc
    runs first, then leaves every live object alone until the capture ends). A capture without
    it lets a collection free a mailbox whose pinned rows an earlier capture copied into, on the
    same stream; torch's pinned allocator then records the block's free on a capturing stream,
    and the next pinned allocation fails ("invalid argument"; ``TestPinnedFreeDuringCapture``)."""
    with freeze_gc(False), torch.cuda.graph(graph, **kwargs):
        yield


E, K, H, TOP_K = 16, 4, 8, 2


class _LinearMoE:
    """Stand-in for the fused-MoE method in no-combine mode: the [tokens, top_k, hidden]
    entries w[t,j] * x[t] @ w13[id] @ w2[id], left for the caller to sum; expert id -1 gives
    zero entries."""

    runner = SimpleNamespace(config=SimpleNamespace(num_experts=E, num_local_experts=K))
    # What the triton runner's quantization does to a row: nothing here, so a row's output
    # does not depend on the other rows of its call.
    quant_info = SimpleNamespace(
        use_fp8_w8a8=False,
        use_int8_w8a8=False,
        block_shape=None,
        per_channel_quant=False,
        a13_scale=None,
        a2_scale=None,
    )

    def __init__(self):
        self.rows = 0
        self.calls = []  # (rows, top_k) of each call

    def get_triton_quant_info(self, layer):
        return self.quant_info

    def apply(self, layer, dispatch_output):
        topk = dispatch_output.topk_output
        ids, w = topk.topk_ids.long(), topk.topk_weights
        valid = ids >= 0
        self.rows += int(valid.sum())
        self.calls.append(tuple(ids.shape))
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


class _CpuTiered(TieredExpertStore):
    _alloc = _CpuStore._alloc


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
_EXPERT_CACHE = "sglang.srt.layers.moe.paged_experts.expert_cache."


def _create(method, layer, contract):
    """Create the base method's expert tensors and runner as FusedMoE does, with
    ``contract`` for the runner and loads that wait (a mapped safetensors checkpoint)."""
    with (
        patch(_METHOD + "_loads_mapped_files", return_value=True),
        patch(_METHOD + "runner_config", side_effect=lambda config, num_slots: config),
        patch(_METHOD + "contract_for", return_value=contract),
        patch(_METHOD + "host_rows", side_effect=lambda rows, e, *a, **k: e),
        patch(_METHOD + "get_exec", return_value=_exec()),
    ):
        method.create_weights(
            layer=layer,
            num_experts=E,
            hidden_size=H,
            intermediate_size_per_partition=H,
            params_dtype=torch.float32,
        )
        method.create_moe_runner(layer, SimpleNamespace(routed_scaling_factor=None))


def _exec(enable_disk=False):
    return SimpleNamespace(
        moe=SimpleNamespace(
            paged_experts_enable_disk=enable_disk, paged_experts_host_memory_gb=None
        )
    )


def _load(method, layer, disk=False, router_top_k=K):
    """The post-load step, with the host store on the CPU, the expert cache as ``disk`` says
    and the model's router top-k ``router_top_k`` (for a layer without one)."""
    model_config = SimpleNamespace(
        hf_text_config=SimpleNamespace(num_experts_per_tok=router_top_k)
    )
    with (
        patch.object(formats, "HostExpertStore", _CpuStore),
        patch(_EXPERT_CACHE + "HostExpertStore", _CpuStore),
        patch(_METHOD + "get_exec", return_value=_exec(disk)),
        patch(_METHOD + "process_model_config", return_value=model_config),
    ):
        method.process_weights_after_loading(layer)


def _copy_row(param, loaded_weight, **kw):
    """A layer's per-expert loader, reduced to copying the row it is given."""
    param.data[0].copy_(loaded_weight)


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
    """A paged method over random experts, its layer created through the base method;
    ``load`` loads the experts and runs the post-load step (host store on the CPU).
    ``auto_size`` builds it with K = ``num_resident`` slots that sizing then grows, as for
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
        for name_, t in full.items():
            method.loads.table[name_].copy_(t)
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

    def test_triton_waves_run_only_their_rows_with_the_full_calls_configs(self):
        """With row-local quantization (none, or FP8 per-tensor with both scales static),
        each triton wave runs only its entries (top-k 1 rows), inside a config lookup for the
        step's full call (tokens x top-k), so the runner keeps that call's kernel configs.
        Control: a dynamic per-tensor scale for either GEMM's input (a max over all rows)
        keeps every wave at the full call, outside any lookup."""
        lookups = []

        @contextlib.contextmanager
        def lookup(num_tokens, top_k):
            lookups.append((num_tokens, top_k))
            yield

        static = torch.ones(1)
        cases = {  # name: (a13_scale, a2_scale, quantized), dynamic = not row-local
            "unquantized": (None, None, False),
            "fp8-static": (static, static, True),
            "fp8-dynamic-input": (None, static, True),
            "fp8-dynamic-intermediate": (static, None, True),
        }
        for name, (a13, a2, quantized) in cases.items():
            dynamic = quantized and (a13 is None or a2 is None)
            with self.subTest(quantization=name):
                method, layer, full = _paged_layer(CONTRACTS[1])
                method.base_method.quant_info = SimpleNamespace(
                    use_fp8_w8a8=quantized,
                    use_int8_w8a8=False,
                    block_shape=None,
                    per_channel_quant=False,
                    a13_scale=a13,
                    a2_scale=a2,
                )
                lookups.clear()
                d = _step(64)
                method.base_method.calls.clear()
                with patch.object(runners, "_config_lookup", lookup):
                    out = method.apply(layer=layer, dispatch_output=d).hidden_states
                torch.testing.assert_close(out, _reference(full, d))
                calls = method.base_method.calls
                self.assertGreater(len(calls), 1)  # several waves
                if dynamic:
                    self.assertEqual(lookups, [])
                    self.assertEqual(set(calls), {(64, TOP_K)})
                else:
                    self.assertEqual(set(lookups), {(64, TOP_K)})
                    self.assertTrue(all(k == 1 for _, k in calls))
                    self.assertLess(sum(r for r, _ in calls), 2 * 64 * TOP_K)

    def test_no_runner_call_exceeds_the_compacted_rows(self):
        """A step whose entries exceed one compacted call (``COMPACT_ROWS``, here 8) runs in
        calls of at most that many rows (padded to the next batch size), also when one wave
        holds every expert it routes to, and matches unpaged. Control: with the default cap
        the one-wave step is one full call."""
        for contract in CONTRACTS:
            for cap, one_call in ((8, False), (runners.COMPACT_ROWS, True)):
                with self.subTest(contract=type(contract).__name__, cap=cap):
                    method, layer, full = _paged_layer(contract)
                    d = _step(64)
                    d.topk_output.topk_ids.copy_(  # experts 0-3: one wave of K = 4
                        torch.stack(
                            [torch.randperm(K)[:TOP_K] for _ in range(64)]
                        ).int()
                    )
                    method.base_method.calls.clear()
                    with patch.object(runners, "COMPACT_ROWS", cap):
                        out = method.apply(layer=layer, dispatch_output=d).hidden_states
                    torch.testing.assert_close(out, _reference(full, d))
                    calls = method.base_method.calls
                    if one_call and isinstance(contract, TritonContract):
                        self.assertEqual(calls, [(64, TOP_K)])
                    elif not one_call:
                        self.assertGreater(len(calls), 1)
                        self.assertTrue(all(r <= 16 and k == 1 for r, k in calls))

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
        seen = {}

        def impl(self, param, loaded_weight, weight_name, shard_id, expert_id):
            seen["tag"] = param.quant_method
            param.data[expert_id].copy_(loaded_weight)

        layer._weight_loader_impl = impl.__get__(layer)
        layer.w2_weight.weight_loader(
            layer.w2_weight, _mapped(torch.ones(H, H)), "experts.w2_weight", "w2", 9
        )
        self.assertEqual(seen, {})  # a view of a mapped file: it waits
        rows = method.loads.rows([8, 9])["w2_weight"]
        self.assertEqual(seen, {"tag": "block"})
        self.assertTrue(torch.equal(rows[1], torch.ones(H, H)))
        self.assertTrue(
            (rows[0] == 0.5).all()
        )  # not loaded: as the base method created

    def test_stacked_checkpoint_goes_to_the_native_loader_as_called(self):
        """A checkpoint tensor holding every expert (e.g. gpt-oss) goes, as called, to the
        layer's native loader (fused for bf16, padded for MXFP4) on a table of every expert,
        after the waiting loads; the padding it leaves keeps the value the base method
        created."""
        method, layer, full = _paged_layer(load=False)
        layer._weight_loader_impl = _copy_row
        layer.w13_weight.weight_loader(
            layer.w13_weight, _mapped(full.w13_weight[3]), "w13_weight", "w13", 3
        )
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
        table = method.loads.table
        self.assertTrue(torch.equal(table["w2_weight"][..., :-1], stacked))
        self.assertTrue((table["w2_weight"][..., -1] == 0.5).all())
        # The load that waited before the stacked one is in the table too.
        self.assertTrue(torch.equal(table["w13_weight"][3], full.w13_weight[3]))
        self.assertFalse(method.loads.waiting)

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
        self.assertEqual(set(method.loads.rows_like), set(method.checkpoint_params))
        self.assertIsNone(layer.w13_blockscale_swizzled)

    def test_repack_runs_per_chunk_and_keeps_its_output(self):
        """A post-load step that repacks each expert into a new tensor runs over every expert,
        K at a time, each chunk from the layer's state before the first (a step that skips
        once it has run still repacks every chunk); the tensors it leaves per expert are
        paged, and the slots hold experts 0..K-1 repacked."""
        layer = _w13_layer(K)
        loads, values = _loads(layer, E + 1)  # a partial last chunk
        store = _post_load(layer, _Repacking(), loads)
        expected = (values * 10 + 1).reshape(E + 1, 2, 3)
        self.assertEqual(
            list(store.host), ["w13_qweight"]
        )  # not the deleted w13_weight
        self.assertTrue(torch.equal(store.host["w13_qweight"], expected))
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
        loads = ExpertLoads(layer, names, E + 1, K)
        for table in loads.table.values():  # a last chunk of 1 expert
            if table.is_floating_point():
                table.copy_(torch.rand(table.shape).add(0.5).to(table.dtype))
            else:
                table.copy_(torch.randint(0, 256, table.shape, dtype=table.dtype))
        global_scale = loads.table["w13_weight_global_scale"]
        global_scale[:, 1] = global_scale[:, 0]  # gate and up share it, as exported
        with patch(
            "sglang.srt.layers.quantization.marlin_utils_fp4."
            "prepare_moe_nvfp4_layer_for_marlin"
        ):
            store = _post_load(layer, scheme, loads)
        self.assertTrue(
            torch.equal(store.host["w13_weight_scale_2"], 1 / global_scale[:, 0])
        )
        self.assertTrue(
            torch.equal(store.host["w13_weight"], loads.table["w13_weight_packed"])
        )

    def test_each_chunk_repacks_from_what_it_loaded(self):
        """A step that reads its own output parameter when present (MXFP4 Marlin reads
        `w13_weight_scale` before the loaded `w13_weight_scale_inv`) sees, in every chunk,
        only what that chunk loaded. Negative control: keeping the last chunk's output,
        later chunks repack it, which the mixing check refuses."""

        class _PrefersOwnOutput:
            def process_weights_after_loading(self, layer):
                source = getattr(layer, "w13_weight_scale", None)
                if source is None:
                    source = layer.w13_weight
                layer.register_parameter(
                    "w13_weight_scale",
                    torch.nn.Parameter(source.data * 2, requires_grad=False),
                )

        for drops in (True, False):
            with self.subTest(drops=drops):
                layer = _w13_layer(K)
                loads, values = _loads(layer, E)
                step = PostLoad(
                    layer, _PrefersOwnOutput(), loads, True, ("w13_weight",)
                )
                if not drops:
                    step._params |= {"w13_weight_scale"}
                    with self.assertRaisesRegex(RuntimeError, "each expert on its own"):
                        _fill(step)
                    continue
                store = _fill(step)
                self.assertTrue(_same(store.host["w13_weight_scale"], values * 2))

    def test_serves_mxfp4_experts_of_fp8_checkpoints(self):
        from sglang.srt.layers.quantization.mxfp4_marlin_moe import (
            Mxfp4MarlinMoEMethod,
        )

        method = Mxfp4MarlinMoEMethod.__new__(Mxfp4MarlinMoEMethod)
        self.assertIsInstance(
            formats.format_for(SimpleNamespace(scheme=None), method),
            formats.Fp8Mxfp4ExpertsFormat,
        )

    def test_rejects_a_repack_that_mixes_experts(self):
        """A post-load step that depends on the other experts of its chunk (here, their
        mean) cannot run K at a time."""

        class _Mixing:
            def process_weights_after_loading(self, layer):
                raw = layer.w13_weight.data
                raw.sub_(raw.mean(0))

        layer = _w13_layer(K)
        loads, _ = _loads(layer, E)
        with self.assertRaisesRegex(RuntimeError, "each expert on its own"):
            _post_load(layer, _Mixing(), loads)

    def test_replays_see_the_layer_as_before_the_step(self):
        """A chunk's loads replay after earlier chunks' steps ran on the layer; a loader that
        reads the layer's state (as FusedMoE's reads its scheme) still sees it as loaded.
        Negative control: without restoring that state first, later chunks replay garbage,
        which the mixing check (experts 1..K again) refuses."""
        for restores in (True, False):
            with self.subTest(restores=restores):
                layer = _w13_layer(K)
                layer.mode = "loaded"
                loads, values = _loads(layer, E, wait=True, reads_mode=layer)
                _Repacking.marks = True
                step = PostLoad(layer, _Repacking(), loads, True, ())
                if restores:
                    store = _fill(step)
                    expected = (values * 10 + 1).reshape(E, 2, 3)
                    self.assertTrue(_same(store.host["w13_qweight"], expected))
                    continue
                real_reset = step._reset  # restore all but the state the loader reads

                def reset_keeping_mode(real_reset=real_reset):
                    mode = layer.mode
                    real_reset()
                    layer.mode = mode

                step._reset = reset_keeping_mode
                with self.assertRaisesRegex(RuntimeError, "each expert on its own"):
                    _fill(step)
        _Repacking.marks = False


class _Repacking:
    """A post-load step that repacks each expert into a new tensor, once per layer state (as
    Marlin's ``is_marlin_converted``); ``calls`` counts the chunks it repacked, ``nan_first``
    makes expert 0 NaN (a valid output ``torch.equal`` never matches), ``validates`` rejects
    a row that was never loaded (-1, as NVFP4A16 checks rows), ``marks`` changes the layer's
    ``mode`` (state a loader may read)."""

    calls = 0
    nan_first = False
    validates = False
    marks = False

    def process_weights_after_loading(self, layer):
        if getattr(layer, "converted", False):
            return
        layer.converted = True
        if type(self).marks:
            layer.mode = "repacked"
        type(self).calls += 1
        raw = layer.w13_weight.data
        if type(self).validates and (raw == -1).any():
            raise ValueError("a row that was never loaded")
        packed = (raw * 10 + 1).reshape(raw.shape[0], 2, -1)
        if type(self).nan_first:
            packed[raw[:, 0] == 0] = float("nan")
        del layer.w13_weight
        layer.register_parameter(
            "w13_qweight", torch.nn.Parameter(packed, requires_grad=False)
        )


class _NoStep:
    """A post-load step that leaves the loaded rows as they are (FP8, bf16)."""

    def process_weights_after_loading(self, layer):
        pass


_same = same_bytes


def _mapped(tensor: torch.Tensor) -> torch.Tensor:
    """A copy of ``tensor`` as a view of a mapped file, mapped private as safetensors maps a
    checkpoint: its bytes are in the file, not in memory of its own."""
    data = tensor.contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()
    with tempfile.TemporaryFile() as f:
        f.write(data or b"\0")
        f.flush()
        buf = mmap.mmap(
            f.fileno(),
            max(1, len(data)),
            flags=mmap.MAP_PRIVATE,
            prot=mmap.PROT_READ | mmap.PROT_WRITE,
        )
    out = torch.frombuffer(buf, dtype=tensor.dtype, count=tensor.numel())
    return out.view(tensor.shape)


def _anonymous(tensor: torch.Tensor) -> torch.Tensor:
    """A copy of ``tensor`` in private anonymous memory (as numpy or a loader's own buffer
    holds a tensor): ``MADV_DONTNEED`` zeroes it."""
    nbytes = tensor.numel() * tensor.element_size()
    buf = mmap.mmap(-1, max(1, nbytes), flags=mmap.MAP_PRIVATE | mmap.MAP_ANONYMOUS)
    out = torch.frombuffer(buf, dtype=tensor.dtype, count=tensor.numel())
    return out.view(tensor.shape).copy_(tensor)


def _w13_layer(num_slots: int, created: float = 0.0) -> torch.nn.Module:
    layer = torch.nn.Module()
    layer.register_parameter(
        "w13_weight",
        torch.nn.Parameter(torch.full((num_slots, 6), created), requires_grad=False),
    )
    return layer


def _loads(
    layer,
    num_experts,
    wait=False,
    skip=(),
    reads_mode=None,
    count=None,
    num_slots=K,
):
    """Loads of experts ``0..num_experts-1`` (values ``e * 6 .. e * 6 + 5``) for ``layer``:
    in a table, or (``wait``) waiting, ``skip`` never written (rows stay as created) and
    ``count`` called per replayed write; ``reads_mode``: the write reads that layer's
    ``mode`` and writes garbage unless it says "loaded"."""
    values = torch.arange(num_experts * 6.0).reshape(num_experts, 6)
    loads = ExpertLoads(layer, ("w13_weight",), num_experts, num_slots)
    if not wait:
        loads.table["w13_weight"].copy_(values)
        return loads, values
    for e in range(num_experts):

        def write(dest, e=e):
            if count is not None:
                count(e)
            if e in skip:
                return
            ok = reads_mode is None or reads_mode.mode == "loaded"
            dest.copy_(values[e] if ok else -values[e] - 7)

        assert loads.defer(e, "w13_weight", write, values[e])
    return loads, values


def _fill(step):
    """``build_store`` with the host store on the CPU."""
    with patch.object(formats, "HostExpertStore", _CpuStore):
        return build_store(step)


def _post_load(layer, base_method, loads, repacks=True):
    step = PostLoad(layer, base_method, loads, repacks, ())
    store = _fill(step)
    step.finish(store)
    return store


def _page_events(start):
    """``start()`` (a load that fills a layer): the experts its loads replay and whose
    checkpoint pages it gives back, as ("read" | "out", expert) in order."""
    events = []
    page_out, rows = ExpertLoads.page_out, ExpertLoads.rows

    def spy_page_out(loads, experts):
        experts = list(experts)
        events.extend(("out", e) for e in experts)
        return page_out(loads, experts)

    def spy_rows(loads, experts, out=None):
        events.extend(("read", e) for e in dict.fromkeys(experts))
        return rows(loads, experts, out)

    with (
        patch.object(ExpertLoads, "page_out", spy_page_out),
        patch.object(ExpertLoads, "rows", spy_rows),
    ):
        start()
    return events


def _given_back_early(events, num_experts):
    """The experts whose pages a start gave back other than once, after their last read."""
    wrong = []
    for e in range(num_experts):
        mine = [kind for kind, x in events if x == e]
        if mine.count("out") != 1 or mine[-1] != "out":
            wrong.append((e, mine))
    return wrong


def _rows_ignoring_keep(loads, experts, out=None):
    """``ExpertLoads.rows`` that gives every replayed expert's pages back, also those the
    mixing check reads again (the loads before ``keep``)."""
    keep, loads.keep = loads.keep, set()
    try:
        return _rows(loads, experts, out)
    finally:
        loads.keep = keep


_rows = ExpertLoads.rows


class TestLoads(CustomTestCase):
    def test_loads_wait_until_their_experts_are_replayed(self):
        """A view of a mapped file waits; ``rows`` replays only the asked experts' loads,
        onto the created values; repeats are copied, not loaded again."""
        method, layer, full = _paged_layer(load=False)
        writes = []

        def impl(param, loaded_weight, **kw):
            writes.append(1)
            param.data[0].copy_(loaded_weight)

        layer._weight_loader_impl = impl
        for e in range(E):
            layer.w2_weight.weight_loader(
                layer.w2_weight, _mapped(full.w2_weight[e]), "w2_weight", "w2", e
            )
        self.assertEqual(writes, [])
        rows = method.loads.rows([3, 5, 3])["w2_weight"]
        self.assertEqual(len(writes), 2)
        self.assertTrue(torch.equal(rows[0], full.w2_weight[3]))
        self.assertTrue(torch.equal(rows[1], full.w2_weight[5]))
        self.assertTrue(torch.equal(rows[2], full.w2_weight[3]))
        self.assertTrue((method.loads.rows([3])["w13_weight"] == 0.5).all())

    def test_only_views_of_mapped_files_wait(self):
        """A tensor torch allocated (e.g. a ``.bin`` checkpoint's) cannot wait: it would hold
        its memory. A real safetensors file's tensor can (the control)."""
        from safetensors import safe_open
        from safetensors.torch import save_file

        from sglang.srt.layers.moe.paged_experts.method import _can_defer

        with tempfile.TemporaryDirectory() as folder:
            path = os.path.join(folder, "w.safetensors")
            save_file({"w": torch.randn(64, 64)}, path)
            with safe_open(path, framework="pt") as f:
                self.assertTrue(_can_defer(f.get_tensor("w")))
        self.assertTrue(_can_defer(_mapped(torch.ones(4))))
        self.assertFalse(_can_defer(torch.ones(4)))

    def test_loads_wait_only_with_the_safetensors_loaders(self):
        """Only the safetensors loaders with mmap give views of the mapped files; other
        loaders' tensors may be copies, on the GPU, or buffers reused for the next tensor
        (the RunAI streamer's), so their loads go to the table at once."""
        for load_format, disable_mmap, waits in (
            ("auto", False, True),
            ("safetensors", False, True),
            ("mistral", False, True),
            ("auto", True, False),
            ("pt", False, False),
            ("npcache", False, False),
            ("runai_streamer", False, False),
            ("fastsafetensors", False, False),
        ):
            model = SimpleNamespace(
                load_format=load_format, weight_loader_disable_mmap=disable_mmap
            )
            with (
                self.subTest(load_format=load_format, disable_mmap=disable_mmap),
                patch(_METHOD + "get_model", return_value=model),
            ):
                self.assertEqual(_loads_mapped_files(), waits)

    def test_a_load_that_cannot_wait_ends_the_waiting(self):
        """It writes the table, after the created values and the waiting loads; later loads
        go there too. Negative control: a waiting load not in the table would leave its row
        as created."""
        method, layer, full = _paged_layer(load=False)
        layer._weight_loader_impl = _copy_row
        loader = layer.w2_weight.weight_loader
        loader(layer.w2_weight, _mapped(full.w2_weight[5]), "w2_weight", "w2", 5)
        self.assertTrue(method.loads.waiting)
        loader(layer.w2_weight, full.w2_weight[6].clone(), "w2_weight", "w2", 6)
        self.assertFalse(method.loads.waiting)
        loader(layer.w2_weight, _mapped(full.w2_weight[7]), "w2_weight", "w2", 7)
        table = method.loads.table["w2_weight"]
        self.assertTrue(torch.equal(table[5:8], full.w2_weight[5:8]))
        self.assertTrue((table[8] == 0.5).all())
        fresh = ExpertLoads(layer, ("w2_weight",), E, K)  # loads never handed over
        self.assertFalse(torch.equal(fresh.table["w2_weight"][5], full.w2_weight[5]))

    def test_replayed_pages_are_given_back_unchanged(self):
        """``madvise`` on a replayed view keeps its bytes (the kernel reads them again), and
        advice the kernel refuses (``MADV_PAGEOUT`` before Linux 5.4) leaves even anonymous
        memory as it is. Negative control: ``MADV_DONTNEED``, the advice it must not fall
        back to, zeroes anonymous memory."""
        view = _mapped(torch.arange(4096.0))
        before = view.clone()
        store_module._madvise(view, store_module._MADV_PAGEOUT)
        store_module._madvise(view, store_module._MADV_WILLNEED)
        self.assertTrue(torch.equal(view, before))

        values = torch.arange(4096.0) + 1
        anonymous = _anonymous(values)
        madvise = store_module._libc().madvise

        def refuses_pageout(start, length, advice):
            if advice == store_module._MADV_PAGEOUT:
                return -1
            return madvise(start, length, advice)

        with patch.object(
            store_module, "_libc", lambda: SimpleNamespace(madvise=refuses_pageout)
        ):
            store_module._madvise(anonymous, store_module._MADV_PAGEOUT)
        self.assertTrue(torch.equal(anonymous, values))
        store_module._madvise(anonymous, 4)  # MADV_DONTNEED
        self.assertFalse(torch.equal(anonymous, values))

    def test_a_build_reads_each_expert_once_and_gives_its_pages_back_after(self):
        """Every expert's loads replay once, also those of experts 1..K that the mixing check
        repacks again (their pages stay until it ran), and each expert's checkpoint pages are
        given back once, after its last read; for a step that repacks and one that does not.
        Negative control: giving the pages back at every replay, as before, reads experts
        1..K after they went."""
        for repacks in (True, False):
            with self.subTest(repacks=repacks):
                layer = _w13_layer(K)
                loads, _ = _loads(layer, E + 1, wait=True)
                base = _Repacking() if repacks else _NoStep()
                events = _page_events(lambda: _post_load(layer, base, loads, repacks))
                self.assertEqual(_given_back_early(events, E + 1), [])
                reads = [e for kind, e in events if kind == "read"]
                mixing = list(range(1, K + 1)) if repacks else []
                self.assertEqual(reads, list(range(E + 1)) + mixing)
        layer = _w13_layer(K)
        loads, _ = _loads(layer, E + 1, wait=True)
        with patch.object(ExpertLoads, "rows", _rows_ignoring_keep):
            events = _page_events(lambda: _post_load(layer, _Repacking(), loads))
        early = _given_back_early(events, E + 1)
        self.assertEqual([e for e, _ in early], list(range(1, K + 1)))

    def test_pages_are_given_back_off_the_loading_thread(self):
        """``page_out`` hands the replayed views to a thread of its own and returns before the
        kernel reclaimed them (it does so synchronously); ``drop`` waits for it and ends the
        thread. Control: ``read_ahead`` advises on the calling thread."""
        layer = _w13_layer(K)
        loads, _ = _loads(layer, E, wait=True)
        loads.rows([0, 1])  # replays experts 0 and 1, then gives their pages back
        started, release, seen = threading.Event(), threading.Event(), []
        madvise = store_module._madvise

        def slow(view, advice):
            seen.append((threading.current_thread(), advice))
            if advice == store_module._MADV_PAGEOUT:
                started.set()
                release.wait(5)
            return madvise(view, advice)

        with patch.object(store_module, "_madvise", slow):
            loads.page_out([2, 3])
            self.assertTrue(started.wait(5))
            pager = seen[0][0]
            self.assertTrue(pager.name.startswith("paged-experts-pageout"))
            loads.read_ahead([4])
            self.assertIn(
                (threading.current_thread(), store_module._MADV_WILLNEED), seen
            )
            done = threading.Thread(target=loads.drop)
            done.start()
            done.join(0.2)
            self.assertTrue(done.is_alive())  # drop waits for the page-out
            release.set()
            done.join(5)
        self.assertFalse(done.is_alive())
        self.assertIsNone(loads._pager)
        pager.join(5)  # the executor's worker exits after shutdown
        self.assertFalse(pager.is_alive())

    def test_madvise_without_linux_is_skipped(self):
        """Off Linux there is no libc ``madvise`` to load, and the advice is skipped; on
        Linux, ``MADV_PAGEOUT`` has its Linux value also where Python's mmap module does not
        export it."""
        store_module._libc.cache_clear()
        self.addCleanup(store_module._libc.cache_clear)
        view = _mapped(torch.arange(16.0))
        with patch.object(store_module, "_LINUX", False):
            self.assertIsNone(store_module._libc())
            store_module._madvise(view, store_module._MADV_PAGEOUT)  # a no-op
        store_module._libc.cache_clear()
        if store_module._LINUX:
            self.assertIsNotNone(store_module._libc())
            self.assertEqual(store_module._MADV_PAGEOUT, 21)
        self.assertTrue(torch.equal(view, torch.arange(16.0)))


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
        # This sums the top-k outputs inside the kernel, so waves cannot be exact.
        backend = MoeRunnerBackend.FLASHINFER_CUTLASS
        with self.assertRaisesRegex(RuntimeError, backend.value):
            contract_for(runner(backend))
        with self.assertRaisesRegex(RuntimeError, "triton or Marlin"):
            contract_for(None)


class TestCompat(CustomTestCase):
    def _check(
        self, num_resident=K, speculative_algorithm=None, load_format="auto", **flags
    ):
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
                method + "get_model",
                return_value=SimpleNamespace(load_format=load_format),
            ),
            patch(method + "check_cuda_graph_backend", return_value=True),
        ):
            check_paged_experts_compat(layer=layer, expert_format=object())

    def test_accepts_a_supported_config(self):
        self._check()

    def test_only_loaders_that_fill_through_load_weights_are_accepted(self):
        """The loaders that fill the experts through model.load_weights and then run the
        post-load step (e.g. pt) are accepted; the others are refused by name with their
        reason: dummy has no real weights, gguf GGUF-quantized experts, the rest (e.g.
        sharded_state, and any new format) load without Paged Experts' loader, and the
        message names the accepted ones."""
        from sglang.srt.configs.load_config import LoadFormat

        accepted = {
            LoadFormat.AUTO,
            LoadFormat.SAFETENSORS,
            LoadFormat.PT,
            LoadFormat.NPCACHE,
            LoadFormat.MISTRAL,
            LoadFormat.FASTSAFETENSORS,
            LoadFormat.INSTANTTENSOR,
            LoadFormat.RUNAI_STREAMER,
        }
        reasons = {
            LoadFormat.DUMMY: "needs the real weights",
            LoadFormat.GGUF: "GGUF-quantized experts",
        }
        for load_format in LoadFormat:
            with self.subTest(load_format=load_format.value):
                if load_format in accepted:
                    self._check(load_format=load_format)
                    self._check(load_format=load_format.value)  # as a string
                    continue
                reason = reasons.get(
                    load_format, r"without Paged Experts' loader .*supported: auto, "
                )
                with self.assertRaisesRegex(
                    RuntimeError,
                    f"--load-format {load_format.value} is not supported: .*{reason}",
                ):
                    self._check(load_format=load_format)

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
        """The schedule namespace. A ``max_running_requests`` given is the user's unless
        ``_max_running_requests_explicitly_set=False`` says SGLang resolved it."""
        over.setdefault(
            "_max_running_requests_explicitly_set",
            over.get("max_running_requests") is not None,
        )
        return SimpleNamespace(
            **{
                "max_total_tokens": None,
                "max_running_requests": None,
                "max_mamba_cache_size": None,
                "mamba_full_memory_ratio": 0.9,
                "page_size": 1,
                "chunked_prefill_size": 4096,
                "max_prefill_tokens": 16384,
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

    def test_a_resolved_max_running_requests_is_no_target(self):
        """A --max-running-requests SGLang resolves itself (DeepSeek-V4's 256) caps the
        request pool but is no target: the KV cache aims for one request, and the Mamba state
        is not raised to 256. Control: the same 256, or 4, set by the user is the target."""
        resolved = dict(_max_running_requests_explicitly_set=False)
        f = self._floor(4096, max_running_requests=256, **resolved)
        self.assertEqual((f.tokens, f.requests), (4096, 1))
        f = self._floor(4096, max_running_requests=4)
        self.assertEqual((f.tokens, f.requests), (4096 * 4, 4))
        args = dict(max_total_tokens=1000, max_running_requests=256)
        self.assertEqual(self._mamba_slots(10**6, **resolved, **args), [2])
        self.assertEqual(self._mamba_slots(10**6, **args), [472])

    def test_the_cache_hook_records_whether_the_user_set_max_running_requests(self):
        """From the raw input: DeepSeek-V4's 256, declared by its model hook before the cache
        hook runs, is the resolved value but not the user's. Control: the same 256 set by
        the user is."""
        from sglang.srt.arg_groups.kv_cache_hook import handle_cache_compatibility
        from sglang.srt.arg_groups.overrides import (
            declare_resolution,
            resolution_result,
        )
        from sglang.srt.server_args import ServerArgs

        field = "_max_running_requests_explicitly_set"
        args = ServerArgs(model_path="dummy")
        declare_resolution(args, "apply_deepseek_v4_defaults", max_running_requests=256)
        handle_cache_compatibility(args)
        self.assertEqual(resolution_result(args, "max_running_requests"), 256)
        self.assertIs(resolution_result(args, field), False)
        args = ServerArgs(model_path="dummy", max_running_requests=256)
        handle_cache_compatibility(args)
        self.assertIs(resolution_result(args, field), True)

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
        reserve_slots=0,
        **schedule,
    ):
        """Two layers built with K = top_k, sized through ``ExpertSlotSizing`` against a static
        budget of ``budget_slots`` slots per layer (each layer's slot bytes), of which a
        prefill step's working memory takes ``reserve_slots`` slots of one layer. Returns K
        per layer, the layers, the decode caps applied and the overrides made."""
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
            patch.object(
                RunnerContract, "prefill_bytes", return_value=reserve_slots * row
            ),
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

    def test_slots_leave_a_prefill_step_its_working_memory(self):
        """A step reserve of 2 slots' bytes takes one round of growth (one slot in each of the
        two layers): K = 3 instead of 4. Control: no reserve, K = 4."""
        for reserve, k in ((0, K), (2, K - 1)):
            with self.subTest(reserve_slots=reserve):
                ks, _, _, _ = self._sized(budget_slots=4, reserve_slots=reserve)
                self.assertEqual(ks, [k, k])

    def test_a_step_reserve_the_smallest_slots_leave_no_room_for_warns(self):
        """Slots stay at K = top_k and the log names the flags; control: a reserve that fits
        logs no warning."""
        with self.assertLogs(sizing.logger, "WARNING") as logs:
            ks, _, _, _ = self._sized(budget_slots=TOP_K, reserve_slots=3)
        self.assertEqual(ks, [TOP_K, TOP_K])
        self.assertIn("--chunked-prefill-size", logs.output[0])
        with self.assertNoLogs(sizing.logger, "WARNING"):
            self._sized(budget_slots=4, reserve_slots=2)

    def test_the_step_reserve_follows_the_chunk_and_the_layer(self):
        """The reserve is the merged [tokens, top_k, hidden] entries plus one compacted call
        of ``COMPACT_ROWS`` rows: doubling the tokens adds one more merged buffer, and the
        hidden width is padded as a runner may pad it. A step carries one prefill chunk, or
        without chunking the larger of a prefill batch and one full request."""
        c = RunnerContract()
        one = c.prefill_bytes(100, 40, 1024, 4, torch.bfloat16)
        two = c.prefill_bytes(100, 40, 2048, 4, torch.bfloat16)
        self.assertEqual(two - one, 1024 * 4 * 256 * 2)
        self.assertGreater(one, 1024 * 4 * 256 * 2)
        for chunk, prefill, minimum, tokens in (
            (4096, 16384, 4096, 4096),
            (-1, 16384, 4096, 16384),
            (None, 2048, 32768, 32768),
        ):
            with patch(
                self._SIZING + "get_schedule",
                return_value=SimpleNamespace(
                    chunked_prefill_size=chunk, max_prefill_tokens=prefill
                ),
            ):
                self.assertEqual(sizing._step_tokens(minimum), tokens)

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
        # Room for all 8 requests: no change. No room even for one: one request's state,
        # and startup names the context that does not fit.
        self.assertEqual(self._mamba_slots(10**6, **args), [16])
        self.assertEqual(self._mamba_slots(5_000, **args), [2])
        # --max-total-tokens is the minimum when set. 2000 tokens: 16 slots fit only without
        # the padding slot (4,000 B left vs 3,000 B with it), so 14.
        args = dict(kv_bytes=2, max_running_requests=8, mamba_full_memory_ratio=9)
        self.assertEqual(self._mamba_slots(20_000, max_total_tokens=2000, **args), [14])
        self.assertEqual(self._mamba_slots(20_000, max_total_tokens=5000, **args), [8])

    def test_kv_minimum_by_size_rule_in_whole_pages(self):
        """A --max-total-tokens that is no page multiple: the KV cache can hold at most
        9,984 of 10,000 tokens at page size 64, which size() accepts, so the Mamba state
        keeps all 8 requests when 9,984 fit (26,990 B less 17 slots of 1000 B = 9,990 B).
        Control: "fewer tokens than asked" would call 9,984 short and drop one request."""
        args = dict(max_running_requests=8, mamba_full_memory_ratio=9, page_size=64)
        self.assertEqual(
            self._mamba_slots(26_990, max_total_tokens=10_000, **args), [16]
        )


def _tiered_layer(
    folder,
    num_rows,
    contract=CONTRACTS[0],
    poison=(),
    auto_size=False,
    mapped=True,
    top_k=K,
):
    """A paged method whose experts live in the expert cache file in ``folder`` and
    ``num_rows`` host rows, built and loaded as FusedMoE does, its loads views of mapped
    files. A start that finds the file replays only the checked experts' loads; ``poison``
    gives the others other values in the checkpoint, which only a start that loads them
    sees."""
    torch.manual_seed(0)
    full = {name: torch.randn(E, H, H) for name in ("w13_weight", "w2_weight")}

    class _Base(_LinearMoE):
        def create_weights(self, layer, num_experts, **kwargs):
            for name in full:
                layer.register_parameter(
                    name,
                    torch.nn.Parameter(
                        torch.zeros(num_experts, H, H), requires_grad=False
                    ),
                )

    method = PagedExpertsMoEMethod(
        base_method=_Base(),
        expert_format=UnquantizedFormat(),
        num_experts=E,
        num_resident=K,
        auto_size=auto_size,
    )
    layer = torch.nn.Module()
    layer.weight_load_method = None
    layer.top_k = (
        top_k  # the router's top-k, as FusedMoE holds it: checked groups that wide
    )
    layer._weight_loader_impl = _copy_row
    entry = ExpertCache(os.path.join(folder, "layer-0.experts"), {"key": "k"})
    with (
        patch.object(formats, "HostExpertStore", _CpuStore),
        patch(_EXPERT_CACHE + "HostExpertStore", _CpuStore),
        patch(_METHOD + "TieredExpertStore", _CpuTiered),
        patch(_METHOD + "get_exec", return_value=_exec(True)),
        patch(_METHOD + "runner_config", side_effect=lambda config, num_slots: config),
        patch(_METHOD + "contract_for", return_value=contract),
        patch.object(PagedExpertsMoEMethod, "_expert_cache", return_value=entry),
        patch(_METHOD + "host_rows", return_value=num_rows),
        patch(_METHOD + "tier_cache", side_effect=lambda cache, *args: cache),
        patch(_METHOD + "_loads_mapped_files", return_value=mapped),
    ):
        method.create_weights(
            layer=layer,
            num_experts=E,
            hidden_size=H,
            intermediate_size_per_partition=H,
            params_dtype=torch.float32,
        )
        method.create_moe_runner(layer, SimpleNamespace(routed_scaling_factor=None))
        for name, t in full.items():
            param = getattr(layer, name)
            for e in range(E):
                value = torch.full_like(t[e], 7.0) if e in poison else t[e]
                param.weight_loader(param, _mapped(value), name, "w", e)
        method.process_weights_after_loading(layer)
    return method, layer, SimpleNamespace(**full)


def _every_second(others, k):
    return list(others)[1::2][:k]


def _rows_then_page_out_kept(loads, experts, out=None):
    """``ExpertLoads.rows`` that gives the kept experts' pages back after every run (the
    checked experts after the first group, before the build replays them)."""
    result = _rows(loads, experts, out)
    loads.page_out(sorted(loads.keep))
    return result


def _rows_alive_between_runs(start, hold=None):
    """Runs ``start`` with ``PostLoad.run`` tracked: whether, at any run after the first, rows
    an earlier run returned (views of the layer's whole K-slot table when it repacks) were
    still alive; ``hold`` (a list) keeps the first run's rows, as a control."""
    run, earlier, alive = PostLoad.run, [], []

    def tracked(step, experts, out=None):
        if earlier:
            gc.collect()
            alive.append(any(ref() is not None for ref in earlier))
        rows = run(step, experts, out)
        if out is None:
            if hold is not None and not hold:
                hold.append(rows)
            earlier.extend(weakref.ref(t) for t in rows.values())
        return rows

    with patch.object(PostLoad, "run", tracked):
        start()
    assert alive, "the step ran only once"
    return any(alive)


# With E experts and groups of K, a cached start checks experts 0..3, 1..4, 12..15 (the last)
# and 6, 8, 10 (sampled: every second of the others): 5, 7, 9, 11 are not loaded.
TIER_UNCHECKED = (5, 7, 9, 11)


class TestDiskTier(CustomTestCase):
    def setUp(self):
        expert_cache._said.clear()
        expert_cache._write_failed = False
        expert_cache._prune_other_keys.cache_clear()
        margin = patch.object(expert_cache, "_FREE_MARGIN_BYTES", 0)  # any test disk
        margin.start()
        self.addCleanup(margin.stop)
        sampling = patch.object(_sampler, "sample", side_effect=_every_second)
        sampling.start()
        self.addCleanup(sampling.stop)

    def _assert_consistent(self, method, layer, full):
        """Each expert in at most one host row, and every row and slot holding the expert the
        maps say."""
        on_gpu = dict(method.policy.resident)  # expert -> slot
        in_rows = {e: r for r, e in enumerate(method.store.row_expert) if e >= 0}
        held = [e for e in method.store.row_expert if e >= 0]
        self.assertEqual(len(held), len(set(held)))
        for e, slot in on_gpu.items():
            self.assertTrue(torch.equal(layer.w2_weight[slot], full.w2_weight[e]))
        for e, row in in_rows.items():
            self.assertTrue(
                torch.equal(method.store.host["w2_weight"][row], full.w2_weight[e])
            )

    def _serve(self, method, layer, full):
        for num_tokens in (1, 2, 64):
            for _ in range(4):
                d = _step(num_tokens)
                out = method.apply(layer=layer, dispatch_output=d).hidden_states
                torch.testing.assert_close(out, _reference(full, d))
                self._assert_consistent(method, layer, full)

    def _corrupt(self, folder, expert):
        layout = expert_cache.ObjectLayout.of(
            {n: ((H, H), torch.float32) for n in ("w13_weight", "w2_weight")}
        )
        with open(os.path.join(folder, "layer-0.experts"), "r+b") as f:
            f.seek(expert_cache._HEADER_BYTES + expert * layout.object_bytes)
            f.write(b"\xff" * 4)

    def test_matches_unpaged_with_experts_on_disk(self):
        """Six host rows and four slots for sixteen experts: waves copy experts from host rows
        into slots and read the rest from the layer's file, with a repack (compacting
        contract) or not (masking contract). A later start does not load the unchecked experts
        (their checkpoint values are poisoned) and serves them from the file."""
        for contract in CONTRACTS:
            with (
                self.subTest(contract=type(contract).__name__),
                tempfile.TemporaryDirectory() as folder,
            ):
                method, layer, full = _tiered_layer(folder, 6, contract)
                self.assertIsInstance(method.store, TieredExpertStore)
                self._serve(method, layer, full)
                self.assertEqual(
                    os.listdir(folder), ["layer-0.experts"]
                )  # the entry only
                method, layer, full = _tiered_layer(
                    folder, 6, contract, poison=TIER_UNCHECKED
                )
                self._serve(method, layer, full)

    def test_a_step_reads_each_expert_on_disk_once(self):
        """A step in several waves serves the experts in host rows before those only on disk,
        so no read evicts a row a later wave of the step still needs: the step reads exactly
        the experts that were only on disk when it started. Negative control: waves ordered
        without the host rows read some experts more than once."""
        for ordered in (True, False):
            extra = 0
            with (
                self.subTest(ordered=ordered),
                tempfile.TemporaryDirectory() as folder,
                (
                    contextlib.nullcontext()
                    if ordered
                    else patch.object(_CpuTiered, "in_host_rows", lambda self: set())
                ),
            ):
                method, layer, full = _tiered_layer(folder, 6)
                for num_tokens in (8, 64, 8, 64, 3, 64):
                    d = _step(num_tokens)
                    needed = set(d.topk_output.topk_ids.flatten().tolist())
                    held = set(method.policy.resident) | set(method.store.row_expert)
                    with patch.object(
                        disk_tier, "read_many", wraps=disk_tier.read_many
                    ) as reads:
                        out = method.apply(layer=layer, dispatch_output=d).hidden_states
                    torch.testing.assert_close(out, _reference(full, d))
                    read = [
                        e
                        for c in reads.call_args_list
                        for _, rs in c.args[0]
                        for e, _ in rs
                    ]
                    if ordered:
                        self.assertEqual(sorted(read), sorted(needed - held))
                    extra += len(read) - len(needed - held)
                    self._assert_consistent(method, layer, full)
            self.assertEqual(extra > 0, not ordered)

    def test_an_entry_the_loaded_chunks_do_not_reproduce_is_rebuilt(self):
        with tempfile.TemporaryDirectory() as folder:
            _tiered_layer(folder, 6)
            self._corrupt(folder, 1)
            with self.assertLogs(expert_cache.logger, "WARNING") as logs:
                method, layer, full = _tiered_layer(folder, 6)
            self.assertIn("differs from the checkpoint", logs.output[0])
            self._serve(method, layer, full)
            method, layer, full = _tiered_layer(folder, 6, poison=TIER_UNCHECKED)
            self._serve(method, layer, full)

    def test_an_unchecked_expert_comes_from_the_file(self):
        """Negative control: a corrupted unchecked expert is served as the file has it."""
        with tempfile.TemporaryDirectory() as folder:
            _tiered_layer(folder, 6)
            self._corrupt(folder, TIER_UNCHECKED[0])
            method, _, full = _tiered_layer(folder, 6, poison=TIER_UNCHECKED)
            row = method.store.row_expert.index(TIER_UNCHECKED[0])
            self.assertFalse(
                torch.equal(
                    method.store.host["w13_weight"][row],
                    full.w13_weight[TIER_UNCHECKED[0]],
                )
            )

    def test_the_checked_pages_go_after_their_last_read(self):
        """As in ``fill_store``: a start that builds the file (none, or a stale one) gives each
        expert's checkpoint pages back once, after the build replayed it, and a start that
        reads the file gives the checked experts' back once, after the check. Negative
        control: giving the checked experts' pages back as soon as the first group ran (before
        the build replays them) is caught."""
        checked = [0, 1, 2, 3, 4, 6, 8, 10, 12, 13, 14, 15]
        with tempfile.TemporaryDirectory() as folder:
            events = _page_events(lambda: _tiered_layer(folder, 6))
            self.assertEqual(_given_back_early(events, E), [])
            events = _page_events(lambda: _tiered_layer(folder, 6))  # reads the file
            self.assertEqual(sorted(e for k, e in events if k == "out"), checked)
            self.assertEqual({e for _, e in events}, set(checked))
            self.assertEqual(
                _given_back_early(events, E), [(e, []) for e in (5, 7, 9, 11)]
            )
            self._corrupt(folder, 1)
            with self.assertLogs(expert_cache.logger, "WARNING"):  # stale: rebuilt
                events = _page_events(lambda: _tiered_layer(folder, 6))
            self.assertEqual(_given_back_early(events, E), [])
        with (
            patch.object(ExpertLoads, "rows", _rows_then_page_out_kept),
            tempfile.TemporaryDirectory() as folder,
        ):
            events = _page_events(lambda: _tiered_layer(folder, 6))
        self.assertNotEqual(_given_back_early(events, E), [])

    def test_host_rows_that_hold_every_expert_read_nothing(self):
        """With R >= E, no expert is read from disk after the start (the slots hold copies of
        rows, so K + R >= E is not enough: the control, R = E - K, reads)."""
        for num_rows in (E, E - K):
            with self.subTest(num_rows=num_rows):
                requests = self._requests_after_the_start(num_rows)
                self.assertEqual(requests == [], num_rows == E)

    def _requests_after_the_start(self, num_rows):
        with tempfile.TemporaryDirectory() as folder:
            method, layer, full = _tiered_layer(folder, num_rows)
            with patch.object(
                disk_tier, "read_many", wraps=disk_tier.read_many
            ) as reads:
                for _ in range(8):
                    d = _step(64)
                    out = method.apply(layer=layer, dispatch_output=d).hidden_states
                    torch.testing.assert_close(out, _reference(full, d))
            # No assertion that reads ran: with R = E nothing reads at all; the R = E - K
            # control shows the patch sees the reads.
            requests = [
                r for c in reads.call_args_list for _, rs in c.args[0] for r in rs
            ]
            if num_rows < E:
                self._assert_consistent(method, layer, full)
            else:  # R = E builds no tier: a plain host store, no row map to check
                self.assertFalse(hasattr(method.store, "row_expert"))
            return requests

    def test_inclusive_rows_match_unpaged_and_never_write_a_row_back(self):
        """Steps over many waves serve the unpaged result, the maps stay consistent, a slot's
        expert may keep its row, and a row's bytes change only when a disk read gives it
        another expert. Negative control: a store that writes the slot's old expert back
        into the row (exclusive rows, as the maps do not say) is caught."""
        self._never_write_back()

        def write_back(store, layer, copies):
            for name, host in store.host.items():
                gpu = getattr(layer, name).data
                for row, slot in copies:
                    held = host[row].clone()
                    host[row].copy_(gpu[slot])
                    gpu[slot].copy_(held)

        with patch.object(_CpuTiered, "apply", write_back):
            with self.assertRaises(AssertionError):
                self._never_write_back()

    def _never_write_back(self):
        rewritten = 0
        dups = 0
        with tempfile.TemporaryDirectory() as folder:
            method, layer, full = _tiered_layer(folder, 6)
            store = method.store
            for num_tokens in (1, 2, 8, 64) * 6:
                rows = store.host["w2_weight"].clone()
                d = _step(num_tokens)
                with patch.object(
                    disk_tier, "read_many", wraps=disk_tier.read_many
                ) as reads:
                    out = method.apply(layer=layer, dispatch_output=d).hidden_states
                torch.testing.assert_close(out, _reference(full, d))
                self._assert_consistent(method, layer, full)
                read = {
                    e
                    for c in reads.call_args_list
                    for _, rs in c.args[0]
                    for e, _ in rs
                }
                for r, e in enumerate(store.row_expert):
                    if e not in read and not torch.equal(
                        store.host["w2_weight"][r], rows[r]
                    ):
                        rewritten += 1
                dups += len(set(method.policy.resident) & set(store.row_expert))
        self.assertEqual(rewritten, 0)
        self.assertGreater(dups, 0)

    def test_a_slot_serves_its_expert_whatever_its_row_holds(self):
        """An expert in a slot and a row is served from the slot: overwriting its row (as a
        disk read into that row would) does not change the output. Control: overwriting the
        slot does."""
        with tempfile.TemporaryDirectory() as folder:
            method, layer, full = _tiered_layer(folder, 6)
            store = method.store
            for _ in range(8):
                method.apply(layer=layer, dispatch_output=_step(64))
            both = sorted(set(method.policy.resident) & set(store.row_expert))
            self.assertTrue(both)
            e = both[0]
            other = next(x for x in method.policy.resident if x != e)
            d = _step(1)
            d.topk_output.topk_ids[:] = torch.tensor([[e, other]], dtype=torch.int32)
            expected = _reference(full, d)
            for target in ("row", "slot"):
                with self.subTest(target=target):
                    if target == "row":
                        store.host["w2_weight"][store.row_expert.index(e)] = 1e3
                    else:
                        layer.w2_weight.data[method.policy.resident[e]] = 1e3
                    out = method.apply(layer=layer, dispatch_output=d).hidden_states
                    self.assertEqual(
                        torch.allclose(out, expected, rtol=1.3e-6, atol=1e-5),
                        target == "row",
                    )

    def test_a_first_build_writes_each_file_once(self):
        """The repack (compacting contract) writes its output, the masking contract its
        replayed loads, straight into the file being built: nothing goes through the background
        writer, only the header is written by a call, and no other file is created. Control:
        an entry kept from memory goes through that writer, which the check sees."""
        for contract in CONTRACTS:
            with (
                self.subTest(contract=type(contract).__name__),
                tempfile.TemporaryDirectory() as folder,
                patch.object(ExpertCache, "write") as write,
                patch.object(expert_cache.os, "pwrite", wraps=os.pwrite) as pwrite,
            ):
                method, layer, full = _tiered_layer(folder, 6, contract)
                write.assert_not_called()
                written = sum(len(c.args[1]) for c in pwrite.call_args_list)
                self.assertEqual(written, expert_cache._HEADER_BYTES)
                self.assertEqual(os.listdir(folder), ["layer-0.experts"])
                self._serve(method, layer, full)
                ExpertCache(os.path.join(folder, "other"), {"key": "k"}).save(
                    {"w": torch.zeros(E, 4)}
                )
                expert_cache.wait_for_writes()
                write.assert_called_once()

    def test_one_slot_table_at_a_time(self):
        """When the step runs again, no rows an earlier run returned are alive: a first build
        with checked groups as wide as K (chunk 0 copied, not run again) or narrower, and a
        start that checks the file. Control: rows the test keeps are seen."""
        for contract in CONTRACTS:
            for top_k in (K, K // 2):
                with (
                    self.subTest(contract=type(contract).__name__, top_k=top_k),
                    tempfile.TemporaryDirectory() as folder,
                ):
                    start = lambda: _tiered_layer(folder, 6, contract, top_k=top_k)
                    self.assertFalse(_rows_alive_between_runs(start))  # builds the file
                    self.assertFalse(_rows_alive_between_runs(start))  # checks it
        with tempfile.TemporaryDirectory() as folder:
            start = lambda: _tiered_layer(folder, 6)
            self.assertTrue(_rows_alive_between_runs(start, hold=[]))

    def test_the_reader_keeps_no_staging_for_every_row(self):
        """Binding reads all R rows in one call; afterwards the reader's staging is at most
        K experts (the reads that follow), not R. Control: a reader that read R experts holds
        R experts of staging until it drops it."""
        with tempfile.TemporaryDirectory() as folder:
            method, layer, full = _tiered_layer(folder, 6)
            reader = method.store.reader
            limit = K * reader.layout.object_bytes
            self.assertLessEqual(len(reader._staging or b""), limit)
            reader.jobs([(e, {}) for e in range(6)])
            self.assertGreater(len(reader._staging), limit)
            reader.drop_staging()
            self.assertIsNone(reader._staging)

    def test_the_tier_needs_loads_that_can_wait(self):
        """Without mapped checkpoint files, or with a load that cannot wait (a stacked
        tensor), the tier cannot replay K experts at a time from the checkpoint: the start
        fails, naming the budget flag. Control: mapped loads start."""
        for contract in CONTRACTS:
            with (
                self.subTest(contract=type(contract).__name__),
                tempfile.TemporaryDirectory() as folder,
            ):
                with self.assertRaisesRegex(RuntimeError, "mapped safetensors"):
                    _tiered_layer(folder, 6, contract, mapped=False)
                with (
                    patch(_METHOD + "_can_defer", return_value=False),
                    self.assertRaisesRegex(RuntimeError, "cannot wait"),
                ):
                    _tiered_layer(folder, 6, contract)
                _tiered_layer(folder, 6, contract)

    def test_auto_size_binds_a_tiered_layer_once(self):
        """With auto, a tiered layer is built once K is final (``materialize``): its rows are
        read and its device state registered then, not also after loading. Control: without
        auto, the post-load step builds it, and a later materialize binds it again (what auto
        did before)."""
        for auto in (True, False):
            with (
                self.subTest(auto=auto),
                tempfile.TemporaryDirectory() as folder,
                patch.object(
                    _CpuTiered,
                    "bind",
                    autospec=True,
                    side_effect=TieredExpertStore.bind,
                ) as bind,
            ):
                method, layer, full = _tiered_layer(folder, 6, auto_size=auto)
                self.assertEqual(bind.call_count, 0 if auto else 1)
                slab = torch.empty(_slot_table_bytes(method, K), dtype=torch.uint8)
                with patch(
                    _METHOD + "runner_config",
                    side_effect=lambda config, num_slots: config,
                ):
                    method.materialize(layer, K, slab, 0)
                self.assertEqual(bind.call_count, 1 if auto else 2)
                self._serve(method, layer, full)


class TestDiskTierGuards(CustomTestCase):
    def _rows(
        self,
        budget_rows,
        num_slots=K,
        enabled=True,
        cache=True,
        exists=True,
        replayssm=False,
        recorder=None,
        unified_memory=False,
        online_compress=False,
        free=1 << 50,
    ):
        from sglang.srt.environ import envs

        exec_ctx = SimpleNamespace(
            moe=SimpleNamespace(
                expert_distribution_recorder_mode=recorder,
                paged_experts_host_memory_gb=None,
            ),
            mamba=SimpleNamespace(
                enable_linear_replayssm=replayssm, enable_linear_replayssm_spec=False
            ),
        )
        model = disk_tier.ModelContext()
        model.budget_measured = True
        model.budget_bytes = budget_rows * H * H * 4
        with (
            tempfile.TemporaryDirectory() as folder,
            patch.object(disk_tier, "_MODEL", model),
            patch.object(
                disk_tier.shutil, "disk_usage", return_value=SimpleNamespace(free=free)
            ),
            patch(
                "sglang.srt.model_executor.cuda_graph_config.cuda_graph_fully_disabled",
                return_value=False,
            ),
            patch(
                "sglang.srt.model_executor.cuda_graph_config.check_cuda_graph_backend",
                return_value=True,
            ),
            patch(
                "sglang.srt.runtime_context.process_model_config",
                return_value=SimpleNamespace(num_hidden_layers=1),
            ),
            patch("sglang.srt.runtime_context.get_exec", return_value=exec_ctx),
            patch(
                "sglang.srt.runtime_context.get_memory",
                return_value=SimpleNamespace(enable_unified_memory=unified_memory),
            ),
            envs.SGLANG_OPT_USE_ONLINE_COMPRESS.override(online_compress),
        ):
            entry = SimpleNamespace(
                path=os.path.join(folder, "layer-0.experts"), exists=lambda n: exists
            )
            layout = {"w": ((H, H), torch.float32)}
            rows = disk_tier.host_rows(layout, E, num_slots, enabled)
            if rows < E:  # once the checkpoint is loaded
                disk_tier.tier_cache(entry if cache else None, layout, E, rows)
            return rows

    def test_a_model_that_fits_needs_no_flag(self):
        self.assertEqual(self._rows(E, enabled=False), E)

    def test_a_model_that_does_not_fit_needs_the_flag(self):
        with self.assertRaisesRegex(RuntimeError, "--paged-experts-enable-disk"):
            self._rows(8, enabled=False)
        self.assertEqual(self._rows(8), 8)  # the control

    def test_the_tier_needs_the_layers_file(self):
        with self.assertRaisesRegex(RuntimeError, "expert cache"):
            self._rows(8, cache=False)

    def test_rejects_fewer_host_rows_than_slots(self):
        """A wave reads up to K experts from disk, each into a host row of its own."""
        with self.assertRaisesRegex(RuntimeError, "fewer than the 4 GPU slots"):
            self._rows(2)

    def test_rejects_replayssm_when_steps_replay(self):
        """Its ring is not restored before a replay."""
        with self.assertRaisesRegex(RuntimeError, "replayssm"):
            # The slots hold copies of rows, so 4 slots + 12 rows still read from disk.
            self._rows(12, replayssm=True)
        # 16 rows hold all 16 experts: no disk tier, nothing is replayed.
        self.assertEqual(self._rows(E, replayssm=True), E)

    def test_rejects_the_expert_distribution_recorder_when_steps_replay(self):
        """A replayed step would be counted twice."""
        with self.assertRaisesRegex(RuntimeError, "expert-distribution-recorder-mode"):
            self._rows(4, recorder="stat")
        self.assertEqual(self._rows(4), 4)

    def test_rejects_the_unified_memory_pool_when_steps_replay(self):
        """Its recurrent state is not in the request pool's mamba_pool, which a replay
        restores: a replayed step would advance it twice."""
        with self.assertRaisesRegex(RuntimeError, "--enable-unified-memory"):
            self._rows(8, unified_memory=True)
        self.assertEqual(self._rows(E, unified_memory=True), E)  # nothing replays
        self.assertEqual(self._rows(8), 8)

    def test_rejects_online_compression_when_steps_replay(self):
        """DeepSeek-V4's online c128 compressor folds each token into running sums in
        place: a replayed step would fold it twice."""
        with self.assertRaisesRegex(RuntimeError, "SGLANG_OPT_USE_ONLINE_COMPRESS"):
            self._rows(8, online_compress=True)
        self.assertEqual(self._rows(E, online_compress=True), E)
        self.assertEqual(self._rows(8), 8)

    def test_too_little_disk_space_for_a_first_build_is_an_error(self):
        """A build writes the files (once, at their size on disk) and keeps the cache's free
        margin; a start whose files exist needs no space (the control)."""
        need = (
            expert_cache.ObjectLayout.of({"w": ((H, H), torch.float32)}).file_bytes(E)
            + expert_cache._FREE_MARGIN_BYTES
        )
        with self.assertRaisesRegex(RuntimeError, "needs about .* keeps 8 GiB free"):
            self._rows(8, exists=False, free=need - 1)
        self.assertEqual(self._rows(8, exists=False, free=need), 8)
        self.assertEqual(self._rows(8, exists=True, free=0), 8)

    def test_the_tier_runs_on_x86_hosts_only(self):
        """The host answers a waiting step with plain stores, which Arm may reorder: a model
        that needs the tier fails there; one that fits does not (the control), nor the tier
        on x86."""
        with patch.object(disk_tier.platform, "machine", return_value="aarch64"):
            with self.assertRaisesRegex(RuntimeError, "x86-64 hosts .* aarch64"):
                self._rows(8)
            self.assertEqual(self._rows(E), E)
        with patch.object(disk_tier.platform, "machine", return_value="x86_64"):
            self.assertEqual(self._rows(8), 8)

    def test_a_larger_later_layer_does_not_pin_past_what_fits(self):
        """Whether every expert fits is judged from the first layer: a later layer of another
        format (bf16 experts a quantization leaves out) that would pin past what was measured
        fails with the flag to set. Control: layers of one size that fit pin every expert."""
        small, large = (
            {"w": ((H, H), torch.float32)},
            {"w": ((4 * H, H), torch.float32)},
        )
        layer_bytes = E * H * H * 4
        for layouts, fails in (((small, small), False), ((small, large), True)):
            model = disk_tier.ModelContext()
            model.budget_measured, model.fits_bytes = True, 2 * layer_bytes
            with (
                self.subTest(fails=fails),
                patch.object(disk_tier, "_MODEL", model),
                patch(
                    "sglang.srt.runtime_context.process_model_config",
                    return_value=SimpleNamespace(num_hidden_layers=2),
                ),
            ):
                self.assertEqual(disk_tier.host_rows(layouts[0], E, K, True, 0), E)
                if fails:
                    with self.assertRaisesRegex(
                        RuntimeError, "--paged-experts-host-memory-gb"
                    ):
                        disk_tier.host_rows(layouts[1], E, K, True, 1)
                else:
                    self.assertEqual(disk_tier.host_rows(layouts[1], E, K, True, 1), E)

    def test_a_full_disk_during_a_build_names_the_directory(self):
        """A layer whose space runs out while its file is reserved fails with where and how
        much, and leaves no staging file behind."""
        with tempfile.TemporaryDirectory() as folder:
            entry = ExpertCache(os.path.join(folder, "layer-0.experts"), {"key": "k"})
            full = OSError(errno.ENOSPC, "No space left on device")
            with (
                patch.object(expert_cache, "_FREE_MARGIN_BYTES", 0),
                patch.object(expert_cache.os, "posix_fallocate", side_effect=full),
                self.assertRaisesRegex(RuntimeError, f"cannot build .* under {folder}"),
            ):
                entry.create({"w": ((H, H), torch.float32)}, E)
            self.assertEqual(os.listdir(folder), [])

    def test_a_build_keeps_the_free_margin(self):
        """Each file a build creates leaves the cache's free margin, before anything is
        written. Control: with the margin free it is created."""
        layout = {"w": ((H, H), torch.float32)}
        nbytes = expert_cache.ObjectLayout.of(layout).file_bytes(E)
        margin = expert_cache._FREE_MARGIN_BYTES
        for free, fails in ((nbytes + margin - 1, True), (nbytes + margin, False)):
            with (
                self.subTest(fails=fails),
                tempfile.TemporaryDirectory() as folder,
                patch.object(
                    expert_cache.shutil,
                    "disk_usage",
                    return_value=SimpleNamespace(free=free),
                ),
            ):
                entry = ExpertCache(
                    os.path.join(folder, "layer-0.experts"), {"key": "k"}
                )
                if fails:
                    with self.assertRaisesRegex(RuntimeError, "keeping 8 GiB free"):
                        entry.create(layout, E)
                    self.assertEqual(os.listdir(folder), [])
                else:
                    entry.create(layout, E)
                    entry.abandon()

    def test_a_failed_build_leaves_no_staging_file(self):
        """A build that fails after its file was created drops it (``abandon``). Control:
        until then the staging file is there."""
        with (
            tempfile.TemporaryDirectory() as folder,
            patch.object(expert_cache, "_FREE_MARGIN_BYTES", 0),
        ):
            entry = ExpertCache(os.path.join(folder, "layer-0.experts"), {"key": "k"})
            entry.create({"w": ((H, H), torch.float32)}, E)
            self.assertEqual(len(os.listdir(folder)), 1)
            entry.abandon()
            self.assertEqual(os.listdir(folder), [])

    def test_a_running_build_keeps_its_key_directory(self):
        """A start whose key differs prunes the model's other keys, but not one a running
        start builds in (its staging file's writer is alive). Control: a staging file whose
        writer is gone does not keep it."""
        expert_cache._prune_other_keys.cache_clear()
        for alive, kept in ((True, True), (False, False)):
            with (
                self.subTest(alive=alive),
                tempfile.TemporaryDirectory() as parent,
                patch.object(expert_cache, "_alive", return_value=alive),
            ):
                mine = os.path.join(parent, "model-" + "a" * 24)
                other = os.path.join(parent, "model-" + "b" * 24)
                os.makedirs(mine)
                os.makedirs(other)
                staging = expert_cache._staging_prefix(12345) + "x-layer-0.experts"
                open(os.path.join(other, staging), "wb").close()
                expert_cache._prune_other_keys(mine)
                expert_cache._prune_other_keys.cache_clear()
                self.assertEqual(os.path.isdir(other), kept)

    def test_layers_routed_by_token_keep_every_expert_in_host_memory(self):
        """DeepSeek-V4's leading hash-routed layers get all their experts when that takes at
        most half the budget, and the others split the rest evenly; without such layers, or
        when they would take more (the controls), every layer gets an even share."""
        num_experts, layers = 16, 10
        for hash_layers, budget, expected in (
            (2, 2 * 16 + 8 * 5, {0: 16, 1: 16, 2: 5, 9: 5, None: 5}),  # 32 of 72
            (2, 40, {0: 4, 1: 4, 2: 4, None: 4}),  # 32 of 40: more than half
            (0, 72, {0: 7, 5: 7, None: 7}),
        ):
            with self.subTest(hash_layers=hash_layers, budget=budget):
                config = SimpleNamespace(
                    num_hidden_layers=layers,
                    hf_text_config=SimpleNamespace(num_hash_layers=hash_layers),
                )
                with (
                    patch.object(disk_tier, "_MODEL", disk_tier.ModelContext()),
                    patch(
                        "sglang.srt.runtime_context.process_model_config",
                        return_value=config,
                    ),
                    patch.object(
                        disk_tier,
                        "host_memory_budget_bytes",
                        side_effect=lambda need, budget=budget: budget,
                    ),
                    patch.object(disk_tier, "_check_replayable"),
                ):
                    got = {
                        layer: disk_tier.host_rows(
                            {"w": ((1,), torch.uint8)},
                            num_experts,
                            1,
                            True,
                            layer_id=layer,
                        )
                        for layer in expected
                    }
                self.assertEqual(got, expected)

    def test_default_budget_stays_within_the_container_limit(self):
        """Unset, the host budget (with --paged-experts-enable-disk) follows the container's
        memory limit, not the host's RAM, and asks for the flag when the limit cannot be
        read."""
        from sglang.srt.mem_cache import host_memory
        from sglang.srt.mem_cache.pool_host import base

        exec_ctx = SimpleNamespace(
            moe=SimpleNamespace(paged_experts_host_memory_gb=None)
        )
        memory = SimpleNamespace(available=512 << 30, total=1024 << 30)
        with (
            patch("sglang.srt.runtime_context.get_exec", return_value=exec_ctx),
            patch.object(host_memory.psutil, "virtual_memory", return_value=memory),
        ):
            with patch.object(
                host_memory, "_cgroup_memory_headroom", return_value=30 << 30
            ):
                self.assertEqual(
                    disk_tier.host_memory_budget_bytes(40 << 30),
                    (30 << 30) - base.HICACHE_HOST_MEMORY_RESERVE_BYTES,
                )
            for error in (OSError, ValueError("Cannot resolve cgroup memory path")):
                with (
                    self.subTest(error=error),
                    patch.object(
                        host_memory, "_cgroup_memory_headroom", side_effect=error
                    ),
                    self.assertRaisesRegex(
                        RuntimeError, "--paged-experts-host-memory-gb"
                    ),
                ):
                    disk_tier.host_memory_budget_bytes(40 << 30)

    def test_unset_budget_pins_every_expert_that_fits(self):
        """Unset, every expert is pinned while they fit what is available (as without the
        disk tier), and only experts that would not fit go to disk, with the reserve kept;
        set, the flag is the budget even when more is available (the control)."""
        from sglang.srt.mem_cache import host_memory
        from sglang.srt.mem_cache.pool_host import base

        reserve = base.HICACHE_HOST_MEMORY_RESERVE_BYTES
        memory = SimpleNamespace(available=512 << 30, total=1024 << 30)
        for gb, need, expected in (
            (None, 30 << 30, None),  # fits the 30 GiB available: no budget
            (None, (30 << 30) + 1, (30 << 30) - reserve),
            (4, 1 << 30, 4 << 30),  # the flag, though everything fits
        ):
            exec_ctx = SimpleNamespace(
                moe=SimpleNamespace(paged_experts_host_memory_gb=gb)
            )
            with (
                self.subTest(gb=gb, need=need),
                patch("sglang.srt.runtime_context.get_exec", return_value=exec_ctx),
                patch.object(host_memory.psutil, "virtual_memory", return_value=memory),
                patch.object(
                    host_memory, "_cgroup_memory_headroom", return_value=30 << 30
                ),
            ):
                self.assertEqual(disk_tier.host_memory_budget_bytes(need), expected)

    def test_without_the_flags_no_budget_is_measured(self):
        """Without --paged-experts-enable-disk and --paged-experts-host-memory-gb a layer
        pins every expert, as before the disk tier: no budget is measured, so nothing new
        is refused. Controls: either flag measures it."""
        from sglang.srt.layers.moe.paged_experts import method as method_module

        for enable_disk, gb, measured in (
            (False, None, False),
            (True, None, True),
            (False, 64.0, True),
        ):
            exec_ctx = SimpleNamespace(
                moe=SimpleNamespace(
                    paged_experts_enable_disk=enable_disk,
                    paged_experts_host_memory_gb=gb,
                )
            )
            budget = Mock(return_value=None)  # measured: everything fits
            with (
                self.subTest(enable_disk=enable_disk, gb=gb),
                patch.object(method_module, "get_exec", return_value=exec_ctx),
                patch.object(disk_tier, "_MODEL", disk_tier.ModelContext()),
                patch.object(disk_tier, "host_memory_budget_bytes", budget),
                patch(
                    "sglang.srt.runtime_context.process_model_config",
                    return_value=SimpleNamespace(num_hidden_layers=1),
                ),
            ):
                rows = method_module._host_rows({"w": ((H, H), torch.float32)}, E, K, 0)
            self.assertEqual(rows, E)
            self.assertEqual(budget.called, measured)


class TestHostMemoryFlag(CustomTestCase):
    def test_host_memory_needs_paged_experts_and_a_positive_size(self):
        """--paged-experts-host-memory-gb alone, or at 0, is refused when the MoE arguments
        resolve. Control: with --enable-paged-experts and a size the check passes (to the
        next step, stubbed)."""
        from sglang.srt.arg_groups import moe_hook
        from sglang.srt.server_args import ServerArgs

        class _Next(Exception):
            pass

        with (
            patch("sglang.srt.arg_groups.serving_hook.get_device", return_value="cuda"),
            patch.object(moe_hook, "run_post_process_pass", side_effect=_Next),
        ):
            for kwargs, error in (
                ({}, "paged-experts-host-memory-gb requires --enable-paged-experts"),
                (
                    {"enable_paged_experts": True, "paged_experts_host_memory_gb": 0},
                    "greater than 0",
                ),
                (
                    {
                        "enable_paged_experts": True,
                        "paged_experts_host_memory_gb": float("nan"),
                    },
                    "finite number",
                ),
                (
                    {
                        "enable_paged_experts": True,
                        "paged_experts_host_memory_gb": float("inf"),
                    },
                    "finite number",
                ),
            ):
                kwargs = {"paged_experts_host_memory_gb": 8, **kwargs}
                with (
                    self.subTest(kwargs=kwargs),
                    self.assertRaisesRegex(ValueError, error),
                ):
                    moe_hook.handle_moe_kernel_config(
                        ServerArgs(model_path="dummy", **kwargs)
                    )
            args = ServerArgs(
                model_path="dummy",
                enable_paged_experts=True,
                paged_experts_host_memory_gb=8,
            )
            with self.assertRaises(_Next):
                moe_hook.handle_moe_kernel_config(args)


class TestExpertReader(CustomTestCase):
    """The disk tier reads single experts of a layer's file into host rows."""

    def setUp(self):
        expert_cache._said.clear()
        expert_cache._write_failed = False
        expert_cache._prune_other_keys.cache_clear()

    def _entry(self, folder):
        host = {
            "big": torch.randn(E, 1024),  # a page per row: read in place
            "small": torch.randn(E, 3, 5).half(),
            "empty": torch.empty(E, 0, dtype=torch.int32),
        }
        entry = ExpertCache(os.path.join(folder, "layer-0.experts"), {"key": "k"})
        entry.write(host)
        return entry, host

    @staticmethod
    def _page_aligned(shape, dtype):
        import math

        nbytes = max(1, math.prod(shape) * dtype.itemsize)
        return (
            torch.frombuffer(mmap.mmap(-1, nbytes), dtype=torch.uint8)[
                : math.prod(shape) * dtype.itemsize
            ]
            .view(dtype)
            .view(shape)
        )

    def _read_back(self, entry, host):
        reader = ExpertReader(entry, host)
        dests = {
            n: self._page_aligned((3, *t.shape[1:]), t.dtype) for n, t in host.items()
        }
        experts = (7, 0, 15)
        read_many(
            [
                (
                    reader,
                    [
                        (e, {n: d[i].data_ptr() for n, d in dests.items() if d.numel()})
                        for i, e in enumerate(experts)
                    ],
                )
            ]
        )
        for name, dest in dests.items():
            self.assertTrue(_same(dest, host[name][list(experts)]))

    def test_rows_come_back_as_written(self):
        """Page-sized rows land in place, small and empty ones through staging."""
        with tempfile.TemporaryDirectory() as folder:
            self._read_back(*self._entry(folder))

    def test_reads_fall_back_when_the_filesystem_refuses_o_direct_reads(self):
        """Some filesystems accept O_DIRECT at open but refuse its reads: the file is then
        read through the page cache, decided once when it is opened."""
        real = os.preadv

        def refuse_direct(fd, buffers, offset):
            if fcntl.fcntl(fd, fcntl.F_GETFL) & expert_cache._O_DIRECT:
                raise OSError(errno.EINVAL, "no O_DIRECT")
            return real(fd, buffers, offset)

        with tempfile.TemporaryDirectory() as folder:
            entry, host = self._entry(folder)
            with patch.object(expert_cache.os, "preadv", side_effect=refuse_direct):
                self._read_back(entry, host)

    def test_a_missing_file_is_an_error(self):
        with tempfile.TemporaryDirectory() as folder:
            entry = ExpertCache(os.path.join(folder, "layer-0.experts"), {"key": "k"})
            with self.assertRaises(FileNotFoundError):
                ExpertReader(entry, {"w": torch.zeros(E, 4)})

    def test_a_failed_read_raises_once_every_read_is_done(self):
        """Of reads in flight, a failed one is raised only after the others ended: none still
        writes into a row once the error reaches the caller. Control: the slow read was still
        running when the other had failed."""
        import concurrent.futures
        import time

        done = threading.Event()

        def fail():
            raise OSError("read failed")

        def slow():
            time.sleep(0.3)
            done.set()

        with concurrent.futures.ThreadPoolExecutor(2) as pool:
            futures = [pool.submit(fail), pool.submit(slow)]
            futures[0].exception()
            self.assertFalse(done.is_set())
            with self.assertRaisesRegex(OSError, "read failed"):
                expert_cache.wait_reads((futures, []))
            self.assertTrue(done.is_set())

    def test_a_mapping_is_private_and_checks_the_header(self):
        """Writes to a mapped entry stay in memory; another key or layout maps nothing."""
        with tempfile.TemporaryDirectory() as folder:
            entry, host = self._entry(folder)
            rows = {n: (tuple(t.shape[1:]), t.dtype) for n, t in host.items()}
            views = entry.map(rows, E)
            self.assertTrue(_same(views["big"], host["big"]))
            views["big"][3].fill_(0)
            again = entry.map(rows, E)
            self.assertTrue(_same(again["big"], host["big"]))
            other = ExpertCache(entry.path, {"key": "other"})
            self.assertIsNone(other.map(rows, E))


class TestResolvePoll(CustomTestCase):
    """The resolve's poll while a replay runs: it sleeps with a fine timer slack, and handles
    each layer's row once, as it lands."""

    @unittest.skipUnless(os.path.exists("/proc/self/status"), "Linux timer slack")
    def test_the_resolving_thread_gets_a_fine_timer_slack(self):
        """After ``_fine_timer_slack`` the thread's slack is 1 ns. Control: a thread that did
        not call it keeps the default (50 us; set first in both, since a thread inherits its
        creator's slack and an earlier resolve on this thread set it)."""
        import ctypes
        import threading

        libc = ctypes.CDLL(None)
        slack = {}

        def probe(name, fine):
            libc.prctl(29, ctypes.c_ulong(50_000), 0, 0, 0)  # PR_SET_TIMERSLACK
            if fine:
                disk_tier._fine_timer_slack()
            slack[name] = libc.prctl(30, 0, 0, 0, 0)  # PR_GET_TIMERSLACK

        for name, fine in (("fine", True), ("control", False)):
            thread = threading.Thread(target=probe, args=(name, fine))
            thread.start()
            thread.join()
        self.assertEqual(slack["fine"], 1)
        self.assertEqual(slack["control"], 50_000)

    def test_newly_landed_layers_come_once_in_layer_order(self):
        import numpy as np

        seen = np.zeros(4, dtype=bool)
        flags = np.zeros(5, dtype=np.int32)  # one more row than layers: ignored
        self.assertEqual(disk_tier._newly_landed(flags, seen), [])
        flags[[2, 4]] = 7
        self.assertEqual(disk_tier._newly_landed(flags, seen), [2])
        flags[[0, 3]] = 7
        self.assertEqual(disk_tier._newly_landed(flags, seen), [0, 3])
        self.assertEqual(disk_tier._newly_landed(flags, seen), [])

    @unittest.skipUnless(torch.cuda.is_available(), "needs a GPU")
    def test_rows_landing_out_of_order_are_each_planned_once(self):
        """While the step runs (a GPU sleep), layers 2, 1, 0 land one by one (layer 1 without a
        miss): the resolve plans 2 then 0, each once, and settles when the step ends. Negative
        control: a scan that forgets what it saw plans a layer more than once."""
        import threading
        import time

        import numpy as np

        from sglang.srt.model_executor.runner_backend_utils import replay_hooks

        def resolve(newly_landed):
            planned = []

            class _Store:
                num_slots, num_experts, num_rows = K, E, 4
                reader = SimpleNamespace(jobs=lambda requests: ([], []))
                layer = None

                def __init__(self, layer):
                    self.layer_id = layer

                def read_maps(self, row):
                    pass

                def plan(self, loads):
                    planned.append(self.layer_id)
                    return [], []

                def maps(self):
                    return [0] * (E + 2 * self.num_rows + 1)

                def apply(self, layer, copies):
                    pass

                def write_maps(self):
                    pass

            with (
                patch.object(replay_hooks, "_HOOKS", []),
                patch(
                    "sglang.srt.runtime_context.process_model_config",
                    return_value=SimpleNamespace(num_hidden_layers=3),
                ),
                patch.object(disk_tier, "_newly_landed", newly_landed),
            ):
                state = disk_tier._DeviceState()
                stores = [_Store(i) for i in range(3)]
                for store in stores:
                    state.register(store, "cuda")
                state.streaming = True
                mailbox = state.groups[(K, E, 4)][2]
                for i in (0, 2):  # a miss in layers 0 and 2
                    mailbox.rows_np[i, :K] = 0
                    mailbox.rows_np[i, 0] = 1
                    mailbox.rows_np[i, K : 3 * K : K] = (9, 1)

                def land():
                    for i in (2, 1, 0):
                        time.sleep(0.03)
                        mailbox.flags_np[i] = 5

                torch.cuda._sleep(int(5e8))  # the step (~0.2-0.3 s)
                thread = threading.Thread(target=land)
                thread.start()
                state._resolve_streaming()
                thread.join()
                torch.cuda.synchronize()
            return planned

        self.assertEqual(resolve(disk_tier._newly_landed), [2, 0])
        forgetful = lambda flags, seen: np.flatnonzero(flags[: len(seen)] != 0).tolist()
        self.assertNotEqual(resolve(forgetful), [2, 0])


class TestReplay(CustomTestCase):
    def test_device_state_serves_layers_of_different_shapes(self):
        """Each layer gets a row of its own shape, and one resolve pages in what every layer
        left on disk, with the row maps the step left read in the same copy."""
        from sglang.srt.model_executor.runner_backend_utils import replay_hooks

        planned = []

        class _Store:
            num_experts, num_rows = E, 4

            reader = SimpleNamespace(jobs=lambda requests: ([], []))
            layer = None

            def __init__(self, num_slots):
                self.num_slots = num_slots

            def read_maps(self, row):
                self.clock = int(row[-1])

            def plan(self, loads):
                planned.append((loads, self.clock))
                return [], []

            def apply(self, layer, copies):
                pass

            def write_maps(self):
                pass

        with (
            patch.object(replay_hooks, "_HOOKS", []),
            patch(
                "sglang.srt.runtime_context.process_model_config",
                return_value=SimpleNamespace(num_hidden_layers=4),
            ),
        ):
            state = disk_tier._DeviceState()
            stores = [_Store(k) for k in (4, 6)]
            rows = [state.register(store, "cpu") for store in stores]
            self.assertEqual([r.numel() for r in rows], [12 + E + 9, 18 + E + 9])
            for k, r, (e, slot, clock) in zip((4, 6), rows, ((9, 1, 5), (11, 5, 7))):
                r[: 3 * k].view(3, k)[:, 0] = torch.tensor(
                    [1, e, slot], dtype=torch.int32
                )
                r[-1] = clock
            self.assertTrue(replay_hooks.needs_replay())
            self.assertEqual(planned, [([(9, 1)], 5), ([(11, 5)], 7)])
            self.assertFalse(replay_hooks.needs_replay())  # flags cleared

    @unittest.skipUnless(torch.cuda.is_available(), "needs a GPU")
    def test_a_replay_publishes_each_layer_and_reads_start_from_what_landed(self):
        """A captured step's plan kernel writes each layer's row to pinned host memory; the
        resolve after a replay plans the reads from those copies, then clears the flags and
        the device's missing flags, so the next replay settles. A read that fails reaches the
        caller and leaves no flag set."""
        from sglang.kernels.ops.moe.paged_experts import paged_experts_tiered_plan
        from sglang.srt.model_executor.runner_backend_utils import replay_hooks

        planned = []

        class _Store:
            num_slots, num_experts, num_rows = K, E, 4

            reader = SimpleNamespace(jobs=lambda requests: ([], []))
            layer = None

            def read_maps(self, row):
                self.clock = int(row[-1])

            def plan(self, loads):
                planned.append((loads, self.clock))
                return [], []

            def maps(self):
                return [0] * (E + 2 * self.num_rows + 1)

            def apply(self, layer, copies):
                pass

            def write_maps(self):
                pass

        with (
            patch.object(replay_hooks, "_HOOKS", []),
            patch(
                "sglang.srt.runtime_context.process_model_config",
                return_value=SimpleNamespace(num_hidden_layers=4),
            ),
        ):
            i32 = dict(dtype=torch.int32, device="cuda")

            def bind(state, store):
                store.state = state.register(store, "cuda")
                (
                    doorbell,
                    store.d_expert_row,
                    store.d_row_expert,
                    store.d_row_stamp,
                    store.d_clock,
                ) = store.state.split([3 * K, E, 4, 4, 1])
                store.doorbell = doorbell.view(3, K)
                store.d_expert_row.fill_(-1)  # every expert on disk
                store.src, store.dst = torch.zeros(K, **i32), torch.zeros(K, **i32)
                store.count = torch.zeros(1, **i32)
                store.rows_out = torch.zeros(K, **i32)

            def step(state, store):
                paged_experts_tiered_plan(
                    store.src,
                    store.dst,
                    store.count,
                    store.d_expert_row,
                    store.d_row_expert,
                    store.d_row_stamp,
                    store.d_clock,
                    store.doorbell,
                    store.rows_out,
                    *(state.publish(store) or ()),
                )

            state = disk_tier._DeviceState()
            stores = [_Store(), _Store()]
            for store in stores:
                bind(state, store)
            stores[0].src[0], stores[0].dst[0] = 9, 1
            stores[0].count.fill_(1)
            stores[0].d_clock.fill_(4)
            stores[1].d_clock.fill_(2)
            graph = torch.cuda.CUDAGraph()
            with _captured(graph):
                for store in stores:
                    step(state, store)
            self.assertTrue(state.streaming)
            graph.replay()
            self.assertTrue(replay_hooks.needs_replay())
            self.assertEqual(planned, [([(9, 1)], 5)])  # the clock the step advanced to
            flags = state.groups[(K, E, 4)][2].flags
            self.assertFalse(bool(flags.any()))
            stores[0].count.zero_()  # the replayed step finds expert 9 resident
            graph.replay()
            self.assertFalse(replay_hooks.needs_replay())  # missing flags cleared
            self.assertEqual(len(planned), 1)

            def fail(offset, segments):
                raise OSError("read failed")

            stores[0].reader = SimpleNamespace(
                jobs=lambda requests: ([(SimpleNamespace(preadv=fail), 0, [])], [])
            )
            stores[0].count.fill_(1)
            graph.replay()
            with self.assertRaisesRegex(OSError, "read failed"):
                replay_hooks.needs_replay()
            self.assertFalse(bool(flags.any()))
            self.assertFalse(bool(stores[0].state[:K].any()))

    @unittest.skipUnless(torch.cuda.is_available(), "needs a GPU")
    def test_a_captured_step_waits_for_its_disk_reads(self):
        """A step that leaves an expert on disk waits at the layer: the host reads it into a
        row and answers, the step copies it into its slot and runs on, and nothing is replayed.
        Unanswered (negative control), the step times out and runs on, the host copies the
        expert in after it, the step replays and the timeout is counted and warned about. The
        read keeps its row and the expert its slot held is dropped. A read that fails is
        answered at once (the step does not wait out its timeout) and raises, and the tier
        refuses later steps; control: unanswered, the failed step waits out the timeout."""
        import ctypes
        import time

        from sglang.srt.model_executor.runner_backend_utils import replay_hooks

        k, e, r, width = 2, 8, 3, 4

        class _Reader:
            """Expert x's row on disk is ``width`` float32 values x."""

            def jobs(self, requests):
                return [
                    (self, x, [(a, width * 4)]) for x, d in requests for a in d.values()
                ], []

            def preadv(self, x, segments):
                data = torch.full((width,), float(x))  # alive until the copy is done
                for address, n in segments:
                    ctypes.memmove(address, data.data_ptr(), n)

        class _Failing(_Reader):
            def preadv(self, x, segments):
                raise OSError("read failed")

        i32 = dict(dtype=torch.int32, device="cuda")
        for mode in ("answered", "unanswered", "failed", "failed-unanswered"):
            answered = mode in ("answered", "failed")
            failed = mode.startswith("failed")
            timeout_ms = 1500 if failed else 200
            with (
                self.subTest(mode=mode),
                patch.object(replay_hooks, "_HOOKS", []),
                patch(
                    "sglang.srt.runtime_context.process_model_config",
                    return_value=SimpleNamespace(num_hidden_layers=2),
                ),
                patch.object(disk_tier, "WAIT_TIMEOUT_MS", timeout_ms),
                patch.object(disk_tier.logger, "warning") as warn,
            ):
                state = disk_tier._DeviceState()
                store = object.__new__(TieredExpertStore)
                store.num_slots, store._num_experts, store.num_rows = k, e, r
                store._reader = _Failing() if failed else _Reader()
                store.await_plan, store.clock = None, 0
                store.rows_out = torch.full((k,), -1, **i32)
                store.pin = True
                store.host = {"w": torch.zeros(r, width, pin_memory=True)}
                store.layer = SimpleNamespace(w=torch.zeros(k, width, device="cuda"))
                for slot in range(k):  # slots hold experts 0, 1; rows experts 2, 3, 4
                    store.layer.w[slot] = slot
                for row in range(r):
                    store.host["w"][row] = 2 + row
                store.row_expert, store.row_stamp = [2, 3, 4], [0, 0, 0]
                store.bind_device_gather(store.layer)
                store.device = store.layer.w.device
                store._device_state = state
                store.state = state.register(store, "cuda")
                (
                    doorbell,
                    store.d_expert_row,
                    store.d_row_expert,
                    store.d_row_stamp,
                    store.d_clock,
                ) = store.state.split([3 * k, e, r, r, 1])
                store.doorbell = doorbell.view(3, k)
                store.clock = 5
                store.write_maps()
                # The step routed to expert 7, only on disk: slot 1 (expert 1) was planned.
                src, dst = torch.tensor([7, 0], **i32), torch.tensor([1, 0], **i32)
                count = torch.ones(1, **i32)
                graph = torch.cuda.CUDAGraph()
                with _captured(graph):
                    store.gather(src, dst, count)
                self.assertIsNotNone(store.await_plan)  # the step waits at the layer
                reply = disk_tier._Mailbox.reply
                with patch.object(
                    disk_tier._Mailbox,
                    "reply",
                    reply if answered else lambda *args: None,
                ):
                    start = time.monotonic()
                    graph.replay()
                    if failed:
                        with self.assertRaisesRegex(OSError, "read failed"):
                            replay_hooks.needs_replay()
                    else:
                        replays = replay_hooks.needs_replay()
                    elapsed = time.monotonic() - start
                torch.cuda.synchronize()
                if failed:
                    if answered:
                        self.assertLess(elapsed, timeout_ms / 2000)
                        with self.assertRaisesRegex(RuntimeError, "earlier step"):
                            state.resolve()
                    else:
                        self.assertGreaterEqual(elapsed, timeout_ms / 1000)
                    continue
                self.assertEqual(replays, not answered)
                self.assertEqual(state.timeouts, 0 if answered else 1)
                self.assertEqual(warn.call_count, 0 if answered else 1)
                self.assertTrue((store.layer.w[1] == 7).all())  # in its slot either way
                # The read expert keeps its row and the slot's old one is dropped.
                held, dropped = 7, 1
                row = store.row_expert.index(held)
                self.assertTrue((store.host["w"][row] == held).all())
                self.assertEqual(store.state[: 3 * k].view(3, k)[0].tolist(), [0, 0])
                device_maps = store.state[3 * k :].tolist()
                self.assertEqual(device_maps, store.maps())
                self.assertEqual(device_maps[dropped], -1)  # not in a row
                self.assertEqual(device_maps[held], row)

    def test_reads_never_target_the_rows_the_steps_copies_read(self):
        """A device-planned step publishes before it copies its host-row hits, so those copies
        may still run while the host reads its misses: the host reads into another row, even
        an empty row stamped with the published clock that the plain order would take first
        (the control: a host-planned store, which waits for the stream before it reads)."""
        for device_planned in (True, False):
            with self.subTest(device_planned=device_planned):
                store = object.__new__(TieredExpertStore)
                store._num_experts, store.num_rows = 8, 2
                store.host = {"w": torch.zeros(2, 4)}
                store.rows_out = torch.zeros(2) if device_planned else None
                # After the step's update (clock 6): row 0 is empty and stamped 6; row 1
                # holds expert 3.
                store.row_expert, store.row_stamp, store.clock = [-1, 3], [6, 0], 6
                copies, reads = store.plan([(7, 1)])
                self.assertEqual(copies[0][0], 1 if device_planned else 0)
                self.assertEqual(reads[0][0], 7)

    def test_a_replay_starts_from_the_saved_recurrent_state(self):
        """Each replay advances the recurrent state of the step's requests from where the step
        started, not where the last replay left it; other slots keep what the step wrote."""
        from sglang.srt.model_executor.runner_backend_utils import replay_hooks

        conv = torch.zeros(2, 6, 3)  # [layers, slots, ...]
        temporal = torch.zeros(2, 6, 4, 2)
        pool = SimpleNamespace(
            mamba_pool=SimpleNamespace(
                mamba_cache=SimpleNamespace(
                    conv=[conv], temporal=temporal, replayssm_d=None
                ),
                _slot_siblings=(),
            ),
            get_mamba_indices=lambda rows: rows + 1,  # request row r -> slot r + 1
            translate_mamba_indices=lambda slots: slots,
        )
        rows = torch.tensor([2, 0, 0])  # padded rows map to slot 1

        def step():
            for state in (conv, temporal):
                state[:, rows + 1] += 1
                state[:, 5] += 1  # a slot outside the batch: not restored

        asks = iter([True, True, False])
        with patch.object(replay_hooks, "_HOOKS", [lambda: next(asks)]):
            restore = replay_hooks.save_recurrent_state(pool, rows)
            step()
            replay_hooks.replay_until_settled(step, restore)
        for state in (conv, temporal):
            self.assertTrue((state[:, [1, 3]] == 1).all())  # three runs, advanced once
            self.assertTrue((state[:, [0, 2, 4]] == 0).all())
            self.assertTrue((state[:, 5] == 3).all())
        with patch.object(replay_hooks, "_HOOKS", []):
            self.assertIsNone(replay_hooks.save_recurrent_state(pool, rows))

    def test_replays_stop_once_the_step_settles(self):
        """A step that settles on the last allowed replay is not an error."""
        from sglang.srt.model_executor.runner_backend_utils import replay_hooks

        asks = iter([True, True, False])
        replays = []
        with (
            patch.object(replay_hooks, "_HOOKS", [lambda: next(asks)]),
            patch.object(replay_hooks, "_MAX_REPLAYS", 2),
        ):
            replay_hooks.replay_until_settled(lambda: replays.append(1))
        self.assertEqual(len(replays), 2)


def _plan_store(num_rows=6, num_experts=16, device_planned=False):
    """A tiered store's host side, rows 0.. holding experts 4.., as ``plan`` sees it."""
    store = object.__new__(TieredExpertStore)
    store._num_experts, store.num_rows = num_experts, num_rows
    store.host = {"w": torch.zeros(num_rows, 4)}
    store.rows_out = torch.zeros(2) if device_planned else None
    store.row_expert = [4 + r for r in range(num_rows)]
    store.row_stamp = [r for r in range(num_rows)]
    store.clock = num_rows
    return store


def _reference_plan(store, loads, drop=None):
    """``TieredExpertStore.plan`` with a Python sort of the free rows, for the equivalence
    test; ``drop`` removes one sort criterion (a control)."""
    where = {e: r for r, e in enumerate(store.row_expert) if e >= 0}
    pending = store.clock if store.rows_out is not None else None
    store.clock += 1
    copies, reads, used = [], [], set()
    disk = []
    for expert, slot in loads:
        row = where.get(expert)
        if row is None:
            disk.append((expert, slot))
            continue
        copies.append((row, slot))
        used.add(row)
        store.row_stamp[row] = store.clock
    if disk:

        def key(r):
            k = (
                store.row_stamp[r] == pending,
                store.row_expert[r] >= 0,
                store.row_stamp[r],
            )
            return k if drop is None else k[:drop] + k[drop + 1 :]

        free = sorted((r for r in range(store.num_rows) if r not in used), key=key)
        assert len(free) >= len(disk), (len(free), len(disk))
        for (expert, slot), row in zip(disk, free):
            reads.append((expert, store._dests(row)))
            copies.append((row, slot))
            store.row_expert[row] = expert
            store.row_stamp[row] = store.clock
    return copies, reads


def _reference_device_plan(store, row, drop=None):
    """``_DeviceState._plan`` with torch ops on the row copy."""
    store.read_maps(row)
    k = store.num_slots
    doorbell = row[: 3 * k].view(3, k)
    mask = doorbell[0].bool()
    loads = zip(*(doorbell[i][mask].tolist() for i in (1, 2)))
    return _reference_plan(store, list(loads), drop)


class TestPlanBookkeeping(CustomTestCase):
    """The per-layer plan with numpy rows and a numpy sort of the free rows plans exactly what
    a Python sort plans."""

    def _run(self, seed, device_planned, drop=None, steps=300):
        """Two stores in the same state, one planned by the reference, one by the fork, over
        ``steps`` random device rows (hits, misses, rows the step stamped with the published
        clock, rows emptied); the first step whose plan or maps differ, or None."""
        import copy
        import random

        rng = random.Random(seed)
        k, e, r = 4, 48, 12
        ref = _plan_store(r, e, device_planned)
        ref.num_slots = k
        new = copy.copy(ref)
        new.row_expert, new.row_stamp = list(ref.row_expert), list(ref.row_stamp)
        new.host = ref.host
        for step in range(steps):
            # The device's step: some rows emptied or stamped with the clock it published.
            for _ in range(rng.randrange(3)):
                row = rng.randrange(r)
                if rng.random() < 0.3:
                    ref.row_expert[row] = -1
                ref.row_stamp[row] = (
                    ref.clock if rng.random() < 0.6 else rng.randrange(ref.clock + 1)
                )
            held = [x for x in ref.row_expert if x >= 0]
            outside = [x for x in range(e) if x not in held]
            n = rng.randrange(1, k + 1)
            hits = rng.sample(held, min(len(held), rng.randrange(n + 1)))
            misses = rng.sample(outside, n - len(hits))
            experts = hits + misses
            rng.shuffle(experts)
            slots = rng.sample(range(k), len(experts))
            doorbell = torch.zeros(3, k, dtype=torch.int32)
            for j, (x, slot) in enumerate(zip(experts, slots)):
                doorbell[:, j] = torch.tensor([1, x, slot])
            maps = torch.tensor(ref.maps(), dtype=torch.int32)
            row = torch.cat([doorbell.flatten(), maps])
            new.row_expert, new.row_stamp, new.clock = (
                [-9] * r,
                [-9] * r,
                -9,
            )  # read_maps sets them
            want = _reference_device_plan(ref, row.clone(), drop)
            got = disk_tier._DeviceState._plan(new, row.numpy().copy())
            state = lambda s: (s.row_expert, s.row_stamp, s.clock, s.maps())
            if want != got or state(ref) != state(new):
                return step
        return None

    def test_plans_and_maps_equal_the_python_plan(self):
        """300 random steps per mode: the same copies, reads (expert, destinations) and maps.
        Negative control: a reference without the published-clock or the empty-row criterion
        differs within the run, so the steps exercise both."""
        for device_planned in (True, False):
            with self.subTest(device_planned=device_planned):
                self.assertIsNone(self._run(1, device_planned))
        self.assertIsNotNone(self._run(1, True, drop=0))
        self.assertIsNotNone(self._run(1, True, drop=1))

    def test_a_tensor_row_plans_as_a_numpy_row(self):
        """``_plan`` takes the row as a tensor (the host-planned resolve) or numpy (the
        resolve while the replay runs)."""
        import copy

        a = _plan_store(device_planned=True)
        a.num_slots = 2
        b = copy.deepcopy(a)
        doorbell = torch.tensor([[1, 1], [12, 4], [0, 1]], dtype=torch.int32)
        row = torch.cat([doorbell.flatten(), torch.tensor(a.maps(), dtype=torch.int32)])
        planned = []
        for store, r in ((a, row.clone()), (b, row.numpy().copy())):
            copies, reads = disk_tier._DeviceState._plan(store, r)
            # the copy has its own host rows: compare which rows are read, not addresses
            planned.append((copies, [e for e, _ in reads], store.maps()))
        self.assertEqual(planned[0], planned[1])
        self.assertEqual(planned[0][1], [12])


@unittest.skipUnless(torch.cuda.is_available(), "needs a GPU")
class TestPinnedFreeDuringCapture(CustomTestCase):
    """A mailbox's pinned rows are written by copies captured on the capture stream, and the
    tiered stores and their device state hold each other, so only the cyclic gc frees them, at a
    point of its choosing. When torch's pinned allocator frees a block, it records an event on
    every stream the block was used on; if that stream is capturing, the record is captured, and
    the next pinned allocation fails ("CUDA error: invalid argument") querying the event, once per
    such event (tests that capture several graphs hit it when a collection fell into a later
    capture). The server's runners capture under ``freeze_gc``; this file's tests capture through
    ``_captured``, which does the same."""

    def _failed_allocations(self, capture) -> int:
        gc.collect()  # earlier tests' garbage goes now, outside any capture
        stream = torch.cuda.Stream()
        mailbox = disk_tier._Mailbox(2, 4, 8, 3)
        row = torch.arange(mailbox.rows.shape[1], dtype=torch.int32, device="cuda")
        graph = torch.cuda.CUDAGraph()
        with capture(graph, stream=stream):
            mailbox.rows[0].copy_(row, non_blocking=True)  # what ``publish`` captures
        graph.replay()
        torch.cuda.synchronize()
        self.assertEqual(mailbox.rows[0].tolist(), row.tolist())
        cycle = SimpleNamespace(mailbox=mailbox)
        cycle.me = cycle  # like the stores and their device state
        freed = weakref.ref(mailbox)
        del mailbox, cycle
        self.assertIsNotNone(freed())  # only a collection frees it
        later = torch.cuda.CUDAGraph()
        with capture(later, stream=stream):
            row.add_(1)
            gc.collect()  # a collection that falls into a later capture on the same stream
        torch.cuda.synchronize()
        gc.collect()
        self.assertIsNone(freed())
        failed = 0
        for _ in range(16):  # each captured event fails one allocation and is dropped
            try:
                torch.zeros(1, pin_memory=True)
                return failed
            except Exception as e:  # torch.AcceleratorError
                self.assertIn("invalid argument", str(e))
                failed += 1
        self.fail("the pinned allocator kept failing")

    def test_a_capture_like_the_servers_frees_nothing_during_it(self):
        self.assertEqual(self._failed_allocations(_captured), 0)

    def test_control_a_plain_capture_lets_a_collection_break_the_pinned_allocator(self):
        # Shows the test above can fail, on a torch whose pinned allocator has the bug: a
        # torch without it passes both (nothing to guard against).
        if self._failed_allocations(torch.cuda.graph) == 0:
            self.skipTest(
                "this torch's pinned allocator survives a free during capture"
            )


class TestSaveRecurrentState(CustomTestCase):
    """A disk-tier replay restores the step's recurrent state first, the pool's slot side
    states included (Qwen4-Exp's PLE short-conv window and N-gram context)."""

    def _pool(self, replayssm=False):
        from sglang.srt.mem_cache.ple_state_pool import NGramPool, ShortConvPool

        slots = 4
        cache = SimpleNamespace(
            conv=[torch.randn(2, slots + 1, 3, 5)],
            temporal=torch.randn(2, slots + 1, 2, 2),
            replayssm_d=torch.zeros(1) if replayssm else None,
        )
        conv = ShortConvPool(
            size=slots,
            state_shape=(6, 3),
            layer_ids=[1, 3],
            dtype=torch.float32,
            device="cpu",
        )
        conv.conv_state.normal_()
        ngram = NGramPool(size=slots, context_len=3, eos_token_id=7, device="cpu")
        ngram.context.random_(0, 1000)
        mamba = SimpleNamespace(mamba_cache=cache, _slot_siblings=[conv, ngram])
        pool = SimpleNamespace(
            mamba_pool=mamba,
            get_mamba_indices=lambda req: req + 1,
            translate_mamba_indices=lambda idx: idx,
        )
        return pool, cache, conv, ngram

    def _states(self, cache, conv, ngram):
        return [cache.conv[0], cache.temporal, conv.conv_state, ngram.context]

    def test_restores_side_states_of_the_steps_slots_only(self):
        from sglang.srt.model_executor.runner_backend_utils import replay_hooks

        pool, cache, conv, ngram = self._pool()
        before = [s.clone() for s in self._states(cache, conv, ngram)]
        with patch.object(replay_hooks, "_HOOKS", [lambda: False]):
            restore = replay_hooks.save_recurrent_state(
                pool, torch.tensor([1])
            )  # slot 2
        for s in self._states(cache, conv, ngram):  # the step advances every slot
            s.add_(1)
        after_step = [s.clone() for s in self._states(cache, conv, ngram)]
        restore()
        for axis, b, a, s in zip(
            (1, 1, 1, 0), before, after_step, self._states(cache, conv, ngram)
        ):
            self.assertTrue(torch.equal(s.select(axis, 2), b.select(axis, 2)))
            # the control: the other slots keep what the step wrote
            for other in (0, 1, 3, 4):
                self.assertTrue(
                    torch.equal(s.select(axis, other), a.select(axis, other))
                )

    def test_without_the_side_states_the_replay_would_see_the_advanced_rows(self):
        """Negative control: restoring the pool's own states only leaves the PLE rows advanced."""
        from sglang.srt.model_executor.runner_backend_utils import replay_hooks

        pool, cache, conv, ngram = self._pool()
        before = ngram.context.clone()
        pool.mamba_pool._slot_siblings = []
        with patch.object(replay_hooks, "_HOOKS", [lambda: False]):
            restore = replay_hooks.save_recurrent_state(pool, torch.tensor([1]))
        ngram.context.add_(1)
        restore()
        self.assertFalse(torch.equal(ngram.context[2], before[2]))

    def test_rejects_replayssm_rings(self):
        from sglang.srt.model_executor.runner_backend_utils import replay_hooks

        pool, *_ = self._pool(replayssm=True)
        with patch.object(replay_hooks, "_HOOKS", [lambda: False]):
            with self.assertRaisesRegex(RuntimeError, "ReplaySSM"):
                replay_hooks.save_recurrent_state(pool, torch.tensor([1]))


if __name__ == "__main__":
    unittest.main()
