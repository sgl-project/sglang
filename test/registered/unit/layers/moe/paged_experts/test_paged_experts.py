import os
import tempfile
import unittest
from types import SimpleNamespace
from typing import NamedTuple, Optional
from unittest.mock import patch

import torch

from sglang.srt.layers.moe.paged_experts import PagedExpertsMoEMethod
from sglang.srt.layers.moe.paged_experts.formats import ExpertFormat, UnquantizedFormat
from sglang.srt.layers.moe.paged_experts.method import (
    _cap_decode_capture_batch_sizes,
    check_paged_experts_compat,
)
from sglang.srt.layers.moe.paged_experts.residency import (
    DeviceResidency,
    LRUPolicy,
    plan_waves,
)
from sglang.srt.layers.moe.paged_experts.runners import RunnerContract, TritonContract
from sglang.srt.layers.moe.paged_experts.sizing import (
    _checkpoint_bytes,
    num_resident_for_budget,
)
from sglang.srt.layers.moe.paged_experts.store import HostExpertStore
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

E, K, H, TOP_K = 16, 4, 8, 2


class _TopK(NamedTuple):
    topk_ids: torch.Tensor
    topk_weights: torch.Tensor
    router_logits: Optional[torch.Tensor] = None


class _Dispatch(NamedTuple):
    hidden_states: torch.Tensor
    topk_output: _TopK
    hidden_states_scale: Optional[torch.Tensor] = None
    hidden_states_pre_quant: Optional[tuple] = None


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

    def process_weights_after_loading(self, layer):
        pass


class _CpuStore(HostExpertStore):
    def _alloc(self, shape, dtype):
        return torch.empty(shape, dtype=dtype)


def _sum_top_k(self, entries, out, routed_scaling_factor):
    out.copy_(entries.sum(1) * routed_scaling_factor)


class _CompactingContract(RunnerContract):
    _sum_top_k = _sum_top_k


class _SkippingContract(TritonContract):
    _sum_top_k = _sum_top_k


CONTRACTS = (_CompactingContract(), _SkippingContract())


def _reference(full, dispatch_output):
    return _LinearMoE().apply(full, dispatch_output).hidden_states.sum(1)


def _step(num_tokens):
    ids = torch.stack([torch.randperm(E)[:TOP_K] for _ in range(num_tokens)]).int()
    weights = torch.rand(num_tokens, TOP_K, dtype=torch.float32)
    x = torch.randn(num_tokens, H, dtype=torch.float32)
    return _Dispatch(x, _TopK(ids, weights))


def _paged_layer(contract=CONTRACTS[0]):
    """A paged method over random experts, loaded into a CPU store."""
    full = {
        "w13_weight": torch.randn(E, H, H, dtype=torch.float32),
        "w2_weight": torch.randn(E, H, H, dtype=torch.float32),
    }
    layer = SimpleNamespace(
        **{
            name: torch.nn.Parameter(torch.empty_like(t[:K]), requires_grad=False)
            for name, t in full.items()
        }
    )
    expert_format = UnquantizedFormat()
    expert_format.contract = contract
    method = PagedExpertsMoEMethod(
        base_method=_LinearMoE(),
        expert_format=expert_format,
        num_experts=E,
        num_resident=K,
    )
    method.store = _CpuStore(
        layer=layer, names=method.format.paged_params, num_experts=E
    )
    for name, t in full.items():
        method.store.host[name].copy_(t)
    method.process_weights_after_loading(layer)
    return method, layer, SimpleNamespace(**full)


class TestPagedForward(CustomTestCase):
    def test_matches_unpaged(self):
        for contract in CONTRACTS:
            for num_tokens in (1, 2, 64):  # one wave, one wave, several waves
                with self.subTest(
                    contract=type(contract).__name__, num_tokens=num_tokens
                ):
                    torch.manual_seed(0)
                    method, layer, full = _paged_layer(contract)
                    for _ in range(8):  # consecutive steps reuse and evict residents
                        d = _step(num_tokens)
                        out = method.apply(layer=layer, dispatch_output=d).hidden_states
                        torch.testing.assert_close(out, _reference(full, d))

    def test_waves_compute_only_their_entries(self):
        torch.manual_seed(0)
        method, layer, _ = _paged_layer(CONTRACTS[0])
        d = _step(64)
        method.apply(layer=layer, dispatch_output=d)
        # Each entry once, plus the padding to a bucketed size; without compaction every wave
        # would compute all of them (here 4 waves).
        self.assertLess(method.base_method.rows, 2 * d.topk_output.topk_ids.numel())

    def test_padded_ids_are_masked(self):
        for contract in CONTRACTS:
            with self.subTest(contract=type(contract).__name__):
                torch.manual_seed(0)
                method, layer, full = _paged_layer(contract)
                d = _step(3)
                ids = d.topk_output.topk_ids.clone()
                ids[1] = -1
                d = d._replace(topk_output=d.topk_output._replace(topk_ids=ids))
                out = method.apply(layer=layer, dispatch_output=d).hidden_states
                weights = d.topk_output.topk_weights.clone()
                weights[1] = 0
                masked = d._replace(topk_output=_TopK(ids.clamp(min=0), weights))
                torch.testing.assert_close(out, _reference(full, masked))

    def test_slots_hold_the_resident_experts(self):
        torch.manual_seed(0)
        method, layer, _ = _paged_layer()
        method.apply(layer=layer, dispatch_output=_step(2))
        for expert, slot in method.policy.resident.items():
            torch.testing.assert_close(
                layer.w2_weight[slot], method.store.host["w2_weight"][expert]
            )


class TestLoading(CustomTestCase):
    def test_rejects_post_load_weight_transform(self):
        method, layer, _ = _paged_layer()
        method.base_method.process_weights_after_loading = lambda layer: (
            layer.w2_weight.data.mul_(2)
        )
        with self.assertRaisesRegex(RuntimeError, "transforms w2_weight"):
            method.process_weights_after_loading(layer)

    def _created_layer(self, extra_params=()):
        """A layer whose expert tensors the paged method created through its base method."""

        class _Base:
            def create_weights(self, layer, num_experts, **kwargs):
                for name in ("w13_weight", "w2_weight", *extra_params):
                    param = torch.nn.Parameter(
                        torch.zeros(num_experts, H, H), requires_grad=False
                    )
                    param.quant_method = "block"
                    layer.register_parameter(name, param)

        method = PagedExpertsMoEMethod(
            base_method=_Base(),
            expert_format=UnquantizedFormat(),
            num_experts=E,
            num_resident=K,
        )
        layer = torch.nn.Module()
        layer.weight_load_method = None
        with patch(
            "sglang.srt.layers.moe.paged_experts.method.HostExpertStore", _CpuStore
        ):
            method.create_weights(
                layer=layer,
                num_experts=E,
                hidden_size=H,
                intermediate_size_per_partition=H,
                params_dtype=torch.float32,
            )
        return method, layer

    def test_host_loader_follows_base_method_and_param_tags(self):
        """Host rows are loaded as natively: the layer's loader follows the wrapped method's
        layout, and sees the parameter's tags (a scale's ``quant_method`` decides its
        layout)."""
        method, layer = self._created_layer()
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

    def test_rejects_online_weight_updates(self):
        from sglang.srt.model_executor.model_runner_components.weight_updater import (
            _unsupported_derived_weight_cache_error,
        )

        _, layer = self._created_layer()
        model = torch.nn.Sequential(layer)
        self.assertIn(
            "--enable-paged-experts", _unsupported_derived_weight_cache_error(model)
        )

    def test_rejects_undeclared_expert_params(self):
        """A per-expert tensor the format does not declare would never reach the slots."""
        with self.assertRaisesRegex(RuntimeError, "w13_weight_scale"):
            self._created_layer(extra_params=("w13_weight_scale",))

    def test_repack_runs_per_chunk_and_keeps_its_output(self):
        """A post-load step that repacks each expert (new shape, new name) is applied to every
        expert of the host store, K at a time; the slots then hold experts 0..K-1 repacked."""

        class _Repacking:
            def process_weights_after_loading(self, layer):
                raw = layer.raw_weight.data
                packed = (raw * 10 + 1).reshape(raw.shape[0], 2, -1)  # per expert
                layer.register_parameter(
                    "packed_weight", torch.nn.Parameter(packed, requires_grad=False)
                )

        class _RepackFormat(ExpertFormat):
            checkpoint_params = ("raw_weight",)
            paged_params = ("packed_weight",)
            repacks_after_loading = True

            @classmethod
            def supports(cls, base_method):
                return True

        layer = torch.nn.Module()
        layer.register_parameter(
            "raw_weight", torch.nn.Parameter(torch.zeros(K, 6), requires_grad=False)
        )
        fmt = _RepackFormat()
        staged = _CpuStore(
            layer=layer,
            names=fmt.checkpoint_params,
            num_experts=E + 1,  # a partial last chunk
            pin=False,
        )
        staged.host["raw_weight"].copy_(torch.arange((E + 1) * 6.0).reshape(E + 1, 6))

        packed = fmt.after_loading(
            layer=layer,
            base_method=_Repacking(),
            store=staged,
            num_slots=K,
            new_store=lambda names: _CpuStore(
                layer=layer, names=names, num_experts=E + 1
            ),
        )

        expected = (staged.host["raw_weight"] * 10 + 1).reshape(E + 1, 2, 3)
        self.assertTrue(torch.equal(packed.host["packed_weight"], expected))
        self.assertTrue(torch.equal(layer.packed_weight.data, expected[:K]))


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


class TestCompat(CustomTestCase):
    def test_rejects_speculative_decoding(self):
        method = "sglang.srt.layers.moe.paged_experts.method."
        moe = SimpleNamespace(
            enable_eplb=False,
            paged_experts_num_resident=None,
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
                return_value=SimpleNamespace(speculative_algorithm="EAGLE"),
            ),
            patch(
                method + "get_model", return_value=SimpleNamespace(load_format="auto")
            ),
            patch(method + "check_cuda_graph_backend", return_value=True),
        ):
            with self.assertRaisesRegex(RuntimeError, "speculative decoding"):
                check_paged_experts_compat(layer=layer, expert_format=object())


class TestSizing(CustomTestCase):
    def test_budget_is_clamped_to_top_k_and_num_experts(self):
        kwargs = dict(
            mem_fraction=0.5,
            nonexpert_bytes=10,
            reserve_bytes=10,
            per_expert_bytes=4,
            top_k=TOP_K,
            num_experts=E,
        )
        # (100 * 0.5 - 20) // 4 = 7
        self.assertEqual(num_resident_for_budget(free_bytes=100, **kwargs), 7)
        self.assertEqual(num_resident_for_budget(free_bytes=10, **kwargs), TOP_K)
        self.assertEqual(num_resident_for_budget(free_bytes=1e6, **kwargs), E)

    def test_checkpoint_bytes_split_routed_experts(self):
        from safetensors.torch import save_file

        with tempfile.TemporaryDirectory() as folder:
            save_file(
                {
                    "model.layers.0.mlp.experts.3.down_proj.weight": torch.zeros(4, 8),
                    # A draft (MTP) layer past num_layers is not served.
                    "model.layers.1.mlp.experts.3.down_proj.weight": torch.zeros(4, 8),
                    "model.layers.1.self_attn.o_proj.weight": torch.zeros(4, 8),
                    "model.layers.0.mlp.shared_experts.down_proj.weight": torch.zeros(
                        2
                    ),
                    "model.embed_tokens.weight": torch.zeros(3, dtype=torch.bfloat16),
                },
                os.path.join(folder, "model.safetensors"),
            )
            self.assertEqual(
                _checkpoint_bytes(
                    files=[os.path.join(folder, "model.safetensors")], num_layers=1
                ),
                (128, 14),
            )

    def test_loader_resolves_the_checkpoint_before_building_the_model(self):
        from sglang.srt.layers.moe.paged_experts import sizing
        from sglang.srt.model_loader.loader import DefaultModelLoader

        loader = DefaultModelLoader.__new__(DefaultModelLoader)
        model_config = SimpleNamespace(
            model_path="org/model", revision=None, hf_config=None
        )
        model_class = type("Model", (), {"fall_back_to_pt_during_load": False})
        prepared = ("/cache/org/model", ["/cache/org/model/a.safetensors"], True)
        loader_module = "sglang.srt.model_loader.loader."
        with (
            patch.object(sizing, "needs_checkpoint", return_value=True),
            patch.object(sizing, "_checkpoint_files", ()),
            patch(
                loader_module + "get_model_architecture",
                return_value=(model_class, "Model"),
            ),
            patch(
                loader_module + "maybe_add_mtp_safetensors", side_effect=lambda f, *_: f
            ),
            patch.object(
                DefaultModelLoader,
                "_prepare_weights",
                return_value=prepared,
                create=True,
            ) as prepare,
        ):
            resolved = loader._resolve_before_init(model_config)
            self.assertEqual(
                sizing._checkpoint_files, ("/cache/org/model/a.safetensors",)
            )
        prepare.assert_called_once_with("org/model", None, False, None)
        # The load after the model is built reuses it instead of preparing again.
        self.assertEqual(
            resolved.source,
            DefaultModelLoader.Source.init_new(model_config, model_class),
        )


if __name__ == "__main__":
    unittest.main()
