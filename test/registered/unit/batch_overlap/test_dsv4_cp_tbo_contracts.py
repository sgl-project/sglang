"""CPU contracts for CP-TBO routing, child geometry, and stream handoff.

Compile the production methods in isolation so these tests do not require
ROCm, AITER, model weights, or an initialized distributed process group.
"""

import ast
import contextlib
import sys
import types
import unittest
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import patch

import torch

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")
ROOT = Path(__file__).resolve().parents[4] / "python/sglang/srt"


def load(relative, name, **namespace):
    path = ROOT / relative
    node = next(
        n
        for n in ast.walk(ast.parse(path.read_text()))
        if isinstance(n, ast.FunctionDef) and n.name == name
    )
    node.decorator_list = []
    tree = ast.Module(
        body=[
            ast.ImportFrom(
                module="__future__", names=[ast.alias(name="annotations")], level=0
            ),
            node,
        ],
        type_ignores=[],
    )
    exec(compile(ast.fix_missing_locations(tree), str(path), "exec"), namespace)
    return namespace[name]


@contextlib.contextmanager
def module(name, **values):
    item = types.ModuleType(name)
    item.__dict__.update(values)
    with patch.dict(sys.modules, {name: item}):
        yield item


class TestCpTboContracts(unittest.TestCase):
    def test_split_boundary_has_no_local_padding_overlap(self):
        plan = load(
            "batch_overlap/two_batch_overlap.py",
            "_compute_cp_aligned_extend_split_plan",
        )
        merge = load(
            "batch_overlap/two_batch_overlap.py",
            "_model_forward_tbo_merge_key",
            torch=torch,
        )
        for cp in (2, 4, 8):
            for length in (32768, 51200, 102400, cp * cp * 3):
                split = plan([length], length, cp * cp)[1]
                self.assertEqual(split % (cp * cp), 0)
                full = torch.arange(length)
                for rank in range(cp):
                    local = full[rank::cp]
                    a, b = full[:split][rank::cp], full[split:][rank::cp]
                    outputs = [
                        dict(
                            hidden_states=x,
                            forward_batch=NS(tbo_cp_local_token_range=r),
                        )
                        for x, r in (
                            (a, (0, split // cp)),
                            (b, (split // cp, length // cp)),
                        )
                    ]
                    result = merge(
                        *outputs,
                        "hidden_states",
                        len(local),
                        "tbo_cp_local_token_range",
                    )
                    torch.testing.assert_close(result, local, rtol=0, atol=0)

    def test_unsupported_parent_alignment_falls_back(self):
        plan = load(
            "batch_overlap/two_batch_overlap.py",
            "_compute_cp_aligned_extend_split_plan",
        )
        for cp in (2, 4, 8):
            self.assertIsNone(plan([32768 + 1], 32769, cp * cp))
            self.assertIsNone(plan([cp * cp], cp * cp, cp * cp))

    def _prepare(self, cp, ranges, lengths=None):
        prepared = []
        children = [
            NS(
                tbo_parent_token_range=r,
                batch_size=1,
                input_ids=torch.arange(r[0], r[1]),
                extend_seq_lens_cpu=[r[1] - r[0]],
            )
            for r in ranges
        ]
        if lengths is not None:
            children[-1].extend_seq_lens_cpu = lengths
        batch = NS(
            tbo_cp_children_ready=True,
            can_run_tbo=True,
            tbo_children=children,
            input_ids=torch.arange(ranges[-1][1]),
            forward_mode=NS(is_extend_without_speculative=lambda: True),
        )
        layer = NS(self_attn=NS(_should_use_cp_tbo_attn_schedule=lambda: True))
        model = NS(model=NS(pp_group=NS(world_size=1), layers=[layer], start_layer=0))
        prepare = load(
            "models/deepseek_v4.py",
            "prepare_forward_batch",
            is_cp_active=lambda b: True,
            is_interleave=lambda: True,
            get_parallel=lambda: NS(attn_cp_size=cp, attn_dp_size=1, attn_tp_size=1),
            get_moe_a2a_backend=lambda: NS(is_none=lambda: True),
            prepare_cp_forward=lambda c: prepared.append(c),
        )
        prepare(model, batch)
        return batch, prepared

    def test_children_have_separate_metadata_and_local_ranges(self):
        for cp in (2, 4, 8):
            batch, prepared = self._prepare(cp, [(0, 16384), (16384, 32768)])
            self.assertTrue(batch.tbo_cp_children_ready)
            self.assertEqual(len(prepared), 2)
            self.assertEqual(prepared[0].tbo_cp_local_token_range, (0, 16384 // cp))
            self.assertEqual(
                prepared[1].tbo_cp_local_token_range, (16384 // cp, 32768 // cp)
            )
            self.assertEqual([c.tbo_cp_subbatch_index for c in prepared], [0, 1])

    def test_invalid_children_are_rejected_before_any_metadata_mutation(self):
        for ranges in ([(0, 16388), (16388, 32768)], [(0, 16384), (16400, 32768)]):
            batch, prepared = self._prepare(4, ranges)
            self.assertFalse(batch.tbo_cp_children_ready)
            self.assertEqual(prepared, [])
        batch, prepared = self._prepare(4, [(0, 16384), (16384, 32768)], [16383])
        self.assertFalse(batch.tbo_cp_children_ready)
        self.assertEqual(prepared, [])

    def test_attention_gate_rejects_other_models_layouts_and_devices(self):
        gate = load(
            "models/deepseek_v4.py",
            "_should_use_cp_tbo_attn_schedule",
            _is_hip=True,
            is_cp_enabled=lambda: True,
            is_interleave=lambda: True,
        )
        name = "sglang.kernels.ops.attention.dsv4.unified_kv_kernels.env_gate"
        with module(
            name, is_unified_kv_triton=lambda: True, is_unified_kv_fp8=lambda: False
        ) as env:
            self.assertTrue(gate(NS(is_dsv41=False, attn_tp_size=1)))
            self.assertFalse(gate(NS(is_dsv41=True, attn_tp_size=1)))
            self.assertFalse(gate(NS(is_dsv41=False, attn_tp_size=2)))
            env.is_unified_kv_fp8 = lambda: True
            self.assertFalse(gate(NS(is_dsv41=False, attn_tp_size=1)))
            env.is_unified_kv_fp8 = lambda: False
            gate.__globals__["_is_hip"] = False
            self.assertFalse(gate(NS(is_dsv41=False, attn_tp_size=1)))

    def test_attention_wait_interleaves_only_after_ready_event(self):
        waited = []
        fn = load(
            "layers/cp/tbo.py", "cp_tbo_all_gather_rerange_output_wait", torch=torch
        )
        with patch.object(
            torch.cuda, "current_stream", return_value=NS(wait_event=waited.append)
        ):
            for cp in (2, 4, 8):
                full = torch.arange(cp * 16 * 3).reshape(cp * 16, 3)
                gathered = torch.cat([full[r::cp] for r in range(cp)])
                event = object()
                handle = NS(
                    output_tensor=gathered,
                    reranged_tensor=torch.empty_like(full),
                    event=event,
                    cp_size=cp,
                )
                actual = fn(handle)
                self.assertIs(waited[-1], event)
                torch.testing.assert_close(actual, full, rtol=0, atol=0)

    def test_attention_launches_share_stream_and_keep_unique_buffers(self):
        comm, compute, calls, buffers = object(), object(), [], {}
        comm = NS(wait_stream=lambda s: calls.append(("wait", s)))
        event = NS(record=lambda s: calls.append(("event", s)))
        group = NS(
            cp_all_gather_into_tensor_async=lambda out, inp, stream: calls.append(
                ("gather", stream)
            )
        )

        def buffer(key, shape, dtype, device):
            buffers.setdefault(key, torch.empty(shape, dtype=dtype, device=device))
            return buffers[key]

        fn = load(
            "layers/cp/tbo.py",
            "cp_tbo_all_gather_rerange_output_async",
            torch=torch,
            get_parallel=lambda: NS(attn_cp_group=group),
            get_dp_tbo_comm_stream=lambda: comm,
            get_tbo_persistent_tensor=buffer,
            _tbo_event=lambda key: event,
            is_allocation_symmetric=lambda: False,
            use_symmetric_memory=lambda *a, **k: contextlib.nullcontext(),
            CpTboAllGatherHandle=NS,
        )
        with (
            module(
                "sglang.srt.layers.attention.dsa.utils",
                is_dsa_prefill_cp_interleave=lambda: True,
            ),
            patch.object(
                torch.cuda, "stream", side_effect=lambda s: contextlib.nullcontext()
            ),
            patch.object(torch.cuda, "current_stream", return_value=compute),
            patch.object(torch.Tensor, "record_stream", return_value=None),
        ):
            handles = [
                fn(torch.ones(16, 4), 4, NS(tbo_cp_subbatch_index=sub), kind=kind)
                for sub in (0, 1)
                for kind in ("kv", "indexer_score_c4", "core_score_c4")
            ]
        self.assertEqual(len(buffers), 12)
        self.assertEqual(len({h.output_tensor.data_ptr() for h in handles}), 6)
        self.assertTrue(all(s is comm for op, s in calls if op in ("gather", "event")))
        self.assertTrue(all(s is compute for op, s in calls if op == "wait"))

    def test_server_guard_only_exempts_deepseek_v4_cp(self):
        cfg = NS(
            enable_two_batch_overlap=True,
            enable_prefill_cp=True,
            cp_strategy="interleave",
            attn_cp_size=4,
            dp_size=1,
            moe_a2a_backend="none",
            attn_dp_size=1,
        )
        model = NS(hf_config=NS(model_type="deepseek_v4"))
        guard = load(
            "arg_groups/validation_hook.py",
            "check_two_batch_overlap",
            resolving_view=lambda a: a,
            model_config_of=lambda a: model,
            attn_dp_enabled_of=lambda a: a.attn_dp_size > 1,
        )
        guard(cfg)
        for model_type in ("deepseek_v41", "deepseek_v3", "llama"):
            model.hf_config.model_type = model_type
            with self.assertRaisesRegex(ValueError, "attn-dp-size"):
                guard(cfg)
        cfg.enable_prefill_cp = False
        cfg.attn_dp_size = 2
        guard(cfg)
        cfg.attn_dp_size = 1
        cfg.enable_two_batch_overlap = False
        guard(cfg)

    def test_family_guard_rejects_unvalidated_cp_tbo_combinations(self):
        def env(value):
            return NS(get=lambda: value, set=lambda value: None)

        values = NS(
            SGLANG_HACK_FLASHMLA_BACKEND=env("unified_kv_triton"),
            SGLANG_DSV4_UNIFIED_KV_FP8=env(False),
            SGLANG_OPT_FLASHMLA_SPARSE_PREFILL=env(False),
        )
        cfg = NS(
            enable_prefill_cp=True,
            cp_strategy="interleave",
            nnodes=1,
            enable_decoder_swa_bounded_replay=False,
            enable_two_batch_overlap=True,
            moe_a2a_backend="none",
            ep_size=1,
            pp_size=1,
            dcp_size=1,
            speculative_algorithm=None,
            enable_mixed_chunk=False,
            cuda_graph_config=NS(prefill=NS(backend="disabled")),
            tp_size=4,
            moe_dense_tp_size=1,
            attn_cp_size=4,
        )
        validate = load(
            "arg_groups/deepseek_v4_hook.py",
            "validate_deepseek_v4_cp",
            resolving_view=lambda a: a,
            resolved_view=lambda a: a,
            get_platform=lambda: NS(is_hip=True),
            is_npu=lambda: False,
            attention_backends_of=lambda a: ("dsv4", "dsv4"),
            model_config_of=lambda a: NS(hf_config=NS(model_type="deepseek_v4")),
            envs=values,
            context_parallel_attn_dp_size=lambda *a: 1,
            declare_resolution=lambda *a, **k: None,
            logger=NS(warning=lambda *a: None),
        )
        with module(
            "sglang.srt.model_executor.cuda_graph_config",
            Backend=NS(DISABLED="disabled"),
        ):
            validate(cfg)
            for field, value in (
                ("pp_size", 2),
                ("dcp_size", 2),
                ("ep_size", 2),
                ("enable_mixed_chunk", True),
                ("speculative_algorithm", "DSPARK"),
            ):
                before = getattr(cfg, field)
                setattr(cfg, field, value)
                with self.assertRaises(ValueError):
                    validate(cfg)
                setattr(cfg, field, before)
            values.SGLANG_DSV4_UNIFIED_KV_FP8 = env(True)
            with self.assertRaisesRegex(ValueError, "BF16 unified KV"):
                validate(cfg)
            cfg.enable_two_batch_overlap = False
            validate(cfg)

    def test_local_query_is_preserved_until_attention_finish(self):
        fn = load(
            "models/deepseek_v4.py",
            "op_attn_cp_finish",
            get_attn_backend=lambda: backend,
            cp_tbo_all_gather_rerange_output_wait=lambda h: h,
        )
        calls = []
        backend = NS(
            forward_indexer_compressor_from_kv_score=lambda *a: calls.append(
                "indexer_k"
            ),
            forward_c4_indexer=lambda **k: calls.append("indexer_q"),
            forward_core_compressor_from_kv_score=lambda *a: calls.append("core_k"),
        )

        class State(dict):
            __getattr__ = dict.__getitem__
            __setattr__ = dict.__setitem__

        q, kv, prepared = object(), object(), object()
        state = State(
            cp_attn_split_active=True,
            forward_batch=object(),
            positions=object(),
            cp_attn_q=q,
            cp_attn_kv_handle=kv,
            cp_attn_indexer_prepared=prepared,
            cp_attn_indexer_score_handle=object(),
            cp_attn_core_score_handle=object(),
        )

        def core(actual_q, actual_kv, *args):
            self.assertIs(actual_q, q)
            self.assertIs(actual_kv, kv)
            self.assertEqual(calls, ["indexer_k", "indexer_q", "core_k"])
            return "result"

        layer = NS(
            indexer=NS(layer_id=2, compressor=object()),
            layer_id=2,
            compressor=object(),
            _forward_core_and_project=core,
        )
        fn(layer, state)
        self.assertEqual(state.hidden_states_after_attn, "result")


if __name__ == "__main__":
    unittest.main()
