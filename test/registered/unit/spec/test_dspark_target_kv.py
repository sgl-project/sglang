"""Checkpoint, RoPE and differentiable encoder invariants for KV-input DSpark."""

import copy
import unittest
from types import SimpleNamespace

import msgspec
import torch

from sglang.srt.speculative.dspark_components.dspark_target_kv_contract import (
    SharedHeadTransform,
    TargetKVDraftContract,
    read_target_kv_draft_contract,
    validate_target_kv_draft_contract,
)
from sglang.srt.speculative.dspark_components.dspark_target_kv_encoder import (
    TargetKVContextEncoder,
)
from sglang.srt.speculative.dspark_components.dspark_target_kv_inject import (
    TargetKVSourceRange,
)
from sglang.srt.training_capture.kv_codec import (
    StandardRopeConfig,
    inverse_standard_rope,
    target_kv_features,
)
from sglang.srt.training_capture.protocol import (
    ContractError,
    canonical_bytes,
    digest_bytes,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.dspark_target_kv_utils import (
    graph_verify_hidden_mode,
    make_minimal_kv_weight_loader,
    make_target_kv_config,
    make_target_kv_contract,
    make_target_kv_injector,
)
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


class TestTargetKVDraftContract(CustomTestCase):
    def test_pd_kv_draft_uses_local_projection_without_transferring_draft_pool(self):
        from sglang.srt.speculative.dspark_components.dspark_worker_v2 import (
            DSparkWorkerV2,
        )
        from sglang.srt.speculative.spec_info import SpeculativeAlgorithm

        pool = object()
        worker = object.__new__(DSparkWorkerV2)
        worker._target_worker = SimpleNamespace(
            model_runner=SimpleNamespace(spec_algorithm=SpeculativeAlgorithm.DSPARK)
        )
        worker._draft_worker = SimpleNamespace(
            model_runner=SimpleNamespace(token_to_kv_pool=pool)
        )
        worker._target_kv_contract = make_target_kv_contract()
        self.assertIs(worker.primary_draft_kv_pool, pool)
        self.assertIsNone(worker.disaggregation_draft_kv_pool)
        worker._target_kv_contract = None
        self.assertIs(worker.disaggregation_draft_kv_pool, pool)

    def test_shared_head_transform_applies_once_in_fp32_before_markov(self):
        transform = SharedHeadTransform.decode(
            '{"logit_scale":0.5,"final_logit_softcapping":2.0}'
        )
        logits = torch.tensor([-8, -2, 0, 4, 8], dtype=torch.bfloat16)
        actual = transform.apply(logits)
        self.assertEqual(actual.dtype, torch.float32)
        torch.testing.assert_close(actual, 2 * torch.tanh(logits.float() / 4))
        with self.assertRaises((ContractError, msgspec.ValidationError)):
            SharedHeadTransform.decode('{"unknown_scale":0.5}')

    def test_legacy_and_explicit_input_modes(self):
        self.assertIsNone(
            read_target_kv_draft_contract({"architectures": ["DSparkDraftModel"]})
        )
        config = make_target_kv_config()
        contract = read_target_kv_draft_contract(config)
        self.assertEqual(contract.feature_size, 32)
        self.assertEqual(contract, make_target_kv_contract())
        for changes in (
            {"input_mode": "target_hidden"},
            {"input_mode": "kv_typo"},
            {"architectures": ["DSparkDraftModel"]},
            {"hidden_size": 16},
            {"enable_confidence_head": True},
        ):
            with self.subTest(changes=changes), self.assertRaises(ContractError):
                read_target_kv_draft_contract(config | changes)

    def test_rejects_silent_window_codec_and_layer_changes(self):
        raw = make_target_kv_config()["target_kv_contract"]
        changes = (
            ("sequence", "input_length", 4),
            ("sequence", "label_shift", 0),
            ("sequence", "mask_token_id", 256),
            ("training", "confidence_policy", "learned"),
            ("training", "tv_tail_policy", "renormalize"),
            ("kv", "selected_layer_ids", [1, 3]),
            ("kv", "rope_config_sha256", "0" * 64),
            ("encoder", "rms_norm_eps", 0),
        )
        for group, key, value in changes:
            altered = copy.deepcopy(raw)
            altered[group][key] = value
            with (
                self.subTest(group=group, key=key),
                self.assertRaises((ContractError, msgspec.ValidationError)),
            ):
                TargetKVDraftContract.decode(altered)

    def test_runtime_binding_excludes_physical_chunking_but_pins_semantics(self):
        contract = make_target_kv_contract()
        args = {
            "target": contract.teacher,
            "draft": contract,
            "pool_codec": contract.kv,
            "target_hidden_size": 8,
            "prediction_count": 3,
            "mask_token_id": 255,
        }
        validate_target_kv_draft_contract(**args)
        validate_target_kv_draft_contract(
            **(
                args
                | {
                    "pool_codec": msgspec.structs.replace(
                        contract.kv, source_page_size=16, storage_chunk_tokens=256
                    )
                }
            )
        )
        for change in (
            {
                "target": msgspec.structs.replace(
                    contract.teacher, weights_revision="other"
                )
            },
            {"target_hidden_size": 16},
            {"prediction_count": 2},
            {"mask_token_id": 254},
            {
                "pool_codec": msgspec.structs.replace(
                    contract.kv, source_k_norm="different"
                )
            },
        ):
            with self.subTest(change=change), self.assertRaises(ContractError):
                validate_target_kv_draft_contract(**(args | change))


class TestTargetKVFeatureMath(CustomTestCase):
    def test_inverse_rope_against_complex_rotation_and_partial_suffix(self):
        positions = torch.tensor([0, 17, 1007], dtype=torch.int64)
        generator = torch.Generator().manual_seed(71)
        source = torch.randn(3, 2, 6, generator=generator)
        for interleaved in (False, True):
            with self.subTest(interleaved=interleaved):
                rope = StandardRopeConfig(
                    type="default",
                    theta=10000.0,
                    rotary_dim=4,
                    interleaved=interleaved,
                    scaling=None,
                )
                pairs = (
                    source[..., :4].reshape(3, 2, 2, 2)
                    if interleaved
                    else torch.stack((source[..., :2], source[..., 2:4]), dim=-1)
                )
                complex_source = torch.view_as_complex(pairs.contiguous())
                phase = positions.double()[:, None, None] * torch.tensor(
                    [1.0, 0.01], dtype=torch.float64
                )
                rotated_pairs = torch.view_as_real(
                    complex_source.to(torch.complex128) * torch.exp(1j * phase)
                ).float()
                rotated = (
                    rotated_pairs.flatten(-2)
                    if interleaved
                    else torch.cat(
                        (rotated_pairs[..., 0], rotated_pairs[..., 1]), dim=-1
                    )
                )
                stored = torch.cat((rotated, source[..., 4:]), dim=-1)
                restored = inverse_standard_rope(stored, positions, rope)
                torch.testing.assert_close(restored, source, rtol=1e-5, atol=1e-5)
                torch.testing.assert_close(
                    restored[..., 4:], source[..., 4:], rtol=0, atol=0
                )

    def test_feature_order_and_unknown_rope_fail_closed(self):
        contract = make_target_kv_contract()
        values = {
            f"target_{component}.{layer}": torch.full(
                (2, 2, 4), value, dtype=torch.bfloat16
            )
            for value, (layer, component) in enumerate(
                ((3, "k"), (3, "v"), (1, "k"), (1, "v")), 1
            )
        }
        features = target_kv_features(
            contract.kv, values, torch.zeros(2, dtype=torch.int64)
        )
        torch.testing.assert_close(
            features, torch.tensor([[1.0] * 8 + [2.0] * 8 + [3.0] * 8 + [4.0] * 8] * 2)
        )
        rope = dict(contract.kv.rope_config, scaling={"factor": 2})
        altered = msgspec.structs.replace(
            contract.kv,
            rope_config=rope,
            rope_config_sha256=digest_bytes(canonical_bytes(rope)),
        )
        with self.assertRaises(msgspec.ValidationError):
            target_kv_features(altered, values, torch.arange(2))
        with self.assertRaises(ContractError):
            target_kv_features(contract.kv, {}, torch.arange(2))

    def test_encoder_has_gradients_target_is_constant_and_future_rows_are_isolated(
        self,
    ):
        torch.manual_seed(17)
        contract = make_target_kv_contract()
        encoder = TargetKVContextEncoder(contract)
        positions = torch.arange(5)
        source = {
            f"target_{component}.{layer.layer_id}": torch.randn(
                5, layer.num_kv_heads, dim, dtype=torch.bfloat16
            ).requires_grad_()
            for layer in contract.kv.layers
            for component, dim in (
                ("k", layer.key_head_dim),
                ("v", layer.value_head_dim),
            )
        }
        output = encoder(source, positions)
        (output * torch.arange(1, 9)).sum().backward()
        for parameter in encoder.parameters():
            self.assertTrue(torch.isfinite(parameter.grad).all())
            self.assertGreater(parameter.grad.abs().sum().item(), 0)
        self.assertTrue(all(value.grad is None for value in source.values()))
        changed = {name: value.detach().clone() for name, value in source.items()}
        for value in changed.values():
            value[3:] = 1000
        torch.testing.assert_close(
            encoder(changed, positions)[:3], output[:3], rtol=0, atol=0
        )
        prefix = encoder(
            {name: value[:3] for name, value in source.items()}, positions[:3]
        )
        torch.testing.assert_close(prefix, output[:3], rtol=1e-6, atol=1e-6)


class TestTargetKVInjector(CustomTestCase):
    def test_encoder_receives_global_heads_once_in_canonical_order(self):
        """Replicated TP heads occupy adjacent ranks, not a second head block."""
        for heads in (2, 8):
            with self.subTest(heads=heads):
                injector, _, _ = make_target_kv_injector()
                full = torch.arange(6 * heads * 2).reshape(6, heads, 2)
                locations = torch.tensor([4, 1, 5])
                replicas = max(1, 4 // heads)
                physical = full.repeat_interleave(replicas, dim=1)
                if replicas > 1:
                    physical[:, 1::replicas] = -999
                injector.sources = {"target_k.3": physical[:, : max(1, heads // 4)]}
                injector.head_replicas = {"target_k.3": replicas}

                def gather(
                    value,
                    *,
                    dim,
                    injector=injector,
                    locations=locations,
                    physical=physical,
                ):
                    self.assertEqual(dim, 1)
                    torch.testing.assert_close(
                        value, injector.sources["target_k.3"][locations]
                    )
                    return physical[locations]

                injector.tp_group = SimpleNamespace(all_gather=gather)
                selected = injector._select_target_kv(locations)
                torch.testing.assert_close(
                    selected["target_k.3"], full[locations], rtol=0, atol=0
                )

    def test_large_cached_prefix_projects_in_bounded_chunks(self):
        injector, writer, batch = make_target_kv_injector()
        injector.sources = {"target_k.3": torch.arange(5000).reshape(2500, 1, 2)}
        injector.model_runner.req_to_token_pool.req_to_token = torch.arange(2500).view(
            1, -1
        )
        batch.reqs = batch.reqs[:1]
        batch.seq_lens_cpu = torch.tensor([2500])
        injector.ensure_context(batch)
        self.assertEqual(
            [positions.numel() for _, positions, _ in writer.writes], [1024, 1024, 452]
        )
        self.assertEqual(batch.reqs[0].dspark_projected_context.end, 2500)
        self.assertEqual(injector.projected_token_ct, 2500)
        torch.testing.assert_close(
            torch.cat([positions for _, positions, _ in writer.writes]),
            torch.arange(2500),
        )

    def test_prefill_incremental_verify_and_request_slot_reuse(self):
        injector, writer, batch = make_target_kv_injector()
        injector.ensure_context(batch)
        self.assertEqual(injector.projected_token_ct, 5)
        self.assertEqual(writer.writes[0][2].tolist(), [7, 3, 9])
        self.assertEqual(writer.writes[0][1].tolist(), [0, 1, 2])
        torch.testing.assert_close(
            writer.writes[0][0]["target_k.3"], injector.sources["target_k.3"][[7, 3, 9]]
        )
        injector.ensure_context(batch)
        self.assertEqual(len(writer.writes), 2)
        window = SimpleNamespace(
            verify_cache_loc_2d=torch.tensor([[2, 8, 4, 1], [15, 11, 14, 13]]),
            positions_2d=torch.tensor([[3, 4, 5, 6], [2, 3, 4, 5]]),
        )
        injector.inject_verify(
            batch=batch, verify_window=window, commit_lens=torch.tensor([1, 3])
        )
        self.assertEqual(writer.writes[2][2].tolist(), [2])
        self.assertEqual(writer.writes[3][2].tolist(), [15, 11, 14])
        self.assertEqual(writer.writes[3][1].tolist(), [2, 3, 4])
        self.assertEqual(
            [req.dspark_projected_context.end for req in batch.reqs], [4, 5]
        )
        # Rejected KV and the still-unforwarded bonus never enter the projection.
        self.assertEqual(injector.projected_token_ct, 9)
        batch.seq_lens_cpu = torch.tensor([4, 5])
        injector.ensure_context(batch)
        self.assertEqual(len(writer.writes), 4)
        batch.reqs[0] = SimpleNamespace(req_pool_idx=0, dspark_projected_context=None)
        injector.ensure_context(batch)
        self.assertEqual(writer.writes[-1][2].tolist(), [7, 3, 9, 2])

    def test_epoch_change_and_retraction_rebuild_prefix(self):
        injector, writer, batch = make_target_kv_injector()
        injector.ensure_context(batch)
        previous = injector.weight_version
        injector.invalidate_all()
        injector.ensure_context(batch)
        self.assertNotEqual(injector.weight_version, previous)
        self.assertEqual(injector.invalidation_ct, 2)
        self.assertEqual(injector.projected_token_ct, 10)
        batch.seq_lens_cpu = torch.tensor([1, 2])
        injector.ensure_context(batch)
        self.assertEqual(writer.writes[-1][2].tolist(), [7])
        self.assertEqual(batch.reqs[0].dspark_projected_context.end, 1)

    def test_gaps_overlaps_stale_weights_and_invalid_lengths_fail_before_write(self):
        injector, writer, batch = make_target_kv_injector()
        injector.ensure_context(batch)
        req = batch.reqs[0]
        state = req.dspark_projected_context
        for start, end, version in (
            (4, 5, injector.weight_version),
            (2, 3, injector.weight_version),
            (3, 5, injector.weight_version),
            (3, 4, "stale"),
        ):
            with (
                self.subTest(start=start, end=end, version=version),
                self.assertRaises(ContractError),
            ):
                injector.inject_target_kv(
                    req,
                    committed_prefix_end=end,
                    source_ranges=(
                        TargetKVSourceRange(
                            start=start,
                            cache_locs=torch.tensor([2]),
                            positions=torch.tensor([3]),
                        ),
                    ),
                    draft_weight_version=version,
                )
        self.assertIs(req.dspark_projected_context, state)
        self.assertEqual(len(writer.writes), 2)


class TestTargetKVWeightLoader(CustomTestCase):
    def test_global_exports_load_into_tp_shards_with_replicated_kv_heads(self):
        """A global checkpoint must not be checked against local parameter shapes."""
        full = make_minimal_kv_weight_loader()
        weights = {
            name: torch.arange(value.numel()).reshape_as(value).float() + 100 * i
            for i, (name, value) in enumerate(full.named_parameters())
        }
        for rank in range(2):
            for split in (False, True):
                with self.subTest(rank=rank, split=split):
                    model = make_minimal_kv_weight_loader(tp_size=2, tp_rank=rank)
                    checkpoint = []
                    expected = {}
                    for name, value in weights.items():
                        if ".qkv_proj." in name:
                            q, k, v = value.split((4, 2, 2))
                            expected[name] = torch.cat(
                                (q[rank * 2 : rank * 2 + 2], k, v)
                            )
                            parts = zip(("q", "k", "v"), (q, k, v), strict=True)
                            if split:
                                checkpoint.extend(
                                    (name.replace("qkv_proj", part + "_proj"), shard)
                                    for part, shard in parts
                                )
                            else:
                                checkpoint.append((name, value))
                        elif ".gate_up_proj." in name:
                            gate, up = value.chunk(2)
                            expected[name] = torch.cat(
                                (
                                    gate[rank * 2 : rank * 2 + 2],
                                    up[rank * 2 : rank * 2 + 2],
                                )
                            )
                            if split:
                                checkpoint.extend(
                                    (
                                        name.replace("gate_up_proj", part + "_proj"),
                                        shard,
                                    )
                                    for part, shard in zip(
                                        ("gate", "up"), (gate, up), strict=True
                                    )
                                )
                            else:
                                checkpoint.append((name, value))
                        else:
                            checkpoint.append((name, value))
                            expected[name] = (
                                value[:, rank * 2 : rank * 2 + 2]
                                if ".o_proj." in name
                                else value
                            )
                    model.load_weights(checkpoint)
                    for name, value in model.named_parameters():
                        torch.testing.assert_close(
                            value, expected[name], rtol=0, atol=0
                        )

    def test_split_gqa_and_mlp_exports_preserve_packed_parameter_layout(self):
        model = make_minimal_kv_weight_loader()
        expected = {
            name: torch.arange(parameter.numel()).reshape_as(parameter).float() + i
            for i, (name, parameter) in enumerate(model.named_parameters())
        }
        split = []
        for name, value in expected.items():
            if ".qkv_proj." in name:
                parts = zip(("q", "k", "v"), value.split((4, 2, 2)), strict=True)
                split.extend(
                    ("model." + name.replace("qkv_proj", part + "_proj"), shard)
                    for part, shard in parts
                )
            elif ".gate_up_proj." in name:
                parts = zip(("gate", "up"), value.chunk(2), strict=True)
                split.extend(
                    ("model." + name.replace("gate_up_proj", part + "_proj"), shard)
                    for part, shard in parts
                )
            else:
                split.append(("model." + name, value))
        model.load_weights(reversed(split))
        for name, parameter in model.named_parameters():
            torch.testing.assert_close(parameter, expected[name], rtol=0, atol=0)

    def test_oversized_packed_and_split_weights_cannot_be_silently_truncated(self):
        """Parallel loaders otherwise narrow oversized exports and discard rows."""
        for packed in (True, False):
            for projection in ("qkv_proj", "gate_up_proj"):
                with self.subTest(packed=packed, projection=projection):
                    model = make_minimal_kv_weight_loader()
                    original = {
                        name: parameter.detach().clone()
                        for name, parameter in model.named_parameters()
                    }
                    weights = []
                    for name, value in original.items():
                        value = torch.full_like(value, 9)
                        if f".{projection}." not in name:
                            weights.append((name, value))
                        elif packed:
                            weights.append((name, torch.cat((value, value[:1]))))
                        else:
                            parts = (
                                zip(
                                    ("q", "k", "v"), value.split((4, 2, 2)), strict=True
                                )
                                if projection == "qkv_proj"
                                else zip(("gate", "up"), value.chunk(2), strict=True)
                            )
                            for part, shard in parts:
                                weights.append(
                                    (
                                        name.replace(projection, part + "_proj"),
                                        torch.cat((shard, shard[:1])),
                                    )
                                )
                    with self.assertRaisesRegex(ContractError, "shape"):
                        model.load_weights(weights)
                    for name, parameter in model.named_parameters():
                        torch.testing.assert_close(
                            parameter, original[name], rtol=0, atol=0
                        )

    def test_late_invalid_tensor_rejects_before_any_parameter_or_cache_changes(self):
        """Bad late weights previously partially overwrote or poisoned the draft."""
        malformed = {
            "shape": torch.ones(1, 2),
            "integer": torch.ones(2, 2, dtype=torch.int64),
            "complex": torch.ones(2, 2, dtype=torch.complex64),
            "nan": torch.tensor([[1.0, float("nan")], [2.0, 3.0]]),
            "infinity": torch.tensor([[1.0, 2.0], [float("inf"), 3.0]]),
            "cast_overflow": torch.tensor([[1.0, 2.0], [-70000.0, 3.0]]),
        }
        for kind, invalid in malformed.items():
            with self.subTest(kind=kind):
                model = make_minimal_kv_weight_loader().half()
                original = {
                    name: parameter.detach().clone()
                    for name, parameter in model.named_parameters()
                }
                fused, stacked = (
                    model._fused_kv_write_cache,
                    model._stacked_ctx_kv_cache,
                )
                weights = [
                    (name, torch.full_like(value, 9, dtype=torch.float32))
                    for name, value in original.items()
                    if name != "markov_head.weight"
                ]
                weights.append(("markov_head.weight", invalid))
                with self.assertRaises(ContractError):
                    model.load_weights(weights)
                for name, parameter in model.named_parameters():
                    torch.testing.assert_close(
                        parameter, original[name], rtol=0, atol=0
                    )
                self.assertIs(model._fused_kv_write_cache, fused)
                self.assertIs(model._stacked_ctx_kv_cache, stacked)

    def test_gated_markov_parameters_are_not_interpreted_as_mlp_shards(self):
        from sglang.srt.models.dspark import GatedMarkovHead

        model = make_minimal_kv_weight_loader()
        model.markov_head = GatedMarkovHead(vocab_size=4, markov_rank=2, hidden_size=2)
        expected = {
            name: torch.full_like(parameter, index + 1)
            for index, (name, parameter) in enumerate(model.named_parameters())
        }
        model.load_weights(expected.items())
        for name, parameter in model.named_parameters():
            torch.testing.assert_close(parameter, expected[name], rtol=0, atol=0)

    def test_complete_checkpoint_loads_and_resets_projection_caches(self):
        for source, destination in (
            (torch.float32, torch.float16),
            (torch.float16, torch.bfloat16),
            (torch.bfloat16, torch.float32),
        ):
            with self.subTest(source=source, destination=destination):
                model = make_minimal_kv_weight_loader().to(dtype=destination)
                expected = {
                    name: torch.full_like(parameter, i + 1, dtype=source)
                    for i, (name, parameter) in enumerate(model.named_parameters())
                }
                model.load_weights(
                    [("model." + name, value) for name, value in expected.items()]
                )
                for name, parameter in model.named_parameters():
                    torch.testing.assert_close(
                        parameter, expected[name].to(destination), rtol=0, atol=0
                    )
                self.assertIsNone(model._fused_kv_write_cache)
                self.assertIs(model._stacked_ctx_kv_cache, False)

    def test_missing_duplicate_foreign_and_partial_shards_reject_before_mutation(self):
        model = make_minimal_kv_weight_loader()
        original = {
            name: parameter.detach().clone()
            for name, parameter in model.named_parameters()
        }
        items = [(name, torch.full_like(value, 99)) for name, value in original.items()]
        qkv_name = "layers.0.self_attn.qkv_proj.weight"
        without_qkv = [(name, value) for name, value in items if name != qkv_name]
        partial = ("layers.0.self_attn.q_proj.weight", torch.ones(2, 2))
        for weights in (
            items[:-1],
            items + items[:1],
            items + [("fc.weight", torch.ones(2, 2))],
            without_qkv + [partial],
            items + [partial],
        ):
            with (
                self.subTest(names=[name for name, _ in weights]),
                self.assertRaises(ContractError),
            ):
                model.load_weights(weights)
            for name, parameter in model.named_parameters():
                torch.testing.assert_close(parameter, original[name], rtol=0, atol=0)


class TestTargetKVManagement(CustomTestCase):
    def test_graph_capture_preserves_hidden_mode_only_for_hidden_input(self):
        from sglang.srt.model_executor.forward_batch_info import CaptureHiddenMode

        self.assertEqual(
            graph_verify_hidden_mode(use_aux_hidden=True), CaptureHiddenMode.FULL
        )
        self.assertEqual(
            graph_verify_hidden_mode(use_aux_hidden=False), CaptureHiddenMode.NULL
        )

    def test_mutation_replies_reject_without_touching_worker_or_offload_state(self):
        from sglang.srt.managers.scheduler_components.weight_updater import (
            SchedulerWeightUpdaterManager,
        )

        manager = SchedulerWeightUpdaterManager(
            tp_worker=None,
            draft_worker=SimpleNamespace(
                draft_model_runner=SimpleNamespace(
                    model_config=SimpleNamespace(
                        hf_config=SimpleNamespace(
                            architectures=["DSparkTargetKVDraftModel"]
                        )
                    )
                )
            ),
            tp_cpu_group=None,
            memory_saver_adapter=None,
            flush_cache=None,
            is_fully_idle=None,
        )
        for method in (
            manager.update_weights_from_disk,
            manager.update_weights_from_distributed,
            manager.update_weights_from_tensor,
            manager.update_weights_from_ipc,
            manager.release_memory_occupation,
            manager.resume_memory_occupation,
        ):
            response = method(None)
            self.assertFalse(response.success)
            self.assertIn("target-KV DSpark", response.message)
        self.assertFalse(manager.offload_tags)
        manager.draft_worker = None
        self.assertIsNone(manager._target_kv_mutation_error("test"))

    def test_memory_response_defaults_and_failure_roundtrip(self):
        from sglang.srt.managers.io_struct import (
            ReleaseMemoryOccupationReqOutput,
            ResumeMemoryOccupationReqOutput,
        )

        for schema in (
            ReleaseMemoryOccupationReqOutput,
            ResumeMemoryOccupationReqOutput,
        ):
            self.assertTrue(schema().success)
            value = schema(success=False, message="bound KV")
            self.assertEqual(
                msgspec.msgpack.decode(msgspec.msgpack.encode(value), type=schema),
                value,
            )


if __name__ == "__main__":
    unittest.main()
