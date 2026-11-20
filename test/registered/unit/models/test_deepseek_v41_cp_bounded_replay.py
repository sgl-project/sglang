"""CP bounded replay preserves compact global rows through logits and DSpark."""

import argparse
import contextlib
import io
import unittest
from types import SimpleNamespace
from unittest import mock

import torch
from torch.fx.experimental.proxy_tensor import make_fx

from sglang.srt.arg_groups import deepseek_v4_hook
from sglang.srt.arg_groups.overrides import resolution_result
from sglang.srt.environ import envs
from sglang.srt.layers.attention.deepseek_v4_backend import (
    DeepseekV4AttnBackend,
    LateLayerTail,
)
from sglang.srt.layers.cp import base as cp_base
from sglang.srt.layers.cp import interleave as cp_interleave
from sglang.srt.layers.cp.bcg import supports_prefill_cp_bcg
from sglang.srt.layers.cp.bcg.bcg import PrefillCPBCGInput, execute_prefill_cp_bcg
from sglang.srt.layers.cp.interleave import InterleaveContextParallelMetadata
from sglang.srt.layers.moe.utils import MoeA2ABackend
from sglang.srt.model_executor.cuda_graph_config import (
    Backend,
    CudaGraphConfig,
    PhaseConfig,
)
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.model_executor.forward_mode import CaptureHiddenMode, ForwardMode
from sglang.srt.model_executor.runner import eager_runner, prefill_cuda_graph_runner
from sglang.srt.model_executor.runner_utils import capture_mode
from sglang.srt.model_executor.runner_utils.prefill_graph import (
    is_in_prefill_graph_scope,
    prefill_graph_scope,
)
from sglang.srt.models import deepseek_v4, deepseek_v4_nextn, deepseek_v4_replay_tail
from sglang.srt.models.deepseek_common import moe_transport
from sglang.srt.models.deepseek_v4 import DeepseekV4ForCausalLM
from sglang.srt.models.deepseek_v4_mhc import HcState
from sglang.srt.runtime_context import (
    get_context,
    get_exec,
    get_flags,
    get_parallel,
    override_platform,
)
from sglang.srt.server_args import ServerArgs
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


def make_model(capture=True, bounded=True):
    model = DeepseekV4ForCausalLM.__new__(DeepseekV4ForCausalLM)
    torch.nn.Module.__init__(model)
    model.capture_aux_hidden_states = capture
    model.model = SimpleNamespace(late_layer_start=3 if bounded else None)
    model.lm_head = object()
    model.logits_processor = mock.Mock(return_value=SimpleNamespace())
    return model


def make_batch():
    return SimpleNamespace(
        input_ids=torch.arange(512),
        forward_mode=ForwardMode.EXTEND,
        attn_cp_metadata=object(),
        return_logprob=False,
        extend_seq_lens_cpu=[302, 210],
        extend_logprob_start_lens_cpu=[302, 210],
    )


def make_tail(*, cp=True, graph=False):
    indices = torch.tensor([300, 301, 510, 511])
    return LateLayerTail(
        token_indices=torch.tensor([75, 127]) if cp else indices,
        positions=torch.arange(2 if cp else 4),
        extend_seq_lens=torch.tensor([2, 2]),
        extend_seq_lens_cpu=[2, 2],
        swa_out_cache_loc=torch.arange(4),
        cp_metadata=object() if cp else None,
        global_token_indices=indices if cp else None,
        real_rows_count=3 if graph else None,
    )


class TestCpBoundedReplayOutput(CustomTestCase):
    def test_real_model_keeps_dspark_cp_tail_compact(self):
        # The actual model producer must satisfy the compact gather contract.
        for num_tokens in (64, 200, 16384):
            for capture in (False, True):
                with self.subTest(num_tokens=num_tokens, capture=capture):
                    local_tokens = num_tokens // 4
                    tail_tokens = min(num_tokens, 128) // 4
                    local_indices = torch.arange(
                        local_tokens - tail_tokens, local_tokens
                    )
                    markers = torch.arange(
                        1, tail_tokens * 2 + 1, dtype=torch.float32
                    ).view(tail_tokens, 2)
                    tail = LateLayerTail(
                        token_indices=local_indices,
                        positions=local_indices * 4,
                        extend_seq_lens=torch.tensor([tail_tokens * 4]),
                        extend_seq_lens_cpu=[tail_tokens * 4],
                        swa_out_cache_loc=torch.arange(tail_tokens * 4),
                        cp_metadata=object(),
                    )
                    body = deepseek_v4.DeepseekV4Model.__new__(
                        deepseek_v4.DeepseekV4Model
                    )
                    torch.nn.Module.__init__(body)
                    body.pp_group = SimpleNamespace(
                        is_first_rank=False, is_last_rank=True
                    )
                    body.hc_mult, body.hidden_size = 1, 2
                    body.dspark_layers_to_capture = [0] if capture else None
                    body.hc_pre_from_prev_sublayer = True
                    body._can_run_tbo = mock.Mock(return_value=False)
                    body.norm = torch.nn.Identity()

                    def layers(*args):
                        packer = args[-1]
                        if packer is not None:
                            packer.slot(markers).copy_(markers + 5000)
                        return markers[:, None, :], None, tail

                    body._forward_layers_hc_pre_from_prev = layers
                    batch = SimpleNamespace()
                    ids = torch.arange(local_tokens)
                    with (
                        mock.patch.object(deepseek_v4, "_is_cuda", True),
                        mock.patch.object(deepseek_v4, "_is_npu", False),
                        mock.patch.object(
                            deepseek_v4,
                            "get_parallel",
                            return_value=SimpleNamespace(attn_dp_size=1),
                        ),
                        mock.patch.object(deepseek_v4, "get_attn_backend"),
                        mock.patch.object(
                            deepseek_v4, "is_cp_active", return_value=True
                        ),
                        mock.patch(
                            "sglang.kernels.ops.layernorm.mhc.mhc_combine",
                            side_effect=lambda pre, *args: pre,
                        ),
                    ):
                        output = body._forward_impl(
                            ids,
                            ids,
                            batch,
                            None,
                            pp_proxy_tensors={
                                "hidden_states": torch.zeros(local_tokens, 1, 2)
                            },
                        )
                    if capture:
                        (hidden, pre), aux = output
                        torch.testing.assert_close(aux, markers + 5000)
                        expected = markers
                    else:
                        hidden, pre = output
                        expected = markers.new_zeros((local_tokens, 2))
                        expected[local_indices] = markers
                    torch.testing.assert_close(hidden, expected)
                    torch.testing.assert_close(pre, expected)

    def test_compact_global_rows_reach_logits_and_dspark(self):
        model, batch, tail = make_model(), make_batch(), make_tail()
        original_metadata = batch.attn_cp_metadata
        local = (torch.zeros(2, 3), torch.zeros(2, 6))
        aux = torch.zeros(2, 9)
        hidden, pre, global_aux = (torch.arange(4 * n).view(4, n) for n in (3, 6, 9))
        backend = SimpleNamespace(
            tail_forward_metadata=SimpleNamespace(late_layer_tail=tail),
            forward_metadata=SimpleNamespace(prefill_graph_tail=False),
        )
        seen_metadata = []

        def gather(value, fb, stream):
            seen_metadata.append(fb.attn_cp_metadata)
            return (hidden, pre) if isinstance(value, tuple) else global_aux

        with (
            mock.patch.object(deepseek_v4, "get_attn_backend", return_value=backend),
            mock.patch.object(
                deepseek_v4, "cp_gather_after_forward", side_effect=gather
            ),
            mock.patch.object(torch.cuda, "current_stream", return_value=None),
            mock.patch.object(
                deepseek_v4.LogitsMetadata,
                "from_forward_batch",
                return_value=SimpleNamespace(),
            ),
        ):
            output = model.postprocess_cp_output((local, aux), batch)

        args, kwargs = model.logits_processor.call_args
        torch.testing.assert_close(args[0], torch.tensor([300, 301, 510, 511]))
        self.assertIs(args[1], hidden)
        self.assertIs(args[4], global_aux)
        self.assertEqual(args[3].extend_seq_lens_cpu, [2, 2])
        self.assertEqual(args[3].extend_logprob_start_lens_cpu, [2, 2])
        self.assertIsNone(kwargs["hidden_states_before_norm"])
        torch.testing.assert_close(
            output.hidden_states_token_indices, tail.global_token_indices
        )
        self.assertEqual(seen_metadata, [tail.cp_metadata] * 2)
        self.assertIs(batch.attn_cp_metadata, original_metadata)

    def test_collective_failure_restores_full_metadata(self):
        model, batch, tail = make_model(), make_batch(), make_tail()
        original_metadata = batch.attn_cp_metadata
        backend = SimpleNamespace(
            tail_forward_metadata=SimpleNamespace(late_layer_tail=tail),
            forward_metadata=SimpleNamespace(prefill_graph_tail=False),
        )
        with (
            mock.patch.object(deepseek_v4, "get_attn_backend", return_value=backend),
            mock.patch.object(
                deepseek_v4,
                "cp_gather_after_forward",
                side_effect=RuntimeError("collective failed"),
            ),
            mock.patch.object(torch.cuda, "current_stream", return_value=None),
            self.assertRaisesRegex(RuntimeError, "collective failed"),
        ):
            model.postprocess_cp_output(((torch.zeros(2, 3), None), None), batch)
        self.assertIs(batch.attn_cp_metadata, original_metadata)
        model.logits_processor.assert_not_called()

    def test_missing_cp_layout_fails_before_collective(self):
        model, batch, tail = make_model(), make_batch(), make_tail(cp=False)
        original_metadata = batch.attn_cp_metadata
        backend = SimpleNamespace(
            tail_forward_metadata=SimpleNamespace(late_layer_tail=tail),
            forward_metadata=SimpleNamespace(prefill_graph_tail=False),
        )
        with (
            mock.patch.object(deepseek_v4, "get_attn_backend", return_value=backend),
            mock.patch.object(deepseek_v4, "cp_gather_after_forward") as gather,
            self.assertRaisesRegex(RuntimeError, "missing its tail layout"),
        ):
            model.postprocess_cp_output(((torch.zeros(2, 3), None), None), batch)
        gather.assert_not_called()
        self.assertIs(batch.attn_cp_metadata, original_metadata)

    def test_full_cp_output_and_non_dspark_keep_original_metadata(self):
        for capture in (False, True):
            with self.subTest(capture=capture):
                model, batch = make_model(capture=capture, bounded=False), make_batch()
                metadata = batch.attn_cp_metadata
                hidden, pre, aux = (torch.ones(512, n) for n in (3, 6, 9))
                raw = ((hidden, pre), aux) if capture else (hidden, pre)
                with (
                    mock.patch.object(
                        deepseek_v4,
                        "cp_gather_after_forward",
                        side_effect=lambda x, *_: x,
                    ),
                    mock.patch.object(torch.cuda, "current_stream", return_value=None),
                ):
                    output = model.postprocess_cp_output(raw, batch)
                args, kwargs = model.logits_processor.call_args
                self.assertIs(args[0], batch.input_ids)
                self.assertIs(args[3], batch)
                self.assertIs(args[4], aux if capture else None)
                self.assertIs(
                    kwargs["hidden_states_before_norm"], None if capture else pre
                )
                self.assertFalse(hasattr(output, "hidden_states_token_indices"))
                self.assertIs(batch.attn_cp_metadata, metadata)

    def test_non_cp_graph_tail_still_trims_padding(self):
        model, batch, tail = make_model(), make_batch(), make_tail(cp=False, graph=True)
        backend = SimpleNamespace(
            tail_forward_metadata=SimpleNamespace(late_layer_tail=tail),
            forward_metadata=SimpleNamespace(prefill_graph_tail=False),
        )
        raw = ((torch.ones(4, 3), torch.ones(4, 6)), torch.ones(4, 9))
        with (
            mock.patch.object(deepseek_v4, "get_attn_backend", return_value=backend),
            mock.patch.object(
                deepseek_v4.LogitsMetadata,
                "from_forward_batch",
                return_value=SimpleNamespace(),
            ),
        ):
            output = model._postprocess_output(raw, batch.input_ids, batch)
        args, _ = model.logits_processor.call_args
        torch.testing.assert_close(args[0], torch.tensor([300, 301, 510]))
        self.assertEqual(args[1].shape[0], 3)
        self.assertEqual(args[4].shape[0], 3)
        torch.testing.assert_close(
            output.hidden_states_token_indices, tail.token_indices[:3]
        )

    def test_prompt_logprob_offsets_use_compact_lengths(self):
        model, batch, tail = make_model(), make_batch(), make_tail(cp=False)
        batch.return_logprob = True
        batch.extend_logprob_start_lens_cpu = [0, 210]
        backend = SimpleNamespace(
            tail_forward_metadata=SimpleNamespace(late_layer_tail=tail),
            forward_metadata=SimpleNamespace(prefill_graph_tail=False),
        )
        with (
            mock.patch.object(deepseek_v4, "get_attn_backend", return_value=backend),
            mock.patch.object(
                deepseek_v4.LogitsMetadata,
                "from_forward_batch",
                return_value=SimpleNamespace(),
            ),
        ):
            model._postprocess_output(
                ((torch.ones(4, 3), torch.ones(4, 6)), torch.ones(4, 9)),
                batch.input_ids,
                batch,
            )
        metadata = model.logits_processor.call_args.args[3]
        self.assertEqual(metadata.extend_logprob_start_lens_cpu, [0, 2])


class TestEagerCpOutputHook(CustomTestCase):
    def test_eagle_nextn_cp_prefill_with_and_without_bounded_replay(self):
        """EAGLE draft prefill must keep full rows through the inherited CP hook."""
        hidden, pre = (torch.arange(512 * n).view(512, n) for n in (3, 6))
        local = hidden[::4], pre[::4]

        @contextlib.contextmanager
        def shard(*_args):
            yield None, None, None

        def logits(input_ids, states, head, batch, aux, **kwargs):
            self.assertIsNone(aux)
            return SimpleNamespace(
                input_ids=input_ids,
                hidden_states=states,
                pre_hc_head=kwargs["hidden_states_before_norm"],
                forward_batch=batch,
            )

        for bounded in (False, True):
            with (
                self.subTest(bounded=bounded),
                get_context().override_server_args(
                    tp_size=4,
                    attn_cp_size=4,
                    enable_prefill_cp=True,
                    speculative_algorithm="EAGLE",
                    enable_decoder_swa_bounded_replay=bounded,
                ),
                get_parallel().override(pp_group=SimpleNamespace(is_last_rank=True)),
                mock.patch.object(
                    deepseek_v4_nextn,
                    "DeepseekV4ModelNextN",
                    return_value=mock.Mock(spec=[], return_value=local),
                ),
                mock.patch.object(deepseek_v4_nextn, "ParallelLMHead"),
                mock.patch.object(
                    deepseek_v4_nextn, "LogitsProcessor", return_value=logits
                ),
            ):
                model = deepseek_v4_nextn.DeepseekV4ForCausalLMNextN(
                    SimpleNamespace(vocab_size=16, hidden_size=3)
                )
                runner = SimpleNamespace(model_runner=SimpleNamespace(model=model))
                batch = make_batch()
                batch.positions = torch.arange(512)
                metadata = batch.attn_cp_metadata

                def gather(value, fb, stream, metadata=metadata):
                    self.assertIs(fb.attn_cp_metadata, metadata)
                    self.assertEqual(tuple(x.shape[0] for x in value), (128, 128))
                    return hidden, pre

                with (
                    mock.patch.object(eager_runner, "cp_shard_model_inputs", shard),
                    mock.patch.object(
                        deepseek_v4, "cp_gather_after_forward", side_effect=gather
                    ),
                    mock.patch.object(torch.cuda, "current_stream", return_value=None),
                ):
                    output = eager_runner.EagerRunner._execute_extend_cp(
                        runner, batch, {"input_embeds": torch.zeros(512, 3)}
                    )

                torch.testing.assert_close(output.input_ids, torch.arange(512))
                torch.testing.assert_close(output.hidden_states, hidden)
                torch.testing.assert_close(output.pre_hc_head, pre)
                self.assertIs(output.forward_batch, batch)
                self.assertIs(batch.attn_cp_metadata, metadata)

    def test_middle_pp_stage_preserves_proxy_with_aux_capture(self):
        """An intermediate PP stage returns its proxy without output processing."""
        proxy = object()
        model = SimpleNamespace(
            prepare_cp_inputs=lambda *_args, **_kwargs: (None, None, {}),
            model=mock.Mock(spec=[], return_value=proxy),
            pp_group=SimpleNamespace(is_last_rank=False),
            capture_aux_hidden_states=True,
            postprocess_cp_output=mock.Mock(side_effect=AssertionError("PP hook")),
        )
        runner = SimpleNamespace(model_runner=SimpleNamespace(model=model))

        @contextlib.contextmanager
        def shard(*_args):
            yield None, None, None

        with mock.patch.object(eager_runner, "cp_shard_model_inputs", shard):
            output = eager_runner.EagerRunner._execute_extend_cp(
                runner, make_batch(), {}
            )
        self.assertIs(output, proxy)

    def test_model_hook_receives_raw_output_before_generic_gather(self):
        raw, result = object(), object()
        model = SimpleNamespace(
            prepare_cp_inputs=lambda *_args, **_kwargs: (None, None, {}),
            model=mock.Mock(spec=[], return_value=raw),
            pp_group=SimpleNamespace(is_last_rank=True),
            postprocess_cp_output=mock.Mock(return_value=result),
        )
        runner = SimpleNamespace(model_runner=SimpleNamespace(model=model))
        batch = make_batch()

        @contextlib.contextmanager
        def shard(*_args):
            yield None, None, None

        with (
            mock.patch.object(eager_runner, "cp_shard_model_inputs", shard),
            mock.patch.object(eager_runner, "cp_gather_after_forward") as generic,
        ):
            actual = eager_runner.EagerRunner._execute_extend_cp(runner, batch, {})
        self.assertIs(actual, result)
        model.postprocess_cp_output.assert_called_once_with(raw, batch)
        generic.assert_not_called()


class TestCpBoundedReplayConfiguration(CustomTestCase):
    def test_explicit_cp_bcg_stays_enabled_and_other_graphs_are_rejected(self):
        """An explicit opt-in must survive resolution, including PD + MegaMoE."""
        for backend, strategy in (
            (Backend.DISABLED, "interleave"),
            (Backend.BREAKABLE, "interleave"),
            (Backend.FULL, "interleave"),
            (Backend.BREAKABLE, "zigzag"),
        ):
            args = ServerArgs(
                model_path="dummy",
                tp_size=4,
                ep_size=4,
                attn_cp_size=4,
                enable_prefill_cp=True,
                cp_strategy=strategy,
                disaggregation_mode="prefill",
                speculative_algorithm="DSPARK",
                moe_a2a_backend="megamoe",
                enable_decoder_swa_bounded_replay=True,
                cuda_graph_config=CudaGraphConfig(prefill=PhaseConfig(backend=backend)),
            )
            with (
                self.subTest(backend=backend, strategy=strategy),
                override_platform(is_cuda=True, is_hip=False),
                mock.patch.object(
                    deepseek_v4_hook,
                    "model_config_of",
                    return_value=SimpleNamespace(
                        hf_config=SimpleNamespace(model_type="deepseek_v41")
                    ),
                ),
                mock.patch(
                    "sglang.srt.layers.attention.dsv4.unified_kv_gate.is_unified_kv_triton",
                    return_value=False,
                ),
            ):
                if backend == Backend.FULL or strategy != "interleave":
                    with self.assertRaises(ValueError):
                        deepseek_v4_hook.validate_deepseek_v41_features(args)
                else:
                    deepseek_v4_hook.validate_deepseek_v41_features(args)
                    self.assertIs(
                        resolution_result(args, "enable_decoder_swa_bounded_replay"),
                        True,
                    )


class TestCpBoundedBCGBridge(CustomTestCase):
    def _backend_and_batch(self, lengths, rank):
        total = sum(lengths)
        prefixes = [17 + i * 131 for i in range(len(lengths))]
        seq_lens = [p + n for p, n in zip(prefixes, lengths)]
        metadata = InterleaveContextParallelMetadata(
            per_rank_actual_token=[(total + 3) // 4] * 4,
            max_rank_len=[(total + 3) // 4] * 4,
            total_seq_lens=total,
            bs=len(lengths),
        )
        positions = torch.cat(
            [torch.arange(p, p + n) for p, n in zip(prefixes, lengths)]
        )
        batch = ForwardBatch(
            input_ids=torch.arange(total),
            seq_lens_sum=sum(seq_lens),
            batch_size=len(lengths) + 4,
            positions=positions,
            # Deliberately pad request tensors past the live CPU lengths.
            out_cache_loc=torch.arange(1, total + 1),
            req_pool_indices=torch.tensor([5, 11, 19][: len(lengths)] + [0] * 4),
            seq_lens=torch.tensor(seq_lens + [0] * 4),
            seq_lens_cpu=torch.tensor(seq_lens + [0] * 4),
            extend_seq_lens_cpu=lengths,
            return_logprob=False,
            capture_hidden_mode=CaptureHiddenMode.FULL,
            forward_mode=ForwardMode.EXTEND,
            attn_cp_metadata=metadata,
            global_num_token_non_padded_cpu=total,
            global_num_token_non_padded=None,
            num_token_non_padded=None,
        )
        backend = DeepseekV4AttnBackend.__new__(DeepseekV4AttnBackend)
        backend.model_runner = SimpleNamespace(
            spec_algorithm=SimpleNamespace(is_dspark=lambda: True)
        )
        backend.encoder_row_floor = None
        backend.low_ratios = (1, 2)
        backend.forward_metadata = SimpleNamespace(
            low_ratio_k_slots=None, prefill_graph_tail=True
        )
        backend.token_to_kv_pool = SimpleNamespace(
            request_window=None,
            translate_loc_from_full_to_swa=lambda locations: locations + 100,
        )
        backend.tail_source_kv_layer_id = None

        def init_metadata(**kwargs):
            self.assertEqual(kwargs["req_pool_indices"].numel(), len(lengths))
            self.assertEqual(kwargs["seq_lens"].numel(), len(lengths))
            self.assertEqual(len(kwargs["seq_lens_cpu"]), len(lengths))
            return SimpleNamespace(
                core_attn_metadata=SimpleNamespace(),
                low_ratio_k_slots=None,
                low_ratio_req_indices=None,
                low_ratio_pos_i64=None,
            )

        # Boundary: production metadata initialization launches CUDA kernels.
        backend.init_forward_metadata_prefill = init_metadata
        backend.tail_forward_metadata = backend._prefill_graph_tail_metadata(
            batch, max(seq_lens)
        )
        return backend, batch

    @contextlib.contextmanager
    def _cp_rank(self, rank):
        with (
            get_context().override_server_args(
                tp_size=4,
                attn_cp_size=4,
                enable_prefill_cp=True,
                cp_strategy="interleave",
                enable_decoder_swa_bounded_replay=True,
            ),
            get_parallel().override(attn_cp_rank=rank),
            get_flags().moe.override(a2a_backend=MoeA2ABackend.MEGAMOE),
            mock.patch.object(cp_base, "_STRATEGY", None),
        ):
            yield

    def test_graph_tail_ownership_uses_original_extend_with_padded_requests(self):
        """Compact-tail interleaving is wrong when request boundaries are uneven."""
        lengths = [259, 131, 2]
        expected = list(range(131, 259)) + list(range(262, 390)) + [390, 391]
        rank_rows = []
        for rank in range(4):
            with self._cp_rank(rank):
                backend, _batch = self._backend_and_batch(lengths, rank)
                tail = backend.tail_forward_metadata.late_layer_tail
                torch.testing.assert_close(
                    tail.global_token_indices, torch.tensor(expected)
                )
                local_expected = [index // 4 for index in expected if index % 4 == rank]
                torch.testing.assert_close(
                    tail.token_indices, torch.tensor(local_expected)
                )
                local = torch.arange(rank, sum(lengths), 4)
                rank_rows.append(tail.rows(local))
                if rank == 0:
                    gather_index = tail.cp_metadata.gather_index
        torch.testing.assert_close(
            torch.cat(rank_rows)[gather_index], torch.tensor(expected)
        )

    def test_bridge_replay_updates_rows_without_changing_output_storage(self):
        """A second request layout in one bucket must clear stale tail rows."""
        with self._cp_rank(2):
            owner = deepseek_v4_replay_tail.DecoderReplayTail.__new__(
                deepseek_v4_replay_tail.DecoderReplayTail
            )
            owner._bridge_buffers = {}
            owner._model = SimpleNamespace(late_layer_vl_routing=True)
            owner._break_context = lambda batch: contextlib.nullcontext()
            # These cases exercise bridge/fallback semantics, not CUDA dispatch.
            owner.plan = mock.Mock(return_value=(False, None))
            bridge = None
            pointers = None
            for lengths, moe in (
                ([259, 131, 2], MoeA2ABackend.MEGAMOE),
                ([130, 260, 2], MoeA2ABackend.NONE),
            ):
                backend, batch = self._backend_and_batch(lengths, 2)
                full_metadata, cp_metadata = (
                    backend.forward_metadata,
                    batch.attn_cp_metadata,
                )
                tail = backend.tail_forward_metadata.late_layer_tail
                local = torch.arange(100, dtype=torch.float32).view(100, 1, 1)
                ids = torch.arange(2, 402, 4)

                def layers(state, *, tail=tail, backend=backend, moe=moe, **kwargs):
                    state = state.residual[:, 0]
                    self.assertEqual(state.shape[0], tail.positions.shape[0])
                    self.assertEqual(backend.tail_source_kv_layer_id, 12)
                    if moe == MoeA2ABackend.NONE:
                        indices = tail.global_token_indices.tolist()
                        cap = max(tail.cp_metadata.per_rank_actual_token)
                        expected_ids = []
                        for rank in range(4):
                            owned = [i for i in indices if i % 4 == rank]
                            expected_ids.extend(owned + [0] * (cap - len(owned)))
                        torch.testing.assert_close(
                            kwargs["input_ids_global"], torch.tensor(expected_ids)
                        )
                    return state + 1000, state + 2000, [state + 3000]

                def gather(local_ids):
                    indices = tail.global_token_indices.tolist()
                    cap = max(tail.cp_metadata.per_rank_actual_token)
                    expected = []
                    for rank in range(4):
                        owned = [i for i in indices if i % 4 == rank]
                        expected.extend(owned + [0] * (cap - len(owned)))
                    expected = torch.tensor(expected).reshape(4, cap, 1)
                    torch.testing.assert_close(local_ids, expected[2])
                    return expected.flatten(0, 1)

                owner._layer_stack = layers
                with (
                    mock.patch.object(
                        deepseek_v4_replay_tail,
                        "attn_cp_interleave_gather",
                        side_effect=gather,
                    ),
                    get_flags().moe.override(a2a_backend=moe),
                    mock.patch.object(
                        deepseek_v4_replay_tail,
                        "get_attn_backend",
                        return_value=backend,
                    ),
                    mock.patch.object(
                        deepseek_v4_replay_tail,
                        "get_tc_piecewise_forward_context",
                        return_value=SimpleNamespace(forward_batch=batch),
                    ),
                ):
                    deepseek_v4_replay_tail._cp_tail_layers_in_prefill_graph(
                        owner,
                        lambda values: HcState(values[0]),
                        True,
                        12,
                        (True, False),
                        bridge,
                        [local],
                        ids,
                    )
                bridge = owner._bridge((True, False), 100)
                current = [value.data_ptr() for value in bridge]
                if pointers is not None:
                    self.assertEqual(current, pointers)
                pointers = current
                expected = torch.zeros(100, 1)
                for value, offset in zip(bridge, (1000, 2000, 3000)):
                    expected.zero_()
                    expected[tail.token_indices, 0] = (
                        tail.token_indices.to(expected.dtype) + offset
                    )
                    torch.testing.assert_close(value, expected)
                self.assertIs(backend.forward_metadata, full_metadata)
                self.assertIs(batch.attn_cp_metadata, cp_metadata)
                self.assertEqual(batch.global_num_token_non_padded_cpu, sum(lengths))
                self.assertIsNone(backend.tail_source_kv_layer_id)

    def test_bridge_failure_restores_full_metadata(self):
        with self._cp_rank(1):
            backend, batch = self._backend_and_batch([259, 131, 2], 1)
            full_metadata, full_cp = backend.forward_metadata, batch.attn_cp_metadata
            owner = deepseek_v4_replay_tail.DecoderReplayTail.__new__(
                deepseek_v4_replay_tail.DecoderReplayTail
            )
            owner._run_layers_eagerly = mock.Mock(
                side_effect=RuntimeError("tail failure")
            )
            owner.plan = mock.Mock(return_value=(False, None))
            with (
                mock.patch.object(
                    deepseek_v4_replay_tail, "get_attn_backend", return_value=backend
                ),
                mock.patch.object(
                    deepseek_v4_replay_tail,
                    "get_tc_piecewise_forward_context",
                    return_value=SimpleNamespace(forward_batch=batch),
                ),
                self.assertRaisesRegex(RuntimeError, "tail failure"),
            ):
                deepseek_v4_replay_tail._cp_tail_layers_in_prefill_graph(
                    owner,
                    lambda values: HcState(values[0]),
                    True,
                    12,
                    (True, False),
                    None,
                    [torch.zeros(100, 1, 2)],
                    torch.arange(100),
                )
            self.assertIs(backend.forward_metadata, full_metadata)
            self.assertIs(batch.attn_cp_metadata, full_cp)
            self.assertIsNone(backend.tail_source_kv_layer_id)

    def test_graph_output_gathers_full_shards_then_compacts_all_dspark_rows(self):
        model, batch, tail = make_model(), make_batch(), make_tail()
        metadata = batch.attn_cp_metadata
        backend = SimpleNamespace(
            tail_forward_metadata=SimpleNamespace(late_layer_tail=tail),
            forward_metadata=SimpleNamespace(prefill_graph_tail=True),
        )
        full_hidden = torch.arange(512 * 3).view(512, 3)
        full_pre = torch.arange(512 * 6).view(512, 6)
        full_aux = torch.arange(512 * 9).view(512, 9)

        def gather(value, fb, _stream):
            self.assertIs(fb.attn_cp_metadata, metadata)
            return (full_hidden, full_pre) if isinstance(value, tuple) else full_aux

        with (
            mock.patch.object(deepseek_v4, "get_attn_backend", return_value=backend),
            mock.patch.object(
                deepseek_v4, "cp_gather_after_forward", side_effect=gather
            ),
            mock.patch.object(torch.cuda, "current_stream", return_value=None),
            mock.patch.object(
                deepseek_v4.LogitsMetadata,
                "from_forward_batch",
                return_value=SimpleNamespace(),
            ),
        ):
            output = model.postprocess_cp_output(
                ((torch.zeros(128, 3), torch.zeros(128, 6)), torch.zeros(128, 9)), batch
            )
        args, _kwargs = model.logits_processor.call_args
        indices = tail.global_token_indices
        torch.testing.assert_close(args[0], indices)
        torch.testing.assert_close(args[1], full_hidden[indices])
        torch.testing.assert_close(args[4], full_aux[indices])
        torch.testing.assert_close(output.hidden_states_token_indices, indices)

    @contextlib.contextmanager
    def _tail_status_fixture(self, lengths):
        with (
            self._cp_rank(1),
            get_flags().dp.override(capturing_prefill_graph=False),
            mock.patch.object(
                deepseek_v4_replay_tail,
                "get_global_expert_distribution_recorder",
                return_value=SimpleNamespace(recording=False),
            ),
        ):
            owner = deepseek_v4_replay_tail.DecoderReplayTail(
                model=None, layer_ids=range(1), row_step=32, max_rows=512
            )
            owner.runner._warm = True
            backend, batch = self._backend_and_batch(lengths, 1)
            backend.prefill_graph_replay_batch = batch
            replay = SimpleNamespace(succeeded=True, batch=batch)

            def layers(**kwargs):
                return kwargs["state"].residual, None, []

            def run(**kwargs):
                # CUDA execution boundary: a planned graph can still miss its key.
                owner._last_run_replayed = replay.succeeded
                return layers(**kwargs)

            def execute():
                deepseek_v4_replay_tail._cp_tail_layers_in_prefill_graph(
                    owner,
                    lambda values: HcState(values[0]),
                    False,
                    12,
                    (False, False),
                    owner._bridge((False, False), 100),
                    [torch.zeros(100, 1, 2)],
                    torch.arange(100),
                )

            with (
                mock.patch.object(owner, "run", side_effect=run),
                mock.patch.object(owner, "_run_layers_eagerly", side_effect=layers),
                mock.patch.object(
                    deepseek_v4_replay_tail, "get_attn_backend", return_value=backend
                ),
                mock.patch.object(
                    deepseek_v4_replay_tail,
                    "get_tc_piecewise_forward_context",
                    return_value=SimpleNamespace(forward_batch=batch),
                ),
            ):
                yield owner, replay, execute

    def test_tail_status_tracks_each_batch_without_separate_messages(self):
        """A replay, then a miss, must report fresh status without standalone logs."""
        for lengths in ([259, 131, 2], [4, 8, 2]):
            with (
                self.subTest(lengths=lengths),
                self._tail_status_fixture(lengths) as (owner, replay, execute),
            ):
                for succeeded in (True, False, True):
                    replay.succeeded = succeeded
                    with self.assertNoLogs(
                        deepseek_v4_replay_tail.logger, level="INFO"
                    ):
                        execute()
                    self.assertIs(replay.batch.tail_cuda_graph_replayed, succeeded)
                # A subsequent oversized/disabled bucket must clear a prior hit.
                owner.runner.max_rows = 0
                with self.assertNoLogs(deepseek_v4_replay_tail.logger, level="INFO"):
                    execute()
                self.assertIs(replay.batch.tail_cuda_graph_replayed, False)

    def test_prefill_runner_returns_static_tail_status_without_stale_hits(self):
        """BCG's private batch must report to the caller and clear the next miss."""
        module = prefill_cuda_graph_runner
        runner = module.PrefillCudaGraphRunner.__new__(module.PrefillCudaGraphRunner)
        static = SimpleNamespace(
            input_ids=torch.arange(4), tail_cuda_graph_replayed=True
        )
        runner.raw_num_tokens = 4
        runner.enable_cp_bcg_capture = True
        runner.backend = SimpleNamespace(replay_session=contextlib.nullcontext)
        runner.model_runner = None
        runner.device_module = None
        output = object()
        calls = []

        def execute(*args, **kwargs):
            if not calls:
                static.tail_cuda_graph_replayed = True
            calls.append(True)
            return output

        with (
            mock.patch.object(runner, "_validate_capture_hidden_mode"),
            mock.patch.object(
                runner, "model_context", side_effect=contextlib.nullcontext
            ),
            mock.patch.object(runner, "load_batch", return_value=(static, None)),
            mock.patch.object(
                runner, "_shape_key", return_value=SimpleNamespace(variant_label=None)
            ),
            mock.patch.object(runner, "_wait_for_layer_loads"),
            mock.patch.object(
                runner, "_finalize_execute_output", side_effect=lambda value: value
            ),
            mock.patch.object(module, "maybe_publish_prefill_shared_read_done"),
            mock.patch.object(module, "execute_prefill_cp_bcg", side_effect=execute),
        ):
            first = SimpleNamespace(tail_cuda_graph_replayed=False)
            second = SimpleNamespace(tail_cuda_graph_replayed=False)
            self.assertIs(runner.execute(first), output)
            self.assertIs(first.tail_cuda_graph_replayed, True)
            self.assertIs(runner.execute(second), output)
            self.assertIs(second.tail_cuda_graph_replayed, False)
            self.assertIs(first.tail_cuda_graph_replayed, True)

    def test_bounded_bcg_selects_cp_adapter_and_prepares_capture_metadata(self):
        """An enabled bounded tail must not bypass CP input/metadata preparation."""
        with self._cp_rank(0), override_platform(is_cuda=True, is_hip=False):
            args = ServerArgs(
                model_path="dummy",
                tp_size=4,
                ep_size=4,
                attn_cp_size=4,
                attention_backend="dsv4",
                enable_prefill_cp=True,
                cp_strategy="interleave",
                enable_decoder_swa_bounded_replay=True,
                cuda_graph_config=CudaGraphConfig(
                    prefill=PhaseConfig(backend=Backend.BREAKABLE)
                ),
            )
            self.assertTrue(supports_prefill_cp_bcg(args))
            for attention, strategy in (
                ("dsa", "interleave"),
                ("trtllm_mha", "zigzag"),
            ):
                unsupported = ServerArgs(
                    model_path="dummy",
                    tp_size=4,
                    attn_cp_size=4,
                    attention_backend=attention,
                    enable_prefill_cp=True,
                    cp_strategy=strategy,
                    enable_decoder_swa_bounded_replay=True,
                )
                self.assertFalse(supports_prefill_cp_bcg(unsupported))
            embedding = torch.nn.Embedding(32, 4)
            runner = SimpleNamespace(
                device="cpu",
                max_num_tokens=16,
                model_runner=SimpleNamespace(
                    server_args=args,
                    dtype=torch.float32,
                    model_config=SimpleNamespace(hidden_size=4),
                    model=SimpleNamespace(get_input_embeddings=lambda: embedding),
                ),
            )
            adapter = PrefillCPBCGInput.create(runner)
            batch = ForwardBatch(
                forward_mode=ForwardMode.EXTEND,
                batch_size=1,
                input_ids=torch.arange(12),
                positions=torch.arange(12),
                req_pool_indices=torch.tensor([1]),
                seq_lens=torch.tensor([12]),
                seq_lens_cpu=torch.tensor([12]),
                seq_lens_sum=12,
                out_cache_loc=torch.arange(1, 13),
                extend_num_tokens=12,
                extend_seq_lens=torch.tensor([12]),
                extend_seq_lens_cpu=[12],
                extend_prefix_lens_cpu=[0],
                extend_start_loc=torch.tensor([0]),
                num_token_non_padded=torch.tensor(12, dtype=torch.int32),
            )
            adapter.prepare(runner, batch, static_num_tokens=12, capture=True)
            self.assertIsInstance(
                batch.attn_cp_metadata, InterleaveContextParallelMetadata
            )
            torch.testing.assert_close(batch.cp_input_ids[:3], torch.tensor([0, 4, 8]))
            torch.testing.assert_close(batch.cp_positions[:3], torch.tensor([0, 4, 8]))
            torch.testing.assert_close(batch.input_ids, torch.arange(12))
            self.assertGreaterEqual(adapter.bucket_local_tokens[12], 3)

    def test_outer_cp_break_selects_tail_graph_and_preserves_bridge_storage(self):
        """The restored outer BCG must dispatch into the new tail runner."""
        with self._cp_rank(1):
            owner = deepseek_v4_replay_tail.DecoderReplayTail(
                model=SimpleNamespace(late_layer_vl_routing=False),
                layer_ids=range(20, 40),
                row_step=32,
                max_rows=512,
            )
            owner.runner._warm = True

            def layers(offset):
                def run(**kwargs):
                    rows = kwargs["state"].residual[:, 0]
                    return rows + offset, rows + offset + 1000, [rows + offset + 2000]

                return run

            owner.run = layers(5000)
            owner._run_layers_eagerly = layers(1000)
            bridge = None
            pointer = None
            for capture, borrow, offset in (
                (False, False, 5000),
                (True, False, 1000),
                (False, True, 1000),
            ):
                with (
                    self.subTest(capture=capture, borrow=borrow),
                    get_flags().dp.override(capturing_prefill_graph=capture),
                    envs.SGLANG_ENABLE_GRAPH_POOL_BORROW.override(borrow),
                ):
                    backend, batch = self._backend_and_batch([259, 131, 2], 1)
                    full_cp = batch.attn_cp_metadata
                    full_metadata = backend.forward_metadata
                    tail = backend.tail_forward_metadata.late_layer_tail
                    local = torch.arange(100).float().view(100, 1, 1)
                    with (
                        mock.patch.object(
                            deepseek_v4_replay_tail,
                            "get_attn_backend",
                            return_value=backend,
                        ),
                        mock.patch.object(
                            deepseek_v4_replay_tail,
                            "get_tc_piecewise_forward_context",
                            return_value=SimpleNamespace(forward_batch=batch),
                        ),
                        deepseek_v4_replay_tail.enable_breakable_cuda_graph(),
                        mock.patch.object(
                            deepseek_v4_replay_tail,
                            "get_global_expert_distribution_recorder",
                            return_value=SimpleNamespace(recording=False),
                        ),
                    ):
                        deepseek_v4_replay_tail._cp_tail_layers_in_prefill_graph(
                            owner,
                            lambda values: HcState(values[0]),
                            True,
                            20,
                            (True, False),
                            bridge,
                            [local],
                            torch.arange(100),
                        )
                    bridge = owner._bridge((True, False), 100)
                    if pointer is not None:
                        self.assertEqual(bridge[0].data_ptr(), pointer)
                    pointer = bridge[0].data_ptr()
                    for value, delta in zip(
                        bridge, (offset, offset + 1000, offset + 2000)
                    ):
                        expected = torch.zeros(100, 1)
                        expected[tail.token_indices, 0] = (
                            tail.token_indices.float() + delta
                        )
                        torch.testing.assert_close(value, expected)
                    self.assertIs(batch.attn_cp_metadata, full_cp)
                    self.assertIs(backend.forward_metadata, full_metadata)
                    self.assertIsNone(backend.tail_source_kv_layer_id)

    def test_active_captures_reject_nested_cp_tail_graphs(self):
        """Live BCG replay may nest a tail graph; actual captures must not."""
        with self._cp_rank(0):
            backend, batch = self._backend_and_batch([259, 131, 2], 0)
            tail = backend.tail_forward_metadata.late_layer_tail
            owner = deepseek_v4_replay_tail.DecoderReplayTail(
                model=None, layer_ids=range(20, 40), row_step=32, max_rows=512
            )
            owner.runner._warm = True
            for mode in ("model", "breakable", "model_in_breakable"):
                with self.subTest(mode=mode), contextlib.ExitStack() as stack:
                    if mode != "model":
                        stack.enter_context(
                            deepseek_v4_replay_tail.enable_breakable_cuda_graph()
                        )
                    if mode != "breakable":
                        stack.enter_context(capture_mode.model_capture_mode())
                    else:
                        # Set the real capture marker without launching CUDA.
                        bcg = deepseek_v4_replay_tail.bcg_module
                        capture = bcg.BreakableCUDAGraphCapture(
                            bcg.BreakableCUDAGraph()
                        )
                        token = bcg._current_capture_var.set(capture)
                        stack.callback(bcg._current_capture_var.reset, token)
                    use_graph, counts = owner.plan(
                        tail,
                        forward_batch=batch,
                        num_tokens=4096,
                        tail_in_graph=False,
                    )
                    self.assertFalse(use_graph)
                    self.assertIsNone(counts)

    def test_outer_capture_does_not_warm_live_cp_tail_graphs(self):
        """Dummy capture buckets must not trigger row-graph capture on health probes."""
        with self._cp_rank(0):
            backend, batch = self._backend_and_batch([259, 131, 2], 0)
            tail = backend.tail_forward_metadata.late_layer_tail
            owner = deepseek_v4_replay_tail.DecoderReplayTail(
                model=None, layer_ids=range(20, 40), row_step=32, max_rows=512
            )
            with (
                get_flags().dp.override(capturing_prefill_graph=True),
                deepseek_v4_replay_tail.enable_breakable_cuda_graph(),
            ):
                use_graph, counts = owner.plan(
                    tail, forward_batch=batch, num_tokens=4096, tail_in_graph=False
                )
            self.assertFalse(use_graph)
            self.assertIsNone(counts)
            self.assertFalse(owner.runner.warm)

    def test_cp_bridge_remains_available_when_row_graphs_are_disabled(self):
        with (
            self._cp_rank(0),
            get_exec().graph.override(
                enable_decoder_tail_cuda_graph=True,
                cuda_graph_config=CudaGraphConfig(
                    prefill=PhaseConfig(backend=Backend.BREAKABLE)
                ),
            ),
            envs.SGLANG_DSV4_DECODER_TAIL_GRAPH_MAX_ROWS.override(0),
            mock.patch.object(deepseek_v4_replay_tail, "is_cuda", return_value=True),
        ):
            model = SimpleNamespace(
                tail_writes_source_kv=True, late_layer_start=2, end_layer=4
            )
            owner = deepseek_v4_replay_tail.DecoderReplayTail.maybe_create(model)
            self.assertIsNotNone(owner)
            self.assertEqual(list(owner._layer_ids), [2, 3])
            self.assertIsNone(owner.runner.bucket_rows(1))

    def test_bcg_output_uses_static_cp_geometry_with_live_logits_rows(self):
        """The raw batch has no CP metadata; a larger bucket must not pad logits."""
        with (
            self._cp_rank(0),
            get_parallel().override(attn_cp_group=SimpleNamespace(world_size=4)),
        ):
            for capture in (False, True):
                for bounded in (False, True):
                    with self.subTest(capture=capture, bounded=bounded):
                        model = make_model(capture=capture, bounded=bounded)
                        model.pp_group = SimpleNamespace(is_last_rank=True)
                        raw = SimpleNamespace(
                            input_ids=torch.arange(9),
                            positions=torch.arange(9),
                            forward_mode=ForwardMode.EXTEND,
                            attn_cp_metadata=None,
                            return_logprob=False,
                            extend_seq_lens_cpu=[9],
                            extend_logprob_start_lens_cpu=[9],
                        )
                        geometry = InterleaveContextParallelMetadata(
                            per_rank_actual_token=[4] * 4,
                            max_rank_len=[4] * 4,
                            per_rank_logical_token=[4] * 4,
                            total_seq_lens=16,
                            bs=1,
                        )
                        static = SimpleNamespace(attn_cp_metadata=geometry)
                        tail = LateLayerTail(
                            token_indices=torch.tensor([2]),
                            positions=torch.tensor([8]),
                            extend_seq_lens=torch.tensor([4]),
                            extend_seq_lens_cpu=[4],
                            swa_out_cache_loc=torch.arange(4),
                            cp_metadata=object(),
                            global_token_indices=torch.arange(5, 9),
                        )
                        backend = SimpleNamespace(
                            forward_metadata=SimpleNamespace(prefill_graph_tail=True),
                            tail_forward_metadata=SimpleNamespace(late_layer_tail=tail),
                        )
                        local = (torch.zeros(4, 3), torch.zeros(4, 6))
                        body_output = (local, torch.zeros(4, 9)) if capture else local
                        runner = SimpleNamespace(
                            prefill_cp_bcg_input=SimpleNamespace(live_local_tokens=4),
                            model_runner=SimpleNamespace(model=model),
                            backend=SimpleNamespace(
                                replay=lambda *_a, body_output=body_output, **_kw: (
                                    body_output
                                )
                            ),
                            _prefill_forward_context=lambda *_a, **_kw: (
                                contextlib.nullcontext()
                            ),
                        )

                        def all_gather(output, shard):
                            self.assertEqual(shard.shape[0], 4)
                            width = output.shape[1]
                            global_rows = torch.arange(
                                16 * width, dtype=output.dtype
                            ).view(16, width)
                            # The communication boundary returns rank-major shards.
                            output.copy_(
                                global_rows.view(4, 4, width)
                                .transpose(0, 1)
                                .reshape(16, width)
                            )

                        with (
                            mock.patch.object(
                                cp_interleave,
                                "attn_cp_all_gather_into_tensor",
                                side_effect=all_gather,
                            ),
                            mock.patch.object(
                                cp_interleave,
                                "is_allocation_symmetric",
                                return_value=False,
                            ),
                            mock.patch.object(
                                deepseek_v4, "get_attn_backend", return_value=backend
                            ),
                            mock.patch.object(
                                torch.cuda, "current_stream", return_value=None
                            ),
                            mock.patch.object(
                                deepseek_v4.LogitsMetadata,
                                "from_forward_batch",
                                return_value=SimpleNamespace(),
                            ),
                        ):
                            output = execute_prefill_cp_bcg(runner, raw, static, 16, 9)
                        args = model.logits_processor.call_args.args
                        indices = (
                            torch.arange(5, 9)
                            if capture and bounded
                            else torch.arange(9)
                        )
                        torch.testing.assert_close(args[0], indices)
                        expected_hidden = torch.arange(
                            16 * 3, dtype=torch.float32
                        ).view(16, 3)
                        torch.testing.assert_close(args[1], expected_hidden[indices])
                        if capture:
                            expected_aux = torch.arange(
                                16 * 9, dtype=torch.float32
                            ).view(16, 9)
                            torch.testing.assert_close(args[4], expected_aux[indices])
                        if capture and bounded:
                            torch.testing.assert_close(
                                output.hidden_states_token_indices, indices
                            )
                        self.assertIsNone(raw.attn_cp_metadata)
                        self.assertIs(static.attn_cp_metadata, geometry)

    def test_bcg_delegates_model_output_before_full_row_trimming(self):
        raw = torch.arange(128)
        batch = make_batch()
        model = SimpleNamespace(
            pp_group=SimpleNamespace(is_last_rank=True),
            postprocess_cp_output=lambda value, fb: (
                value[-1].item(),
                len(fb.input_ids),
            ),
        )
        runner = SimpleNamespace(
            model_runner=SimpleNamespace(model=model),
            prefill_cp_bcg_input=SimpleNamespace(live_local_tokens=2),
            backend=SimpleNamespace(replay=lambda *_args, **_kwargs: raw),
            _prefill_forward_context=lambda *_args, **_kwargs: contextlib.nullcontext(),
        )
        output = execute_prefill_cp_bcg(runner, batch, batch, 512, 512)
        self.assertEqual(output, (127, 512))


class TestCPTailKVStore(CustomTestCase):
    def test_cp_store_uses_live_global_positions_and_cache_locations(self):
        """Tail-graph KV writes must gather before applying global RoPE/store."""
        for use_tail in (False, True):
            with self.subTest(tail=use_tail):
                positions = torch.arange(2)
                global_positions = torch.arange(
                    500 if use_tail else 100, 508 if use_tail else 108
                )
                qkv = torch.arange(8).reshape(2, 4).float()
                projected = qkv[:, 2:]
                gathered = torch.cat([projected + rank * 10 for rank in range(4)])
                cache = torch.full((32,), -1.0)
                locations = torch.arange(12, 20)
                tail = (
                    SimpleNamespace(pos_global=global_positions) if use_tail else None
                )
                backend = SimpleNamespace(
                    forward_metadata=SimpleNamespace(late_layer_tail=tail),
                    get_swa_out_cache_loc=lambda batch, locations=locations: locations,
                )
                attention = SimpleNamespace(
                    is_dsv41=True,
                    q_lora_rank=2,
                    layer_id=20,
                    kv_norm=SimpleNamespace(weight=SimpleNamespace(data=torch.ones(2))),
                    eps=1e-6,
                    freqs_cis=torch.ones(1024, 1),
                )

                def gather(
                    kv, batch, stream, *, projected=projected, gathered=gathered
                ):
                    torch.testing.assert_close(kv, projected)
                    return gathered

                def store(*, cache=cache, **kwargs):
                    cache[kwargs["swa_loc"]] = kwargs["kv"][:, 0] + kwargs["positions"]

                batch = SimpleNamespace(
                    positions=torch.arange(100, 108), attn_cp_metadata=object()
                )
                with (
                    mock.patch.object(deepseek_v4, "is_cp_active", return_value=True),
                    mock.patch.object(
                        deepseek_v4,
                        "cp_materialize_global_token_order",
                        side_effect=gather,
                    ),
                    mock.patch.object(
                        deepseek_v4,
                        "get_token_to_kv_pool",
                        return_value=SimpleNamespace(
                            set_swa_key_buffer_radix_fused_norm_rope=store
                        ),
                    ),
                    mock.patch.object(torch.cuda, "current_stream", return_value=None),
                ):
                    deepseek_v4.MQALayer._compute_kv_to_cache(
                        attention,
                        torch.zeros(2, 2),
                        positions,
                        batch,
                        backend,
                        qkv_a=qkv,
                    )
                expected = torch.full_like(cache, -1)
                expected[locations] = gathered[:, 0] + global_positions
                torch.testing.assert_close(cache, expected)


class TestTailGraphStartup(CustomTestCase):
    """Serving must replay startup graphs without recording new CUDA work."""

    @contextlib.contextmanager
    def _fixture(self, cp_size, chunk=16384, moe_backend=MoeA2ABackend.MEGAMOE):
        from sglang.srt.mem_cache.memory_pool import ReqToTokenPool
        from sglang.srt.model_executor.forward_context import (
            ForwardContext,
            forward_context,
            get_forward_context,
        )
        from sglang.srt.model_executor.runner.row_bucket_graph_runner import _Graph
        from sglang.srt.models import deepseek_v4_tail_graph_startup as startup

        pool = ReqToTokenPool(16, 16384, "cpu", False)
        pool.req_to_token.fill_(-77)
        allocations, frees, records, calls = [], [], [], []
        next_page = 256

        def alloc_extend(**kw):
            nonlocal next_page
            slots = []
            for length in kw["seq_lens_cpu"].tolist():
                slots.append(torch.arange(next_page, next_page + length))
                next_page += ((length + 255) // 256) * 256
            value = torch.cat(slots)
            self.assertEqual(value.unique().numel(), value.numel())
            allocations.append(value)
            return value

        allocator = SimpleNamespace(
            alloc_extend=alloc_extend,
            free=lambda slots: frees.append(slots),
        )
        backend = SimpleNamespace(
            forward_metadata=object(),
            tail_forward_metadata=object(),
            prefill_graph_replay_batch=None,
        )
        owner = deepseek_v4_replay_tail.DecoderReplayTail(
            model=SimpleNamespace(late_layer_vl_routing=False),
            layer_ids=range(20, 40),
            row_step=128 // cp_size,
            max_rows=2048 // cp_size,
        )
        model_runner = SimpleNamespace(
            device="cpu",
            req_to_token_pool=pool,
            token_to_kv_pool_allocator=allocator,
            max_running_requests=16,
            model_config=SimpleNamespace(context_len=16384),
            spec_algorithm=None,
            is_draft_worker=False,
            attn_backend=backend,
            runner_ids=SimpleNamespace(bind_runner_slots=mock.Mock()),
            _prepare_eager_forward_batch=lambda batch: None,
        )
        device = SimpleNamespace(
            synchronize=mock.Mock(),
            mem_get_info=lambda: (2**30, 2**31),
        )
        arithmetic = []

        def layers(state, **kwargs):
            x = state.streams[:, 0]
            count = get_forward_context().moe_num_token_non_padded()
            if moe_backend.is_none():
                self.assertIsNone(count)
                count = kwargs["forward_batch"].num_token_non_padded
            bias = 1000 if is_in_prefill_graph_scope() else 0
            arithmetic.append((x, count, bias))
            return x + count + bias, None, []

        def record(body, rows, count, capture_batch, start, free_before):
            outputs = body()
            x, bound, bias = arithmetic[-1]
            graph = SimpleNamespace(replay=lambda: outputs[0].copy_(x + bound + bias))
            records.append((rows, bound.data_ptr()))
            return _Graph(graph, count, outputs, (capture_batch,))

        def make_inputs(batch):
            global_rows = sum(min(n, 128) for n in batch.extend_seq_lens_cpu)
            physical = (global_rows + cp_size - 1) // cp_size
            logical = [
                (global_rows + cp_size - 1 - rank) // cp_size for rank in range(cp_size)
            ]
            cp = InterleaveContextParallelMetadata(
                per_rank_actual_token=[physical] * cp_size,
                per_rank_logical_token=logical,
                local_index=torch.arange(logical[0]),
                gather_index=torch.arange(physical * cp_size),
            )
            batch.attn_cp_metadata = cp
            cp_tail = LateLayerTail(
                token_indices=torch.arange(logical[0]),
                positions=torch.arange(physical),
                extend_seq_lens=torch.tensor([global_rows]),
                extend_seq_lens_cpu=[global_rows],
                swa_out_cache_loc=torch.arange(global_rows),
                cp_metadata=cp,
            )
            values = torch.arange(physical).float()[:, None, None]
            return cp_tail, {
                "state": HcState(values),
                "positions": cp_tail.positions,
                "input_ids": torch.arange(physical),
                "input_ids_global": (
                    torch.arange(physical * cp_size) if moe_backend.is_none() else None
                ),
                "forward_batch": batch,
                "capture_dspark": False,
            }

        def execute(batch):
            # This is the GPU model boundary. Keep the actual plan/run/count
            # ownership and row runner, emulating only the layer arithmetic.
            cp_tail, inputs = make_inputs(batch)
            planned, _ = owner.plan(
                cp_tail,
                forward_batch=batch,
                num_tokens=sum(batch.extend_seq_lens_cpu) // cp_size,
                tail_in_graph=False,
            )
            calls.append(
                (sum(batch.extend_seq_lens_cpu), owner._startup_capture, planned)
            )
            if planned:
                with forward_context(get_forward_context(), forward_batch=batch):
                    return owner.run(**inputs)
            return None

        with (
            get_context().override_server_args(
                chunked_prefill_size=chunk,
                tp_size=cp_size,
                attn_cp_size=cp_size,
                enable_prefill_cp=True,
                cp_strategy="interleave",
            ),
            get_parallel().override(
                tp_rank=0,
                attn_cp_size=cp_size,
                attn_cp_rank=0,
                attn_tp_size=1,
                attn_dp_size=1,
                pp_size=1,
            ),
            get_flags().moe.override(a2a_backend=moe_backend),
            get_flags().dp.override(capturing_prefill_graph=False),
            mock.patch.object(
                cp_base,
                "_STRATEGY",
                cp_interleave.InterleaveCPStrategy(cp_size=cp_size),
            ),
            mock.patch.object(
                startup,
                "get_global_expert_distribution_recorder",
                return_value=SimpleNamespace(recording=False),
            ),
            mock.patch.object(
                deepseek_v4_replay_tail,
                "get_global_expert_distribution_recorder",
                return_value=SimpleNamespace(recording=False),
            ),
            mock.patch(
                "sglang.srt.layers.attention.dsa.utils.process_model_config",
                return_value=SimpleNamespace(
                    hf_config=SimpleNamespace(architectures=["DeepseekV4ForCausalLM"])
                ),
            ),
            mock.patch.object(startup, "is_unified_kv_triton", return_value=False),
            mock.patch.object(torch, "get_device_module", return_value=device),
            mock.patch.object(torch.cuda, "mem_get_info", return_value=(1, 2)),
            mock.patch.object(owner, "_layer_stack", side_effect=layers),
            mock.patch.object(
                owner, "_break_context", side_effect=lambda b: contextlib.nullcontext()
            ),
            mock.patch.object(owner.runner, "_record", side_effect=record),
            mock.patch(
                "sglang.srt.model_executor.runner.row_bucket_graph_runner.graph_pool_replay_scope",
                side_effect=contextlib.nullcontext,
            ),
            forward_context(ForwardContext(attn_backend=backend)),
        ):
            yield SimpleNamespace(
                owner=owner,
                runner=model_runner,
                eager=SimpleNamespace(execute=execute),
                pool=pool,
                allocations=allocations,
                frees=frees,
                records=records,
                calls=calls,
                startup=startup,
                make_inputs=make_inputs,
            )

    def test_default_tail_graphs_skip_startup_but_keep_cp_bcg_bridge(self):
        """Default CP BCG must retain its eager tail without allocating tail graphs."""
        with (
            self._fixture(4) as f,
            get_exec().graph.override(
                cuda_graph_config=CudaGraphConfig(
                    prefill=PhaseConfig(backend=Backend.BREAKABLE)
                )
            ),
            envs.SGLANG_DSV4_DECODER_TAIL_GRAPH_MAX_ROWS.override(2048),
            envs.SGLANG_ENABLE_GRAPH_POOL_BORROW.override(False),
            mock.patch.object(deepseek_v4_replay_tail, "is_cuda", return_value=True),
        ):
            model = SimpleNamespace(
                tail_writes_source_kv=True, late_layer_start=20, end_layer=40
            )
            owner = deepseek_v4_replay_tail.DecoderReplayTail.maybe_create(model)
            self.assertIsNotNone(owner)
            self.assertEqual(list(owner._layer_ids), list(range(20, 40)))
            self.assertEqual(owner.runner.max_rows, 0)
            self.assertEqual(owner.capture_at_startup(f.runner, f.eager), (0.0, 0.0))
            self.assertEqual(f.allocations, [])
            self.assertEqual(f.calls, [])
            self.assertEqual(owner.runner._graphs, {})
            self.assertEqual(owner.runner._static, {})

    def test_tail_graph_cli_is_independent_of_prefill_backend(self):
        """Disabling prefix graphs must neither opt into nor override tail graphs."""
        parser = argparse.ArgumentParser()
        ServerArgs.add_cli_args(parser)
        model = SimpleNamespace(
            tail_writes_source_kv=True, late_layer_start=20, end_layer=40
        )
        for backend in (Backend.DISABLED, Backend.BREAKABLE):
            for enabled in (False, True):
                with self.subTest(backend=backend, enabled=enabled):
                    cli = parser.parse_args(
                        [
                            "--model-path",
                            "dummy",
                            "--cuda-graph-backend-prefill",
                            backend,
                        ]
                        + (["--enable-decoder-tail-cuda-graph"] if enabled else [])
                    )
                    with (
                        self._fixture(4),
                        get_exec().graph.override(
                            enable_decoder_tail_cuda_graph=cli.enable_decoder_tail_cuda_graph,
                            cuda_graph_config=CudaGraphConfig(
                                prefill=PhaseConfig(
                                    backend=cli.cuda_graph_backend_prefill
                                )
                            ),
                        ),
                        envs.SGLANG_DSV4_DECODER_TAIL_GRAPH_MAX_ROWS.override(2048),
                        envs.SGLANG_ENABLE_GRAPH_POOL_BORROW.override(False),
                        mock.patch.object(
                            deepseek_v4_replay_tail, "is_cuda", return_value=True
                        ),
                    ):
                        owner = deepseek_v4_replay_tail.DecoderReplayTail.maybe_create(
                            model
                        )
                        if backend == Backend.DISABLED and not enabled:
                            self.assertIsNone(owner)
                        else:
                            self.assertIsNotNone(owner)
                            self.assertEqual(
                                owner.runner.max_rows, 512 if enabled else 0
                            )
                            self.assertEqual(
                                owner.runner.bucket_rows(1), 32 if enabled else None
                            )
                        self.assertEqual(
                            get_exec().graph.cuda_graph_config.prefill.backend, backend
                        )

    def test_startup_covers_cp_buckets_and_serving_never_recaptures(self):
        """A prefilled cache must serve small padded tails with fresh MoE counts."""
        for cp_size in (2, 4, 8):
            with self.subTest(cp_size=cp_size), self._fixture(cp_size) as f:
                old_metadata = f.runner.attn_backend.forward_metadata
                f.owner.capture_at_startup(f.runner, f.eager)
                self.assertEqual(
                    f.calls[:2], [(16384, False, False), (16384, False, False)]
                )
                self.assertTrue(
                    all(capture and plan for _, capture, plan in f.calls[2:])
                )
                expected = list(range(2048 // cp_size, 0, -128 // cp_size))
                self.assertEqual([rows for rows, _ in f.records], expected)
                self.assertTrue(f.owner._startup_prepared)
                self.assertFalse(f.owner._startup_capture)
                self.assertIs(f.runner.attn_backend.forward_metadata, old_metadata)
                self.assertEqual(len(f.pool.free_slots), 16)
                self.assertEqual(len(f.allocations), len(f.frees))
                self.assertTrue(torch.all(f.pool.req_to_token == -77))
                counts = len(f.records)
                pointers = [
                    value.data_ptr()
                    for buffers in f.owner.runner._static.values()
                    for value in buffers
                ]
                for tokens in (1, 65, 129, 1024):
                    with f.startup._startup_batch(
                        f.runner, [min(tokens, 128)] * max(1, tokens // 128)
                    ) as batch:
                        _, inputs = f.make_inputs(batch)
                        value, _, _ = f.owner.run(**inputs)
                        logical = batch.attn_cp_metadata.per_rank_logical_token[0]
                        torch.testing.assert_close(
                            value,
                            inputs["state"].streams[:, 0] + logical,
                        )
                        self.assertTrue(f.owner._last_run_replayed)
                        # New state/DSpark variants must fall back, never capture.
                        inputs["capture_dspark"] = True
                        value, _, _ = f.owner.run(**inputs)
                        self.assertFalse(f.owner._last_run_replayed)
                        torch.testing.assert_close(
                            value,
                            inputs["state"].streams[:, 0] + logical,
                        )
                self.assertEqual(len(f.records), counts)
                self.assertEqual(
                    [
                        value.data_ptr()
                        for buffers in f.owner.runner._static.values()
                        for value in buffers
                    ],
                    pointers,
                )
                # Startup orchestration can be called again on a role switch.
                self.assertEqual(
                    f.owner.capture_at_startup(f.runner, f.eager), (0.0, 0.0)
                )
                self.assertEqual(len(f.records), counts)

    def test_tp_moe_startup_covers_cp_buckets(self):
        """NONE/Humming CP workers must prepare their tails before serving."""
        for cp_size in (2, 4, 8):
            with (
                self.subTest(cp_size=cp_size),
                self._fixture(cp_size, moe_backend=MoeA2ABackend.NONE) as f,
            ):
                f.owner.capture_at_startup(f.runner, f.eager)
                self.assertTrue(f.owner._startup_prepared)
                self.assertEqual(
                    [rows for rows, _ in f.records],
                    list(range(2048 // cp_size, 0, -128 // cp_size)),
                )
                captured = len(f.records)
                with f.startup._startup_batch(f.runner, [65]) as batch:
                    _, inputs = f.make_inputs(batch)
                    value, _, _ = f.owner.run(**inputs)
                    self.assertTrue(f.owner._last_run_replayed)
                    torch.testing.assert_close(
                        value,
                        inputs["state"].streams[:, 0]
                        + batch.attn_cp_metadata.per_rank_logical_token[0],
                    )
                self.assertEqual(len(f.records), captured)
                self.assertEqual(len(f.pool.free_slots), 16)
                self.assertTrue(torch.all(f.pool.req_to_token == -77))

    def test_tp_moe_fixed_buffer_replay_refreshes_ids_counts_and_padding(self):
        """New requests must not reuse capture IDs or route stale/NaN CP padding.

        FX records the CPU tensor operations once, retaining static input/count
        addresses. Replays execute those operations without rerunning Python
        transport or layer preparation, like CUDA graph replay.
        """
        from sglang.srt.model_executor.runner.row_bucket_graph_runner import (
            RowBucketGraphRunner,
            _Graph,
        )

        for vl_routing in (False, True):
            with (
                self.subTest(vl_routing=vl_routing),
                self._fixture(4, moe_backend=MoeA2ABackend.NONE) as f,
            ):
                f.owner._model.late_layer_vl_routing = vl_routing
                f.owner.runner = RowBucketGraphRunner(
                    row_step=4, max_rows=8, label="CPU CP tail"
                )
                recordings, layer_calls, layouts = [], [], []

                def gather(x):
                    return torch.cat([x] * 4)

                def layers(state, **kw):
                    batch = kw["forward_batch"]
                    count = batch.num_token_non_padded
                    # Emulate an attention scratch buffer with dirty pad rows.
                    # The transport must clear these on every recorded layer.
                    x = state.streams[:, 0]
                    x = torch.where(
                        (torch.arange(x.shape[0]) < count)[:, None],
                        x,
                        torch.full_like(x, float("nan")),
                    )
                    transport = moe_transport.MoeRowTransport(
                        None,
                        x,
                        batch,
                        kw["input_ids"],
                        kw["input_ids_global"],
                        use_cp=True,
                        cp_return_contiguous=True,
                    )
                    ids = transport.input_ids_global
                    layouts.append(
                        (
                            transport.input.shape[0],
                            None if ids is None else ids.shape[0],
                        )
                    )
                    layer_calls.append(count.data_ptr())
                    with transport.gathered_rows:
                        self.assertIsNone(batch.moe_num_token_non_padded())
                        # A cross-row reduction makes any poisoned padding visible.
                        result = transport.input + transport.input.sum() / 100
                        if vl_routing:
                            result = result + ids[:, None]
                    return transport.return_output(result), None, []

                def record(body, rows, count, capture_batch, start, free_before):
                    program = make_fx(body)()
                    outputs = program()

                    def replay():
                        for output, current in zip(outputs, program()):
                            if output is not None:
                                output.copy_(current)

                    recordings.append(rows)
                    return _Graph(
                        SimpleNamespace(replay=replay), count, outputs, (capture_batch,)
                    )

                parallel = SimpleNamespace(
                    attn_cp_size=4,
                    attn_cp_rank=0,
                    tp_size=4,
                    tp_group=SimpleNamespace(all_reduce=lambda x: x * 4),
                )
                with (
                    mock.patch.object(f.owner, "_layer_stack", side_effect=layers),
                    mock.patch.object(f.owner.runner, "_record", side_effect=record),
                    mock.patch.object(
                        deepseek_v4_replay_tail,
                        "attn_cp_interleave_gather",
                        side_effect=gather,
                    ),
                    mock.patch.object(
                        moe_transport, "attn_cp_interleave_gather", side_effect=gather
                    ),
                    mock.patch.object(
                        moe_transport, "get_parallel", return_value=parallel
                    ),
                    f.startup._startup_batch(f.runner, [128]) as batch,
                ):
                    original_count = batch.num_token_non_padded

                    def run(rows, logical, offset):
                        cp = InterleaveContextParallelMetadata(
                            per_rank_actual_token=[rows] * 4,
                            per_rank_logical_token=[logical, rows, rows, rows],
                            local_index=torch.arange(rows),
                            gather_index=torch.arange(rows * 4).flip(0),
                        )
                        batch.attn_cp_metadata = cp
                        x = torch.arange(rows).float()[:, None, None] + offset
                        ids = torch.arange(rows) + 10 * offset
                        value, _, _ = f.owner.run(
                            state=HcState(x),
                            positions=torch.arange(rows),
                            input_ids=ids,
                            # These live global IDs deliberately have a different
                            # row axis from the bucket and must not be copied as local.
                            input_ids_global=torch.full((rows * 4,), -999),
                            forward_batch=batch,
                            capture_dspark=False,
                        )
                        expected = x[:logical, 0] + x[:logical].sum() * 4 / 100
                        if vl_routing:
                            expected = expected + ids[:logical, None]
                        torch.testing.assert_close(value[:logical], expected * 4)
                        self.assertTrue(f.owner._last_run_replayed)
                        self.assertIs(batch.num_token_non_padded, original_count)
                        self.assertIs(batch.attn_cp_metadata, cp)
                        self.assertIsNone(cp.moe_local_token_count)

                    for bcg in (False, True):
                        with (
                            prefill_graph_scope(full_graph=False)
                            if bcg
                            else contextlib.nullcontext()
                        ):
                            f.owner._startup_capture = True
                            run(8, 8, 1)
                            run(4, 4, 2)
                            f.owner._startup_capture = False
                            calls_before = len(layer_calls)
                            # Shrink, grow, change logical count at the same physical
                            # size, then use a different bucket and fresh metadata.
                            for rows, logical, offset in (
                                (3, 2, 3),
                                (1, 1, 4),
                                (4, 3, 5),
                                (7, 5, 6),
                            ):
                                run(rows, logical, offset)
                            self.assertEqual(len(layer_calls), calls_before)
                    self.assertEqual(recordings, [8, 4, 8, 4])
                    self.assertEqual(
                        layouts,
                        [
                            (32, 32 if vl_routing else None),
                            (16, 16 if vl_routing else None),
                        ]
                        * 2,
                    )

    def test_tail_capture_progress_is_visible_only_on_rank_zero(self):
        """Startup reports completed row buckets, without one progress bar per rank."""
        for rank in (0, 1):
            with self.subTest(rank=rank), self._fixture(4) as f:
                stream = io.StringIO()
                real_tqdm = f.startup.tqdm.tqdm

                def progress(values, **kwargs):
                    return real_tqdm(values, file=stream, mininterval=0, **kwargs)

                with (
                    get_parallel().override(tp_rank=rank),
                    mock.patch.object(f.startup.tqdm, "tqdm", side_effect=progress),
                ):
                    f.owner.capture_at_startup(f.runner, f.eager)
                if rank == 0:
                    self.assertIn("Capturing decoder tail shape", stream.getvalue())
                    self.assertIn("num_tokens=128", stream.getvalue())
                    self.assertIn("local_rows=32", stream.getvalue())
                    self.assertIn("16/16", stream.getvalue())
                else:
                    self.assertEqual(stream.getvalue(), "")
                self.assertEqual(len(f.records), 16)

    def test_startup_prepares_outer_bcg_state_variant(self):
        """A BCG prefix's different state layout must not miss at first replay."""
        for backend in (MoeA2ABackend.MEGAMOE, MoeA2ABackend.NONE):
            with self.subTest(backend=backend):
                self._check_outer_bcg_state_variant(backend)

    def _check_outer_bcg_state_variant(self, backend):
        with self._fixture(4, moe_backend=backend) as f:
            executions = []

            def execute(batch):
                self.assertTrue(batch.can_run_dp_prefill_cuda_graph)
                _, inputs = f.make_inputs(batch)
                with (
                    prefill_graph_scope(full_graph=False),
                    deepseek_v4_replay_tail.enable_breakable_cuda_graph(),
                ):
                    result = f.owner.run(**inputs)
                executions.append(inputs["positions"].shape[0])
                return result

            bcg = SimpleNamespace(
                prefill_backend_name=Backend.BREAKABLE,
                enable_cp_bcg_capture=True,
                capture_num_tokens=[2048, 16384],
                can_run_graph=lambda batch: True,
                execute=execute,
            )
            f.owner.capture_at_startup(f.runner, f.eager, bcg)
            expected = list(range(512, 0, -32))
            self.assertEqual(executions, expected)
            self.assertEqual(
                [rows for rows, _ in f.records], [n for n in expected for _ in range(2)]
            )
            captured = len(f.records)
            with f.startup._startup_batch(f.runner, [259]) as batch:
                _, inputs = f.make_inputs(batch)
                with prefill_graph_scope(full_graph=False):
                    value, _, _ = f.owner.run(**inputs)
                self.assertTrue(f.owner._last_run_replayed)
                torch.testing.assert_close(value, inputs["state"].streams[:, 0] + 1032)
            self.assertEqual(len(f.records), captured)

    def test_reachable_buckets_respect_request_limit_without_chunking(self):
        """A low request limit cannot capture oversized buckets using smaller metadata."""
        with self._fixture(4, chunk=-1) as f:
            f.runner.max_running_requests = 2
            f.owner.capture_at_startup(f.runner, f.eager)
            self.assertEqual(f.calls[0], (8192, False, False))
            self.assertEqual([rows for rows, _ in f.records], [64, 32])
            self.assertEqual(len(f.pool.free_slots), 16)

    def test_runtime_first_use_falls_back_without_capture(self):
        """The former lazy path must not record graphs on a first user request."""
        with self._fixture(4) as f:
            with f.startup._startup_batch(f.runner, [128]) as batch:
                _, inputs = f.make_inputs(batch)
                value, _, _ = f.owner.run(**inputs)
                torch.testing.assert_close(value, inputs["state"].streams[:, 0] + 32)
                with self.assertRaisesRegex(
                    RuntimeError, "not captured during startup"
                ):
                    f.owner.runner.run(
                        key=(),
                        live=[torch.ones(1, 1)],
                        forward_batch=batch,
                        body=lambda tensors, batch: tensors,
                        context=lambda batch: contextlib.nullcontext(),
                        allow_capture=False,
                    )
            self.assertEqual(f.records, [])
            self.assertEqual(f.owner.runner._static, {})
            self.assertFalse(f.owner._last_run_replayed)

    def test_startup_failure_releases_temporary_kv_and_request_rows(self):
        """Failed warmup or capture cannot leak KV pages or capture permission."""
        for phase in ("warmup", "capture"):
            with self.subTest(phase=phase), self._fixture(4) as f:
                old_metadata = f.runner.attn_backend.forward_metadata
                original = f.eager.execute

                def execute(batch, phase=phase, f=f, original=original):
                    if phase == "warmup" or f.owner._startup_capture:
                        raise RuntimeError("startup failed")
                    return original(batch)

                f.eager.execute = execute
                with self.assertRaisesRegex(RuntimeError, "startup failed"):
                    f.owner.capture_at_startup(f.runner, f.eager)
                self.assertEqual(len(f.pool.free_slots), 16)
                self.assertTrue(torch.all(f.pool.req_to_token == -77))
                self.assertEqual(len(f.allocations), len(f.frees))
                self.assertFalse(f.owner._startup_capture)
                self.assertFalse(f.owner._startup_warming)
                self.assertFalse(f.owner._startup_prepared)
                self.assertIs(f.runner.attn_backend.forward_metadata, old_metadata)

    def test_draft_and_standalone_models_skip_target_tail_startup(self):
        """DSpark draft and standalone models must finish graph initialization without a target body."""
        from sglang.srt.model_executor.model_runner_components import (
            cuda_graph_setup as setup,
        )

        for case in (
            "dspark_draft",
            "draft_with_body",
            "standalone_target",
            "deferred_target",
        ):
            with self.subTest(case=case):
                order = []
                tail = SimpleNamespace(
                    capture_at_startup=lambda *args, order=order: (
                        order.append("tail") or (0.5, 5.0)
                    )
                )
                model = (
                    SimpleNamespace(model=SimpleNamespace(decoder_replay_tail=tail))
                    if case == "draft_with_body"
                    else torch.nn.Module()
                )
                draft = case in ("dspark_draft", "draft_with_body")
                phase = "draft_prefill" if draft else "prefill"
                eager, prefill_runner = object(), object()
                prefill = setup.GraphCapture(
                    runner=prefill_runner,
                    memory_phase=phase,
                    memory_usage_gb=1.0,
                    capture_time=3.0,
                )
                runner = SimpleNamespace(model=model, is_draft_worker=draft)

                def capture_prefill(*, order=order, prefill=prefill, **kwargs):
                    order.append("prefill")
                    return prefill

                with (
                    mock.patch.object(
                        setup.GraphSharedOutput, "create_for_model_runner"
                    ),
                    mock.patch.object(setup, "EagerRunner", return_value=eager),
                    mock.patch.object(setup, "refresh_deep_gemm_layout_memory_budget"),
                    mock.patch.object(
                        setup, "capture_prefill_graph", side_effect=capture_prefill
                    ),
                    mock.patch.object(
                        setup,
                        "finalize_cuda_graph_capture",
                        side_effect=lambda _, order=order: order.append("finalize"),
                    ),
                ):
                    result = setup.capture_cuda_graphs(
                        model_runner=runner,
                        capture_decode_cuda_graph=False,
                        defer_distributed_setup=case == "deferred_target",
                    )
                self.assertEqual(order, ["prefill", "finalize"])
                self.assertIs(result.prefill, prefill)
                self.assertIs(result.decode.runner, eager)
                self.assertEqual(result.memory_usage[phase], 1.0)
                self.assertEqual(result.time_usage[phase], 3.0)

    def test_tail_capture_precedes_finalization_and_counts_in_prefill_stats(self):
        """Tail capture must complete before hooks/readiness and enter startup totals."""
        from sglang.srt.model_executor.model_runner_components import (
            cuda_graph_setup as setup,
        )

        order = []
        prefill_runner, decode_runner = object(), object()
        eager = object()
        tail = SimpleNamespace(
            capture_at_startup=lambda *args: order.append("tail") or (0.5, 5.0)
        )
        model_runner = SimpleNamespace(
            model=SimpleNamespace(model=SimpleNamespace(decoder_replay_tail=tail)),
            is_draft_worker=False,
            maybe_init_uocr_greedy_graph=lambda: None,
        )

        def capture(phase, runner, memory, seconds):
            order.append(phase)
            return setup.GraphCapture(
                runner=runner,
                memory_phase=phase,
                memory_usage_gb=memory,
                capture_time=seconds,
            )

        with (
            mock.patch.object(setup.GraphSharedOutput, "create_for_model_runner"),
            mock.patch.object(setup, "EagerRunner", return_value=eager),
            mock.patch.object(setup, "refresh_deep_gemm_layout_memory_budget"),
            mock.patch.object(
                setup,
                "capture_prefill_graph",
                side_effect=lambda **kw: capture("prefill", prefill_runner, 1.0, 3.0),
            ),
            mock.patch.object(
                setup,
                "capture_decode_graph",
                side_effect=lambda **kw: capture("decode", decode_runner, 2.0, 4.0),
            ),
            mock.patch.object(
                setup,
                "current_platform",
                SimpleNamespace(capabilities=SimpleNamespace(graph_capture=True)),
            ),
            mock.patch.object(
                setup,
                "finalize_cuda_graph_capture",
                side_effect=lambda _: order.append("finalize"),
            ),
        ):
            result = setup.capture_cuda_graphs(model_runner=model_runner)
        self.assertEqual(order, ["prefill", "decode", "tail", "finalize"])
        self.assertIs(result.prefill.runner, prefill_runner)
        self.assertIs(result.decode.runner, decode_runner)
        self.assertEqual(result.memory_usage["prefill"], 1.5)
        self.assertEqual(result.time_usage["prefill"], 8.0)


if __name__ == "__main__":
    unittest.main()
