"""CPU contracts of the EAGLE binding and of SGLang's side of the EAGLE entries: a
covered step's view is the running forward's own metadata and pools, sealed at their
first read and checked after capture; the structs carry data only and the tolerated
lists name real members; offered entries run only at the batch sizes every rank
serves, an unservable setup is refused, and without an offer SGLang's path runs."""

import contextlib
import typing
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, call, patch

import msgspec
import torch

from sglang.kernels.plugin_kernels.nvidia import spec
from sglang.kernels.plugin_kernels.nvidia.spec import Step
from sglang.srt.environ import envs
from sglang.srt.kernel_plugins.nvidia import binding as runtime
from sglang.srt.kernel_plugins.nvidia import eagle_binding as eb
from sglang.srt.kernel_plugins.nvidia import speculative_runtime
from sglang.srt.layers.attention.hybrid_linear_attn_backend import (
    HybridLinearAttnBackend,
)
from sglang.srt.layers.attention.linear.gdn_backend import GDNAttnBackend
from sglang.srt.layers.attention.qwen_sparse_attn_backend import (
    QSAMTPSharedSparseIndices,
    QSAPoolView,
    QSAStepView,
    QwenSparseAttnBackend,
)
from sglang.srt.layers.attention.trtllm_mha_backend import TRTLLMHAAttnBackend
from sglang.srt.layers.radix_attention import RadixAttention
from sglang.srt.managers import overlap_utils
from sglang.srt.managers.schedule_batch import ScheduleBatch
from sglang.srt.mem_cache.kv.decl import STATE_PART
from sglang.srt.mem_cache.kv.router import KVLayerRouter, PlainPoolRouter
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.model_executor.forward_mode import CaptureHiddenMode, ForwardMode
from sglang.srt.runtime_context import get_context, publish, reset_context
from sglang.srt.server_args import ServerArgs
from sglang.srt.speculative import eagle_megakernel, eagle_worker_v2
from sglang.srt.speculative.eagle_megakernel import EagleMegakernel
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, stage="base-a", runner_config="cpu")


def _backend(cls, **fields):
    """A backend of cls holding only the given fields."""
    backend = cls.__new__(cls)
    backend.__dict__.update(fields)
    return backend


class KvPool:
    start_layer, layer_num = 3, 2
    layer_transfer_counter = None

    def __init__(self):
        layer_ids = (7, 3)
        self.k = {i: torch.zeros(4, 2) for i in layer_ids}
        self.v = {i: torch.ones(4, 2) for i in layer_ids}
        view = SimpleNamespace(
            get_key_buffer=lambda index: self.k[layer_ids[index]],
            get_value_buffer=lambda index: self.v[layer_ids[index]],
        )
        self.layer_router = KVLayerRouter.from_layer_maps(
            self, (("full", view, {3: 1, 7: 0}),)
        )


class QsaPool(KvPool):
    qsa_compress_ratio, qsa_compressed_page_size, qsa_token_topk = 4, 16, 8

    def __init__(self):
        super().__init__()
        self.compressed = {i: torch.zeros(3, 1, 2) for i in (3, 7)}
        self.key_state = {i: torch.zeros(8, 1, 2) for i in (3, 7)}
        self.qsa_rope_position_buffer = torch.zeros(8, 3, dtype=torch.int64)

    def get_qsa_compressed_k_buffer(self, layer_id):
        return self.compressed[layer_id]

    def get_qsa_key_state_buffer(self, layer_id):
        return self.key_state[layer_id]


def _qsa(pool):
    return _backend(
        QwenSparseAttnBackend,
        forward_metadata=SimpleNamespace(),
        token_to_kv_pool=pool,
        _mtp_shared_sparse_indices=None,
    )


def _state_pool():
    state = SimpleNamespace(
        conv=[torch.zeros(2, 2)],
        temporal=torch.ones(2, 2),
        intermediate_conv_window=[torch.full((2, 2), 2.0)],
        intermediate_ssm=torch.full((2, 2), 3.0),
        **{
            f"replayssm_{name}": torch.full((2, 2), float(i + 4))
            for i, name in enumerate(("d", "k", "g", "rawv", "rawk", "beta"))
        },
    )
    return SimpleNamespace(
        mamba_cache=state,
        get_speculative_mamba2_params_all_layers=lambda: state,
        replayssm_spec_write_pos=torch.zeros(2, dtype=torch.int32),
        replayssm_cache_base=torch.ones(2, dtype=torch.int32),
        replayssm_is_flush=torch.zeros(2, dtype=torch.bool),
    )


def _trtllm(pool):
    metadata = SimpleNamespace(
        page_table=torch.zeros(2, 3, dtype=torch.int32),
        cache_seqlens_int32=torch.zeros(2, dtype=torch.int32),
        cu_seqlens_q=torch.zeros(3, dtype=torch.int32),
        cu_seqlens_k=torch.zeros(3, dtype=torch.int32),
        max_seq_len_q=7,
    )
    return _backend(
        TRTLLMHAAttnBackend,
        forward_metadata=metadata,
        token_to_kv_pool=pool,
        data_type=torch.bfloat16,
        use_sliding_window_kv_pool=False,
    )


def _hybrid(pool=None, full=_trtllm):
    pool = KvPool() if pool is None else pool
    pool.layer_router.bind_part(
        sub_pool="mamba", part=STATE_PART, view=_state_pool(), indices={0: 1, 2: 0}
    )
    gdn = _backend(
        GDNAttnBackend,
        forward_metadata=SimpleNamespace(
            mamba_cache_indices=torch.zeros(2, dtype=torch.int32),
            query_start_loc=torch.zeros(3, dtype=torch.int32),
            mamba_track_indices=None,
            replayssm_ring_indices=None,
        ),
        verify_intermediate_state_indices=torch.arange(2, dtype=torch.int32),
        pad_slot_id=-1,
    )
    return _backend(
        HybridLinearAttnBackend,
        full_attn_backend=full(pool),
        linear_attn_backend=gdn,
        token_to_kv_pool=pool,
        req_to_token_pool=SimpleNamespace(),
    )


def _view(binding, backend):
    with patch.object(eb, "get_attn_backend", return_value=backend):
        return binding.forward_view()


def _model(**entries):
    """A model whose built megakernel offers these entries."""
    model = torch.nn.Module()
    offered = {
        "draft_sampler": None,
        "draft_loop": None,
        "draft_extend": None,
        "begin_draft_round": None,
        "graph_variants": None,
        "speculative_runtime_hooks": None,
        **entries,
    }
    model.__dict__["_megakernel"] = runtime._InstalledMegakernel(
        built=SimpleNamespace(widths=(), **offered), binding=None, prepare=None
    )
    return model


def _megakernel(**entries):
    """SGLang's side of a draft offering these entries."""
    return EagleMegakernel(
        draft_worker=SimpleNamespace(
            cuda_graph_runner=None,
            draft_runner=SimpleNamespace(
                model=_model(**entries),
                model_config=SimpleNamespace(
                    hidden_size=4, spec_hidden_size=4, dtype=torch.float32
                ),
                spec_algorithm=SimpleNamespace(is_standalone=lambda: False),
            ),
        ),
    )


def _init(
    *,
    draft=None,
    enabled=True,
    topk=1,
    num_steps=2,
    rejection_sampling=False,
    dp_attention=False,
    target_backend=None,
    draft_backend=None,
    hot_token_id=None,
):
    """init_eagle_megakernel, with --enable-nvidia-megakernel unless enabled is False,
    for a draft worker whose model's megakernel offers these entries; by default its
    target runs GDN and trtllm_mha under a hybrid backend and its draft trtllm_mha."""
    if draft_backend is None:
        draft_backend = _trtllm(KvPool())
    draft_worker = SimpleNamespace(
        topk=topk,
        target_worker=SimpleNamespace(
            model_runner=SimpleNamespace(
                attn_backend=_hybrid() if target_backend is None else target_backend
            ),
        ),
        draft_runner=SimpleNamespace(
            model=_model(**(draft or {})),
            model_config=None,
            attn_backend=draft_backend,
        ),
        # DraftBackendFactory builds no multi-step backend for one step
        draft_attn_backend=(
            SimpleNamespace(attn_backends=(draft_backend,)) if num_steps > 1 else None
        ),
        draft_extend_attn_backend=None,
        hot_token_id=hot_token_id,
    )
    spec = SimpleNamespace(speculative_use_rejection_sampling=rejection_sampling)
    parallel = SimpleNamespace(attn_dp_enabled=dp_attention)
    with (
        patch.object(eagle_megakernel, "megakernel_enabled", return_value=enabled),
        patch.object(eagle_megakernel, "get_spec", return_value=spec),
        patch.object(eagle_megakernel, "get_parallel", return_value=parallel),
    ):
        return eagle_megakernel.init_eagle_megakernel(draft_worker=draft_worker)


def _ranks(other=None):
    """megakernel_agree's gather as rank 0 of two ranks; other maps this rank's value
    to rank 1's, by default the same."""
    return patch.object(
        runtime,
        "_gather_tp",
        side_effect=lambda value, group=None: [
            value,
            value if other is None else other(value),
        ],
    )


def _batch(*rids):
    reqs = [SimpleNamespace(rid=rid) for rid in rids]
    return SimpleNamespace(reqs=reqs, seq_lens=torch.zeros(len(rids)))


def _decode_forward_batch(committed, seq_lens):
    """ForwardBatch.init_new of a decode batch whose requests committed these KV
    lengths, on a CPU stand-in of a runner."""
    n = len(committed)
    batch = ScheduleBatch(
        reqs=[
            SimpleNamespace(
                rid=str(i),
                lora_id=None,
                token_type_ids=None,
                kv=SimpleNamespace(kv_committed_len=length),
            )
            for i, length in enumerate(committed)
        ],
        device="cpu",
        forward_mode=ForwardMode.DECODE,
        input_ids=torch.zeros(n, dtype=torch.int64),
        req_pool_indices=torch.arange(n),
        seq_lens=torch.tensor(seq_lens),
        seq_lens_cpu=torch.tensor(seq_lens),
        out_cache_loc=torch.zeros(n, dtype=torch.int64),
    )
    runner = SimpleNamespace(
        device="cpu",
        is_draft_worker=True,
        model_config=SimpleNamespace(
            requires_mm_token_modalities=False, model_is_mrope=False
        ),
        runner_ids=SimpleNamespace(
            own_plan=lambda batch: SimpleNamespace(
                bind=lambda batch, reader, cols=None: None
            )
        ),
        prefill_attention_backend_str="torch_native",
        ngram_embedding_manager=SimpleNamespace(enabled=False),
        lora_manager=None,
        attn_backend=SimpleNamespace(
            get_cpu_graph_seq_len_fill_value=lambda: 1,
            get_cuda_graph_seq_len_fill_value=lambda: 1,
        ),
    )
    return ForwardBatch.init_new(
        batch,
        runner,
        capture_hidden_mode=CaptureHiddenMode.NULL,
        return_hidden_states_before_norm=False,
    )


def _leaves(hint):
    args = typing.get_args(hint)
    if not args:
        return [hint]
    return [leaf for arg in args if arg is not Ellipsis for leaf in _leaves(arg)]


class TestEagleBinding(CustomTestCase):
    def test_real_eagle_binding_runs_covered_dispatch_and_capture_tail(self):
        for arch, backend in (
            ("Qwen3_5ForCausalLM", _hybrid()),
            ("Qwen3_5ForCausalLMMTP", _trtllm(KvPool())),
        ):
            with self.subTest(architecture=arch):
                binding = runtime._binding_class([arch])(model=torch.nn.Module())
                self.assertIsInstance(binding, eb.EagleBinding)
                built = SimpleNamespace(
                    plan=MagicMock(return_value=Step(mode="decode", rows=2)),
                    forward=lambda step, *args, **kwargs: step.view,
                )
                original = MagicMock(return_value="standard")
                dispatch = runtime._Dispatch(
                    built=built, original=original, binding=binding
                )
                forward = dispatch.wrapped()
                batch = SimpleNamespace(batch_size=2, mamba_track_mask=None)
                ids = torch.zeros(2, dtype=torch.int64)
                check = runtime._CaptureCheck(
                    spec=SimpleNamespace(name="eagle"), dispatch=dispatch
                )
                runner = SimpleNamespace(capture_bs=(2,))
                with patch.object(eb, "get_attn_backend", return_value=backend):
                    for capturing in (False, True):
                        with patch.object(
                            torch.cuda,
                            "is_current_stream_capturing",
                            return_value=capturing,
                        ):
                            with patch.object(
                                binding,
                                "forward_view",
                                wraps=binding.forward_view,
                            ) as forward_view:
                                view = forward(ids, None, batch)
                        self.assertIsInstance(view, eb.EagleForwardView)
                        forward_view.assert_called_once_with(batch)
                    original.assert_not_called()
                    check(runner, None, batch, 2)
                    self.assertTrue(check.is_done)
                    backend.token_to_kv_pool.k[3] = torch.zeros(4, 2)
                    check = runtime._CaptureCheck(
                        spec=SimpleNamespace(name="eagle"), dispatch=dispatch
                    )
                    with self.assertRaisesRegex(RuntimeError, "kv was reallocated"):
                        check(runner, None, batch, 2)

    def test_a_qwen3_5_view_is_the_running_forward_and_tolerated_members_exist(self):
        """A covered step is captured once and replays SGLang's refills of the
        backends' metadata, so the view must hold those very tensors."""
        for arch in ("Qwen3_5MoeForConditionalGeneration", "Qwen3_5ForCausalLMMTP"):
            self.assertIs(runtime._binding_class([arch]), eb.EagleBinding)
        backend = _hybrid()
        view = _view(eb.EagleBinding(model=torch.nn.Module()), backend)
        attn = backend.full_attn_backend.forward_metadata
        gdn = backend.linear_attn_backend
        self.assertIs(view.attn.cache_seqlens, attn.cache_seqlens_int32)
        self.assertIs(view.attn.cu_seqlens_k, attn.cu_seqlens_k)
        self.assertIs(view.gdn.state_indices, gdn.forward_metadata.mamba_cache_indices)
        self.assertIs(
            view.gdn.intermediate_indices, gdn.verify_intermediate_state_indices
        )
        self.assertEqual(view.kv.layer_ids, (3, 7))
        self.assertIs(view.kv.k[1], backend.token_to_kv_pool.k[7])
        self.assertIs(view.kv.v[1], backend.token_to_kv_pool.v[7])
        self.assertEqual(view.kv.k_scale, (1.0, 1.0))
        self.assertEqual(view.mamba.layer_ids, (2, 0))
        state_pool = backend.token_to_kv_pool.layer_router.part_view(STATE_PART)
        state = state_pool.mamba_cache
        self.assertIs(view.mamba.conv[0], state.conv[0])
        self.assertIs(view.mamba.temporal, state.temporal)
        self.assertIs(
            view.mamba.intermediate_conv_window[0], state.intermediate_conv_window[0]
        )
        self.assertIs(view.mamba.intermediate_ssm, state.intermediate_ssm)
        for name in ("d", "k", "g", "rawv", "rawk", "beta"):
            self.assertIs(
                getattr(view.mamba, f"ring_{name}"), getattr(state, f"replayssm_{name}")
            )
        self.assertIs(view.mamba.ring_write_pos, state_pool.replayssm_spec_write_pos)
        self.assertIs(view.mamba.ring_cache_base, state_pool.replayssm_cache_base)
        self.assertIs(view.mamba.ring_is_flush, state_pool.replayssm_is_flush)
        # a model without GDN layers, such as the MTP draft
        draft = _view(eb.EagleBinding(model=torch.nn.Module()), _trtllm(KvPool()))
        self.assertIsNone(draft.gdn)
        self.assertIsNone(draft.mamba)
        # a plugin pins these names: a rename here must fail first
        members = {
            "EagleBinding": set(dir(eb.EagleBinding)),
            "LayerStateView": set(dir(eb.LayerStateView)),
            "ForwardBatch": set(ForwardBatch.__dataclass_fields__)
            | set(dir(ForwardBatch)),
        }
        for name in eb.TOLERATED_CALLABLES + eb.TOLERATED_READS:
            owner, _, attr = name.partition(".")
            if owner in members:
                self.assertIn(attr, members[owner], name)

    def test_the_gdn_step_view_holds_the_forwards_state_slots_and_track_mask(self):
        """PLE and GDN track writes go to the forward's resolved slots (the request
        table may be reused meanwhile), for the requests its track mask selects."""
        backend = _hybrid()
        gdn = backend.linear_attn_backend
        gdn.forward_metadata.mamba_track_indices = torch.tensor([5, 6])
        forward_batch = SimpleNamespace(mamba_track_mask=torch.tensor([True, False]))
        with patch.object(eb, "get_attn_backend", return_value=backend):
            view = eb.EagleBinding(model=torch.nn.Module()).forward_view(forward_batch)
        slots = gdn.forward_state_slots(2, track_ids=torch.zeros(2))
        self.assertEqual(view.gdn.state_indices.data_ptr(), slots.working.data_ptr())
        self.assertEqual(view.gdn.track_indices.data_ptr(), slots.track.data_ptr())
        self.assertIs(view.gdn.track_mask, forward_batch.mamba_track_mask)

    def test_a_backend_with_its_own_views_supplies_the_step_and_pool_data(self):
        """A sparse full-attention backend hands a megakernel its own metadata and
        pool buffers; the binding forwards them without reading their fields."""
        pool = QsaPool()
        backend = _hybrid(pool, full=_qsa)
        qsa = backend.full_attn_backend
        binding = eb.EagleBinding(model=torch.nn.Module())
        view = _view(binding, backend)
        self.assertIsInstance(view.attn, QSAStepView)
        self.assertIs(view.attn.metadata, qsa.forward_metadata)
        self.assertIsNone(view.attn.mtp_indices)
        self.assertIsInstance(view.kv, QSAPoolView)
        self.assertEqual(view.kv.layer_ids, (3, 7))
        self.assertIs(view.kv.k[1], pool.k[7])
        self.assertIs(view.kv.v[0], pool.v[3])
        self.assertIs(view.kv.compressed_k[1], pool.compressed[7])
        self.assertIs(view.kv.key_state[0], pool.key_state[3])
        self.assertIs(view.kv.rope_positions, pool.qsa_rope_position_buffer)
        self.assertEqual(
            (view.kv.compress_ratio, view.kv.compressed_page_size, view.kv.token_topk),
            (4, 16, 8),
        )
        # a draft's decode steps get its draft extend's selection
        shared = QSAMTPSharedSparseIndices(
            layer_ids=(3, 7), num_requests=2, token_topk=8, tail_width=4, device="cpu"
        )
        qsa.set_mtp_shared_sparse_indices(shared)
        # the step view reads the metadata as the backend's own forward does
        qsa.forward_metadata = None
        qsa.init_forward_metadata = lambda fb: setattr(qsa, "forward_metadata", fb.m)
        with patch.object(eb, "get_attn_backend", return_value=backend):
            later = binding.forward_view(
                SimpleNamespace(m="eager", mamba_track_mask=None)
            )
        self.assertEqual(later.attn.metadata, "eager")
        self.assertIs(later.attn.mtp_indices, shared.indices)
        self.assertIs(later.attn.mtp_captured_len, shared.captured_len)
        # the pool view is sealed at its first read, as KvPoolBuffers is
        self.assertIs(later.kv, view.kv)
        with patch.object(eb, "get_attn_backend", return_value=backend):
            binding.assert_live()
            pool.compressed[7] = torch.zeros(3, 1, 2)
            with self.assertRaisesRegex(RuntimeError, "kv was reallocated"):
                binding.assert_live()

    def test_a_builds_prepare_sees_the_pools_a_forward_will(self):
        """A build sizes and allocates its own per-slot state in prepare, before any
        capture, from the same sealed pool views its forwards get."""
        backend = _hybrid()
        binding = eb.EagleBinding(model=torch.nn.Module())
        runner = SimpleNamespace(
            attn_backend=backend,
            req_to_token_pool=SimpleNamespace(req_to_token=torch.zeros(9, 4)),
        )
        prepared = binding.preparation_view(runner)
        self.assertEqual(prepared.num_request_rows, 9)
        view = _view(binding, backend)
        self.assertIs(prepared.mamba, view.mamba)
        self.assertIs(prepared.parts, view.parts)
        draft = binding.preparation_view(
            SimpleNamespace(
                attn_backend=_trtllm(KvPool()),
                req_to_token_pool=runner.req_to_token_pool,
            )
        )
        self.assertIsNone(draft.mamba)

    def test_the_layer_view_waits_for_each_layers_load_back(self):
        """A megakernel that reads the pools inside its kernels waits per layer, as
        the pools' getters do, for a hierarchical-cache load-back to land."""
        backend = _hybrid()
        pool = backend.token_to_kv_pool
        parts = _view(eb.EagleBinding(model=torch.nn.Module()), backend).parts
        self.assertEqual(parts.layer_ids(), (3, 7))
        self.assertEqual(parts.part_indices(STATE_PART), {0: 1, 2: 0})
        state_pool = pool.layer_router.part_view(STATE_PART)
        self.assertIs(parts.part_view(STATE_PART), state_pool)
        view, view_layer_id = parts.read(7)
        self.assertIs(view.get_key_buffer(view_layer_id), pool.k[7])
        self.assertEqual(parts.read(2, STATE_PART), (state_pool, 0))
        pool.layer_transfer_counter = MagicMock()
        parts.wait_layer(7)
        parts.read(3)
        # a bound part waits only once the host stack loads it layer by layer
        parts.wait_layer(2, STATE_PART)
        pool.layer_router.load_parts_layer_wise()
        parts.wait_layer(2, STATE_PART)
        self.assertEqual(
            pool.layer_transfer_counter.wait_until.call_args_list,
            [call(7 - pool.start_layer), call(0), call(2 - pool.start_layer)],
        )

    def test_pool_views_keep_layer_offsets_and_fp8_scales(self):
        """Routed views use local indices; plain pools use model layer IDs.
        Neither choice may change which layer's FP8 scales accompany its KV."""
        keys = {i: torch.zeros(4, 2) for i in (3, 4)}
        values = {i: torch.ones(4, 2) for i in (3, 4)}
        plain = SimpleNamespace(
            start_layer=3,
            layer_num=2,
            layer_transfer_counter=None,
            get_key_buffer=keys.__getitem__,
            get_value_buffer=values.__getitem__,
        )
        plain.layer_router = PlainPoolRouter(plain, "full")
        for pool, layer_ids, expected_keys, expected_values in (
            (plain, (3, 4), keys, values),
            (KvPool(), (3, 7), None, None),
        ):
            with self.subTest(layer_ids=layer_ids):
                if expected_keys is None:
                    expected_keys, expected_values = pool.k, pool.v
                model = torch.nn.ModuleList()
                for i, layer_id in enumerate(layer_ids):
                    layer = RadixAttention.__new__(RadixAttention)
                    torch.nn.Module.__init__(layer)
                    layer.layer_id = layer_id
                    layer.k_scale_float, layer.v_scale_float = 0.25 + i, 0.5 + i
                    model.append(layer)
                backend = _trtllm(pool)
                backend.data_type = torch.float8_e4m3fn
                view = eb.kv_pool_buffers(pool=pool, backend=backend, model=model)
                self.assertEqual(view.layer_ids, layer_ids)
                self.assertEqual(view.k_scale, (0.25, 1.25))
                self.assertEqual(view.v_scale, (0.5, 1.5))
                for index, layer_id in enumerate(layer_ids):
                    self.assertIs(view.k[index], expected_keys[layer_id])
                    self.assertIs(view.v[index], expected_values[layer_id])

    def test_prefill_view_uses_committed_state_without_speculative_buffers(self):
        from sglang.srt.layers.attention.linear.kernels.gdn_triton import (
            TritonGDNKernel,
        )
        from sglang.srt.layers.attention.mamba.mamba2_metadata import ForwardMetadata

        backend = _hybrid()
        linear = backend.linear_attn_backend
        state = SimpleNamespace(
            conv=[torch.zeros(2, 4, 3)], temporal=torch.zeros(2, 4, 8)
        )
        backend.req_to_token_pool = SimpleNamespace(
            mamba_map={4: 0, 6: 1},
            mamba_pool=SimpleNamespace(mamba_cache=state),
            get_speculative_mamba2_params_all_layers=MagicMock(
                side_effect=AssertionError("prefill must not read speculative scratch")
            ),
        )
        meta = ForwardMetadata(
            query_start_loc=torch.tensor([0, 3, 5], dtype=torch.int32),
            mamba_cache_indices=torch.tensor([1, 2], dtype=torch.int32),
        )
        linear.forward_metadata = meta
        linear.kernel_dispatcher = SimpleNamespace(extend_kernel=TritonGDNKernel())
        linear.requires_contiguous_prefill_state = True
        binding = eb.EagleBinding(model=torch.nn.Module())
        forward_batch = SimpleNamespace(extend_prefix_lens=torch.tensor([0, 64]))
        meta.track_chunk_idx = torch.tensor([1], dtype=torch.int32)
        meta.track_ssm_h_batch_src = torch.tensor([1], dtype=torch.int64)
        with patch.object(eb, "get_attn_backend", return_value=backend):
            view = binding.prefill_view(forward_batch)
        self.assertIs(view.state.temporal, state.temporal)
        self.assertIs(view.state_indices, meta.mamba_cache_indices)
        # what a prefill covering the full-attention layers and resumed or
        # checkpointed GDN state reads
        attn = backend.full_attn_backend.forward_metadata
        self.assertIs(view.attn.page_table, attn.page_table)
        router = backend.token_to_kv_pool.layer_router
        self.assertIs(view.parts.part_view(STATE_PART), router.part_view(STATE_PART))
        self.assertIs(view.extend_prefix_lens, forward_batch.extend_prefix_lens)
        self.assertIs(view.track_chunk_idx, meta.track_chunk_idx)
        self.assertIs(view.track_ssm_h_batch_src, meta.track_ssm_h_batch_src)
        self.assertEqual((view.is_graph, view.prefill_graph_metadata), (False, None))
        # a prefill graph's capture or replay reads the bucket's static tables
        linear.prefill_graph_metadata = SimpleNamespace()
        with patch.object(eb, "get_attn_backend", return_value=backend):
            view = binding.prefill_view(forward_batch)
        self.assertTrue(view.is_graph)
        self.assertIs(view.prefill_graph_metadata, linear.prefill_graph_metadata)

    def test_existing_forward_wrapper_selects_the_prefill_data_view(self):
        views = SimpleNamespace(
            forward_view=MagicMock(), prefill_view=MagicMock(return_value="prefill")
        )
        built = SimpleNamespace(
            plan=lambda *a, **k: Step(mode="prefill", rows=5),
            forward=lambda step, original, *a, **k: step.view,
        )
        wrapped = runtime._Dispatch(
            built=built, original=MagicMock(), binding=views
        ).wrapped()
        with patch.object(torch.cuda, "is_current_stream_capturing", return_value=True):
            forward_batch = SimpleNamespace(batch_size=5)
            self.assertEqual(wrapped(torch.zeros(5), None, forward_batch), "prefill")
        views.prefill_view.assert_called_once_with(forward_batch)
        views.forward_view.assert_not_called()

    def test_prefill_without_an_offered_plan_uses_stock(self):
        original = MagicMock(return_value="stock")
        views = SimpleNamespace(prefill_view=MagicMock(), forward_view=MagicMock())
        wrapped = runtime._Dispatch(
            built=SimpleNamespace(plan=lambda *a, **k: None),
            original=original,
            binding=views,
        ).wrapped()
        args = (torch.zeros(5), None, SimpleNamespace(batch_size=5))
        with patch.object(
            torch.cuda, "is_current_stream_capturing", return_value=False
        ):
            self.assertEqual(wrapped(*args), "stock")
        original.assert_called_once_with(*args)
        views.prefill_view.assert_not_called()
        views.forward_view.assert_not_called()

    def test_a_reallocated_pool_fails_assert_live_by_name(self):
        backend = _hybrid()
        binding = eb.EagleBinding(model=torch.nn.Module())
        first = _view(binding, backend)
        self.assertIs(_view(binding, backend).kv, first.kv)  # sealed at the first read
        with patch.object(eb, "get_attn_backend", return_value=backend):
            binding.assert_live()
            backend.token_to_kv_pool.k[3] = torch.zeros(4, 2)
            with self.assertRaisesRegex(RuntimeError, "kv was reallocated"):
                binding.assert_live()
        # a rebuilt router over the same buffers
        backend = _hybrid()
        binding = eb.EagleBinding(model=torch.nn.Module())
        self.assertIs(_view(binding, backend).parts, _view(binding, backend).parts)
        pool, old = backend.token_to_kv_pool, backend.token_to_kv_pool.layer_router
        router = KVLayerRouter.from_layer_maps(
            pool, (("full", old.view("full"), {3: 1, 7: 0}),)
        )
        state = old.part_view(STATE_PART)
        router.bind_part(
            sub_pool="mamba", part=STATE_PART, view=state, indices={0: 1, 2: 0}
        )
        with patch.object(eb, "get_attn_backend", return_value=backend):
            binding.assert_live()
            pool.layer_router = router
            with self.assertRaisesRegex(RuntimeError, "parts was reallocated"):
                binding.assert_live()

    def test_a_graph_variant_selects_on_a_lone_requests_committed_kv_length(self):
        """select picks a captured graph from host state alone: init_new hands it a
        one-request batch's committed KV length, which the KV its step reads never
        falls below (seq_lens may run a verify ahead), and nothing for several."""
        publish(ServerArgs(model_path="dummy"), role="test")
        self.addCleanup(reset_context)
        override = get_context().override_server_args(device="cpu")
        override.install()
        self.addCleanup(override.restore)
        for committed, seq_lens, bound in (
            ([70000], [70007], 70000),
            ([70000, 5], [70007, 12], None),
        ):
            with self.subTest(committed=committed):
                forward_batch = _decode_forward_batch(committed, seq_lens)
                self.assertEqual(forward_batch.attention_kv_len_lower_bound, bound)

    def test_entry_structs_carry_data_only(self):
        structs = {
            cls
            for cls in vars(eb).values()
            if isinstance(cls, type) and issubclass(cls, msgspec.Struct)
        }
        # msgspec.Struct: a struct an attention backend declares for its own view
        allowed = {torch.Tensor, int, float, bool, type(None), msgspec.Struct}
        allowed |= structs
        # the layer view's methods are tolerated callables, its router is not data
        for cls in structs - {eb.LayerStateView}:
            for field, hint in typing.get_type_hints(cls).items():
                for leaf in _leaves(hint):
                    self.assertIn(leaf, allowed, f"{cls.__name__}.{field}")


class TestEagleEntries(CustomTestCase):
    def test_a_draft_step_fold_is_offered_only_by_a_build_that_has_one(self):
        fold = object()
        self.assertIn("draft_step_fold", spec.ENTRIES)
        self.assertIs(
            runtime.megakernel_entry(_model(draft_step_fold=fold), "draft_step_fold"),
            fold,
        )
        self.assertIsNone(runtime.megakernel_entry(_model(), "draft_step_fold"))
        self.assertIsNone(
            runtime.megakernel_entry(torch.nn.Module(), "draft_step_fold")
        )
        # a build that predates the entry offers none
        self.assertNotIn(
            "draft_step_fold", _model().__dict__["_megakernel"].built.__dict__
        )
        self.assertEqual(
            runtime._offered_entries(
                _model(draft_step_fold=fold).__dict__["_megakernel"].built
            ),
            ("draft_step_fold",),
        )
        # it is not one of the EAGLE draft-loop entries, so it keeps the stock dispatch
        self.assertIsNone(_init(draft={"draft_step_fold": fold}))

    def test_always_present_dispatch_preserves_the_stock_path(self):
        stock_result = object()
        with patch.object(eagle_megakernel, "init_eagle_megakernel", return_value=None):
            dispatch = eagle_megakernel.init_eagle_draft_runtime(draft_worker=object())
        self.assertIs(dispatch, eagle_megakernel.STOCK_EAGLE_DRAFT_RUNTIME)
        self.assertIs(
            dispatch.run_draft_loop(
                runner=object(),
                forward_batch=object(),
                stock=lambda: stock_result,
            ),
            stock_result,
        )
        self.assertIs(
            _megakernel().run_draft_loop(
                runner=object(),
                forward_batch=SimpleNamespace(batch_size=1),
                stock=lambda: stock_result,
            ),
            stock_result,
        )

    def test_without_the_flag_or_an_offered_entry_sglang_keeps_its_own_path(self):
        self.assertIsNone(_init())
        self.assertIsNone(_init(draft={"draft_loop": print}, enabled=False))
        self.assertIsInstance(_init(draft={"draft_loop": print}), EagleMegakernel)
        runner = SimpleNamespace(capture_bs=[1, 2])
        self.assertEqual(_megakernel().bind_draft_graphs(runner), frozenset())

    def test_every_unservable_setup_is_refused_before_any_capture(self):
        """The entries hand over a multi-step topk-1 chain's trtllm_mha and GDN
        metadata for this rank's own batch; any other setup is refused at startup,
        with all its reasons."""
        offer = {"draft_loop": print}
        sliding = _hybrid()
        sliding.full_attn_backend.use_sliding_window_kv_pool = True
        with self.assertRaises(ValueError) as refused:
            _init(
                draft=offer,
                topk=2,
                rejection_sampling=True,
                dp_attention=True,
                target_backend=sliding,
                draft_backend=SimpleNamespace(),
                hot_token_id=torch.arange(4),
            )
        for reason in (
            "chain draft (topk 1)",
            "rejection sampling",
            "speculative token map",
            "DP attention",
            "sliding-window pool",
            "the draft must run trtllm_mha, not SimpleNamespace",
        ):
            self.assertIn(reason, str(refused.exception))
        with self.assertRaisesRegex(ValueError, "hybrid .* not TRTLLMHAAttnBackend"):
            _init(draft=offer, target_backend=_trtllm(KvPool()))
        with self.assertRaisesRegex(ValueError, "multi-step draft"):
            _init(draft={"draft_extend": print}, num_steps=1)

    def test_a_target_verify_megakernel_without_draft_entries_is_refused(self):
        """A target megakernel covering only the verify step reads the same
        ReplaySSM ring while its draft offers no entry, so SGLang's side of the
        entries never runs: the install refuses it before the build, and so before
        any capture. Static pools keep it."""
        for unified in (False, True):
            with self.subTest(unified_memory=unified):
                model = torch.nn.Module()
                original = model.forward
                built = SimpleNamespace(
                    widths=(2, 4),
                    summary=lambda: "target_verify",
                    plan=MagicMock(return_value=None),
                    forward=MagicMock(),
                    draft_sampler=None,
                    draft_loop=None,
                    draft_extend=None,
                    begin_draft_round=None,
                    graph_variants=None,
                    speculative_runtime_hooks=None,
                )
                target_spec = SimpleNamespace(
                    name="qwen3_5_verify",
                    build=MagicMock(return_value=built),
                    prepare=None,
                    max_rows=4,
                    numerics=SimpleNamespace(kind=SimpleNamespace(value="exact")),
                    binding="",
                    prefill_graph=False,
                )
                hooks = []
                refused = (
                    self.assertRaisesRegex(ValueError, "ReplaySSM speculative ring")
                    if unified
                    else contextlib.nullcontext()
                )
                with (
                    self._target_install_context(unified_memory=unified),
                    patch.object(
                        runtime, "_find_spec_on_every_rank", return_value=target_spec
                    ),
                    refused,
                ):
                    runtime.install_megakernel(
                        model=model,
                        hf_config=SimpleNamespace(
                            architectures=["Qwen3_5MoeForCausalLM"]
                        ),
                        capture_tail_hooks=hooks,
                        is_draft=False,
                    )
                if unified:
                    target_spec.build.assert_not_called()
                    self.assertEqual(model.forward, original)
                    self.assertEqual(hooks, [])
                else:
                    target_spec.build.assert_called_once()
                    self.assertNotEqual(model.forward, original)
                    self.assertEqual(len(hooks), 1)

    @staticmethod
    @contextlib.contextmanager
    def _target_install_context(*, unified_memory):
        """install_megakernel's runtime reads on one CPU rank with speculative
        ReplaySSM on, and unified memory as given."""
        memory = SimpleNamespace(enable_unified_memory=unified_memory)
        exec_ = SimpleNamespace(
            mamba=SimpleNamespace(enable_linear_replayssm_spec=True)
        )
        with (
            patch.object(runtime, "megakernel_enabled", return_value=True),
            patch.object(runtime, "msa_serves", return_value=False),
            patch.object(runtime, "_requirement_reasons", return_value=[]),
            patch.object(runtime, "_model_disable_reasons", return_value=[]),
            patch.object(runtime, "get_memory", return_value=memory),
            patch.object(runtime, "get_exec", return_value=exec_),
            patch.object(
                runtime,
                "get_device",
                return_value=SimpleNamespace(watchdog_timeout=None),
            ),
            patch.object(runtime, "_tokens_per_request", return_value=2),
            patch.object(runtime, "_decode_graphs_enabled", return_value=True),
            patch.object(runtime, "_gather_tp", side_effect=lambda x, group=None: [x]),
            patch.object(torch.cuda, "mem_get_info", return_value=(0, 0)),
        ):
            yield

    def test_the_draft_worker_takes_its_megakernel_once_both_backends_exist(self):
        """The draft's graph runners read the worker's megakernel when they capture,
        and its init reads both of the draft's attention backends:
        init_attention_backends sets it once both exist."""
        order = MagicMock()
        worker = eagle_worker_v2.EagleDraftWorker.__new__(
            eagle_worker_v2.EagleDraftWorker
        )
        worker.__dict__.update(
            draft_owns_attention=False,
            draft_worker=order.draft_worker,
            init_attention_backend=order.init_attention_backend,
            parallel_layout=None,
        )
        # The parallel and MoE contexts read process groups and runtime flags this
        # CPU test lacks.
        with (
            patch.object(eagle_worker_v2, "draft_tp_context", contextlib.nullcontext),
            patch.object(eagle_worker_v2, "draft_pp_context", contextlib.nullcontext),
            patch.object(
                eagle_worker_v2,
                "speculative_moe_backend_context",
                contextlib.nullcontext,
            ),
            patch.object(
                eagle_worker_v2,
                "speculative_moe_a2a_backend_context",
                contextlib.nullcontext,
            ),
            patch.object(
                eagle_megakernel, "init_eagle_megakernel", order.init_eagle_megakernel
            ),
        ):
            worker.init_attention_backends()
        self.assertEqual(
            order.mock_calls,
            [
                call.draft_worker.init_attention_backends(),
                call.init_attention_backend(),
                call.init_eagle_megakernel(draft_worker=worker),
            ],
        )
        self.assertIs(worker.draft_runtime, order.init_eagle_megakernel.return_value)

    def test_each_factory_runs_once_and_serves_only_what_every_rank_serves(self):
        """A rank on SGLang's code at a batch size would wait in collectives the
        entry's step skips, so a size not every rank serves must be refused."""
        calls = []

        def factory(**buffers):
            calls.append(buffers)
            return lambda bs: print if bs <= 2 else None

        mk = _megakernel(draft_loop=factory, draft_extend=factory)
        draft_runner = SimpleNamespace(
            max_bs=4, speculative_num_steps=3, device="cpu", capture_bs=[1, 2, 4]
        )
        extend_runner = SimpleNamespace(
            max_bs=4, captured_req_width=7, device="cpu", capture_bs=[1, 2, 4]
        )
        with _ranks():
            self.assertEqual(mk.bind_draft_graphs(draft_runner), {1, 2})
            self.assertEqual(mk.bind_extend_graphs(extend_runner), {1, 2})
        loop, extend = calls
        self.assertEqual(loop["draft_tokens"].shape, (4, 3))
        self.assertEqual((loop["num_steps"], loop["max_bs"]), (3, 4))
        self.assertEqual((extend["tokens_per_request"], extend["max_bs"]), (7, 4))
        self.assertEqual(extend["hidden"].shape, (4, 4))
        with (
            _ranks(lambda sizes: sizes[:1]),
            self.assertRaisesRegex(RuntimeError, "draft_loop batch sizes differ"),
        ):
            mk.bind_draft_graphs(draft_runner)

    def test_a_draft_round_reuses_a_served_extend_only_for_the_input_it_made(self):
        """The megakernel may reuse what its draft extend left behind only when that
        extend produced this round's input: not after another extend, not for other
        requests (a PD decode instance's first round), and only once. The extend's
        tokens remain graph-owned until the unchanged next round consumes them."""
        rounds, outs = [], {}

        def draft_extend(**buffers):
            outs.update(buffers)
            return lambda bs: print if bs == 1 else None

        mk = _megakernel(
            draft_loop=lambda **_: lambda bs: print if bs == 1 else None,
            draft_extend=draft_extend,
            begin_draft_round=rounds.append,
        )
        draft_runner = SimpleNamespace(
            max_bs=2, speculative_num_steps=3, device="cpu", capture_bs=[1, 2]
        )
        extend_runner = SimpleNamespace(
            max_bs=2, captured_req_width=4, device="cpu", capture_bs=[1, 2]
        )
        with _ranks():
            mk.bind_draft_graphs(draft_runner)
            mk.bind_extend_graphs(extend_runner)
        draft_input = SimpleNamespace(
            topk_index=torch.zeros((1, 1), dtype=torch.int64),
            hidden_states=torch.zeros((1, 4)),
        )

        def served_extend(*rids):
            mk.note_draft_extend()
            extend_runner.bs = extend_runner.raw_bs = len(rids)
            return mk.extend_tokens(
                runner=extend_runner,
                batch=_batch(*rids),
                stock=lambda: (None, None, None),
            )

        def begin(*rids):
            mk.begin_draft_round(
                runner=draft_runner, batch=_batch(*rids), draft_input=draft_input
            )
            return rounds[-1].from_served_extend

        outs["topk_index"][0] = 5  # what the extend's replay wrote
        _, _, topk_index = served_extend("a")
        self.assertEqual(topk_index.data_ptr(), outs["topk_index"].data_ptr())
        self.assertEqual(topk_index.tolist(), [[5]])
        draft_input.topk_index = topk_index
        self.assertTrue(begin("a"))
        self.assertIs(rounds[-1].topk_index, topk_index)
        outs["topk_index"].zero_()  # a later replay rewrites the shared view
        self.assertEqual(topk_index.tolist(), [[0]])
        self.assertFalse(begin("a"))
        served_extend("a")
        mk.note_draft_extend()  # an extend the megakernel did not serve
        self.assertFalse(begin("a"))
        served_extend("a")
        self.assertFalse(begin("b"))
        calls = len(rounds)
        begin("a", "b")
        self.assertEqual(len(rounds), calls)  # batch size 2 is not served
        self.assertEqual(
            served_extend("a", "b"), (None, None, None)
        )  # SGLang's stock sampler there


class TestSpeculativeRuntimeContract(CustomTestCase):
    def test_strategy_passes_the_local_policy_to_runtime(self):
        output = spec.PostVerifyOutput(
            token_ids=torch.tensor([1]),
            accepted_lengths=torch.tensor([1]),
            accepted_indices=torch.tensor([0]),
        )
        local_keys = lambda view: view.values.argmax(-1)
        calls = []

        def run(io, policy):
            calls.append((io, policy))
            return output

        io = object()
        strategy = spec.PostVerifyStrategy(run=run, local_keys=local_keys)
        self.assertIs(strategy(io), output)
        self.assertEqual(calls, [(io, local_keys)])


def _defaults() -> spec.SpeculativeRuntimeHooks:
    return speculative_runtime.default_speculative_runtime_hooks()


class TestSpeculativeRuntimeHooks(CustomTestCase):
    def test_switch_off_returns_stock_identity_without_provider_work(self):
        defaults = _defaults()
        model = _model(
            speculative_runtime_hooks=lambda _: (_ for _ in ()).throw(
                AssertionError("factory must not run")
            )
        )
        with (
            envs.SGLANG_ENABLE_NV_MEGA_RUNTIME_OPTIMIZATION.override(False),
            patch.object(
                runtime,
                "built_megakernel",
                side_effect=AssertionError("off must not inspect a provider"),
            ),
            patch.object(runtime, "megakernel_agree") as agree,
        ):
            selected = runtime.bind_speculative_runtime_hooks(model, defaults)
        self.assertIs(selected, defaults)
        agree.assert_not_called()

    def test_missing_provider_keeps_exact_defaults(self):
        defaults = _defaults()
        models = (torch.nn.Module(), _model())
        for model in models:
            with (
                self.subTest(has_built=runtime.built_megakernel(model) is not None),
                envs.SGLANG_ENABLE_NV_MEGA_RUNTIME_OPTIMIZATION.override(True),
                patch.object(runtime, "_gather_tp", return_value=[False]),
            ):
                selected = runtime.bind_speculative_runtime_hooks(model, defaults)
            self.assertIs(selected, defaults)

    def test_factory_wraps_defaults_and_changes_only_local_keys(self):
        defaults = _defaults()
        custom_keys = lambda view: torch.zeros(
            view.values.shape[0], dtype=torch.int64, device=view.values.device
        )

        def factory(stock):
            return spec.SpeculativeRuntimeHooks(
                post_verify=spec.PostVerifyStrategy(
                    run=stock.post_verify.run,
                    local_keys=custom_keys,
                )
            )

        model = _model(speculative_runtime_hooks=factory)
        with (
            envs.SGLANG_ENABLE_NV_MEGA_RUNTIME_OPTIMIZATION.override(True),
            patch.object(runtime, "_gather_tp", return_value=[True]),
        ):
            selected = runtime.bind_speculative_runtime_hooks(model, defaults)
        self.assertIs(selected.post_verify.run, defaults.post_verify.run)
        self.assertIs(selected.post_verify.local_keys, custom_keys)

    def test_default_post_verify_uses_the_stock_signature(self):
        io = SimpleNamespace(
            device=torch.device("cpu"),
            is_idle=False,
            seq_lens=torch.tensor([3]),
            sampling_info=SimpleNamespace(),
            req_pool_indices=torch.tensor([0]),
            return_logprob=False,
            verify_tree=object(),
            logits=object(),
            grammar_mask=None,
            sparse_target_max_top_k=None,
            watermark=None,
            watermark_full_mask=True,
            simulated_accept_length=None,
            simulated_accept_length_rng=None,
            sparse_bonus_argmax=False,
        )
        expected = (
            torch.tensor([1]),
            torch.tensor([0]),
            torch.tensor([0]),
        )
        with (
            patch(
                "sglang.srt.speculative.eagle_utils.LOCAL_KEYS_SUPPORTED",
                False,
                create=True,
            ),
            patch(
                "sglang.srt.speculative.eagle_utils.eagle_sample",
                return_value=expected,
            ) as sample,
        ):
            output = speculative_runtime.default_post_verify(io, lambda view: None)
        self.assertEqual(
            sample.call_args.kwargs,
            {
                "uno_target_max_top_k": None,
                "watermark_state": None,
                "watermark_full_mask": True,
                "simulated_acc_len": None,
                "simulated_acc_len_rng": None,
                "sparse_bonus_argmax": False,
            },
        )
        self.assertTrue(torch.equal(output.token_ids, expected[0]))
        batch = sample.call_args.args[1]
        self.assertIsInstance(batch, speculative_runtime.StockVerifyBatch)
        self.assertIs(batch.forward_mode, ForwardMode.TARGET_VERIFY)
        self.assertIs(batch.sampling_info, io.sampling_info)
        self.assertIs(batch.req_pool_indices, io.req_pool_indices)


class TestSpecRelayBypass(CustomTestCase):
    @staticmethod
    def _future_map():
        future_map = overlap_utils.FutureMap.__new__(overlap_utils.FutureMap)
        future_map._deferred_relay = None
        future_map.output_tokens_buf = torch.empty(8, dtype=torch.int64)
        future_map.spec_algo = SimpleNamespace(is_ngram=lambda: False)
        return future_map

    @staticmethod
    def _draft_input(indices):
        return SimpleNamespace(
            future_indices=indices,
            bonus_tokens=torch.tensor([7, 8], dtype=torch.int32),
            topk_p=torch.tensor([[0.7], [0.8]]),
            topk_index=torch.tensor([[3], [4]]),
            hidden_states=torch.arange(8, dtype=torch.float32).view(2, 4),
            draft_probs=None,
            dsa_topk_indices=None,
            future_dsa_topk_indices_available=False,
        )

    def test_unchanged_batch_consumes_the_draft_input_in_place(self):
        future_map = self._future_map()
        future_map._stash = MagicMock()
        indices = torch.tensor([2, 5])
        draft_input = self._draft_input(indices)
        payload = overlap_utils.RelayPayload.from_draft_input(draft_input)
        topk_p, topk_index, hidden = (
            draft_input.topk_p,
            draft_input.topk_index,
            draft_input.hidden_states,
        )
        future_map.defer_relay(indices, payload, draft_input)
        batch = SimpleNamespace(
            spec_info=draft_input,
            prefill_input_ids_cpu=None,
            input_ids=torch.tensor([0]),
            enable_overlap=True,
            spec_algorithm=SimpleNamespace(is_none=lambda: False),
        )

        overlap_utils.resolve_forward_inputs(batch, future_map)

        future_map._stash.assert_not_called()
        self.assertIsNone(future_map._deferred_relay)
        self.assertIs(draft_input.topk_p, topk_p)
        self.assertIs(draft_input.topk_index, topk_index)
        self.assertIs(draft_input.hidden_states, hidden)
        self.assertEqual(draft_input.bonus_tokens.dtype, torch.int64)
        self.assertEqual(draft_input.bonus_tokens.tolist(), [7, 8])

    def test_changed_batch_flushes_before_it_can_overwrite_the_outputs(self):
        future_map = self._future_map()
        future_map._stash = MagicMock()
        indices = torch.tensor([2, 5])
        draft_input = self._draft_input(indices)
        payload = overlap_utils.RelayPayload.from_draft_input(draft_input)
        future_map.defer_relay(indices, payload, draft_input)
        other = self._draft_input(indices.clone())
        batch = SimpleNamespace(
            spec_info=other,
            prefill_input_ids_cpu=None,
            input_ids=torch.tensor([0]),
            enable_overlap=False,
        )

        overlap_utils.resolve_forward_inputs(batch, future_map)

        future_map._stash.assert_called_once_with(indices, payload)
        self.assertIsNone(future_map._deferred_relay)


class TestDraftOutputWidth(CustomTestCase):
    def test_recurrent_output_uses_the_declared_hidden_stream(self):
        supplied = {}

        def extend(**buffers):
            supplied.update(buffers)
            return lambda bs: (lambda io: None) if bs == 1 else None

        mk = _megakernel(draft_extend=extend)
        mk._draft_config.spec_hidden_size = 16
        runner = SimpleNamespace(
            max_bs=1, device="cpu", captured_req_width=4, capture_bs=(1,)
        )
        with patch.object(eagle_megakernel, "megakernel_agree"):
            self.assertEqual(mk.bind_extend_graphs(runner), {1})
        self.assertEqual(supplied["hidden"].shape, (1, 16))
        self.assertEqual(supplied["hidden"].dtype, torch.float32)

    def test_no_recurrent_hidden_stream_keeps_stock(self):
        factory = MagicMock()
        mk = _megakernel(draft_extend=factory)
        mk._draft_runner.spec_algorithm = SimpleNamespace(is_standalone=lambda: True)
        self.assertEqual(mk.bind_extend_graphs(SimpleNamespace()), frozenset())
        factory.assert_not_called()

    def _check_author_alias(self, cases):
        for declared_width, buffer_dtype, reuse in cases:
            with self.subTest(width=declared_width, dtype=buffer_dtype):
                supplied = {}

                def extend(**buffers):
                    supplied.update(buffers)
                    return lambda bs: (lambda io: None) if bs == 1 else None

                mk = _megakernel(draft_extend=extend)
                mk._draft_config.spec_hidden_size = declared_width
                buffers = SimpleNamespace(
                    topk_index=torch.zeros((2, 1), dtype=torch.int64),
                    topk_p=torch.ones((2, 1), dtype=torch.float32),
                    hidden_states=torch.zeros((2, 4), dtype=buffer_dtype),
                )
                mk._draft_worker.cuda_graph_runner = SimpleNamespace(buffers=buffers)
                runner = SimpleNamespace(
                    max_bs=1, device="cpu", captured_req_width=4, capture_bs=(1,)
                )
                with (
                    patch.object(
                        overlap_utils, "spec_relay_bypass_enabled", return_value=True
                    ),
                    patch.object(
                        eagle_megakernel,
                        "get_parallel",
                        return_value=SimpleNamespace(moe_ep_size=1),
                    ),
                    patch.object(eagle_megakernel, "megakernel_agree"),
                ):
                    self.assertEqual(mk.bind_extend_graphs(runner), {1})
                self.assertEqual(supplied["hidden"].shape, (1, declared_width))
                self.assertEqual(supplied["hidden"].dtype, torch.float32)
                for name, original in (
                    ("topk_index", buffers.topk_index),
                    ("topk_p", buffers.topk_p),
                    ("hidden", buffers.hidden_states),
                ):
                    self.assertEqual(
                        supplied[name].data_ptr() == original.data_ptr(), reuse
                    )
                self.assertIs(
                    mk._extend_graphs[id(runner)].out.hidden, supplied["hidden"]
                )

    def test_compatible_author_alias_preserves_buffer_identity(self):
        self._check_author_alias(((4, torch.float32, True),))

    def test_incompatible_author_alias_uses_owned_stream_buffers(self):
        self._check_author_alias(
            ((16, torch.float32, False), (4, torch.float64, False))
        )

    def test_missing_hidden_stream_returns_stock_before_alias_access(self):
        factory = MagicMock()
        mk = _megakernel(draft_extend=factory)
        mk._draft_runner.spec_algorithm = SimpleNamespace(is_standalone=lambda: True)
        with patch.object(mk, "_draft_inputs_as_extend_outputs") as alias:
            self.assertEqual(mk.bind_extend_graphs(SimpleNamespace()), frozenset())
        alias.assert_not_called()
        factory.assert_not_called()


if __name__ == "__main__":
    unittest.main()
