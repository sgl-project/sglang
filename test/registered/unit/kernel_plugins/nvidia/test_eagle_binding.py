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

from sglang.srt.kernel_plugins.nvidia import binding as runtime
from sglang.srt.kernel_plugins.nvidia import eagle_binding as eb
from sglang.srt.layers.attention.hybrid_linear_attn_backend import (
    HybridLinearAttnBackend,
)
from sglang.srt.layers.attention.linear.gdn_backend import GDNAttnBackend
from sglang.srt.layers.attention.trtllm_mha_backend import TRTLLMHAAttnBackend
from sglang.srt.managers.schedule_batch import ScheduleBatch
from sglang.srt.model_executor.forward_batch_info import (
    CaptureHiddenMode,
    ForwardBatch,
    ForwardMode,
)
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

    def __init__(self):
        self.k = {i: torch.zeros(4, 2) for i in (3, 4)}

    def get_key_buffer(self, layer_id):
        return self.k[layer_id]

    def get_value_buffer(self, layer_id):
        return self.k[layer_id]


def _req_pool():
    tensor = torch.zeros(2)
    state = SimpleNamespace(
        conv=[tensor],
        temporal=tensor,
        intermediate_conv_window=[tensor],
        intermediate_ssm=tensor,
        **{f"replayssm_{n}": None for n in ("d", "k", "g", "rawv", "rawk", "beta")},
    )
    return SimpleNamespace(
        mamba_map={0: 0, 2: 1},
        get_speculative_mamba2_params_all_layers=lambda: state,
        mamba_pool=SimpleNamespace(
            replayssm_spec_write_pos=None,
            replayssm_cache_base=None,
            replayssm_is_flush=None,
        ),
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


def _hybrid():
    pool = KvPool()
    gdn = _backend(
        GDNAttnBackend,
        forward_metadata=SimpleNamespace(
            mamba_cache_indices=torch.zeros(2, dtype=torch.int32),
            query_start_loc=torch.zeros(3, dtype=torch.int32),
            mamba_track_indices=None,
        ),
        verify_intermediate_state_indices=torch.arange(2, dtype=torch.int32),
        pad_slot_id=-1,
    )
    return _backend(
        HybridLinearAttnBackend,
        full_attn_backend=_trtllm(pool),
        linear_attn_backend=gdn,
        token_to_kv_pool=pool,
        req_to_token_pool=_req_pool(),
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
        **entries,
    }
    model.__dict__["_megakernel"] = SimpleNamespace(widths=(), **offered)
    return model


def _megakernel(**entries):
    """SGLang's side of a draft offering these entries."""
    return EagleMegakernel(
        draft_worker=SimpleNamespace(
            draft_runner=SimpleNamespace(
                model=_model(**entries),
                model_config=SimpleNamespace(hidden_size=4, dtype=torch.float32),
            )
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
        kv_index_translator=SimpleNamespace(
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
        self.assertEqual(view.kv.layer_ids, (3, 4))
        self.assertIs(view.kv.k[1], backend.token_to_kv_pool.k[4])
        self.assertEqual(view.kv.k_scale, (1.0, 1.0))
        self.assertEqual(view.mamba.layer_ids, (0, 2))
        # a model without GDN layers, such as the MTP draft
        draft = _view(eb.EagleBinding(model=torch.nn.Module()), _trtllm(KvPool()))
        self.assertIsNone(draft.gdn)
        self.assertIsNone(draft.mamba)
        # a plugin pins these names: a rename here must fail first
        members = {
            "EagleBinding": set(dir(eb.EagleBinding)),
            "ForwardBatch": set(ForwardBatch.__dataclass_fields__),
        }
        for name in eb.TOLERATED_CALLABLES + eb.TOLERATED_READS:
            owner, _, attr = name.partition(".")
            if owner in members:
                self.assertIn(attr, members[owner], name)

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
        allowed = {torch.Tensor, int, float, bool, type(None)} | structs
        for cls in structs:
            for field, hint in typing.get_type_hints(cls).items():
                for leaf in _leaves(hint):
                    self.assertIn(leaf, allowed, f"{cls.__name__}.{field}")


class TestEagleEntries(CustomTestCase):
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
        self.assertIs(worker.megakernel, order.init_eagle_megakernel.return_value)

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
        tokens are SGLang's own copy, which the next replay leaves alone."""
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
            return mk.extend_tokens(runner=extend_runner, batch=_batch(*rids))

        def begin(*rids):
            mk.begin_draft_round(
                runner=draft_runner, batch=_batch(*rids), draft_input=draft_input
            )
            return rounds[-1].from_served_extend

        outs["topk_index"][0] = 5  # what the extend's replay wrote
        _, topk_index = served_extend("a")
        outs["topk_index"].zero_()  # the next replay rewrites it
        self.assertEqual(topk_index.tolist(), [[5]])
        self.assertTrue(begin("a"))
        self.assertFalse(begin("a"))
        served_extend("a")
        mk.note_draft_extend()  # an extend the megakernel did not serve
        self.assertFalse(begin("a"))
        served_extend("a")
        self.assertFalse(begin("b"))
        calls = len(rounds)
        begin("a", "b")
        self.assertEqual(len(rounds), calls)  # batch size 2 is not served
        self.assertIsNone(served_extend("a", "b"))  # SGLang's top-1 there


if __name__ == "__main__":
    unittest.main()
