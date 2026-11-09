"""DSpark verify on vocabulary-sharded target logits (SGLANG_ENABLE_SHARDED_VERIFY).

TP is simulated on one device: a fake group's ``all_gather_into_tensor`` joins the
ranks' tensors in rank order. The accept on full rows is the reference. Triton
kernels run only under TRITON_INTERPRET=1 or on CUDA; elsewhere the greedy key
exchange is replaced by the argmax over the joined rows, and the sampled accept's
kernels by recorders.
"""

import os
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import msgspec
import torch

from sglang.kernels.ops.speculative.dspark import sharded_accept_hip
from sglang.srt.environ import envs
from sglang.srt.layers import logits_processor as lp
from sglang.srt.layers.logits_processor import (
    VERIFY_FULL_LOGITS,
    VERIFY_LOGITS_VARIANTS,
    VERIFY_VOCAB_SHARDS,
    LogitsProcessor,
    LogitsProcessorOutput,
)
from sglang.srt.layers.vocab_parallel_embedding import VocabParallelEmbedding
from sglang.srt.model_executor.runner.decode_cuda_graph_runner import (
    DecodeCudaGraphRunner,
)
from sglang.srt.speculative import eagle_worker_common, sharded_verify
from sglang.srt.speculative.dspark_components import dspark_verify, dspark_worker_v2
from sglang.srt.speculative.dspark_components.dspark_draft import DraftBlockResult
from sglang.srt.speculative.dspark_components.dspark_observability import (
    DsparkStepObservers,
)
from sglang.srt.speculative.dspark_components.dspark_verify import (
    DsparkVerifyEpilogue,
    TargetVerifyExecutor,
    accept_draft_tokens,
    full_logits_unless_kept,
)
from sglang.srt.speculative.dspark_components.dspark_worker_v2 import (
    DSparkWorkerV2,
    target_vocab_shards_refusal,
)
from sglang.srt.speculative.sharded_verify import (
    VocabShard,
    gather_full_logits,
    key_splits,
    reduce_sharded_token_keys,
    shard_keys,
    sharded_target_tokens,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

TP, WIDTH, BS, STRIDE = 4, 40, 3, 6
VOCAB = TP * WIDTH
_TRITON = torch.cuda.is_available() or os.environ.get("TRITON_INTERPRET") == "1"


class _FakeGroup:
    """all_gather_into_tensor over every rank's tensor, held on one device."""

    def __init__(self, per_rank):
        self.per_rank, self.world_size = list(per_rank), len(per_rank)
        self.gathers = 0

    def all_gather_into_tensor(self, output, input):
        self.gathers += 1
        output.copy_(torch.cat([t.reshape(-1) for t in self.per_rank]))


class _NoSync:
    def sync(self, site, values):
        return values

    def enabled(self, site):
        return False


def _split(full):
    return [part.contiguous() for part in full.split(WIDTH, dim=1)]


def _target_shard(rank=0):
    return VocabShard(start=rank * WIDTH, valid=WIDTH, vocab_size=VOCAB)


def _greedy_case(seed=0):
    """Candidates and full target rows whose argmax matches the drafts for 0, 2 and
    all positions; row 0 holds a tie across shards (token 5 against 2 * WIDTH + 7)."""
    torch.manual_seed(seed)
    candidates = torch.randint(0, VOCAB, (BS, STRIDE))
    predict = torch.randint(0, VOCAB, (BS, STRIDE))
    predict[1, :2] = candidates[1, 1:3]
    predict[2, :-1] = candidates[2, 1:]
    full = torch.randn(BS * STRIDE, VOCAB)
    full.scatter_(1, predict.view(-1, 1), 50.0)
    full[0, 5] = full[0, 2 * WIDTH + 7] = 60.0
    return candidates, full.bfloat16()


def _argmax_exchange(full):
    """The key exchange's result over the joined rows, for CPU runs."""

    def tokens(*, logits, shard, tp_group):
        assert logits.shape[1] == WIDTH and shard.vocab_size == VOCAB
        return torch.argmax(full.float()[:, :VOCAB], dim=-1)

    return tokens


def _tp_group(per_rank=()):
    return patch.object(
        dspark_verify,
        "get_parallel",
        return_value=SimpleNamespace(tp_group=_FakeGroup(per_rank)),
    )


def _greedy_block(greedy=True):
    return DraftBlockResult(
        draft_tokens=torch.zeros(BS, STRIDE - 1, dtype=torch.int64),
        corrected_logits=None,
        greedy_mask=torch.full((BS,), greedy),
        temperatures=torch.ones(BS),
        greedy_draft=True,
    )


def _sampling(**overrides):
    fields = dict(
        temperatures=torch.ones(BS, 1),
        is_all_greedy=False,
        is_any_greedy=True,
        need_top_k_sampling=False,
        need_top_p_sampling=False,
        need_min_p_sampling=False,
        acc_additive_penalties=None,
        acc_scaling_penalties=None,
        acc_linear_penalties=None,
        logit_bias=None,
        sampling_seed=None,
        has_watermark_candidates=False,
        sampling_mask_batch_indices=None,
        has_custom_logit_processor=False,
        penalizer_orchestrator=None,
        grammar_mask=None,
    )
    fields.update(overrides)
    return SimpleNamespace(**fields)


def _bare_processor(*, eligible=True, softcapping=None):
    proc = LogitsProcessor.__new__(LogitsProcessor)
    torch.nn.Module.__init__(proc)
    proc._sharded_verify_eligible = eligible
    proc.sharded_verify = False
    proc._sharded_verify_temperature_keys = False
    proc.verify_logits_variants = VERIFY_LOGITS_VARIANTS
    proc.final_logit_softcapping = softcapping
    proc.vocab_size = VOCAB
    return proc


class TestSwitch(CustomTestCase):
    def test_switch_defaults_off(self):
        with patch.dict(os.environ):
            os.environ.pop("SGLANG_ENABLE_SHARDED_VERIFY", None)
            self.assertFalse(envs.SGLANG_ENABLE_SHARDED_VERIFY.get())

    def test_vocab_shards_follow_either_switch(self):
        for nv_mega, sharded in ((False, False), (True, False), (False, True)):
            with (
                self.subTest(nv_mega=nv_mega, sharded=sharded),
                envs.SGLANG_ENABLE_NV_MEGA_RUNTIME_OPTIMIZATION.override(nv_mega),
                envs.SGLANG_ENABLE_SHARDED_VERIFY.override(sharded),
            ):
                self.assertEqual(
                    lp._runtime_optimized_vocab_shards_enabled(), nv_mega or sharded
                )


class TestShardsOnlyGraphs(CustomTestCase):
    def test_opt_in_formats(self):
        proc = _bare_processor()
        LogitsProcessor.enable_sharded_verify(proc, temperature_keys=True)
        self.assertTrue(proc.sharded_verify)
        self.assertEqual(proc.verify_logits_variants, VERIFY_LOGITS_VARIANTS)
        LogitsProcessor.enable_sharded_verify(
            proc, temperature_keys=True, full_logits_graphs=False
        )
        self.assertTrue(proc._sharded_verify_temperature_keys)
        self.assertEqual(proc.verify_logits_variants, (VERIFY_VOCAB_SHARDS,))
        ineligible = _bare_processor(eligible=False)
        LogitsProcessor.enable_sharded_verify(ineligible, full_logits_graphs=False)
        self.assertFalse(ineligible.sharded_verify)

    def test_replay_picks_the_captured_format(self):
        resolve = DecodeCudaGraphRunner._resolve_logits_variant
        for keeps in (False, True):
            proc = SimpleNamespace(verify_uses_vocab_shards=lambda fb, k=keeps: k)
            both = SimpleNamespace(
                _verify_logits_processor=proc,
                logits_graph_variants=VERIFY_LOGITS_VARIANTS,
            )
            shards_only = SimpleNamespace(
                _verify_logits_processor=proc,
                logits_graph_variants=(VERIFY_VOCAB_SHARDS,),
            )
            with self.subTest(keeps=keeps):
                self.assertEqual(
                    resolve(both, None),
                    VERIFY_VOCAB_SHARDS if keeps else VERIFY_FULL_LOGITS,
                )
                self.assertEqual(resolve(both, None, padded=True), VERIFY_VOCAB_SHARDS)
                self.assertEqual(resolve(shards_only, None), VERIFY_VOCAB_SHARDS)
        stock = SimpleNamespace(
            _verify_logits_processor=None, logits_graph_variants=(None,)
        )
        self.assertIsNone(resolve(stock, None))


def _torch_keys(logits, start):
    """sharded_verify's int64 keys of one shard, one split, in torch."""
    x = logits.float()
    x = torch.where(x == x, x, torch.full_like(x, float("-inf"))) + 0.0
    bits = x.view(torch.int32)
    ordered = torch.where(bits < 0, bits ^ 0x7FFFFFFF, bits).to(torch.int64)
    gid = torch.arange(start, start + x.shape[1], dtype=torch.int64)
    return ((ordered << 32) | (0xFFFFFFFF - gid)).amax(dim=1)


class TestKeyExchange(CustomTestCase):
    def test_rocm_uses_the_collective_only(self):
        _, full = _greedy_case()
        keys = [_torch_keys(part, r * WIDTH) for r, part in enumerate(_split(full))]
        group = _FakeGroup(keys)
        nvlink = MagicMock(side_effect=AssertionError("NVLink exchange on ROCm"))
        with (
            patch.object(sharded_verify, "_is_hip", True),
            patch.object(sharded_verify, "_ll_keys", nvlink),
            patch.object(sharded_verify, "_mc_gather_keys", nvlink),
        ):
            got = reduce_sharded_token_keys(keys[0], group, splits=1)
        nvlink.assert_not_called()
        self.assertEqual(group.gathers, 1)
        self.assertTrue(torch.equal(got, torch.argmax(full.float(), dim=-1)))

    def test_cuda_still_tries_the_nvlink_exchanges_first(self):
        keys = torch.zeros(BS * STRIDE, dtype=torch.int64)
        tokens = torch.arange(BS * STRIDE)
        with (
            patch.object(sharded_verify, "_is_hip", False),
            patch.object(sharded_verify, "_ll_keys", return_value=tokens) as ll,
        ):
            got = reduce_sharded_token_keys(keys, _FakeGroup([keys] * TP), splits=1)
        ll.assert_called_once()
        self.assertIs(got, tokens)

    @unittest.skipUnless(_TRITON, "needs CUDA or TRITON_INTERPRET=1")
    def test_shard_tokens_match_argmax(self):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        _, full = _greedy_case()
        full = full.to(device)
        full[1].fill_(1.5)
        full[2].zero_()
        parts = _split(full)
        splits = key_splits(WIDTH)
        keys = [
            shard_keys(part, _target_shard(r), splits) for r, part in enumerate(parts)
        ]
        with patch.object(sharded_verify, "_is_hip", True):
            got = sharded_target_tokens(parts[0], _target_shard(0), _FakeGroup(keys))
        self.assertTrue(torch.equal(got, torch.argmax(full.float(), dim=-1)))


class TestGatherFallback(CustomTestCase):
    def test_gather_full_logits_on_cpu(self):
        _, full = _greedy_case()
        parts = _split(full)
        got = gather_full_logits(parts[0], _target_shard(0), _FakeGroup(parts))
        self.assertEqual(got.dtype, torch.float32)
        self.assertTrue(torch.equal(got, full.float()))

    def test_full_logits_unless_kept(self):
        _, full = _greedy_case()
        parts = _split(full)
        parallel = SimpleNamespace(tp_group=_FakeGroup(parts))

        def call(out, logits, keep):
            return full_logits_unless_kept(
                logits_output=out, logits=logits, keep_target_shards=keep
            )

        with patch.object(dspark_verify, "get_parallel", return_value=parallel):
            kept = SimpleNamespace(next_token_logits_vocab_shard=_target_shard(0))
            self.assertIs(call(kept, parts[0], True), parts[0])
            self.assertIsNotNone(kept.next_token_logits_vocab_shard)
            gathered = SimpleNamespace(next_token_logits_vocab_shard=_target_shard(0))
            got = call(gathered, parts[0], False)
            self.assertTrue(torch.equal(got, full.float()))
            self.assertIsNone(gathered.next_token_logits_vocab_shard)
            stock = SimpleNamespace(next_token_logits_vocab_shard=None)
            logits = full.float()
            for keep in (False, True):
                self.assertIs(call(stock, logits, keep), logits)
        self.assertEqual(parallel.tp_group.gathers, 1)

    def test_run_non_compact_gathers_before_the_adjustments(self):
        _, full = _greedy_case()
        parts = _split(full)
        seen = []
        executor = TargetVerifyExecutor(
            target_worker=SimpleNamespace(
                model_runner=SimpleNamespace(
                    attn_backend=SimpleNamespace(),
                    model_config=SimpleNamespace(
                        hf_text_config=SimpleNamespace(model_type="test")
                    ),
                ),
                random_seed=0,
            ),
            gamma=STRIDE - 1,
            verify_num_draft_tokens=STRIDE,
            model_runner=None,
            kv_injector=None,
            tp_sync=_NoSync(),
        )
        window = SimpleNamespace(
            positions_2d=torch.zeros(BS, STRIDE, dtype=torch.int64),
            verify_cache_loc=torch.zeros(BS * STRIDE, dtype=torch.int64),
            iteration_ids=None,
        )
        for keep in (False, True):
            out = SimpleNamespace(
                next_token_logits=parts[0],
                next_token_logits_vocab_shard=_target_shard(0),
            )
            result = dspark_verify.TargetVerifyResult(
                logits_output=out, can_run_cuda_graph=False
            )
            batch = SimpleNamespace(seq_lens_cpu=None, seq_lens_sum=0)
            with (
                self.subTest(keep=keep),
                patch.object(
                    dspark_verify,
                    "get_parallel",
                    return_value=SimpleNamespace(tp_group=_FakeGroup(parts)),
                ),
                patch.object(
                    executor, "_verify_backend_self_adds_seq_lens", return_value=True
                ),
                patch.object(executor, "_forward_prepared_verify", return_value=result),
                patch.object(
                    dspark_verify,
                    "apply_dflash_verify_logits_adjustments",
                    side_effect=lambda **kw: seen.append(kw["next_token_logits"].shape),
                ),
            ):
                executor.run_non_compact(
                    batch=batch,
                    draft_input=SimpleNamespace(nxt_kv_lens_cpu=None),
                    verify_ids_2d=torch.zeros(BS, STRIDE, dtype=torch.int64),
                    verify_window=window,
                    sampling_info=_sampling(),
                    keep_target_shards=keep,
                )
            self.assertEqual(seen[-1], (BS * STRIDE, WIDTH if keep else VOCAB))
            self.assertEqual(out.next_token_logits_vocab_shard is None, not keep)


class TestAcceptOnShards(CustomTestCase):
    def _accept(self, *, target_logits, target_shard=None, verify_lens=None, **kw):
        return accept_draft_tokens(
            candidates=kw.pop("candidates"),
            target_logits=target_logits,
            draft_block=kw.pop("draft_block", _greedy_block()),
            sampling_info=kw.pop("sampling_info", None),
            draft_input=None,
            gamma=STRIDE - 1,
            verify_num_draft_tokens=STRIDE,
            cutoff_layout=(
                None
                if verify_lens is None
                else SimpleNamespace(verify_lens=verify_lens)
            ),
            target_shard=target_shard,
        )

    def test_greedy_on_shards_equals_full_rows(self):
        ragged = torch.tensor([STRIDE, 2, 3])
        for seed in (0, 1):
            for verify_lens in (None, ragged):
                candidates, full = _greedy_case(seed)
                with self.subTest(seed=seed, ragged=verify_lens is not None):
                    want = self._accept(
                        candidates=candidates,
                        target_logits=full.float(),
                        verify_lens=verify_lens,
                    )
                    exchange = MagicMock(side_effect=_argmax_exchange(full))
                    with (
                        _tp_group(),
                        patch.object(dspark_verify, "sharded_target_tokens", exchange),
                    ):
                        got = self._accept(
                            candidates=candidates,
                            target_logits=_split(full)[0],
                            target_shard=_target_shard(0),
                            verify_lens=verify_lens,
                        )
                    exchange.assert_called_once()
                    for g, w in zip(got, want):
                        self.assertTrue(torch.equal(g.long(), w.long()))

    def _sampled_block(self, draft_start=0):
        return DraftBlockResult(
            draft_tokens=torch.zeros(BS, STRIDE - 1, dtype=torch.int64),
            corrected_logits=torch.zeros(BS, STRIDE - 1, WIDTH),
            greedy_mask=torch.tensor([True, False, False]),
            temperatures=torch.ones(BS),
            greedy_draft=False,
            vocab_shard=SimpleNamespace(
                start=draft_start,
                width=WIDTH,
                gather=SimpleNamespace(gather_stacked=lambda t: t),
            ),
            noise=SimpleNamespace(key=7, offset=torch.zeros(1, dtype=torch.int64)),
        )

    def test_mixed_batch_reads_the_target_shard(self):
        candidates, full = _greedy_case()
        sampled = (
            torch.tensor([4, 1, 2], dtype=torch.int32),
            torch.tensor([9, 8, 7]),
            torch.zeros(BS, dtype=torch.int32),
        )
        greedy_want = self._accept(candidates=candidates, target_logits=full.float())
        for target_shard in (None, _target_shard(0)):
            recorder = MagicMock(return_value=sampled)
            logits = full.float() if target_shard is None else _split(full)[0]
            with (
                self.subTest(on_shards=target_shard is not None),
                _tp_group(),
                patch.object(dspark_verify, "accept_sampling_sharded", recorder),
                patch.object(
                    dspark_verify, "sharded_target_tokens", _argmax_exchange(full)
                ),
            ):
                got = self._accept(
                    candidates=candidates,
                    target_logits=logits,
                    target_shard=target_shard,
                    draft_block=self._sampled_block(),
                    sampling_info=_sampling(),
                )
            kwargs = recorder.call_args.kwargs
            self.assertIs(kwargs["target_logits"], logits)
            want_start = None if target_shard is None else 0
            self.assertEqual(kwargs["target_start"], want_start)
            self.assertEqual((kwargs["vocab_start"], kwargs["width"]), (0, WIDTH))
            correct_len, bonus, _ = got
            # Row 0 is greedy (the accept on the joined rows), rows 1-2 sampled.
            self.assertEqual(int(correct_len[0]), int(greedy_want[0][0]))
            self.assertEqual(int(bonus[0]), int(greedy_want[1][0]))
            self.assertEqual(correct_len[1:].tolist(), [1, 2])
            self.assertEqual(bonus[1:].tolist(), [8, 7])

    def test_sampled_rows_need_matching_shards(self):
        candidates, full = _greedy_case()
        with self.assertRaises(ValueError):
            self._accept(
                candidates=candidates,
                target_logits=_split(full)[0],
                target_shard=_target_shard(0),
                draft_block=self._sampled_block(draft_start=WIDTH),
                sampling_info=_sampling(is_any_greedy=False),
            )
        with self.assertRaises(ValueError):
            self._accept(
                candidates=candidates,
                target_logits=_split(full)[0],
                target_shard=_target_shard(0),
                draft_block=_greedy_block(greedy=False),
                sampling_info=_sampling(is_any_greedy=False),
            )


class _Launch:
    """Records ``kernel[grid](*args, **kwargs)``."""

    def __init__(self):
        self.calls = []

    def __getitem__(self, grid):
        return lambda *args, **kwargs: self.calls.append(args)


class TestSampledAcceptLaunch(CustomTestCase):
    def _launch(self, *, target_logits, vocab_start, target_start=None):
        stats, accept, finish = _Launch(), _Launch(), _Launch()
        with (
            patch.object(sharded_accept_hip, "_stats_kernel", stats),
            patch.object(sharded_accept_hip, "_accept_kernel", accept),
            patch.object(sharded_accept_hip, "_markov_sample_finish_kernel", finish),
        ):
            sharded_accept_hip.accept_sampling_sharded(
                candidates=torch.zeros(BS, STRIDE, dtype=torch.int64),
                target_logits=target_logits,
                draft_logits=torch.zeros(BS, STRIDE - 1, WIDTH),
                temperatures=torch.ones(BS),
                vocab_start=vocab_start,
                width=WIDTH,
                gather_stacked=lambda t: t,
                key=7,
                offset=torch.zeros(1, dtype=torch.int64),
                target_start=target_start,
            )
        # (TARGET_S, TARGET_START, START) of each kernel
        return stats.calls[0][5:8], accept.calls[0][11:14]

    def test_full_rows_keep_todays_addressing(self):
        rows = torch.zeros(BS * STRIDE, VOCAB)
        stats, accept = self._launch(target_logits=rows, vocab_start=2 * WIDTH)
        self.assertEqual(stats, (VOCAB, 2 * WIDTH, 2 * WIDTH))
        self.assertEqual(accept, (VOCAB, 2 * WIDTH, 2 * WIDTH))

    def test_shard_reads_from_column_zero(self):
        shard = torch.zeros(BS * STRIDE, WIDTH)
        stats, accept = self._launch(
            target_logits=shard, vocab_start=2 * WIDTH, target_start=0
        )
        self.assertEqual(stats, (WIDTH, 0, 2 * WIDTH))
        self.assertEqual(accept, (WIDTH, 0, 2 * WIDTH))
        with self.assertRaises(AssertionError):
            self._launch(target_logits=shard, vocab_start=2 * WIDTH)


class TestEpilogueOnShards(CustomTestCase):
    def _capture(self, logits, vocab_shard, candidates, fused):
        """The static verify graph's in-graph accept, through its capture hook."""
        epilogue = DsparkVerifyEpilogue(
            max_bs=BS + 1,
            verify_num_draft_tokens=STRIDE,
            device="cpu",
            tp_sync=_NoSync(),
            fused_accept_tail=fused,
        )
        rows = candidates.numel()
        epilogue.capture_hook(
            SimpleNamespace(
                model_runner=SimpleNamespace(is_draft_worker=False),
                ragged_verify_mode=False,
            ),
            LogitsProcessorOutput(
                next_token_logits=logits,
                hidden_states=torch.zeros(rows, 4),
                next_token_logits_vocab_shard=vocab_shard,
            ),
            SimpleNamespace(
                forward_mode=SimpleNamespace(is_target_verify=lambda: True),
                batch_size=BS,
                input_ids=candidates.view(-1),
                seq_lens=torch.tensor([7, 11, 13]),
                req_pool_indices=torch.arange(BS),
                out_cache_loc=torch.zeros(rows, dtype=torch.int64),
                positions=torch.zeros(rows, dtype=torch.int64),
            ),
            rows,
        )
        return msgspec.structs.asdict(epilogue.read_accept(BS))

    def test_in_graph_greedy_accept_reads_the_shard_keys(self):
        candidates, full = _greedy_case()
        for fused in (False, True):
            with self.subTest(fused=fused):
                want = self._capture(full.float(), None, candidates, fused)
                exchange = MagicMock(side_effect=_argmax_exchange(full))
                with (
                    _tp_group(),
                    patch.object(dspark_verify, "sharded_target_tokens", exchange),
                ):
                    got = self._capture(
                        _split(full)[0], _target_shard(0), candidates, fused
                    )
                exchange.assert_called_once()
                for field, value in want.items():
                    self.assertTrue(torch.equal(got[field], value), field)


def _lm_head(added=0):
    head = VocabParallelEmbedding.__new__(VocabParallelEmbedding)
    torch.nn.Module.__init__(head)
    head.shard_indices = SimpleNamespace(
        org_vocab_start_index=0,
        num_org_elements=WIDTH,
        num_added_elements_padded=added,
    )
    return head


def _worker(**overrides):
    proc = overrides.pop("proc", _bare_processor())
    model = SimpleNamespace(
        modules=lambda: [torch.nn.Identity(), proc],
        lm_head=overrides.pop("lm_head", _lm_head()),
    )
    fields = dict(
        target_worker=SimpleNamespace(
            model_runner=SimpleNamespace(
                model=model,
                model_config=SimpleNamespace(
                    hf_text_config=SimpleNamespace(model_type="deepseek_v41")
                ),
            )
        ),
        _observers=SimpleNamespace(reads_target_logits=False),
        _simulate_acc_len=0.0,
        draft_model=SimpleNamespace(
            markov_head=SimpleNamespace(
                vocab_shard=SimpleNamespace(start=0, width=WIDTH)
            )
        ),
        _sharded_accept=True,
        _rejection_noise=None,
        _target_tp_rank=0,
        _sampling_accept_graph=object(),
        _target_temperature_keys=False,
    )
    fields.update(overrides)
    return SimpleNamespace(**fields), proc


def _spec(**overrides):
    fields = dict(
        speculative_use_rejection_sampling=False,
        speculative_use_block_verification=False,
        speculative_accept_threshold_single=1.0,
        speculative_accept_threshold_acc=1.0,
    )
    fields.update(overrides)
    return SimpleNamespace(**fields)


class TestWorkerOptIn(CustomTestCase):
    def _enable(self, worker, *, switch=True, tp_size=TP, spec=None):
        with (
            envs.SGLANG_ENABLE_SHARDED_VERIFY.override(switch),
            patch.object(
                dspark_worker_v2,
                "get_parallel",
                return_value=SimpleNamespace(tp_size=tp_size),
            ),
            patch.object(dspark_worker_v2, "get_spec", return_value=spec or _spec()),
        ):
            return DSparkWorkerV2._enable_target_vocab_shards(worker)

    def test_switch_off_touches_nothing(self):
        worker, proc = _worker()
        region = worker._sampling_accept_graph
        self.assertIsNone(self._enable(worker, switch=False))
        self.assertFalse(proc.sharded_verify)
        self.assertFalse(proc._sharded_verify_temperature_keys)
        self.assertEqual(proc.verify_logits_variants, VERIFY_LOGITS_VARIANTS)
        self.assertIs(worker._sampling_accept_graph, region)
        self.assertFalse(worker._target_temperature_keys)

    def test_opt_in(self):
        worker, proc = _worker()
        shard = self._enable(worker)
        self.assertEqual(shard, _target_shard(0))
        self.assertTrue(proc.sharded_verify)
        self.assertEqual(proc.verify_logits_variants, (VERIFY_VOCAB_SHARDS,))
        self.assertTrue(proc._sharded_verify_temperature_keys)
        self.assertTrue(worker._target_temperature_keys)
        self.assertIsNone(worker._sampling_accept_graph)

    def test_sampled_rows_stay_gathered_without_a_matching_draft_shard(self):
        cases = {
            "no sharded draft accept": dict(_sharded_accept=False),
            "other draft columns": dict(
                draft_model=SimpleNamespace(
                    markov_head=SimpleNamespace(
                        vocab_shard=SimpleNamespace(start=WIDTH, width=WIDTH)
                    )
                )
            ),
            "rejection noise": dict(_rejection_noise=object()),
        }
        for name, overrides in cases.items():
            with self.subTest(name):
                worker, proc = _worker(**overrides)
                self.assertIsNotNone(self._enable(worker))
                self.assertFalse(proc._sharded_verify_temperature_keys)
        worker, proc = _worker()
        self._enable(worker, spec=_spec(speculative_use_block_verification=True))
        self.assertFalse(proc._sharded_verify_temperature_keys)

    def test_refusals(self):
        def refusal(**overrides):
            kw = dict(
                tp_size=TP,
                model_type="deepseek_v41",
                processors=[_bare_processor()],
                simulate_acc_len=0.0,
                observers_read_logits=False,
            )
            kw.update(overrides)
            return target_vocab_shards_refusal(**kw)

        self.assertIsNone(refusal())
        refused = {
            "tp1": dict(tp_size=1),
            "model": dict(model_type="qwen3"),
            "no processor": dict(processors=[]),
            "two processors": dict(processors=[_bare_processor(), _bare_processor()]),
            "softcap": dict(processors=[_bare_processor(softcapping=30.0)]),
            "simulated acceptance": dict(simulate_acc_len=2.0),
            "observer": dict(observers_read_logits=True),
        }
        for name, overrides in refused.items():
            with self.subTest(name):
                self.assertIsNotNone(refusal(**overrides))

    def test_refused_opt_in_keeps_the_stock_path(self):
        cases = {
            "tp1": (dict(), dict(tp_size=1)),
            "observer": (
                dict(_observers=SimpleNamespace(reads_target_logits=True)),
                dict(),
            ),
            "added vocabulary": (dict(lm_head=_lm_head(added=8)), dict()),
            "lm head of another type": (dict(lm_head=torch.nn.Identity()), dict()),
            "ineligible processor": (dict(proc=_bare_processor(eligible=False)), {}),
        }
        for name, (worker_kw, enable_kw) in cases.items():
            with self.subTest(name):
                worker, proc = _worker(**worker_kw)
                region = worker._sampling_accept_graph
                self.assertIsNone(self._enable(worker, **enable_kw))
                self.assertFalse(proc.sharded_verify)
                self.assertIs(worker._sampling_accept_graph, region)
                self.assertFalse(worker._target_temperature_keys)


class TestObserversThatReadTargetRows(CustomTestCase):
    def test_each_sink_that_reads_the_rows_refuses_shards(self):
        reads = DsparkStepObservers.reads_target_logits.fget
        quiet = dict(_sts_collect_path="", _block_accept_recorder=None)
        with envs.SGLANG_DSPARK_DEBUG_CONFIDENCE_METRICS.override(False):
            self.assertFalse(reads(SimpleNamespace(**quiet)))
            for sink in (
                dict(_sts_collect_path="/tmp/sts"),
                dict(_block_accept_recorder=object()),
            ):
                with self.subTest(sink=next(iter(sink))):
                    self.assertTrue(reads(SimpleNamespace(**{**quiet, **sink})))
        with envs.SGLANG_DSPARK_DEBUG_CONFIDENCE_METRICS.override(True):
            self.assertTrue(reads(SimpleNamespace(**quiet)))


class TestPerStepKeep(CustomTestCase):
    def _keeps(self, *, opted_in=True, temperature_keys=True, batch=None, **sampling):
        worker = SimpleNamespace(
            _target_vocab_shard=_target_shard(0) if opted_in else None,
            _target_temperature_keys=temperature_keys,
        )
        draft_block = DraftBlockResult(
            draft_tokens=torch.zeros(BS, STRIDE - 1, dtype=torch.int64),
            corrected_logits=None,
            greedy_mask=torch.zeros(BS, dtype=torch.bool),
            temperatures=torch.ones(BS),
            greedy_draft=False,
            vocab_shard=SimpleNamespace(start=0, width=WIDTH),
        )
        info = None if sampling.pop("none", False) else _sampling(**sampling)
        with patch.object(dspark_worker_v2, "get_spec", return_value=_spec()):
            return DSparkWorkerV2._keeps_target_vocab_shards(
                worker,
                batch=batch or SimpleNamespace(has_grammar=False, return_logprob=False),
                sampling_info=info,
                draft_block=draft_block,
            )

    def test_keep_or_gather_table(self):
        greedy = dict(is_all_greedy=True)
        table = {
            "greedy": (True, greedy),
            "no sampling info": (True, dict(none=True)),
            "temperature": (True, {}),
            "temperature, no keys": (False, dict(temperature_keys=False)),
            "greedy, no keys": (True, dict(temperature_keys=False, **greedy)),
            "top-k": (False, dict(need_top_k_sampling=True)),
            "top-p": (False, dict(need_top_p_sampling=True)),
            "min-p": (False, dict(need_min_p_sampling=True)),
            "seed": (False, dict(sampling_seed=torch.zeros(BS))),
            "greedy with seed": (True, dict(sampling_seed=torch.zeros(BS), **greedy)),
            "penalty": (False, dict(acc_additive_penalties=torch.zeros(BS, 1))),
            "linear penalty": (False, dict(acc_linear_penalties=torch.zeros(BS, 1))),
            "logit bias": (False, dict(logit_bias=torch.zeros(BS, 1), **greedy)),
            "custom processor": (False, dict(has_custom_logit_processor=True)),
            "watermark": (False, dict(has_watermark_candidates=True, **greedy)),
            "sampling mask": (
                False,
                dict(sampling_mask_batch_indices=torch.zeros(1), **greedy),
            ),
            "logprobs": (
                False,
                dict(batch=SimpleNamespace(has_grammar=False, return_logprob=True)),
            ),
            "grammar": (
                False,
                dict(batch=SimpleNamespace(has_grammar=True, return_logprob=False)),
            ),
            "not opted in": (False, dict(opted_in=False, **greedy)),
        }
        for name, (want, kw) in table.items():
            with self.subTest(name):
                self.assertEqual(self._keeps(**kw), want)


class TestEagleBinding(CustomTestCase):
    def test_eagle_opts_in_on_cuda_only(self):
        proc = _bare_processor()
        proc.enable_sharded_verify = MagicMock()
        model_cls = type("Qwen3_5ForCausalLM", (), {"modules": lambda self: [proc]})
        worker = SimpleNamespace(model_runner=SimpleNamespace(model=model_cls()))
        for cuda in (True, False):
            proc.enable_sharded_verify.reset_mock()
            with (
                self.subTest(cuda=cuda),
                patch.object(eagle_worker_common, "is_cuda", return_value=cuda),
                patch.object(
                    eagle_worker_common, "import_module", return_value=SimpleNamespace()
                ),
            ):
                eagle_worker_common.bind_eagle_runtime_hooks(
                    worker, supports_sharded_verify=True
                )
            self.assertEqual(proc.enable_sharded_verify.called, cuda)


if __name__ == "__main__":
    unittest.main()
