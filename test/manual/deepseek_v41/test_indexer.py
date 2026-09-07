import unittest

import torch
from args_util import to_v41_args
from dsv41_indexer import Indexer
from dsv41_shared import SharedAttentionRuntime
from ref_loader import RefTestCase, assert_equal, randomize_, requires_ref, small_args

from sglang.srt.layers.attention.dsv4.indexer import select_candidate_blocks

# Mirrors the released layout: ratio-2 layers first (owner 1, reader 2), then ratio-1
# layers (owner 3 is the candidate source, reader 4 filters by its candidates).
TWO_LEVEL = dict(
    n_layers=5,
    compress_ratios=(0, 2, 2, 1, 1),
    kv_source_layers=(1, 3),
    index_source_layers=(1, 2, 3, 4),
    candidate_source_layer=3,
    candidate_topk_blocks=2,
    candidate_block_size=4,
)


@requires_ref
class TestIndexer(RefTestCase):
    def test_select_candidate_blocks(self):
        logits = torch.randn(2, 6, 23, dtype=torch.float32)
        compress_lens = torch.tensor([3, 7, 8, 12, 20, 23]).unsqueeze(-1)
        logits = logits.masked_fill(torch.arange(23) >= compress_lens, -torch.inf)
        expected = self.model.select_candidate_blocks(logits, compress_lens, 2, 4)
        actual = select_candidate_blocks(
            logits, compress_lens, topk_blocks=2, block_size=4
        )
        assert_equal(actual, expected)
        assert_equal(
            select_candidate_blocks(logits[:, -1], 23, topk_blocks=2, block_size=4),
            self.model.select_candidate_blocks(logits[:, -1], 23, 2, 4),
        )

    def _build(self, args, layer_id, seed):
        model = self.model
        ref = model.Indexer(args, layer_id)
        randomize_(ref, seed=seed)
        ours = Indexer(to_v41_args(args), layer_id)
        ours.load_state_dict(ref.state_dict())
        freqs = model.precompute_freqs_cis(
            args.rope_head_dim,
            args.max_seq_len,
            args.original_seq_len,
            args.compress_rope_theta,
            args.rope_factor,
            args.beta_fast,
            args.beta_slow,
        )
        ref.freqs_cis = freqs
        return ref, ours, freqs

    def test_ratio_2_owner_and_reader(self):
        self._run_pair(owner=1, reader=2)

    def test_ratio_1_candidate_source_and_reader(self):
        self._run_pair(owner=3, reader=4)

    def _run_pair(self, owner, reader):
        """The owner publishes index keys (and candidates when it is the source); the
        reader scores with its own weights against the shared keys."""
        model = self.model
        args = small_args(model, **TWO_LEVEL)
        layers = [
            self._build(args, layer_id, seed)
            for seed, layer_id in enumerate((owner, reader))
        ]
        shared = SharedAttentionRuntime()
        ratio, bsz = args.compress_ratios[owner], args.max_batch_size
        prefill_len, total = 21, 27
        x = torch.randn(bsz, total, args.dim)
        qr = torch.randn(bsz, total, args.q_lora_rank)
        latent = torch.randn(bsz, total // ratio, args.head_dim)
        steps = [(0, prefill_len)] + [
            (pos, pos + 1) for pos in range(prefill_len, total)
        ]
        for start, end in steps:
            # The owner publishes a latent only when a group completes.
            if start == 0:
                lat = latent[:, : prefill_len // ratio]
            elif (start + 1) % ratio == 0:
                lat = latent[:, start // ratio : start // ratio + 1]
            else:
                lat = None
            for li, (ref, ours, freqs) in enumerate(layers):
                is_owner = li == 0
                expected = ref(
                    x[:, start:end],
                    qr[:, start:end],
                    lat if is_owner else None,
                    start,
                    100,
                )
                actual = ours(
                    x[:, start:end],
                    qr[:, start:end],
                    lat if is_owner else None,
                    start_pos=start,
                    offset=100,
                    freqs_cis=freqs,
                    shared=shared,
                )
                assert_equal(
                    actual,
                    expected,
                    msg=f"layer {(owner, reader)[li]} start_pos={start}",
                )
        assert_equal(shared.index_k, model.shared_attn.index_k, msg="index_k")
        if layers[0][1].is_candidate_source:
            assert_equal(
                shared.candidates, model.shared_attn.candidates, msg="candidates"
            )


if __name__ == "__main__":
    unittest.main()
