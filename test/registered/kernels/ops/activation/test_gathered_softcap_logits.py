import unittest

import torch

from sglang.kernels.ops.activation.softcap import (
    gathered_softcap_logits,
    softcap_inplace_logits,
    softcap_row_argmax,
)
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=10, stage="extra-a", runner_config="1-gpu-small")


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
class TestGatheredSoftcapLogits(unittest.TestCase):
    def test_matches_permute_then_softcap(self):
        torch.manual_seed(0)
        for world, tokens, shard, vocab in [
            (4, 256, 50512, 202048),
            (4, 7, 1000, 3998),
            (8, 3, 64, 512),
        ]:
            gathered = torch.randn(world, tokens, shard, device="cuda") * 30
            ref = gathered.movedim(0, 1).reshape(tokens, world * shard)
            ref = ref[:, :vocab].contiguous()
            softcap_inplace_logits(ref, 20.0)
            out = torch.empty(tokens, vocab, device="cuda")
            gathered_softcap_logits(gathered, 20.0, out)
            self.assertTrue(torch.equal(out, ref), (world, tokens, shard, vocab))

    def test_sharded_argmax_matches_full_argmax(self):
        torch.manual_seed(0)
        for world, tokens, vocab in [(4, 256, 202048), (4, 9, 4001), (8, 5, 700)]:
            shard = -(-vocab // world)
            full = torch.randn(tokens, vocab, device="cuda") * 40
            full[0, 3] = full[0, vocab - 2] = 1e4
            full[1, shard + 1] = full[1, 2 * shard] = 1e4
            padded = torch.nn.functional.pad(full, (0, world * shard - vocab))
            ref = full.clone()
            softcap_inplace_logits(ref, 20.0)
            parts = torch.stack(
                [
                    softcap_row_argmax(
                        padded[:, r * shard : (r + 1) * shard].contiguous(),
                        20.0,
                        max(0, min(shard, vocab - r * shard)),
                        r * shard,
                    )
                    for r in range(world)
                ]
            )
            best = parts[..., 0].argmax(dim=0, keepdim=True)
            got = parts[..., 1].gather(0, best).squeeze(0).long()
            self.assertTrue(
                torch.equal(got, ref.argmax(dim=-1)), (world, tokens, vocab)
            )


if __name__ == "__main__":
    unittest.main()
