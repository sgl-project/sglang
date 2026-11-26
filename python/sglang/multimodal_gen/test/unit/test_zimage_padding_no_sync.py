import unittest

import torch

from sglang.multimodal_gen.runtime.models.dits.zimage import ZImageTransformer2DModel

_replace = ZImageTransformer2DModel._replace_padding_with_token


def _mask_assign_reference(tensor, valid_lens, pad_token):
    """The previous implementation: a boolean-mask write."""
    positions = torch.arange(tensor.shape[1], device=tensor.device).unsqueeze(0)
    lengths = torch.as_tensor(valid_lens, device=tensor.device).reshape(-1, 1)
    out = tensor.clone()
    out[positions >= lengths] = pad_token.to(tensor.dtype)
    return out


@unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
class TestZImagePaddingWithoutSync(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.tensor = torch.randn(3, 40, 64, device="cuda", dtype=torch.bfloat16)
        self.pad = torch.randn(64, device="cuda")

    def _check(self, valid_lens, as_tensor):
        lens = torch.tensor(valid_lens, device="cuda") if as_tensor else valid_lens
        expected = _mask_assign_reference(self.tensor, valid_lens, self.pad)
        torch.cuda.set_sync_debug_mode("error")
        try:
            out = _replace(self.tensor, lens, self.pad)
        finally:
            torch.cuda.set_sync_debug_mode("default")
        self.assertTrue(torch.equal(out, expected))

    def test_list_lengths(self):
        self._check([40, 17, 0], as_tensor=False)

    def test_tensor_lengths(self):
        self._check([40, 17, 0], as_tensor=True)

    def test_no_padding_returns_input(self):
        self.assertIs(_replace(self.tensor, [40, 40, 40], self.pad), self.tensor)


if __name__ == "__main__":
    unittest.main()
