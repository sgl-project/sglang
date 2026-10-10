"""Standalone CUDA checks; imports only the kernel module, not a model/server."""

import importlib.util
import unittest
from pathlib import Path

import torch

_path = (
    Path(__file__).resolve().parents[3]
    / "python/sglang/kernels/ops/embeddings/engram_hash.py"
)
_spec = importlib.util.spec_from_file_location("engram_hash_test_kernel", _path)
kernel = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(kernel)


def fixture(lengths, padding=0, explicit_history=False, image=False, dtype=torch.int32):
    torch.manual_seed(27)
    bs, n, layers, heads, vocab = len(lengths), 4, 2, 8, 256
    real = sum(lengths)
    ids = torch.randint(0, vocab, (real + padding,), device="cuda")
    lens = torch.tensor(lengths, dtype=dtype, device="cuda")
    starts = (lens.cumsum(0) - lens).to(dtype)
    slots = torch.randperm(bs + 3, device="cuda")[:bs].to(dtype)
    history = torch.randint(0, vocab, (bs + 4, n - 1), device="cuda", dtype=torch.int32)
    source_history = (
        torch.randint_like(history[:bs], 0, vocab) if explicit_history else history
    )
    positions = torch.zeros_like(ids)
    offset = 0
    for r, length in enumerate(lengths):
        positions[offset : offset + length] = torch.arange(length, device="cuda") + (
            0 if r % 2 else 19
        )
        offset += length
    if image:
        if real:
            ids[0] = 4096
        if real > 2:
            ids[2] = 7
        source_history[:, -1] = 4096
    primes = torch.arange(
        16000001, 16000001 + layers * (n - 1) * heads, device="cuda"
    ).reshape(layers, n - 1, heads)
    offsets = primes.flatten(1).cumsum(1) - primes.flatten(1)
    return dict(
        input_ids=ids,
        positions=positions,
        history=source_history,
        commit_history=history,
        req_slots=slots,
        starts=starts,
        lengths=lens,
        num_real=real,
        history_via_slots=not explicit_history,
        out_cache_loc=torch.ones_like(ids),
        token_map=torch.randperm(vocab, device="cuda"),
        multipliers=torch.randint(0, 2**48, (layers, n), device="cuda") * 2 + 1,
        primes=primes,
        offsets=offsets,
        pad_id=2,
        image_token_id=7 if image else None,
        mm_pad_shift=4096,
    )


def reference(kwargs):
    """Original eager row preparation, hash call, and PyTorch history update."""
    lens = kwargs["lengths"].to(torch.int64)
    starts = kwargs["starts"].to(torch.int64)
    slots = kwargs["req_slots"]
    row = torch.repeat_interleave(
        torch.arange(slots.numel(), device="cuda"), lens, output_size=kwargs["num_real"]
    )
    hash_ids, tokens = kernel.engram_hash_ids(
        kwargs["input_ids"],
        kwargs["positions"],
        mode=kernel.MODE_EXTEND,
        history=kwargs["history"],
        req_slots=slots if kwargs["history_via_slots"] else None,
        row=row,
        starts=starts,
        **{
            k: kwargs[k]
            for k in (
                "num_real",
                "token_map",
                "multipliers",
                "primes",
                "offsets",
                "pad_id",
                "image_token_id",
                "mm_pad_shift",
            )
        },
    )
    if kwargs["input_ids"].numel():
        pad_row = kwargs["commit_history"].shape[0] - 1
        commit_rows = torch.where(lens > 0, slots, pad_row)
        last = (starts + lens - 1).clamp(0, kwargs["input_ids"].numel() - 1)
        if kwargs["out_cache_loc"] is not None:
            commit_rows = torch.where(
                kwargs["out_cache_loc"][last] == 0, pad_row, commit_rows
            )
        kwargs["commit_history"][commit_rows] = (
            tokens[last, :3].flip(-1).to(torch.int32)
        )
    return hash_ids


def clone_inputs(kwargs):
    result = {
        k: v.clone() if isinstance(v, torch.Tensor) else v for k, v in kwargs.items()
    }
    if kwargs["history"] is kwargs["commit_history"]:
        result["history"] = result["commit_history"]
    return result


@unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
class ExtendHashTest(unittest.TestCase):
    def check_case(self, lengths, **options):
        inputs = fixture(lengths, **options)
        if inputs["num_real"] > 1:
            # A request with an invalid final output slot must not update history.
            inputs["out_cache_loc"][inputs["num_real"] - 1] = 0
        baseline, candidate = clone_inputs(inputs), clone_inputs(inputs)
        for _ in range(2):
            expected = reference(baseline)
            actual = kernel.engram_hash_extend_and_commit(**candidate)
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            # The reserved padding row is deliberately no longer written.
            torch.testing.assert_close(
                candidate["commit_history"][:-1],
                baseline["commit_history"][:-1],
                rtol=0,
                atol=0,
            )
            baseline["out_cache_loc"].fill_(1)
            candidate["out_cache_loc"].fill_(1)

    def test_shapes_history_padding_and_multimodal(self):
        for lengths in ([8192], [1], [2], [0, 3, 0, 5, 0], [3, 1, 7], [1] * 64, [0, 0]):
            for explicit_history in (False, True):
                for image in (False, True):
                    with self.subTest(
                        lengths=lengths, explicit_history=explicit_history, image=image
                    ):
                        self.check_case(
                            lengths,
                            padding=7,
                            explicit_history=explicit_history,
                            image=image,
                        )

    def test_int64_metadata_and_no_output_mask(self):
        inputs = fixture([5, 0, 9], dtype=torch.int64)
        inputs["out_cache_loc"] = None
        a, b = clone_inputs(inputs), clone_inputs(inputs)
        torch.testing.assert_close(
            kernel.engram_hash_extend_and_commit(**a), reference(b), rtol=0, atol=0
        )
        torch.testing.assert_close(
            a["commit_history"][:-1], b["commit_history"][:-1], rtol=0, atol=0
        )


if __name__ == "__main__":
    unittest.main()
