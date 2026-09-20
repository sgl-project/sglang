"""Exercise the FlashMLA wrapper on CPU without importing device libraries."""

import ast
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch

BACKEND_PATH = (
    Path(__file__).resolve().parents[5]
    / "python/sglang/srt/layers/attention/flashmla_backend.py"
)


class _Mode:
    def is_target_verify(self):
        return True

    def is_draft_extend_v2(self):
        return False


def _load_forward_extend(flash_mla):
    tree = ast.parse(BACKEND_PATH.read_text())
    backend = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "FlashMLABackend"
    )
    method = next(
        node
        for node in backend.body
        if isinstance(node, ast.FunctionDef) and node.name == "forward_extend"
    )
    module = ast.Module(
        body=[
            ast.ImportFrom(
                module="__future__",
                names=[ast.alias(name="annotations")],
                level=0,
            ),
            method,
        ],
        type_ignores=[],
    )
    namespace = {
        "torch": torch,
        "PAGE_SIZE": 64,
        "ForwardMode": SimpleNamespace(EXTEND=object()),
        "flash_mla_with_kvcache": flash_mla,
        "concat_mla_absorb_q_general": lambda q, rope: torch.cat((q, rope), -1),
    }
    exec(
        compile(ast.fix_missing_locations(module), str(BACKEND_PATH), "exec"),
        namespace,
    )
    return namespace["forward_extend"]


def _reference_attention(q, keys, *, prefix_len, scale):
    scores = torch.einsum("thd,kd->htk", q, keys) * scale
    positions = torch.arange(keys.shape[0])
    mask = positions[None, :] <= prefix_len + torch.arange(q.shape[0])[:, None]
    scores.masked_fill_(~mask[None, :, :], float("-inf"))
    return torch.einsum("htk,kd->thd", scores.softmax(-1), keys[:, :2])


class TestDSparkFlashMLALayout(unittest.TestCase):
    def _run(
        self,
        *,
        bs,
        width,
        padding=0,
        identical=False,
        split_rope=False,
        dspark=True,
        steps=1,
        spec_width=None,
    ):
        generator = torch.Generator().manual_seed(17)
        queries = torch.randn(bs, width, 2, 4, generator=generator)
        cache = torch.randn(bs, 64, 1, 4, generator=generator)
        prefix_lens = torch.arange(bs, dtype=torch.int32) + 5
        if identical:
            queries[:] = queries[0].clone()
            cache[:] = cache[0].clone()
            prefix_lens[:] = prefix_lens[0].clone()
        expected = torch.stack(
            [
                _reference_attention(
                    queries[i],
                    cache[i, : int(prefix_lens[i]) + width, 0],
                    prefix_len=int(prefix_lens[i]),
                    scale=0.5,
                )
                for i in range(bs)
            ]
        ).reshape(bs * width, 4)

        calls = []

        def flash_mla(**kwargs):
            q = kwargs["q"]
            calls.append(q.clone())
            out = []
            for i in range(bs):
                page = int(kwargs["block_table"][i, 0])
                length = int(kwargs["cache_seqlens"][i])
                # FlashMLA aligns a causal Q window to the end of the KV window.
                out.append(
                    _reference_attention(
                        q[i],
                        kwargs["k_cache"][page, :length, 0],
                        prefix_len=length - q.shape[1],
                        scale=kwargs["softmax_scale"],
                    )
                )
            return torch.stack(out), None

        forward = _load_forward_extend(flash_mla)
        backend = SimpleNamespace(
            speculative_num_steps=steps,
            num_draft_tokens=width,
            kv_cache_dim=4,
            kv_lora_rank=2,
            is_fp8_kvcache=False,
            forward_metadata=SimpleNamespace(
                block_kv_indices=torch.arange(bs).view(bs, 1),
                flashmla_metadata=None,
                num_splits=None,
            ),
            token_to_kv_pool=SimpleNamespace(get_key_buffer=lambda _: cache),
        )
        batch = SimpleNamespace(
            forward_mode=_Mode(),
            batch_size=bs,
            seq_lens=prefix_lens,
            out_cache_loc=None,
            spec_algorithm=SimpleNamespace(is_dspark=lambda: dspark),
            spec_info=SimpleNamespace(
                draft_token_num=width if spec_width is None else spec_width
            ),
        )
        layer = SimpleNamespace(
            layer_id=0,
            tp_q_head_num=2,
            head_dim=4,
            v_head_dim=2,
            scaling=0.5,
        )
        flat_q = queries.reshape(bs * width, 8)
        if padding:
            flat_q = torch.cat((flat_q, torch.full((padding, 8), float("nan"))))
        if split_rope:
            heads = flat_q.view(-1, 2, 4)
            q = heads[..., :2].reshape(-1, 4)
            rope = heads[..., 2:].reshape(-1, 4)
        else:
            q, rope = flat_q, None
        output = forward(
            backend, q, None, None, layer, batch, save_kv_cache=False, q_rope=rope
        )
        self.assertEqual(tuple(calls[0].shape), (bs, width, 2, 4))
        torch.testing.assert_close(calls[0], queries, rtol=0, atol=0)
        torch.testing.assert_close(output[: bs * width], expected)
        self.assertEqual(tuple(output.shape), (bs * width + padding, 4))
        if padding:
            torch.testing.assert_close(output[bs * width :], torch.zeros(padding, 4))

    def test_dspark_full_window_single_and_multi_batch(self):
        for bs in (1, 2, 3, 8):
            for width in (3, 6, 8):
                with self.subTest(bs=bs, width=width):
                    self._run(bs=bs, width=width)

    def test_dspark_identical_requests(self):
        self._run(bs=4, width=8, identical=True)

    def test_dspark_trims_only_trailing_padding(self):
        self._run(bs=3, width=8, padding=8)

    def test_dspark_absorbed_query_with_split_rope(self):
        self._run(bs=3, width=8, padding=8, split_rope=True)

    def test_eagle_layout_is_unchanged(self):
        self._run(bs=3, width=3, padding=2, dspark=False, steps=2)

    def test_non_dspark_target_verify_keeps_step_width(self):
        self._run(
            bs=2,
            width=3,
            padding=4,
            dspark=False,
            steps=2,
            spec_width=5,
        )


if __name__ == "__main__":
    unittest.main()
