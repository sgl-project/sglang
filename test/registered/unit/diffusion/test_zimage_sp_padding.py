"""Global model alignment must not add attention-visible SP image rows."""

import unittest
from contextlib import ExitStack
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.multimodal_gen.configs.pipeline_configs.zimage import ZImagePipelineConfig
from sglang.multimodal_gen.configs.sample.zimage import ZImageSamplingParams
from sglang.multimodal_gen.runtime.models.dits.zimage import (
    ZImageAttention,
    ZImageTransformer2DModel,
)
from sglang.multimodal_gen.runtime.pipelines_core.schedule_batch import Req
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

MODULE = "sglang.multimodal_gen.configs.pipeline_configs.zimage"
MODEL = "sglang.multimodal_gen.runtime.models.dits.zimage"


class TestZImageSPPadding(CustomTestCase):
    def setUp(self):
        self.config = ZImagePipelineConfig()
        self.config.vae_config.post_init()
        self.config._allow_unpadded_sp = True
        self.batch = Req(sampling_params=ZImageSamplingParams(height=1088, width=1920))
        self.batch.prompt_embeds = [torch.zeros(19, 8)]
        self.batch.negative_prompt_embeds = [torch.zeros(11, 8)]
        self.batch.prompt_seq_lens = [[19]]
        self.batch.negative_prompt_seq_lens = [[11]]
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.stack.enter_context(torch.no_grad())
        self.stack.enter_context(patch(f"{MODULE}.get_sp_world_size", return_value=8))
        self.stack.enter_context(
            patch(f"{MODULE}.get_ulysses_parallel_world_size", return_value=8)
        )

    def _plan(self, rank=0):
        with patch(f"{MODULE}.get_sp_parallel_rank", return_value=rank):
            return self.config._get_zimage_sp_plan(self.batch)

    def test_real_rows_rope_and_spatial_round_trip(self):
        image = torch.arange(136 * 240).reshape(1, 1, 1, 136, 240).float()
        shards = []
        for rank in range(8):
            with patch(f"{MODULE}.get_sp_parallel_rank", return_value=rank):
                plan = self._plan(rank)
                self.assertTrue(plan["image_sp_unpadded"])
                self.assertEqual(plan["img_seq_target"], 1020)
                shard, sharded = self.config.shard_latents_for_sp(self.batch, image)
                self.assertTrue(sharded)
                shards.append(shard)
                kwargs = self.config.prepare_pos_cond_kwargs(
                    self.batch, torch.device("cpu"), lambda ids: ids, torch.float32
                )
                caption_ids, image_ids = kwargs["freqs_cis"]
                self.assertEqual(caption_ids.shape, (32, 3))
                self.assertEqual(image_ids.shape, (1020, 3))
                self.assertEqual(image_ids[0].tolist(), [33, 0, rank * 15])
                self.assertEqual(image_ids[-1].tolist(), [33, 67, rank * 15 + 14])
                model = ZImageTransformer2DModel.__new__(ZImageTransformer2DModel)
                model.out_channels = 1
                tokens = model.patchify_and_embed(
                    [shard[0]],
                    [torch.zeros(32, 8)],
                    2,
                    1,
                    image_seq_len_target=kwargs["image_seq_len_target"],
                    image_sp_unpadded=kwargs["image_sp_unpadded"],
                )
                self.assertEqual(tokens[0].shape[1], 1020)
                self.assertEqual(tokens[5], [1020])
                restored = model.unpatchify(list(tokens[0].unbind(0)), tokens[2], 2, 1)
                torch.testing.assert_close(restored[0], shard[0], rtol=0, atol=0)
        torch.testing.assert_close(torch.cat(shards, dim=4), image, rtol=0, atol=0)
        self.assertEqual(8 * 1020 + 32, 8192)

    def test_layout_transitions_and_fallbacks(self):
        self.assertEqual(self._plan()["img_seq_target"], 1020)
        self.batch.height = 1024
        self.batch.width = 1024
        self.assertTrue(self._plan()["image_sp_unpadded"])
        self.assertEqual(self._plan()["img_seq_target"], 512)
        self.batch.width = 1104
        plan = self._plan()
        self.assertTrue(plan["image_sp_unpadded"])
        self.assertEqual(plan["shard_axis"], "h")
        self.assertEqual(plan["img_seq_target"], 552)
        for height, width in ((1072, 1904), (1088, 1904)):
            self.batch.height, self.batch.width = height, width
            self.assertFalse(self._plan()["image_sp_unpadded"])
        self.batch.height, self.batch.width = 1088, 1920
        self.batch.prompt_seq_lens = [[45]]
        self.assertFalse(self._plan()["image_sp_unpadded"])
        self.assertEqual(self._plan()["img_seq_target"], 1024)
        self.batch.prompt_seq_lens = [[19]]
        self.assertEqual(self._plan()["img_seq_target"], 1020)
        with patch(f"{MODULE}.get_ulysses_parallel_world_size", return_value=4):
            self.assertFalse(self._plan()["image_sp_unpadded"])
        with torch.enable_grad():
            self.assertFalse(self._plan()["image_sp_unpadded"])
        for name in ("enable_teacache", "enable_spectrum", "enable_cache_dit"):
            setattr(self.batch, name, True)
            self.assertFalse(self._plan()["image_sp_unpadded"])
            setattr(self.batch, name, False)
        self.config._allow_unpadded_sp = False
        self.assertEqual(self._plan()["img_seq_target"], 1024)

    @staticmethod
    def _bare_transformer():
        class Embed(torch.nn.Module):
            def forward(self, x):
                return x, None

        class Final(torch.nn.Module):
            def forward(self, x, conditioning):
                return x

        model = ZImageTransformer2DModel.__new__(ZImageTransformer2DModel)
        torch.nn.Module.__init__(model)
        model.all_patch_size = [2]
        model.all_f_patch_size = [1]
        model.out_channels = 1
        model.t_embedder = torch.nn.Identity()
        model.all_x_embedder = torch.nn.ModuleDict({"2-1": Embed()})
        model.cap_embedder = Embed()
        model.all_final_layer = torch.nn.ModuleDict({"2-1": Final()})
        model.noise_refiner = model.context_refiner = model.layers = []
        model.x_pad_token = torch.full((1, 4), -99.0)
        model.cap_pad_token = torch.full((1, 4), -88.0)
        model._rope_caches = {}
        model._unified_freqs_cis = None
        model._sharded_unified_freqs = None
        return model

    @staticmethod
    def _enter_ulysses8(stack, *, allows_lossless):
        stack.enter_context(patch(f"{MODEL}.get_sp_world_size", return_value=8))
        stack.enter_context(
            patch(f"{MODEL}.get_ulysses_parallel_world_size", return_value=8)
        )
        stack.enter_context(
            patch(
                f"{MODEL}.get_sp_group",
                return_value=SimpleNamespace(ulysses_group=None),
            )
        )
        stack.enter_context(
            patch("torch.cuda.is_current_stream_capturing", return_value=False)
        )
        stack.enter_context(
            patch(f"{MODEL}.request_allows_lossless", return_value=allows_lossless)
        )

    def test_forward_redistribution_and_cache_transitions(self):
        model = self._bare_transformer()
        with ExitStack() as stack:
            self._enter_ulysses8(stack, allows_lossless=True)
            previous_tables = None
            for target_len in (1020, 1024, 1020):
                image = torch.arange(136 * 240).reshape(1, 1, 1, 136, 240).float()
                images = list(image.chunk(8, dim=4))
                tokens = [
                    model.patchify_and_embed(
                        [part[0]],
                        [torch.zeros(32, 4)],
                        2,
                        1,
                        image_seq_len_target=target_len,
                        image_sp_unpadded=target_len == 1020,
                    )[0]
                    for part in images
                ]
                tokens = [
                    model._replace_padding_with_token(part, [1020], model.x_pad_token)
                    for part in tokens
                ]
                image_freqs = [
                    torch.arange(target_len * 4).reshape(target_len, 4).float()
                    + rank * 10000
                    for rank in range(8)
                ]
                cap_freq = torch.zeros(32, 4)
                global_tokens = torch.cat([*tokens, torch.zeros(1, 32, 4)], dim=1)
                for rank in range(8):
                    gathered_values = iter((tokens, image_freqs, image_freqs))

                    def gather(outputs, value, group):
                        for output, source in zip(
                            outputs, next(gathered_values), strict=True
                        ):
                            output.copy_(source)

                    # A unified shard is target_len + 4 rows, so rank r must take
                    # the 4 * r rows that sit in front of its own shard. Both sides
                    # are checked against the global sequence.
                    def exchange(output, value, out_splits, in_splits, group=None):
                        chunk = target_len + 4
                        if rank + 1 < 8:
                            sent = global_tokens[
                                :, (rank + 1) * target_len : (rank + 1) * chunk
                            ]
                            torch.testing.assert_close(
                                value, sent.transpose(0, 1), rtol=0, atol=0
                            )
                            self.assertEqual(in_splits[rank + 1], 4 * (rank + 1))
                        else:
                            self.assertEqual(value.shape[0], 0)
                        start = rank * target_len
                        output.copy_(
                            global_tokens[:, start : start + 4 * rank].transpose(0, 1)
                        )

                    with (
                        patch(f"{MODEL}.get_ulysses_parallel_rank", return_value=rank),
                        patch("torch.distributed.all_gather", side_effect=gather),
                        patch(
                            "torch.distributed.all_to_all_single", side_effect=exchange
                        ),
                    ):
                        output = model.forward(
                            images[rank],
                            [torch.zeros(32, 4)],
                            torch.zeros(1),
                            image_seq_len_target=target_len,
                            image_sp_unpadded=target_len == 1020,
                            freqs_cis=(
                                (cap_freq, cap_freq),
                                (image_freqs[rank], image_freqs[rank]),
                            ),
                        )
                    torch.testing.assert_close(output, -images[rank], rtol=0, atol=0)
                    expected = torch.cat([*image_freqs, cap_freq]).chunk(8)[rank]
                    torch.testing.assert_close(
                        model._sharded_unified_freqs[2][0], expected, rtol=0, atol=0
                    )
                    self.assertIsNot(model._sharded_unified_freqs[2], previous_tables)
                    previous_tables = model._sharded_unified_freqs[2]

    def test_exact_quality_keeps_replicated_caption_layout(self):
        """Joint sharding only reorders rounding, so an exact request runs the
        replicated caption layout, still on unrounded local image rows."""
        model = self._bare_transformer()
        image = torch.arange(136 * 240).reshape(1, 1, 1, 136, 240).float()
        images = list(image.chunk(8, dim=4))
        image_freq = torch.arange(1020 * 4).reshape(1020, 4).float()
        cap_freq = torch.zeros(32, 4)
        with ExitStack() as stack:
            self._enter_ulysses8(stack, allows_lossless=False)
            stack.enter_context(
                patch(f"{MODEL}.get_ulysses_parallel_rank", return_value=3)
            )
            gather = stack.enter_context(patch("torch.distributed.all_gather"))
            exchange = stack.enter_context(patch("torch.distributed.all_to_all_single"))
            output = model.forward(
                images[3],
                [torch.zeros(32, 4)],
                torch.zeros(1),
                image_seq_len_target=1020,
                image_sp_unpadded=True,
                freqs_cis=((cap_freq, cap_freq), (image_freq, image_freq)),
            )

        torch.testing.assert_close(output, -images[3], rtol=0, atol=0)
        gather.assert_not_called()
        exchange.assert_not_called()
        self.assertIsNone(model._sharded_unified_freqs)

    def test_compile_and_graph_configs_keep_local_padding(self):
        for compile_enabled, graph_enabled in (
            (True, False),
            (False, True),
            (False, False),
        ):
            self.config.validate_server_args(
                SimpleNamespace(
                    enable_torch_compile=compile_enabled,
                    enable_breakable_cuda_graph=graph_enabled,
                )
            )
            plan = self._plan()
            self.assertEqual(
                plan["image_sp_unpadded"], not (compile_enabled or graph_enabled)
            )
            self.assertEqual(
                plan["img_seq_target"],
                1024 if compile_enabled or graph_enabled else 1020,
            )

    def test_head_padding_reuse_and_functional_gradients(self):
        attention = ZImageAttention.__new__(ZImageAttention)
        torch.nn.Module.__init__(attention)
        attention._head_padding_buffers = [None, None, None]
        tensor = torch.arange(2 * 3 * 30 * 4).reshape(2, 3, 30, 4).float()
        padded = attention._pad_heads(tensor, slot=0, pad_heads=2, reuse=True)
        self.assertEqual(padded.shape, (2, 3, 32, 4))
        torch.testing.assert_close(padded[:, :, :30], tensor, rtol=0, atol=0)
        self.assertEqual(torch.count_nonzero(padded[:, :, 30:]).item(), 0)
        updated = attention._pad_heads(tensor + 1, slot=0, pad_heads=2, reuse=True)
        self.assertIs(updated, padded)
        torch.testing.assert_close(updated[:, :, :30], tensor + 1, rtol=0, atol=0)
        self.assertEqual(torch.count_nonzero(updated[:, :, 30:]).item(), 0)
        resized = attention._pad_heads(tensor[:, :1], slot=0, pad_heads=2, reuse=True)
        self.assertIsNot(resized, padded)
        with torch.enable_grad():
            source = tensor.clone().requires_grad_()
            functional = attention._pad_heads(source, slot=0, pad_heads=2, reuse=False)
            functional.sum().backward()
            torch.testing.assert_close(
                source.grad, torch.ones_like(source), rtol=0, atol=0
            )
        self.assertIs(attention._head_padding_buffers[0], resized)

    def test_patchification_requires_explicit_admission(self):
        model = ZImageTransformer2DModel.__new__(ZImageTransformer2DModel)
        args = ([torch.zeros(1, 1, 136, 30)], [torch.zeros(32, 8)], 2, 1)
        padded = model.patchify_and_embed(*args, image_seq_len_target=1020)
        self.assertEqual(padded[0].shape[1], 1024)
        with self.assertRaisesRegex(ValueError, "real image length"):
            model.patchify_and_embed(
                *args, image_seq_len_target=1024, image_sp_unpadded=True
            )


if __name__ == "__main__":
    unittest.main()
