"""Cover the GLM-5.3 PP transport contract for multimodal MTP prefill.

Exercise the real model wrapper with CPU tensors and mocked model computation;
no checkpoint, GPU, process group, or server is required.
"""

import unittest
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.model_executor.forward_batch_info import PPProxyTensors
from sglang.srt.models import glm5_next
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestGlm5NextPPMmEmbeddings(CustomTestCase):
    def _forward_stage(
        self,
        first,
        last,
        embeddings=None,
        proxy=None,
        *,
        extend=True,
        eagle=True,
        speculative=True,
        capture_aux=False,
    ):
        batch = SimpleNamespace(
            mm_input_embeds=None,
            contains_mm_inputs=lambda: embeddings is not None,
            forward_mode=SimpleNamespace(is_extend=lambda: extend),
            spec_algorithm=(
                SimpleNamespace(is_eagle=lambda: eagle) if speculative else None
            ),
        )
        stage = SimpleNamespace(
            is_mrope_enabled=False,
            capture_aux_hidden_states=capture_aux,
            pp_group=SimpleNamespace(is_last_rank=last),
            model=None,
            lm_head=None,
            logits_processor=lambda ids, states, head, fb, aux: SimpleNamespace(
                mm_input_embeds=fb.mm_input_embeds,
                auxiliary=aux,
            ),
        )

        def model_forward(**kwargs):
            fb = kwargs["forward_batch"]
            if first:
                # prepare_mm_inputs creates a separate copy for the draft.
                fb.mm_input_embeds = (
                    embeddings.clone() if embeddings is not None else None
                )
            hidden = torch.zeros(2, 4)
            result = hidden if last else PPProxyTensors({"hidden_states": hidden})
            return (result, [hidden]) if capture_aux else result

        attention_context = SimpleNamespace(
            maybe_input_scattered=lambda fb: nullcontext()
        )
        with (
            patch.object(
                glm5_next, "get_attn_tp_context", return_value=attention_context
            ),
            patch.object(
                glm5_next, "general_mm_embed_routine", side_effect=model_forward
            ),
        ):
            output = glm5_next.Glm5NextForConditionalGeneration.forward(
                stage,
                torch.tensor([1, 2]),
                torch.tensor([0, 1]),
                batch,
                pp_proxy_tensors=proxy,
            )
        return output, batch

    def _relay(self, embeddings, stages):
        proxy = None
        for index in range(stages):
            proxy, batch = self._forward_stage(
                index == 0,
                index == stages - 1,
                embeddings,
                proxy,
            )
            torch.testing.assert_close(batch.mm_input_embeds, embeddings)
        torch.testing.assert_close(proxy.mm_input_embeds, embeddings)
        return proxy

    def test_pp2_embedding_reaches_logits_output(self):
        embeddings = torch.arange(8, dtype=torch.float32).reshape(2, 4)
        self._relay(embeddings, 2)

    def test_pp4_intermediate_stages_forward_embeddings(self):
        embeddings = torch.arange(8, dtype=torch.float32).reshape(2, 4)
        self._relay(embeddings, 4)

    def test_successive_prefill_batches_keep_their_own_embeddings(self):
        first = self._relay(torch.ones(2, 4), 4)
        second = self._relay(torch.full((2, 4), 7.0), 4)
        # Draft prefill may overwrite its final next-token row in-place.
        second.mm_input_embeds[-1].zero_()
        torch.testing.assert_close(first.mm_input_embeds, torch.ones(2, 4))

    def test_text_only_has_no_extra_proxy_payload(self):
        proxy, _ = self._forward_stage(True, False)
        self.assertEqual(set(proxy.tensors), {"hidden_states"})
        output, batch = self._forward_stage(False, True, proxy=proxy)
        self.assertIsNone(batch.mm_input_embeds)
        self.assertIsNone(output.mm_input_embeds)

    def test_non_speculative_and_non_eagle_do_not_relay(self):
        for options in ({"speculative": False}, {"eagle": False}):
            with self.subTest(options=options):
                proxy, _ = self._forward_stage(True, False, torch.ones(2, 4), **options)
                self.assertNotIn("mm_input_embeds", proxy.tensors)

    def test_decode_does_not_relay_embeddings(self):
        proxy, _ = self._forward_stage(True, False, torch.ones(2, 4), extend=False)
        self.assertNotIn("mm_input_embeds", proxy.tensors)

    def test_auxiliary_hidden_states_preserve_embedding_payload(self):
        embeddings = torch.ones(2, 4)
        proxy, _ = self._forward_stage(True, False, embeddings, capture_aux=True)
        output, _ = self._forward_stage(
            False, True, embeddings, proxy, capture_aux=True
        )
        torch.testing.assert_close(output.mm_input_embeds, embeddings)
        self.assertEqual(len(output.auxiliary), 1)


if __name__ == "__main__":
    unittest.main()
