"""BeforeDenoisingStage for LongCat-Image (T2I) and LongCat-Image-Edit (I2I)."""

import re
from typing import List

import torch

from sglang.multimodal_gen.runtime.distributed import get_local_torch_device
from sglang.multimodal_gen.runtime.managers.forward_context import set_forward_context
from sglang.multimodal_gen.runtime.pipelines_core.schedule_batch import Req
from sglang.multimodal_gen.runtime.pipelines_core.stages.base import PipelineStage
from sglang.multimodal_gen.runtime.server_args import ServerArgs
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger

logger = init_logger(__name__)

# Exposed at module level so LongCatImagePipelineConfig can import without circular deps
TOKENIZER_MAX_LENGTH = 512


# --- Utility functions (adapted from diffusers LongCatImagePipeline) ---


def _get_prompt_language(prompt):
    pattern = re.compile(r"[\u4e00-\u9fff]")
    return "zh" if bool(pattern.search(prompt)) else "en"


def _split_quotation(prompt, quote_pairs=None):
    """Split prompt on quoted substrings, returning list of (text, is_quoted) tuples."""
    word_internal_quote_pattern = re.compile(r"[a-zA-Z]+'[a-zA-Z]+")
    matches = word_internal_quote_pattern.findall(prompt)
    mapping = []

    for i, word_src in enumerate(set(matches)):
        word_tgt = "longcat_$##$_longcat" * (i + 1)
        prompt = prompt.replace(word_src, word_tgt)
        mapping.append([word_src, word_tgt])

    if quote_pairs is None:
        quote_pairs = [
            ("'", "'"),
            ('"', '"'),
            ("\u2018", "\u2019"),
            ("\u201c", "\u201d"),
        ]
    pattern = "|".join(
        [
            re.escape(q1) + r"[^" + re.escape(q1 + q2) + r"]*?" + re.escape(q2)
            for q1, q2 in quote_pairs
        ]
    )
    parts = re.split(f"({pattern})", prompt)

    result = []
    for part in parts:
        for word_src, word_tgt in mapping:
            part = part.replace(word_tgt, word_src)
        if re.match(pattern, part):
            if len(part):
                result.append((part, True))
        else:
            if len(part):
                result.append((part, False))
    return result


def _prepare_pos_ids(
    modality_id=0,
    token_type="text",
    start=(0, 0),
    num_token=None,
    height=None,
    width=None,
):
    if token_type == "text":
        assert num_token
        pos_ids = torch.zeros(num_token, 3)
        pos_ids[..., 0] = modality_id
        pos_ids[..., 1] = torch.arange(num_token) + start[0]
        pos_ids[..., 2] = torch.arange(num_token) + start[1]
    elif token_type == "image":
        assert height and width
        pos_ids = torch.zeros(height, width, 3)
        pos_ids[..., 0] = modality_id
        pos_ids[..., 1] = pos_ids[..., 1] + torch.arange(height)[:, None] + start[0]
        pos_ids[..., 2] = pos_ids[..., 2] + torch.arange(width)[None, :] + start[1]
        pos_ids = pos_ids.reshape(height * width, 3)
    else:
        raise KeyError(
            f'Unknown token_type {token_type}, only support "text" or "image".'
        )
    return pos_ids


class LongCatImageBeforeDenoisingStage(PipelineStage):
    """Pre-processing stage for LongCat-Image (T2I).

    Handles:
    - Optional prompt rewriting via Qwen2.5-VL autoregressive decoding
    - Text encoding with Qwen2.5-VL text encoder
    - Position IDs (txt_ids, img_ids) and RoPE pre-computation
    - CPU offload of text_encoder after encoding

    Latent preparation and timestep scheduling are handled by the standard
    LatentPreparationStage and TimestepPreparationStage that follow this stage.
    """

    def __init__(
        self,
        text_encoder,
        tokenizer,
        text_processor,
        transformer,
    ):
        super().__init__()
        self.text_encoder = text_encoder
        self.tokenizer = tokenizer
        self.text_processor = text_processor
        self.transformer = transformer

        prefix_str = (
            "<|im_start|>system\nAs an image captioning expert, generate a descriptive text prompt "
            "based on an image content, suitable for input to a text-to-image model.<|im_end|>\n"
            "<|im_start|>user\n"
        )
        suffix_str = "<|im_end|>\n<|im_start|>assistant\n"

        # Tokenize once at construction time; reused by every _encode_prompt call.
        self._prefix_ids: List[int] = tokenizer(prefix_str, add_special_tokens=False)[
            "input_ids"
        ]
        self._suffix_ids: List[int] = tokenizer(suffix_str, add_special_tokens=False)[
            "input_ids"
        ]

    def _rewire_prompt(self, prompt: List[str], device: torch.device) -> List[str]:
        """Rewrite prompts using greedy decoding on the SGLang text_encoder.

        Mirrors diffusers LongCatImagePipeline.rewire_prompt().
        Uses text_processor.apply_chat_template to build the rewrite request,
        then runs greedy autoregressive decoding directly on self.text_encoder
        (which has lm_head and supports use_cache / past_key_values).
        """
        from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.longcat_image_system_messages import (
            SYSTEM_PROMPT_EN,
            SYSTEM_PROMPT_ZH,
        )

        all_text = []
        for each_prompt in prompt:
            language = _get_prompt_language(each_prompt)
            if language == "zh":
                question = (
                    SYSTEM_PROMPT_ZH
                    + f"\n用户输入为：{each_prompt}\n改写后的prompt为："
                )
            else:
                question = (
                    SYSTEM_PROMPT_EN + f"\nUser Input: {each_prompt}\nRewritten prompt:"
                )
            message = [
                {
                    "role": "user",
                    "content": [{"type": "text", "text": question}],
                }
            ]
            text = self.text_processor.apply_chat_template(
                message, tokenize=False, add_generation_prompt=True
            )
            all_text.append(text)

        inputs = self.text_processor(
            text=all_text, padding=True, return_tensors="pt"
        ).to(device)

        generated_ids = self._greedy_generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_new_tokens=TOKENIZER_MAX_LENGTH,
            device=device,
        )

        prompt_len = inputs["input_ids"].shape[1]
        generated_ids_trimmed = generated_ids[:, prompt_len:]
        rewritten = self.text_processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        logger.info("Rewritten prompts: %s", rewritten)
        return rewritten

    @torch.no_grad()
    def _greedy_generate(
        self,
        input_ids: torch.LongTensor,
        attention_mask: torch.Tensor,
        max_new_tokens: int,
        device: torch.device,
    ) -> torch.LongTensor:
        """Greedy autoregressive decoding using self.text_encoder with KV cache.

        Uses use_cache=True and DynamicCache for O(N) decoding. This is safe because
        LocalAttention.forward() takes the attn_mask path (F.scaled_dot_product_attention)
        when attention_mask is provided, bypassing sglang's attn_metadata entirely.
        past_key_values.update() runs before LocalAttention is called, so there is no
        incompatibility with sglang's custom attention backend.
        """
        from transformers import DynamicCache

        # Ensure text_encoder is on GPU before running (may have been CPU-offloaded).
        if next(self.text_encoder.parameters()).device.type == "cpu":
            self.text_encoder.to(device)

        eos_token_id = self.tokenizer.eos_token_id
        prompt_len = input_ids.shape[1]

        generated = []
        past_key_values = DynamicCache()

        # Prefill: process the full prompt in one shot
        cache_position = torch.arange(prompt_len, device=device)
        with set_forward_context(current_timestep=0, attn_metadata=None):
            outputs = self.text_encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                use_cache=True,
                cache_position=cache_position,
                output_hidden_states=False,
                logits_to_keep=1,
            )
        past_key_values = outputs.past_key_values

        next_token = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)  # [B, 1]
        if (next_token.squeeze(-1) == eos_token_id).all():
            return input_ids
        generated.append(next_token)

        # Decode: one new token per step, KV cache grows by 1 each iteration
        for step in range(1, max_new_tokens):
            if (next_token.squeeze(-1) == eos_token_id).all():
                break

            cur_len = prompt_len + step
            cur_attention_mask = torch.ones(
                input_ids.shape[0],
                cur_len,
                dtype=attention_mask.dtype,
                device=device,
            )
            cache_position = torch.tensor([cur_len - 1], device=device)

            with set_forward_context(current_timestep=0, attn_metadata=None):
                outputs = self.text_encoder(
                    input_ids=next_token,
                    attention_mask=cur_attention_mask,
                    past_key_values=past_key_values,
                    use_cache=True,
                    cache_position=cache_position,
                    output_hidden_states=False,
                    logits_to_keep=1,
                )
            past_key_values = outputs.past_key_values

            next_token = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)
            generated.append(next_token)

        generated_ids = torch.cat([input_ids, torch.cat(generated, dim=1)], dim=1)
        return generated_ids

    def _encode_prompt(self, prompt: List[str], device: torch.device) -> torch.Tensor:
        """Encode text prompts to embeddings using the Qwen2.5-VL text encoder."""
        batch_all_tokens = []

        for each_prompt in prompt:
            all_tokens = []
            for clean_prompt_sub, matched in _split_quotation(each_prompt):
                if matched:
                    # Intentional: tokenize each character in quoted text individually,
                    # mirroring diffusers LongCatImagePipeline._encode_prompt behavior.
                    for sub_word in clean_prompt_sub:
                        tokens = self.tokenizer(sub_word, add_special_tokens=False)[
                            "input_ids"
                        ]
                        all_tokens.extend(tokens)
                else:
                    tokens = self.tokenizer(clean_prompt_sub, add_special_tokens=False)[
                        "input_ids"
                    ]
                    all_tokens.extend(tokens)

            if len(all_tokens) > TOKENIZER_MAX_LENGTH:
                logger.warning(
                    "Prompt truncated: max_sequence_length=%d, input_token_nums=%d",
                    TOKENIZER_MAX_LENGTH,
                    len(all_tokens),
                )
                all_tokens = all_tokens[:TOKENIZER_MAX_LENGTH]
            batch_all_tokens.append(all_tokens)

        text_tokens_and_mask = self.tokenizer.pad(
            {"input_ids": batch_all_tokens},
            max_length=TOKENIZER_MAX_LENGTH,
            padding="max_length",
            return_attention_mask=True,
            return_tensors="pt",
        )

        prefix_len = len(self._prefix_ids)
        suffix_len = len(self._suffix_ids)
        mask_dtype = text_tokens_and_mask.attention_mask[0].dtype
        ids_dtype = text_tokens_and_mask.input_ids.dtype

        prefix_tokens_mask = torch.ones(prefix_len, dtype=mask_dtype)
        suffix_tokens_mask = torch.ones(suffix_len, dtype=mask_dtype)
        prefix_tokens_t = torch.tensor(self._prefix_ids, dtype=ids_dtype)
        suffix_tokens_t = torch.tensor(self._suffix_ids, dtype=ids_dtype)

        batch_size = text_tokens_and_mask.input_ids.size(0)
        prefix_tokens_batch = prefix_tokens_t.unsqueeze(0).expand(batch_size, -1)
        suffix_tokens_batch = suffix_tokens_t.unsqueeze(0).expand(batch_size, -1)
        prefix_mask_batch = prefix_tokens_mask.unsqueeze(0).expand(batch_size, -1)
        suffix_mask_batch = suffix_tokens_mask.unsqueeze(0).expand(batch_size, -1)

        input_ids = torch.cat(
            (prefix_tokens_batch, text_tokens_and_mask.input_ids, suffix_tokens_batch),
            dim=-1,
        ).to(device)
        attention_mask = torch.cat(
            (prefix_mask_batch, text_tokens_and_mask.attention_mask, suffix_mask_batch),
            dim=-1,
        ).to(device)

        # Ensure SGLang encoder is on GPU (may have been offloaded during prompt rewrite)
        if next(self.text_encoder.parameters()).device.type == "cpu":
            self.text_encoder.to(device)

        with set_forward_context(current_timestep=0, attn_metadata=None):
            text_output = self.text_encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=False,
                output_hidden_states=True,
            )
        prompt_embeds = text_output.hidden_states[-1].detach()
        prompt_embeds = prompt_embeds[:, prefix_len:-suffix_len, :]
        return prompt_embeds

    @torch.no_grad()
    def forward(self, batch: Req, server_args: ServerArgs) -> Req:
        device = get_local_torch_device()
        dtype = torch.bfloat16

        prompt = batch.prompt
        if isinstance(prompt, str):
            prompt = [prompt]

        negative_prompt = getattr(batch, "negative_prompt", "") or ""
        if isinstance(negative_prompt, str):
            negative_prompt = [negative_prompt]

        guidance_scale = getattr(batch, "guidance_scale", 4.5)
        do_cfg = guidance_scale > 1.0
        enable_prompt_rewrite = getattr(batch, "enable_prompt_rewrite", True)

        # 1. Optionally rewrite prompt for richer detail (default: True, matches diffusers)
        if enable_prompt_rewrite:
            logger.info(
                "Prompt rewriting is enabled (enable_prompt_rewrite=True). "
                "This runs autoregressive decoding on the 28B text encoder (up to %d tokens). "
                "Pass --enable-prompt-rewrite false to skip.",
                TOKENIZER_MAX_LENGTH,
            )
            prompt = self._rewire_prompt(prompt, device)

        # 2. Encode prompt
        prompt_embeds = self._encode_prompt(prompt, device).to(dtype)
        txt_ids = _prepare_pos_ids(
            modality_id=0,
            token_type="text",
            start=(0, 0),
            num_token=prompt_embeds.shape[1],
        ).to(device)

        if do_cfg:
            negative_prompt_embeds = self._encode_prompt(negative_prompt, device).to(
                dtype
            )
            negative_txt_ids = _prepare_pos_ids(
                modality_id=0,
                token_type="text",
                start=(0, 0),
                num_token=negative_prompt_embeds.shape[1],
            ).to(device)
        else:
            negative_prompt_embeds = None
            negative_txt_ids = txt_ids

        # Offload text_encoder to CPU after encoding to free GPU memory for the transformer.
        # text_encoder_cpu_offload only works for encoders with _fsdp_shard_conditions,
        # which Qwen2.5-VL doesn't have, so we do it manually here.
        if server_args.text_encoder_cpu_offload:
            self.text_encoder.to("cpu")
            torch.cuda.empty_cache()

        # 3. Populate batch for downstream stages
        batch.prompt_embeds = [prompt_embeds]
        batch.negative_prompt_embeds = (
            [negative_prompt_embeds]
            if negative_prompt_embeds is not None
            else [torch.zeros_like(prompt_embeds)]
        )
        # Use CPU generator to match diffusers behavior: randn_tensor generates on CPU
        # then moves to device, so CPU and CUDA generators produce different values
        # for the same seed.
        batch.generator = torch.Generator(device="cpu").manual_seed(batch.seed)

        # Position IDs for the transformer (img_ids set by LatentPreparationStage
        # via maybe_prepare_latent_ids, stored as batch.latent_ids)
        batch.txt_ids = txt_ids
        batch.negative_txt_ids = negative_txt_ids

        # CFG renorm params
        batch.enable_cfg_renorm = getattr(batch, "enable_cfg_renorm", True)
        batch.cfg_renorm_min = getattr(batch, "cfg_renorm_min", 0.0)

        return batch


class LongCatImageRoPEStage(PipelineStage):
    """Pre-compute RoPE embeddings after latents (and thus img_ids) are ready.

    Must run after LatentPreparationStage so that batch.latent_ids is populated.
    Pre-computing once avoids 3x float64 get_1d_rotary_pos_embed calls every step.
    """

    def __init__(self, transformer):
        super().__init__()
        self.transformer = transformer

    def forward(self, batch: Req, server_args: ServerArgs) -> Req:
        device = get_local_torch_device()
        txt_ids = batch.txt_ids
        img_ids = (
            batch.latent_ids
        )  # set by LatentPreparationStage via maybe_prepare_latent_ids
        batch.img_ids = img_ids
        ids = torch.cat((txt_ids, img_ids), dim=0).to(device)
        batch.image_rotary_emb = self.transformer.pos_embed(ids)
        return batch
