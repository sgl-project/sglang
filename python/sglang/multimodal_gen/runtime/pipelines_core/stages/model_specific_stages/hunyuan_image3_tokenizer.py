"""Tokenizer wrapper for HunyuanImage-3 within sglang.

Provides ``HunyuanImage3TokenizerWrapper`` which wraps a base
``PreTrainedTokenizerFast`` (loaded via ``AutoTokenizer`` *without*
``trust_remote_code``) and implements the multimodal ``apply_chat_template``
entry point needed for AR tokenization.

The logic mirrors the ``TokenizerWrapper`` in vllm-omni but is entirely
self-contained — no external dependency on vllm-omni.
"""

import random
import warnings
from collections import defaultdict
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any

import torch
import torch.nn.functional as F

from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger

logger = init_logger(__name__)


# ---------------------------------------------------------------------------
# ImageInfo — lightweight copy compatible with the wrapper
# ---------------------------------------------------------------------------

class ImageInfo:
    """Stores image metadata for the tokenizer (mirrors vllm-omni's class)."""

    def __init__(
        self,
        image_type: str = None,
        image_width: int = None,
        image_height: int = None,
        token_width: int = None,
        token_height: int = None,
        image_token_length: int = None,
        base_size: int = None,
        ratio_index: int = None,
        **kwargs,
    ):
        self.image_type = image_type
        self.image_width = image_width
        self.w = image_width
        self.image_height = image_height
        self.h = image_height
        self.token_width = token_width
        self.tk_w = token_width
        self.token_height = token_height
        self.tk_h = token_height
        self.image_token_length = (
            image_token_length
            if image_token_length is not None
            else (
                token_width * token_height
                if token_width is not None and token_height is not None
                else None
            )
        )
        self.base_size = base_size
        self.ratio_index = ratio_index

        self.add_timestep_token = kwargs.get("add_timestep_token", True)
        self.add_guidance_token = kwargs.get("add_guidance_token", False)
        self.use_front_boi_token = kwargs.get("use_front_boi_token", True)
        self.add_image_shape_token = kwargs.get("add_image_shape_token", True)

    def __getitem__(self, key: str):
        if hasattr(self, key):
            return getattr(self, key)
        raise KeyError(f"Key '{key}' not found in ImageInfo")

    def __contains__(self, key: str) -> bool:
        return hasattr(self, key)

    @property
    def meta_info(self):
        if self.image_type in ["vae", "gen_image"]:
            return dict(
                token_length=self.image_token_length,
                add_timestep_token=self.add_timestep_token,
                add_guidance_token=self.add_guidance_token,
                use_front_boi_token=self.use_front_boi_token,
                add_image_shape_token=self.add_image_shape_token,
                base_size=self.base_size,
                ratio_idx=self.ratio_index,
                token_height=self.token_height,
                token_width=self.token_width,
                image_height=self.image_height,
                image_width=self.image_width,
            )
        raise ValueError(f"Unknown image type '{self.image_type}'")


# ---------------------------------------------------------------------------
# TokenizerEncodeOutput
# ---------------------------------------------------------------------------

@dataclass
class TokenizerEncodeOutput:
    tokens: torch.Tensor = None
    timestep_scatter_index: torch.Tensor = None
    guidance_scatter_index: torch.Tensor = None
    text_slices: list = None
    gen_image_slices: list = None
    text_mask: torch.Tensor = None
    gen_image_mask: torch.Tensor = None
    real_pos: torch.Tensor = None
    all_image_slices: list = None
    cond_timestep_scatter_index: torch.Tensor = None
    gen_timestep_scatter_index: torch.Tensor = None
    think_recaption_end_pos: list = None
    uncond_cfg_start_pos: list = None


# ---------------------------------------------------------------------------
# Conversation template
# ---------------------------------------------------------------------------

class _Conversation:
    roles: list = ["User", "Assistant"]
    sep: str = "\n\n"


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _default(value, default_value):
    return value if value is not None else default_value


# ---------------------------------------------------------------------------
# Main wrapper
# ---------------------------------------------------------------------------

class HunyuanImage3TokenizerWrapper:
    """Wraps a base HF tokenizer with multimodal ``apply_chat_template``.

    Mirrors the vllm-omni ``TokenizerWrapper`` but is self-contained.
    """

    def __init__(self, tokenizer):
        from transformers import AutoTokenizer

        if isinstance(tokenizer, str):
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer)
        else:
            self.tokenizer = tokenizer

        # Short names for special tokens
        self.bos_token_id = self.tokenizer.bos_token_id
        self.eos_token_id = self.tokenizer.eos_token_id
        self.pad_token_id = self.tokenizer.pad_token_id
        self.boi_token_id = self.tokenizer.convert_tokens_to_ids("<boi>")
        self.eoi_token_id = self.tokenizer.convert_tokens_to_ids("<eoi>")
        self.img_token_id = self.tokenizer.convert_tokens_to_ids("<img>")
        self.cfg_token_id = self.tokenizer.convert_tokens_to_ids("<cfg>")
        self.end_answer_token_id = self.tokenizer.convert_tokens_to_ids("</answer>")
        self.end_recaption_token_id = self.tokenizer.convert_tokens_to_ids("</recaption>")
        self.end_think_token_id = self.tokenizer.convert_tokens_to_ids("</think>")
        self.ratio_token_offset = self.tokenizer.convert_tokens_to_ids("<img_ratio_0>")
        self.special_token_map = self.tokenizer.added_tokens_encoder

    # -- padding helper -----------------------------------------------------

    @staticmethod
    def _pad(tensors, dim=0, pad_val=0):
        max_len = max(t.shape[dim] for t in tensors)
        out = []
        for t in tensors:
            if t.shape[dim] < max_len:
                t = F.pad(t, (0, max_len - t.shape[dim]), value=pad_val)
            out.append(t)
        return out

    # -- CoT section parser -------------------------------------------------

    def _get_cot_sections(self, cot_text, uncond_kwargs, drop_think=False):
        """Parse <think>/</think> or <recaption>/</recaption> blocks."""
        if not cot_text:
            return []
        if "<think>" in cot_text and "</think>" in cot_text:
            before = cot_text.split("<think>")[0]
            think = cot_text.split("<think>")[1].split("</think>")[0]
            after = cot_text.split("</think>")[1]
            return (
                self._get_cot_sections(before, uncond_kwargs, drop_think)
                + (
                    [
                        dict(type="text", text="<think>"),
                        dict(type="text", text=think, **uncond_kwargs),
                        dict(type="text", text="</think>"),
                    ]
                    if not drop_think
                    else []
                )
                + self._get_cot_sections(after, uncond_kwargs, drop_think)
            )
        if "<recaption>" in cot_text and "</recaption>" in cot_text:
            before = cot_text.split("<recaption>")[0]
            recaption = cot_text.split("<recaption>")[1].split("</recaption>")[0]
            after = cot_text.split("</recaption>")[1]
            return (
                self._get_cot_sections(before, uncond_kwargs, drop_think)
                + [
                    dict(type="text", text="<recaption>"),
                    dict(type="text", text=recaption, **uncond_kwargs),
                    dict(type="text", text="</recaption>"),
                ]
                + self._get_cot_sections(after, uncond_kwargs, drop_think)
            )
        return [dict(type="text", text=cot_text, **uncond_kwargs)]

    # -- encode_text --------------------------------------------------------

    def encode_text(
        self,
        *texts,
        uncond_enabled=None,
        uncond_p=None,
        max_length=None,
        pad=None,
        return_lengths=False,
    ):
        """Encode text(s), optionally replacing with ``<cfg>`` tokens."""
        if pad is not None:
            assert max_length is not None

        if uncond_enabled is None:
            uncond_enabled = [True] * len(texts)
        elif isinstance(uncond_enabled, bool):
            uncond_enabled = [uncond_enabled] * len(texts)
        assert len(uncond_enabled) == len(texts)

        do_uncond_drop = (uncond_p is not None) and (random.random() < uncond_p)
        text_tokens, lengths = [], []
        cum_length = 0
        for text, uncond_flag in zip(texts, uncond_enabled):
            if max_length is not None and cum_length >= max_length:
                warnings.warn(f"Text exceeds max_length({max_length}), truncating.")
                break
            if isinstance(text, str):
                text_token = self.tokenizer.encode(text, add_special_tokens=False)
            else:
                text_token = text
            if uncond_flag and do_uncond_drop:
                text_token = [self.cfg_token_id] * len(text_token)
            if max_length is not None and (cum_length + len(text_token)) > max_length:
                text_token = text_token[: max_length - cum_length]
            text_tokens.extend(text_token)
            lengths.append(len(text_token))
            cum_length += len(text_token)

        if pad is not None and (pad_length := max_length - len(text_tokens)) > 0:
            if pad == "left":
                text_tokens = [self.pad_token_id] * pad_length + text_tokens
            elif pad == "right":
                text_tokens = text_tokens + [self.pad_token_id] * pad_length
            else:
                raise ValueError(f"Unsupported pad: {pad}")

        if return_lengths:
            return text_tokens, lengths
        return text_tokens

    # -- encode_sequence ----------------------------------------------------

    def encode_sequence(
        self,
        template,
        token_source,
        total_length=None,
        add_timestep_token=False,
        add_guidance_token=False,
        add_eos=True,
        use_front_boi_token=True,
        add_pad=True,
        add_bos=True,
        drop_last="auto",
        add_image_shape_token=False,
    ):
        """Assemble token sequence from *template* (e.g. ``text-text-gen_image-text``)."""
        keys = template.split("-")
        modal_length = len(keys)
        index_indicator = {k: 0 for k in token_source}
        for v in token_source.values():
            assert isinstance(v, (list, tuple))

        # Validate key counts match
        assert set(keys) == set(token_source.keys())
        _key_counts = {k: 0 for k in keys}
        for k in keys:
            _key_counts[k] += 1
        for k, c in _key_counts.items():
            assert len(token_source[k]) == c

        token_seq = []
        token_count = 0
        extra = defaultdict(list)

        if add_bos:
            token_seq.append(self.bos_token_id)
            token_count += 1

        drop_last_break = False
        for i, key in enumerate(keys):
            source = token_source[key][index_indicator[key]]

            if key == "text":
                token_seq.extend(source)
                extra["<text>_start"].append(token_count)
                if "<cfg>_start" not in extra and len(source) > 0 and source[0] == self.cfg_token_id:
                    extra["<cfg>_start"].append(token_count)
                token_count += len(source)
                extra["<text>_end"].append(token_count - 1)
                if len(source) > 0 and source[-1] == self.end_think_token_id:
                    extra["<think>_end"].append(token_count - 1)
                if len(source) > 0 and source[-1] == self.end_recaption_token_id:
                    extra["<recaption>_end"].append(token_count - 1)

            elif key == "gen_image":
                if isinstance(source, int):
                    source = {"length": source}
                extra_count = (
                    2
                    + (1 if source.get("timestep", add_timestep_token) else 0)
                    + (1 if source.get("guidance", add_guidance_token) else 0)
                    + (2 if source.get("image_shape", add_image_shape_token) else 0)
                )
                if drop_last is True and total_length is not None and token_count + extra_count + source["length"] > total_length:
                    drop_last_break = True
                    break
                # <boi>
                if source.get("front_boi", use_front_boi_token):
                    token_seq.append(self.boi_token_id)
                    extra["boi"].append(token_count)
                    token_count += 1
                # image meta (timestep, guidance, image_shape)
                token_count = self._add_image_meta_info_token(
                    token_seq, token_count, extra,
                    add_timestep_token=source.get("timestep", add_timestep_token),
                    add_guidance_token=source.get("guidance", add_guidance_token),
                    add_image_shape_token=source.get("image_shape", add_image_shape_token),
                    base_size=source.get("base_size"),
                    ratio_idx=source.get("ratio_idx"),
                    image_type=key,
                )
                if not source.get("front_boi", use_front_boi_token):
                    token_seq.append(self.boi_token_id)
                    extra["boi"].append(token_count)
                    token_count += 1
                # <img> * N + <eoi>
                token_seq.extend([self.img_token_id] * source["length"] + [self.eoi_token_id])
                extra["<img>_start"].append(token_count)
                extra["<all_img>_start"].append(token_count)
                token_count += source["length"]
                extra["<img>_end"].append(token_count - 1)
                extra["<all_img>_end"].append(token_count - 1)
                extra["eoi"].append(token_count)
                token_count += 1

            else:
                raise ValueError(f"Unsupported key: {key}")
            index_indicator[key] += 1

        # EOS
        if add_eos is True and not drop_last_break:
            token_seq.append(self.eos_token_id)
            extra["eos"].append(token_count)
            token_count += 1
        elif add_eos == "auto" and not drop_last_break:
            if token_seq[-1] != self.eos_token_id and (total_length is None or token_count < total_length):
                token_seq.append(self.eos_token_id)
                extra["eos"].append(token_count)
                token_count += 1

        # Truncate / pad to total_length
        if total_length:
            if token_count > total_length and drop_last:
                for sk, ek in [
                    ("<img>_start", "<img>_end"),
                ]:
                    if sk in extra and ek in extra:
                        assert all(
                            s > total_length or e + 1 < total_length
                            for s, e in zip(extra[sk], extra[ek])
                        ), "Clip position in the middle of image tokens!"
                token_seq = token_seq[:total_length]
            pad_num = max(0, total_length - len(token_seq))
            if add_pad and pad_num:
                token_seq.extend([self.pad_token_id] * pad_num)
                extra["first_pad"].append(token_count)

        return token_seq, extra

    def _add_image_meta_info_token(
        self, token_seq, token_count, extra_token_pos,
        add_timestep_token=False, add_image_shape_token=False,
        add_guidance_token=False, base_size=None, ratio_idx=None,
        image_type=None,
    ):
        if add_image_shape_token:
            token_seq.extend([
                self.special_token_map[f"<img_size_{base_size}>"],
                self.special_token_map[f"<img_ratio_{ratio_idx}>"],
            ])
            token_count += 2
        if add_timestep_token:
            token_seq.extend([self.special_token_map["<timestep>"]])
            extra_token_pos["timestep"].append(token_count)
            if image_type == "gen_image":
                extra_token_pos["gen_timestep"].append(token_count)
            token_count += 1
        if add_guidance_token:
            token_seq.extend([self.special_token_map["<guidance>"]])
            extra_token_pos["guidance"].append(token_count)
            token_count += 1
        return token_count

    # -- encode_general -----------------------------------------------------

    def encode_general(self, sections, max_token_length=None,
                       add_eos="auto", use_text_mask=True,
                       add_pad="auto", add_bos=True, drop_last="auto"):
        """Encode a list of section dicts into a ``TokenizerEncodeOutput``."""
        sections = deepcopy(sections)
        template = "-".join(s["type"] for s in sections)

        token_source = defaultdict(list)
        text_mask_specs = []
        for section in sections:
            if section["type"] == "text":
                text = self.encode_text(
                    section.get("text", section.get("tokens")),
                    uncond_enabled=section.get("uncond_enabled"),
                    uncond_p=section.get("uncond_p"),
                    max_length=section.get("max_length"),
                )
                token_source["text"].append(text)
                text_mask_specs.append(dict(
                    ignore=section.get("ignore", False),
                    start_offset=section.get("start_offset", 0),
                    end_offset=section.get("end_offset", 0),
                ))
            elif section["type"] == "gen_image":
                token_source["gen_image"].append(dict(
                    length=section["token_length"],
                    timestep=section.get("add_timestep_token", False),
                    guidance=section.get("add_guidance_token", False),
                    front_boi=section.get("use_front_boi_token", False),
                    image_shape=section.get("add_image_shape_token", False),
                    base_size=section.get("base_size"),
                    ratio_idx=section.get("ratio_idx"),
                ))
            else:
                raise ValueError(f"Invalid section type: {section['type']}")

        full_token_seq, extra = self.encode_sequence(
            template=template, token_source=dict(token_source),
            total_length=max_token_length, add_eos=add_eos,
            add_pad=add_pad, add_bos=add_bos, drop_last=drop_last,
        )
        full_tensor = torch.tensor(full_token_seq, dtype=torch.long)

        # Scatter indices
        timestep_idx = torch.tensor(extra["timestep"], dtype=torch.long) if "timestep" in extra else None
        gen_ts_idx = torch.tensor(extra["gen_timestep"], dtype=torch.long) if "gen_timestep" in extra else None

        # Image slices / mask
        gen_image_slices = []
        gen_image_mask = None
        if "<img>_start" in extra and "<img>_end" in extra:
            gen_image_slices = [slice(s, e + 1) for s, e in zip(extra["<img>_start"], extra["<img>_end"])]
            gen_image_mask = torch.zeros_like(full_tensor, dtype=torch.bool)
            for sl in gen_image_slices:
                gen_image_mask[sl] = True

        # All image slices
        all_image_slices = []
        if "<all_img>_start" in extra and "<all_img>_end" in extra:
            all_image_slices = [slice(s, e + 1) for s, e in zip(extra["<all_img>_start"], extra["<all_img>_end"])]

        # Text slices
        text_slices = []
        if "<text>_start" in extra and "<text>_end" in extra:
            text_slices = [slice(s, e + 1) for s, e in zip(extra["<text>_start"], extra["<text>_end"])]

        # Text mask
        text_mask = None
        if use_text_mask:
            text_mask = torch.zeros_like(full_tensor, dtype=torch.float32)
            for sl, spec in zip(text_slices, text_mask_specs):
                if not spec["ignore"]:
                    real = slice(sl.start + spec["start_offset"], sl.stop + spec["end_offset"])
                    text_mask[real] = 1.0

        real_pos = torch.tensor(extra.get("first_pad", [full_tensor.shape[0]]), dtype=torch.long)
        think_end = extra.get("<think>_end", [None])[0]
        recaption_end = extra.get("<recaption>_end", [None])[0]

        return TokenizerEncodeOutput(
            tokens=full_tensor,
            timestep_scatter_index=timestep_idx,
            text_slices=text_slices,
            gen_image_slices=gen_image_slices,
            text_mask=text_mask,
            gen_image_mask=gen_image_mask,
            real_pos=real_pos,
            all_image_slices=all_image_slices,
            gen_timestep_scatter_index=gen_ts_idx,
            think_recaption_end_pos=[recaption_end or think_end],
            uncond_cfg_start_pos=[extra.get("<cfg>_start", [None])[0]],
        )

    # -- apply_chat_template ------------------------------------------------

    def apply_chat_template(
        self,
        batch_prompt=None,
        batch_message_list=None,
        mode="gen_text",
        batch_gen_image_info=None,
        batch_cond_image_info=None,
        batch_system_prompt=None,
        batch_cot_text=None,
        max_length=None,
        bot_task="auto",
        image_base_size=1024,
        sequence_template="pretrain",
        cfg_factor=1,
        add_assistant_prefix=None,
        drop_think=False,
    ):
        """Main entry point — mirrors vllm-omni ``TokenizerWrapper.apply_chat_template``."""
        assert bot_task in ["image", "auto", "think", "recaption", "img_ratio"]

        if batch_message_list is None:
            batch_size = len(batch_prompt)
            if not isinstance(batch_system_prompt, list):
                batch_system_prompt = [batch_system_prompt] * batch_size
            if not isinstance(batch_gen_image_info, list):
                batch_gen_image_info = [batch_gen_image_info] * batch_size
            batch_cot_text = batch_cot_text or [None] * batch_size
            batch_cond_image_info = batch_cond_image_info or [[] for _ in range(batch_size)]

            batch_message_list = []
            for prompt, sys_p, cot, img_info, cond_imgs in zip(
                batch_prompt, batch_system_prompt, batch_cot_text,
                batch_gen_image_info, batch_cond_image_info,
            ):
                ml = []
                if sys_p:
                    ml.append(dict(role="system", type="text", content=sys_p, context_type="str"))
                if len(cond_imgs) > 0:
                    ml.extend([
                        dict(role="user", type="joint_image", content=c, context_type="image_info")
                        for c in cond_imgs
                    ])
                ml.append(dict(role="user", type="text", content=prompt, context_type="str"))
                if cot is not None:
                    ml.append(dict(role="assistant", type="text", content=cot, context_type="str"))
                if mode == "gen_image":
                    ml.append(dict(role="assistant", type="gen_image", content=img_info, context_type="image_info"))
                batch_message_list.append(ml)

        output, sections = self._apply_general_template(
            message_list=batch_message_list,
            max_length=max_length,
            add_assistant_prefix=_default(add_assistant_prefix, mode != "gen_image"),
            bot_task=bot_task,
            sequence_template=sequence_template,
            cfg_factor=cfg_factor,
            batchify=True,
            image_base_size=image_base_size,
            drop_think=drop_think,
        )
        return dict(output=output, sections=sections)

    # -- apply_general_template (internal) ----------------------------------

    def _apply_general_template(
        self, message_list, max_length=None,
        add_assistant_prefix=False, answer="auto",
        bot_task="auto", sequence_template="instruct",
        uncond_p=0.0, cfg_factor=1, batchify=False,
        image_base_size=1024, drop_think=False,
    ):
        if batchify:
            return self._batch_gen_infer(
                infer_fn=self._apply_general_template,
                prompt_list=[[]],
                infer_fn_kwargs_list=[
                    dict(
                        message_list=ml_i, max_length=max_length,
                        add_assistant_prefix=add_assistant_prefix,
                        answer=answer, bot_task=bot_task,
                        sequence_template=sequence_template,
                        image_base_size=image_base_size,
                        drop_think=drop_think,
                    )
                    for ml_i in message_list
                ],
                do_classifier_free_guidance=cfg_factor > 1,
            )

        conv = _Conversation()
        uncond_kwargs = dict(uncond_enabled=uncond_p == 1.0, uncond_p=uncond_p)

        # Answer tags
        if (answer == "auto" and sequence_template == "instruct") or answer is True:
            answer_prefix, answer_suffix = "<answer>", "</answer>"
        else:
            answer_prefix, answer_suffix = "", ""

        # Template formatting tokens
        if sequence_template == "pretrain":
            system_suffix = user_prefix = user_suffix = bot_prefix = bot_suffix = ""
        else:
            system_suffix = conv.sep
            user_prefix = f"{conv.roles[0]}: "
            user_suffix = conv.sep
            bot_prefix = f"{conv.roles[1]}: "
            bot_suffix = conv.sep

        # Build sections
        sections: list[dict] = []
        cur_idx = 0
        final_role = None
        while cur_idx < len(message_list):
            for role, pfx, sfx, apfx, asfx in [
                ("system", "", system_suffix, "", ""),
                ("user", user_prefix, user_suffix, "", ""),
                ("assistant", bot_prefix, bot_suffix, answer_prefix, answer_suffix),
            ]:
                sub, cur_idx = self._process_successive(
                    message_list, cur_idx, role, pfx, sfx, apfx, asfx,
                    uncond_kwargs=uncond_kwargs,
                )
                sections.extend(sub)
                if sub:
                    final_role = role

        # Optional trailing assistant prefix
        if add_assistant_prefix:
            if final_role == "assistant":
                _bot_prefix = ""
                if sections and sections[-1].get("text") == bot_suffix:
                    sections = sections[:-1]
            else:
                _bot_prefix = bot_prefix
            bot_response_prefix = {
                "auto": _bot_prefix,
                "image": "",
                "think": f"{_bot_prefix}<think>",
                "recaption": f"{_bot_prefix}<recaption>",
                "img_ratio": f"{_bot_prefix}{answer_prefix}<boi><img_size_{image_base_size}>",
            }[bot_task]
            sections.append(dict(type="text", text=bot_response_prefix))

        output = self.encode_general(sections=sections, use_text_mask=False, add_eos=False, add_pad=False)

        if max_length is not None and output.tokens.shape[-1] > max_length:
            raise ValueError(
                f"Encoded length {output.tokens.shape[-1]} exceeds max_length {max_length}."
            )
        return output, sections

    # -- process successive messages of the same role -----------------------

    def _process_successive(self, message_list, cur_idx, role,
                            prefix, suffix, answer_prefix="", answer_suffix="",
                            uncond_kwargs=None):
        if uncond_kwargs is None:
            uncond_kwargs = {}
        sub_sections: list[dict] = []
        while cur_idx < len(message_list) and message_list[cur_idx]["role"] == role:
            msg = message_list[cur_idx]
            if msg["type"] == "text":
                text = msg["content"]
                if role == "system":
                    sub_sections.append(dict(type="text", text=text))
                elif role == "assistant":
                    if ("<think>" in text and "</think>" in text) or (
                        "<recaption>" in text and "</recaption>" in text
                    ):
                        sub_sections.extend(self._get_cot_sections(text, uncond_kwargs))
                    else:
                        sub_sections.append(dict(type="text", text=text, **uncond_kwargs))
                else:
                    # User text: no answer tags, but apply uncond_kwargs for CFG
                    sub_sections.append(
                        dict(type="text", text=f"{answer_prefix}{text}{answer_suffix}", **uncond_kwargs)
                    )
            elif msg["type"] == "gen_image":
                info = msg["content"]
                assert isinstance(info, ImageInfo), f"Expected ImageInfo, got {type(info)}"
                if role == "assistant":
                    sub_sections.append(dict(type="text", text=answer_prefix))
                sub_sections.append(dict(type=msg["type"], **info.meta_info))
                if role == "assistant":
                    sub_sections.append(dict(type="text", text=answer_suffix))
            else:
                raise ValueError(f"Unknown message type: {msg['type']}")
            cur_idx += 1

        if sub_sections:
            sub_sections.insert(0, dict(type="text", text=prefix))
            sub_sections.append(dict(type="text", text=suffix))
        return sub_sections, cur_idx

    # -- batch_gen_infer (CFG batching) ------------------------------------

    def _batch_gen_infer(self, infer_fn, prompt_list, infer_fn_kwargs_list,
                         do_classifier_free_guidance=False,
                         condition_repeat_times=1, uncondition_repeat_times=1):
        if infer_fn_kwargs_list is None:
            infer_fn_kwargs_list = [{} for _ in prompt_list]

        cond_results_list = None
        uncond_results_list = None
        output_type_list = []

        for prompt, kw in zip(prompt_list, infer_fn_kwargs_list):
            if not isinstance(prompt, (list, tuple)):
                prompt = [prompt]
            cond_kw = {**kw, "uncond_p": 0.0} if do_classifier_free_guidance else kw
            results = infer_fn(*prompt, **cond_kw)
            output_type_list.append((type(results), len(results) if isinstance(results, (list, tuple)) else 1))
            if not isinstance(results, (list, tuple)):
                results = (results,)
            if cond_results_list is None:
                cond_results_list = [[] for _ in results]
                uncond_results_list = [[] for _ in results]
            for i, r in enumerate(results):
                cond_results_list[i].append(r)

            if do_classifier_free_guidance:
                uncond_kw = {**kw, "uncond_p": 1.0}
                uncond_results = infer_fn(*prompt, **uncond_kw)
                if isinstance(uncond_results, TokenizerEncodeOutput):
                    uncond_results_list.append(uncond_results)
                else:
                    if not isinstance(uncond_results, (list, tuple)):
                        uncond_results = (uncond_results,)
                    for i, r in enumerate(uncond_results):
                        uncond_results_list[i].append(r)

        assert all(output_type_list[0] == n for n in output_type_list)

        def _make_batch(cond_items, uncond_items):
            first = cond_items[0]
            if isinstance(first, torch.Tensor):
                return torch.stack(self._pad(
                    cond_items * condition_repeat_times + uncond_items * uncondition_repeat_times,
                ))
            if first is None:
                return None
            if isinstance(first, list):
                return cond_items * condition_repeat_times + uncond_items * uncondition_repeat_times
            if isinstance(first, TokenizerEncodeOutput):
                merged = {}
                for key in list(first.__dataclass_fields__.keys()):
                    vals = [getattr(c, key) for c in cond_items] * condition_repeat_times + \
                           [getattr(u, key) for u in uncond_items] * uncondition_repeat_times
                    if isinstance(vals[0], torch.Tensor):
                        if "mask" in key:
                            pv = 0.0
                        elif key == "tokens":
                            pv = self.special_token_map.get("<pad>", self.pad_token_id)
                        else:
                            pv = False
                        merged[key] = torch.stack(self._pad(vals, pad_val=pv), dim=0)
                    elif isinstance(vals[0], list):
                        merged[key] = vals
                    elif vals[0] is None:
                        merged[key] = None
                    else:
                        merged[key] = vals
                return TokenizerEncodeOutput(**merged)
            raise TypeError(f"Cannot batch {type(first)}")

        stacked = [_make_batch(c, u) for c, u in zip(cond_results_list, uncond_results_list)]
        _, num = output_type_list[0]
        if num == 1:
            return stacked[0]
        return tuple(stacked)
