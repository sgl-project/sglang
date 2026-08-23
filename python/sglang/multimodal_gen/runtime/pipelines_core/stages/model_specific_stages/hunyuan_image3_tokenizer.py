"""HunyuanImage-3 tokenizer wrapper — mirrors vllm-omni TokenizerWrapper."""

import random
from collections import Counter, defaultdict
from copy import deepcopy
from dataclasses import dataclass

import torch
import torch.nn.functional as F

from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger

logger = init_logger(__name__)


class ImageInfo:
    """Image metadata for the tokenizer."""

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
        self.image_height = image_height
        self.token_width = token_width
        self.token_height = token_height
        self.image_token_length = image_token_length if image_token_length is not None else (token_width * token_height if token_width is not None and token_height is not None else None)
        self.base_size = base_size
        self.ratio_index = ratio_index

        self.add_timestep_token = kwargs.get("add_timestep_token", True)
        self.add_guidance_token = kwargs.get("add_guidance_token", False)
        self.use_front_boi_token = kwargs.get("use_front_boi_token", True)
        self.add_image_shape_token = kwargs.get("add_image_shape_token", True)

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


@dataclass
class TokenizerEncodeOutput:
    tokens: torch.Tensor = None
    gen_image_slices: list = None
    gen_image_mask: torch.Tensor = None
    gen_timestep_scatter_index: torch.Tensor = None


_ROLES = ["User", "Assistant"]
_SEP = "\n\n"


class HunyuanImage3TokenizerWrapper:
    """Wraps a base HF tokenizer with multimodal apply_chat_template."""

    def __init__(self, tokenizer):
        from transformers import AutoTokenizer

        if isinstance(tokenizer, str):
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer)
        else:
            self.tokenizer = tokenizer

        self.bos_token_id = self.tokenizer.bos_token_id
        self.eos_token_id = self.tokenizer.eos_token_id
        self.pad_token_id = self.tokenizer.pad_token_id
        self.boi_token_id = self.tokenizer.convert_tokens_to_ids("<boi>")
        self.eoi_token_id = self.tokenizer.convert_tokens_to_ids("<eoi>")
        self.img_token_id = self.tokenizer.convert_tokens_to_ids("<img>")
        self.cfg_token_id = self.tokenizer.convert_tokens_to_ids("<cfg>")
        self.special_token_map = self.tokenizer.added_tokens_encoder

    @staticmethod
    def _pad(tensors, dim=0, pad_val=0):
        max_len = max(t.shape[dim] for t in tensors)
        return [F.pad(t, (0, max_len - t.shape[dim]), value=pad_val) if t.shape[dim] < max_len else t for t in tensors]

    def encode_text(self, *texts, uncond_enabled=None, uncond_p=None, max_length=None):
        if uncond_enabled is None:
            uncond_enabled = [True] * len(texts)
        elif isinstance(uncond_enabled, bool):
            uncond_enabled = [uncond_enabled] * len(texts)
        assert len(uncond_enabled) == len(texts)

        do_uncond_drop = (uncond_p is not None) and (random.random() < uncond_p)
        text_tokens = []
        cum_length = 0
        for text, uncond_flag in zip(texts, uncond_enabled):
            if max_length is not None and cum_length >= max_length:
                break
            text_token = self.tokenizer.encode(text, add_special_tokens=False)
            if uncond_flag and do_uncond_drop:
                text_token = [self.cfg_token_id] * len(text_token)
            if max_length is not None and (cum_length + len(text_token)) > max_length:
                text_token = text_token[: max_length - cum_length]
            text_tokens.extend(text_token)
            cum_length += len(text_token)
        return text_tokens

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

        assert set(keys) == set(token_source.keys())
        key_counts = Counter(keys)
        for k, c in key_counts.items():
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
                token_count += len(source)

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
                if source.get("front_boi", use_front_boi_token):
                    token_seq.append(self.boi_token_id)
                    extra["boi"].append(token_count)
                    token_count += 1
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
                token_seq.extend([self.img_token_id] * source["length"] + [self.eoi_token_id])
                extra["<img>_start"].append(token_count)
                token_count += source["length"]
                extra["<img>_end"].append(token_count - 1)
                token_count += 1

            else:
                raise ValueError(f"Unsupported key: {key}")
            index_indicator[key] += 1

        if add_eos is True and not drop_last_break:
            token_seq.append(self.eos_token_id)
            extra["eos"].append(token_count)
            token_count += 1
        elif add_eos == "auto" and not drop_last_break:
            if token_seq[-1] != self.eos_token_id and (total_length is None or token_count < total_length):
                token_seq.append(self.eos_token_id)
                extra["eos"].append(token_count)
                token_count += 1

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

    def encode_general(self, sections, max_token_length=None,
                       add_eos="auto", add_pad="auto",
                       add_bos=True, drop_last="auto"):
        sections = deepcopy(sections)
        template = "-".join(s["type"] for s in sections)

        token_source = defaultdict(list)
        for section in sections:
            if section["type"] == "text":
                text = self.encode_text(
                    section["text"],
                    uncond_enabled=section.get("uncond_enabled"),
                    uncond_p=section.get("uncond_p"),
                    max_length=section.get("max_length"),
                )
                token_source["text"].append(text)
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

        gen_ts_idx = torch.tensor(extra["gen_timestep"], dtype=torch.long) if "gen_timestep" in extra else None

        gen_image_slices = []
        gen_image_mask = None
        if "<img>_start" in extra and "<img>_end" in extra:
            gen_image_slices = [slice(s, e + 1) for s, e in zip(extra["<img>_start"], extra["<img>_end"])]
            gen_image_mask = torch.zeros_like(full_tensor, dtype=torch.bool)
            for sl in gen_image_slices:
                gen_image_mask[sl] = True

        return TokenizerEncodeOutput(
            tokens=full_tensor,
            gen_image_slices=gen_image_slices,
            gen_image_mask=gen_image_mask,
            gen_timestep_scatter_index=gen_ts_idx,
        )

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
            add_assistant_prefix=(add_assistant_prefix if add_assistant_prefix is not None else (mode != "gen_image")),
            bot_task=bot_task,
            sequence_template=sequence_template,
            cfg_factor=cfg_factor,
            batchify=True,
            image_base_size=image_base_size,
            drop_think=drop_think,
        )
        return dict(output=output, sections=sections)

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

        uncond_kwargs = dict(uncond_enabled=uncond_p == 1.0, uncond_p=uncond_p)

        if (answer == "auto" and sequence_template == "instruct") or answer is True:
            answer_prefix, answer_suffix = "<answer>", "</answer>"
        else:
            answer_prefix, answer_suffix = "", ""

        if sequence_template == "pretrain":
            system_suffix = user_prefix = user_suffix = bot_prefix = bot_suffix = ""
        else:
            system_suffix = _SEP
            user_prefix = f"{_ROLES[0]}: "
            user_suffix = _SEP
            bot_prefix = f"{_ROLES[1]}: "
            bot_suffix = _SEP

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

        if add_assistant_prefix:
            if final_role == "assistant":
                _bot_prefix = ""
                if sections and sections[-1].get("text") == bot_suffix:
                    sections = sections[:-1]
            else:
                _bot_prefix = bot_prefix
            sections.append(dict(type="text", text=_bot_prefix))

        output = self.encode_general(sections=sections, add_eos=False, add_pad=False)

        if max_length is not None and output.tokens.shape[-1] > max_length:
            raise ValueError(
                f"Encoded length {output.tokens.shape[-1]} exceeds max_length {max_length}."
            )
        return output, sections

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
                    sub_sections.append(dict(type="text", text=text, **uncond_kwargs))
                else:
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

    def _batch_gen_infer(self, infer_fn, prompt_list, infer_fn_kwargs_list,
                         do_classifier_free_guidance=False):
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
                return torch.stack(self._pad(cond_items + uncond_items))
            if first is None:
                return None
            if isinstance(first, list):
                return cond_items + uncond_items
            if isinstance(first, TokenizerEncodeOutput):
                merged = {}
                for key in list(first.__dataclass_fields__.keys()):
                    vals = [getattr(c, key) for c in cond_items] + [getattr(u, key) for u in uncond_items]
                    if isinstance(vals[0], torch.Tensor):
                        pv = 0.0 if "mask" in key else (self.special_token_map.get("<pad>", self.pad_token_id) if key == "tokens" else False)
                        merged[key] = torch.stack(self._pad(vals, pad_val=pv), dim=0)
                    else:
                        merged[key] = vals if vals[0] is not None else None
                return TokenizerEncodeOutput(**merged)
            raise TypeError(f"Cannot batch {type(first)}")

        stacked = [_make_batch(c, u) for c, u in zip(cond_results_list, uncond_results_list)]
        _, num = output_type_list[0]
        if num == 1:
            return stacked[0]
        return tuple(stacked)
