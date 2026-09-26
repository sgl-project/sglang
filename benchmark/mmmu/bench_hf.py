import argparse
import re

import PIL
import torch
from data_utils import save_json
from eval_utils import (
    EvalArgs,
    eval_result,
    get_sampling_params,
    prepare_samples,
    process_result,
)
from tqdm import tqdm
from transformers import AutoModel, AutoProcessor, GenerationConfig

_IMAGE_TAG_RE = re.compile(r"<image\s+(\d+)>")


def _walk_prompt(prompt, image_paths):
    """Yield ``("text", str)`` and ``("image", path)`` parts in order by
    resolving each ``<image N>`` placeholder against ``image_paths``. If the
    prompt has no placeholders, the first image is anchored at the front."""
    parts = []
    last = 0
    used = 0
    for m in _IMAGE_TAG_RE.finditer(prompt):
        if m.start() > last:
            parts.append(("text", prompt[last : m.start()]))
        idx = int(m.group(1)) - 1
        if 0 <= idx < len(image_paths):
            parts.append(("image", image_paths[idx]))
            used += 1
        else:
            parts.append(("text", m.group(0)))
        last = m.end()
    if last < len(prompt):
        parts.append(("text", prompt[last:]))
    if used == 0 and image_paths:
        parts.insert(0, ("image", image_paths[0]))
    return parts


@torch.no_grad()
def eval_mmmu(args):
    eval_args = EvalArgs.from_cli_args(args)

    sampling_params = get_sampling_params(eval_args)
    generation_config = GenerationConfig(
        max_new_tokens=sampling_params["max_new_tokens"],
        do_sample=False,
    )

    try:
        from transformers import AutoModelForImageTextToText

        model = AutoModelForImageTextToText.from_pretrained(
            args.model_path,
            torch_dtype="auto",
            trust_remote_code=True,
        )
    except Exception as first_exception:
        try:
            # check if the model is belongs to internvl
            if "InternVL" in args.model_path:
                from transformers import AutoTokenizer

                from sglang.srt.multimodal.internvl_utils import image_to_pixel_values

                tokenizer = AutoTokenizer.from_pretrained(args.model_path)
                model = AutoModel.from_pretrained(
                    args.model_path,
                    torch_dtype="auto",
                    trust_remote_code=True,
                )
                generation_config_internvl = dict(
                    max_new_tokens=sampling_params["max_new_tokens"], do_sample=False
                )

            else:
                model = AutoModel.from_pretrained(
                    args.model_path,
                    torch_dtype="auto",
                    trust_remote_code=True,
                    init_tts=False,
                )
        except Exception as second_exception:
            raise RuntimeError(
                f"Failed to load model: First attempt failed with {first_exception}, "
                f"second attempt failed with {second_exception}"
            ) from second_exception

    model = model.eval().cuda()

    processor = AutoProcessor.from_pretrained(
        args.model_path, torch_dtype="auto", device_map="auto", trust_remote_code=True
    )

    samples = prepare_samples(eval_args)
    if getattr(args, "limit", None):
        total = len(samples)
        samples = samples[: args.limit]
        print(f"--limit {args.limit}: keeping {len(samples)} of {total} samples")
    out_samples = dict()

    answer_dict = {}
    for sample in tqdm(samples):
        prompt = sample["final_input_prompt"]
        image_paths = sample.get("image_paths") or [sample["image_path"]]
        assert image_paths and image_paths[0] is not None
        parts = _walk_prompt(prompt, image_paths)

        if "InternVL" in args.model_path:
            images = [PIL.Image.open(p).convert("RGB") for p in image_paths]
            pv = [
                image_to_pixel_values(
                    im, input_size=448, max_num=12, use_thumbnail=True
                )
                for im in images
            ]
            pixel_values = torch.cat(pv, dim=0).to(device="cuda", dtype=torch.bfloat16)
            contents = "".join(
                "<image>\n" if kind == "image" else val for kind, val in parts
            )
            response = model.chat(
                tokenizer, pixel_values, contents, generation_config_internvl
            )
            sample["original_response"] = response
            process_result(response, sample, answer_dict, out_samples)
            continue

        contents = [
            (
                {"type": "image", "image": val}
                if kind == "image"
                else {"type": "text", "text": val}
            )
            for kind, val in parts
        ]
        messages = [{"role": "user", "content": contents}]
        try:
            model_inputs = processor.tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                return_dict=True,
                add_generation_prompt=True,
                return_tensors="pt",
            ).to(model.device)
            input_len = model_inputs["input_ids"].shape[-1]
            generation = model.generate(
                **model_inputs, generation_config=generation_config
            )
            generation = generation[0][input_len:]
            response = processor.decode(generation, skip_special_tokens=True)
        except:
            contents = [
                PIL.Image.open(val) if kind == "image" else val for kind, val in parts
            ]
            messages = [{"role": "user", "content": contents}]
            response = model.chat(
                msgs=messages,
                tokenizer=processor.tokenizer,
                sampling=False,
                max_new_tokens=sampling_params["max_new_tokens"],
                use_tts_template=False,
                generate_audio=False,
                temperature=0.0,
            )
        sample["original_response"] = response
        process_result(response, sample, answer_dict, out_samples)

    args.output_path = f"{args.model_path}_answer_hf.json"
    save_json(args.output_path, out_samples)
    eval_result(
        model_answer_path=args.output_path,
        answer_dict=answer_dict,
        eval_output_path=f"{args.model_path}_val_hf.json",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-path",
        type=str,
        help="The path of the model weights. This can be a local folder or a Hugging Face repo ID.",
        required=True,
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="If set, only evaluate this many samples (debug / quick runs).",
    )
    EvalArgs.add_cli_args(parser)
    args = parser.parse_args()

    eval_mmmu(args)
