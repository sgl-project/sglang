import argparse

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
                from internvl_utils import load_image
                from transformers import AutoTokenizer

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
    out_samples = dict()

    answer_dict = {}
    for sample in tqdm(samples):
        prompt = sample["final_input_prompt"]
        image = sample["image"]
        prefix = prompt.split("<")[0]
        suffix = prompt.split(">")[1]
        assert image is not None

        if "InternVL" in args.model_path:
            pixel_values = load_image(sample["image_path"]).to(torch.bfloat16).cuda()
            contents = ""
            if prefix:
                contents += prefix
            contents += "<image>\n"
            if suffix:
                contents += suffix
            response = model.chat(
                tokenizer, pixel_values, contents, generation_config_internvl
            )
            print(f"response: {response}")
            process_result(response, sample, answer_dict, out_samples)
            continue

        contents = []
        if prefix:
            contents += [{"type": "text", "text": prefix}]
        contents += [
            {
                "type": "image",
                "image": sample["image_path"],
            }
        ]
        if suffix:
            contents += [{"type": "text", "text": suffix}]
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
            contents = []
            if prefix:
                contents += [prefix]
            image = PIL.Image.open(sample["image_path"])
            contents += [image]
            if suffix:
                contents += [suffix]
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
        print(f"response: {response}")
        process_result(response, sample, answer_dict, out_samples)

    args.output_path = f"{args.model_path}_val_hf.json"
    save_json(args.output_path, out_samples)
    eval_result(model_answer_path=args.output_path, answer_dict=answer_dict)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-path",
        type=str,
        help="The path of the model weights. This can be a local folder or a Hugging Face repo ID.",
        required=True,
    )
    EvalArgs.add_cli_args(parser)
    args = parser.parse_args()

    eval_mmmu(args)
