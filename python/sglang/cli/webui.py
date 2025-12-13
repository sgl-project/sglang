import argparse

import gradio as gr

from sglang.cli.utils import get_is_diffusion_model, get_model_path
from sglang.multimodal_gen import DiffGenerator
from sglang.multimodal_gen.configs.sample.sampling_params import SamplingParams
from sglang.multimodal_gen.runtime.entrypoints.cli.generate import (
    add_multimodal_gen_generate_args,
)
from sglang.multimodal_gen.runtime.server_args import ServerArgs


def webui(args, extra_argv):
    # If help is requested, show generate subcommand help without requiring --model-path
    if any(h in extra_argv for h in ("-h", "--help")):
        parser = argparse.ArgumentParser(description="SGLang Multimodal Generation")
        add_multimodal_gen_generate_args(parser)
        parser.parse_args(extra_argv)
        return

    model_path = get_model_path(extra_argv)
    is_diffusion_model = get_is_diffusion_model(model_path)

    if not is_diffusion_model:
        raise Exception(
            f"Generate subcommand is not yet supported for model: {model_path}"
        )

    parser = argparse.ArgumentParser(description="SGLang Multimodal Generation")
    add_multimodal_gen_generate_args(parser)
    parsed_args = parser.parse_args(extra_argv)

    args = parsed_args

    args.request_id = "mocked_fake_id_for_webui"

    server_args = ServerArgs.from_cli_args(args)
    assert (
        server_args.prompt_file_path is None
    ), "prompt_file_path must be None, we don't support prompt file when using webui"

    # generator and sampling_params_kwargs will be reused in gradio_generate function
    generator = DiffGenerator.from_pretrained(
        model_path=server_args.model_path, server_args=server_args
    )
    sampling_params_kwargs = SamplingParams.get_cli_args(args)

    def gradio_generate(
        prompt,
        negative_prompt,
        seed,
        width,
        height,
        num_inference_steps,
        guidance_scale,
        enable_teacache,
    ):
        # use global variable sampling_params_kwargs to avoid pass this param, because gradio does not support this.
        # return PIL.Image.Image | np.ndarray
        sampling_params_kwargs["prompt"] = prompt
        sampling_params_kwargs["negative_prompt"] = negative_prompt
        sampling_params_kwargs["seed"] = seed
        sampling_params_kwargs["width"] = width
        sampling_params_kwargs["height"] = height
        sampling_params_kwargs["guidance_scale"] = guidance_scale
        sampling_params_kwargs["num_inference_steps"] = num_inference_steps
        sampling_params_kwargs["enable_teacache"] = enable_teacache

        results = generator.generate(sampling_params_kwargs=sampling_params_kwargs)
        frames = results["frames"][0]
        return frames

    default_negative_prompt = "Bright tones, overexposed, static, blurred details, subtitles, style, works, paintings, images, static, overall gray, worst quality, low quality, JPEG compression residue, ugly, incomplete, extra fingers, poorly drawn hands, poorly drawn faces, deformed, disfigured, misshapen limbs, fused fingers, still picture, messy background, three legs, many people in the background, walking backwards"

    with gr.Blocks() as demo:
        gr.Markdown("# 🚀 Sglang Diffusion Application")

        with gr.Row():
            with gr.Column():
                prompt = gr.Textbox(label="Prompt", value="A curious raccoon")
                negative_prompt = gr.Textbox(
                    label="Negative_prompt", value=default_negative_prompt
                )

            run_btn = gr.Button("Generate", variant="primary")

        with gr.Row():
            with gr.Column():
                seed = gr.Number(label="seed", precision=0, value=1234)
                width = gr.Number(label="width", precision=0, value=512)
                height = gr.Number(label="height", precision=0, value=512)
                num_inference_steps = gr.Slider(
                    minimum=0, maximum=50, value=20, step=1, label="num_inference_steps"
                )

                guidance_scale = gr.Slider(
                    minimum=0.0, maximum=10, value=5, step=0.01, label="guidance_scale"
                )
                enable_teacache = gr.Checkbox(label="enable_teacache", value=False)

            image_out = gr.Image(
                label="Generated Image",
            )

        run_btn.click(
            fn=gradio_generate,
            inputs=[
                prompt,
                negative_prompt,
                seed,
                width,
                height,
                num_inference_steps,
                guidance_scale,
                enable_teacache,
            ],
            outputs=[image_out],
        )

    demo.launch()


""" example script
sglang webui --model-path black-forest-labs/FLUX.1-dev

"""
