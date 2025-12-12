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

    def gradio_generate(prompt_in):
        # use global variable sampling_params_kwargs to avoid pass this param, because gradio does not support this.
        # return PIL.Image.Image | np.ndarray
        sampling_params_kwargs["prompt"] = prompt_in
        results = generator.generate(sampling_params_kwargs=sampling_params_kwargs)
        frames = results["frames"][0]
        return frames

    with gr.Blocks() as demo:
        gr.Markdown("# 🚀 Diffusion Generate Demo")

        with gr.Row():
            prompt_in = gr.Textbox(label="Prompt")
            run_btn = gr.Button("Generate")

        image_out = gr.Image(
            label="Generated Image",
        )

        run_btn.click(fn=gradio_generate, inputs=[prompt_in], outputs=[image_out])

    demo.launch()


""" example script
sglang webui  --model-path black-forest-labs/FLUX.1-dev

"""
