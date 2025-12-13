import argparse
import os

import gradio as gr

from sglang.multimodal_gen.configs.sample.sampling_params import SamplingParams
from sglang.multimodal_gen.runtime.entrypoints.openai.utils import post_process_sample
from sglang.multimodal_gen.runtime.entrypoints.utils import prepare_request
from sglang.multimodal_gen.runtime.server_args import ServerArgs
from sglang.multimodal_gen.runtime.sync_scheduler_client import sync_scheduler_client


def add_webui_args(parser: argparse.ArgumentParser):
    """Add the arguments for the generate command."""
    parser = ServerArgs.add_cli_args(parser)
    parser = SamplingParams.add_cli_args(parser)
    return parser


def run_sgl_diffusion_webui(server_args: ServerArgs):
    # init
    sync_scheduler_client.initialize(server_args)

    # server_args will be reused in gradio_generate function
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
        """
        NOTE: The input and output of function which is called by gradio button must be gradio components
        So we use global variable sampling_params_kwargs to avoid pass this param, because gradio does not support this.
        return PIL.Image.Image | np.ndarray
        """
        sampling_params_kwargs = dict(
            prompt=prompt,
            negative_prompt=negative_prompt,
            seed=seed,
            width=width,
            height=height,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
            enable_teacache=enable_teacache,
        )
        sampling_params = SamplingParams.from_user_sampling_params_args(
            server_args.model_path,
            server_args=server_args,
            **sampling_params_kwargs,
        )
        batch = prepare_request(
            server_args=server_args,
            sampling_params=sampling_params,
        )
        result = sync_scheduler_client.forward([batch])
        save_file_path = str(os.path.join(batch.output_path, batch.output_file_name))
        frames = post_process_sample(
            result.output[0],
            batch.data_type,
            batch.fps,
            batch.save_output,
            save_file_path,
        )
        return frames[0]

    with gr.Blocks() as demo:
        gr.Markdown("# 🚀 SGLang Diffusion Application")

        with gr.Row():
            with gr.Column():
                prompt = gr.Textbox(label="Prompt", value="A curious raccoon")
                negative_prompt = gr.Textbox(
                    label="Negative_prompt",
                    value="Bright tones, overexposed, static, blurred details, subtitles, style, works, paintings, images, static, overall gray, worst quality, low quality, JPEG compression residue, ugly, incomplete, extra fingers, poorly drawn hands, poorly drawn faces, deformed, disfigured, misshapen limbs, fused fingers, still picture, messy background, three legs, many people in the background, walking backwards",
                )

            run_btn = gr.Button("Generate", variant="primary")

        with gr.Row():
            with gr.Column():
                seed = gr.Number(label="seed", precision=0, value=1234)
                width = gr.Number(label="width", precision=0, value=1280)
                height = gr.Number(label="height", precision=0, value=720)
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

    demo.launch(server_port=server_args.webui_port)
