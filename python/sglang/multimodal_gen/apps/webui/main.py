import argparse
import os

from sglang.multimodal_gen.configs.sample.sampling_params import (
    DataType,
    SamplingParams,
)
from sglang.multimodal_gen.runtime.entrypoints.utils import (
    post_process_sample,
    prepare_request,
)
from sglang.multimodal_gen.runtime.scheduler_client import sync_scheduler_client
from sglang.multimodal_gen.runtime.server_args import ServerArgs


def add_webui_args(parser: argparse.ArgumentParser):
    """Add the arguments for the generate command."""
    parser = ServerArgs.add_cli_args(parser)
    parser = SamplingParams.add_cli_args(parser)
    return parser


def run_sgl_diffusion_webui(server_args: ServerArgs):
    # import gradio in function to avoid CI crash
    import gradio as gr

    # init client
    sync_scheduler_client.initialize(server_args)

    # server_args will be reused in gradio_generate function
    def gradio_generate(
        prompt,
        negative_prompt,
        seed,
        num_frames,
        frames_per_second,
        width,
        height,
        num_inference_steps,
        guidance_scale,
        enable_teacache,
    ):
        """
        NOTE: The input and output of function which is called by gradio button must be gradio components
        So we use global variable sampling_params_kwargs to avoid pass this param, because gradio does not support this.
        return [ np.ndarray, None ] | [None, np.ndarray]
        """
        sampling_params_kwargs = dict(
            prompt=prompt,
            negative_prompt=negative_prompt,
            seed=seed,
            num_frames=num_frames,
            fps=frames_per_second,
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
        if batch.data_type == DataType.VIDEO:
            # gradio video need video path to show video
            return None, save_file_path
        else:
            return frames[0], None

    with gr.Blocks() as demo:
        gr.Markdown("# 🚀 SGLang Diffusion Application")
        launched_model = gr.Textbox(label="Model", value=server_args.model_path)

        with gr.Row():
            with gr.Column(scale=4):
                prompt = gr.Textbox(label="Prompt", value="A curious raccoon")
                negative_prompt = gr.Textbox(
                    label="Negative_prompt",
                    value="Bright tones, overexposed, static, blurred details, subtitles, style, works, paintings, images, static, overall gray, worst quality, low quality, JPEG compression residue, ugly, incomplete, extra fingers, poorly drawn hands, poorly drawn faces, deformed, disfigured, misshapen limbs, fused fingers, still picture, messy background, three legs, many people in the background, walking backwards",
                )
            with gr.Column(scale=1):
                seed = gr.Number(label="seed", precision=0, value=1234)
                run_btn = gr.Button("Generate", variant="primary", size="lg")

        with gr.Row():
            with gr.Column():
                num_frames = gr.Slider(
                    minimum=1, maximum=161, value=81, step=1, label="num_frames"
                )
                frames_per_second = gr.Slider(
                    minimum=4, maximum=60, value=16, step=1, label="frames_per_second"
                )
                width = gr.Number(label="width", precision=0, value=720)
                height = gr.Number(label="height", precision=0, value=480)
                num_inference_steps = gr.Slider(
                    minimum=0, maximum=50, value=20, step=1, label="num_inference_steps"
                )
                guidance_scale = gr.Slider(
                    minimum=0.0, maximum=10, value=5, step=0.01, label="guidance_scale"
                )
                enable_teacache = gr.Checkbox(label="enable_teacache", value=False)

            with gr.Tabs() as tabs:
                with gr.TabItem("Image output", id=1):
                    image_out = gr.Image(
                        label="Generated Image",
                    )
                with gr.TabItem("Video output", id=2):
                    video_out = gr.Video(label="Generated Video")

        run_btn.click(
            fn=gradio_generate,
            inputs=[
                prompt,
                negative_prompt,
                seed,
                num_frames,
                frames_per_second,
                width,
                height,
                num_inference_steps,
                guidance_scale,
                enable_teacache,
            ],
            outputs=[image_out, video_out],
        )

        _, local_url, _ = demo.launch(
            server_port=server_args.webui_port,
            quiet=True,
            prevent_thread_lock=True,
        )

        # print banner
        delimiter = "=" * 80
        url = local_url or f"http://localhost:{server_args.webui_port}"
        print(
            f"""
{delimiter}
\033[1mSGLang Diffusion WebUI available at:\033[0m \033[1;4;92m{url}\033[0m
{delimiter}
"""
        )

        demo.block_thread()
