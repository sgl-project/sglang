import argparse
import multiprocessing as mp
import os

import gradio as gr

from sglang.cli.utils import get_is_diffusion_model, get_model_path
from sglang.multimodal_gen.configs.sample.sampling_params import SamplingParams
from sglang.multimodal_gen.runtime.entrypoints.cli.generate import (
    add_multimodal_gen_generate_args,
)
from sglang.multimodal_gen.runtime.entrypoints.openai.utils import post_process_sample
from sglang.multimodal_gen.runtime.entrypoints.utils import prepare_request
from sglang.multimodal_gen.runtime.launch_server import (
    launch_http_server_only,
    launch_server,
)
from sglang.multimodal_gen.runtime.server_args import ServerArgs
from sglang.multimodal_gen.runtime.sync_scheduler_client import sync_scheduler_client


def sgl_diffusion_webui(args, extra_argv):
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
    args = parser.parse_args(extra_argv)
    args.request_id = "mock_sgl_diffusion_webui"

    sampling_params_kwargs = SamplingParams.get_cli_args(args)

    # launch sgl server and client
    server_args = ServerArgs.from_cli_args(args)
    server_args.post_init_serve()
    launch_server(server_args, launch_http_server=False)
    sync_scheduler_client.initialize(server_args)

    def launch_http_server_function():
        return_msg = "Launching diffusion http server"
        try:
            # launch http server in another process
            server_process = mp.Process(
                target=launch_http_server_only,
                args=(server_args,),
                name="FastAPI-Server",
            )
            server_process.daemon = True
            server_process.start()
            return_msg = "Launch sglang diffusion http server in another process"
        except Exception as e:
            return_msg = f"Launch sglang diffusion http server failed because of {e}"
        return return_msg

    # sampling_params_kwargs will be reused in gradio_generate function
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
        sampling_params_kwargs["prompt"] = prompt
        sampling_params_kwargs["negative_prompt"] = negative_prompt
        sampling_params_kwargs["seed"] = seed
        sampling_params_kwargs["width"] = width
        sampling_params_kwargs["height"] = height
        sampling_params_kwargs["guidance_scale"] = guidance_scale
        sampling_params_kwargs["num_inference_steps"] = num_inference_steps
        sampling_params_kwargs["enable_teacache"] = enable_teacache

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
        gr.Markdown("# 🚀 Sglang Diffusion Application")

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

        with gr.Row():
            launch_http_btn = gr.Button("Launch http server", variant="primary")
            launch_log = gr.Textbox(
                label="Launch http server logging", value="Not launched yet"
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

        launch_http_btn.click(
            fn=launch_http_server_function,
            outputs=[launch_log],
        )

    demo.launch()


""" example script
sglang webui --model-path black-forest-labs/FLUX.1-dev

"""
