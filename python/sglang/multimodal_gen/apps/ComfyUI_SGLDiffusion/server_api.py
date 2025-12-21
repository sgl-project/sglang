"""
SGLang Diffusion Server API client.
Provides a low-level interface for interacting with SGLang Diffusion HTTP server.
"""

import base64
import io
import os
import time
from typing import Any, Dict, Optional

import requests
from PIL import Image


class SGLDiffusionServerAPI:
    """Client for SGLang Diffusion HTTP server API."""

    def __init__(self, base_url: str, api_key: str = "sk-proj-1234567890"):
        """
        Initialize the API client.

        Args:
            base_url: Base URL of the SGLang Diffusion server (e.g., "http://localhost:30010/v1")
            api_key: API key for authentication (default: "sk-proj-1234567890")
        """
        # Ensure base_url doesn't end with /v1 if it's already there
        if base_url.endswith("/v1"):
            self.base_url = base_url
        elif base_url.endswith("/v1/"):
            self.base_url = base_url.rstrip("/")
        else:
            self.base_url = f"{base_url.rstrip('/')}/v1"

        self.api_key = api_key
        self.headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        }

    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the model served by this server.

        Returns:
            Dictionary containing model information including:
            - model_path: Path to the model
            - task_type: Type of task (e.g., "T2V", "I2I")
            - pipeline_name: Name of the pipeline
            - num_gpus: Number of GPUs
            - dit_precision: DiT model precision
            - vae_precision: VAE model precision
        """
        try:
            # Remove /v1 from base_url for /models endpoint
            models_url = self.base_url.removesuffix("/v1") + "/models"
            response = requests.get(models_url, headers=self.headers, timeout=30)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"Failed to get model info: {str(e)}")

    def generate_image(
        self,
        prompt: str,
        image_path: Optional[str] = None,
        mask_path: Optional[str] = None,
        size: Optional[str] = None,
        width: Optional[int] = None,
        height: Optional[int] = None,
        n: int = 1,
        negative_prompt: Optional[str] = None,
        guidance_scale: Optional[float] = None,
        num_inference_steps: Optional[int] = None,
        seed: Optional[int] = None,
        enable_teacache: bool = False,
        response_format: str = "b64_json",
        quality: Optional[str] = "auto",
        style: Optional[str] = "vivid",
        background: Optional[str] = "auto",
        output_format: Optional[str] = None,
        generator_device: Optional[str] = "cuda",
    ) -> Dict[str, Any]:
        """
        Generate or edit an image using SGLang Diffusion API.
        If image_path is provided, calls the edit endpoint; otherwise calls the generation endpoint.

        Args:
            prompt: Text prompt for image generation/editing
            image_path: Optional path to input image file for editing. If provided, uses edit API.
            mask_path: Optional path to mask image file (only used when image_path is provided)
            size: Image size in format "WIDTHxHEIGHT" (e.g., "1024x1024")
            width: Image width (used if size is not provided)
            height: Image height (used if size is not provided)
            n: Number of images to generate (1-10)
            negative_prompt: Negative prompt to avoid certain elements
            guidance_scale: Classifier-free guidance scale
            num_inference_steps: Number of denoising steps
            seed: Random seed for reproducible generation
            enable_teacache: Enable TEA cache acceleration
            response_format: Response format ("b64_json" or "url")
            quality: Image quality ("auto", "standard", "hd") - only for generation
            style: Image style ("vivid" or "natural") - only for generation
            background: Background type ("auto", "transparent", "opaque")
            output_format: Output format ("png", "jpeg", "webp")
            generator_device: Device for random generator ("cuda" or "cpu")

        Returns:
            Dictionary containing the API response with generated/edited image data
        """
        if not prompt:
            raise ValueError("Prompt cannot be empty")

        # Determine size
        if size is None:
            if width is not None and height is not None:
                size = f"{width}x{height}"
            else:
                size = "1024x1024"

        # Build common parameters
        common_params = self._build_image_common_params(
            prompt=prompt,
            size=size,
            n=n,
            response_format=response_format,
            negative_prompt=negative_prompt,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
            seed=seed,
            enable_teacache=enable_teacache,
            background=background,
            output_format=output_format,
            generator_device=generator_device,
        )

        # If image_path is provided, use edit endpoint
        if image_path:
            if not os.path.exists(image_path):
                raise FileNotFoundError(f"Image file not found: {image_path}")

            # Prepare multipart form data for edit
            files: Dict[str, Any] = {}
            data = common_params.copy()

            # Add image file
            files["image"] = (
                os.path.basename(image_path),
                open(image_path, "rb"),
                self._get_content_type(image_path),
            )

            # Add mask file if provided
            if mask_path:
                if not os.path.exists(mask_path):
                    raise FileNotFoundError(f"Mask file not found: {mask_path}")
                files["mask"] = (
                    os.path.basename(mask_path),
                    open(mask_path, "rb"),
                    self._get_content_type(mask_path),
                )

            # Prepare headers for multipart form data
            headers = {
                "Authorization": f"Bearer {self.api_key}",
            }

            try:
                response = requests.post(
                    f"{self.base_url}/images/edits",
                    files=files,
                    data=data,
                    headers=headers,
                    timeout=300,  # 5 minutes timeout for generation
                )
                response.raise_for_status()
                return response.json()
            except requests.exceptions.RequestException as e:
                raise RuntimeError(f"Failed to edit image: {str(e)}")
            finally:
                # Close file handles
                for file_tuple in files.values():
                    if isinstance(file_tuple, tuple) and len(file_tuple) > 1:
                        file_tuple[1].close()
        else:
            # Use generation endpoint - add generation-specific parameters
            payload = common_params.copy()
            if quality:
                payload["quality"] = quality
            if style:
                payload["style"] = style

            try:
                response = requests.post(
                    f"{self.base_url}/images/generations",
                    json=payload,
                    headers=self.headers,
                    timeout=300,  # 5 minutes timeout for generation
                )
                response.raise_for_status()
                return response.json()
            except requests.exceptions.RequestException as e:
                raise RuntimeError(f"Failed to generate image: {str(e)}")

    def generate_video(
        self,
        prompt: str,
        size: Optional[str] = None,
        width: Optional[int] = None,
        height: Optional[int] = None,
        seconds: Optional[int] = 4,
        fps: Optional[int] = None,
        num_frames: Optional[int] = None,
        negative_prompt: Optional[str] = None,
        guidance_scale: Optional[float] = None,
        num_inference_steps: Optional[int] = None,
        seed: Optional[int] = None,
        enable_teacache: bool = False,
        generator_device: Optional[str] = "cuda",
        input_reference: Optional[str] = None,
        output_path: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Generate a video using SGLang Diffusion API and wait for completion.

        Args:
            prompt: Text prompt for video generation
            size: Video size in format "WIDTHxHEIGHT" (e.g., "1280x720")
            width: Video width (used if size is not provided)
            height: Video height (used if size is not provided)
            seconds: Duration of the video in seconds
            fps: Frames per second
            num_frames: Number of frames (overrides seconds * fps if provided)
            negative_prompt: Negative prompt to avoid certain elements
            guidance_scale: Classifier-free guidance scale
            num_inference_steps: Number of denoising steps
            seed: Random seed for reproducible generation
            enable_teacache: Enable TEA cache acceleration
            generator_device: Device for random generator ("cuda" or "cpu")
            input_reference: Path to input reference image for image-to-video

        Returns:
            Dictionary containing completed video job information with file_path
        """
        if not prompt:
            raise ValueError("Prompt cannot be empty")

        # Determine size
        if size is None:
            if width is not None and height is not None:
                size = f"{width}x{height}"
            else:
                size = "720x1280"

        # Prepare request payload
        payload: Dict[str, Any] = {
            "prompt": prompt,
            "size": size,
        }

        # Add optional parameters
        if seconds is not None:
            payload["seconds"] = seconds
        if fps is not None:
            payload["fps"] = fps
        if num_frames is not None:
            payload["num_frames"] = num_frames
        if negative_prompt:
            payload["negative_prompt"] = negative_prompt
        if guidance_scale is not None:
            payload["guidance_scale"] = guidance_scale
        if num_inference_steps is not None:
            payload["num_inference_steps"] = num_inference_steps
        if seed is not None and seed >= 0:
            payload["seed"] = seed
        if enable_teacache:
            payload["enable_teacache"] = True
        if generator_device:
            payload["generator_device"] = generator_device
        if input_reference:
            payload["input_reference"] = input_reference
        if output_path:
            payload["output_path"] = output_path

        try:
            # Create video generation job
            response = requests.post(
                f"{self.base_url}/videos",
                json=payload,
                headers=self.headers,
                timeout=30,
            )
            response.raise_for_status()
            video_job = response.json()
            video_id = video_job.get("id")

            # Wait for completion with fixed polling
            poll_interval = 5  # 5 seconds
            max_wait_time = 3600  # 1 hour
            max_consecutive_errors = 5
            consecutive_errors = 0
            start_time = time.time()

            while time.time() - start_time < max_wait_time:
                try:
                    status_response = requests.get(
                        f"{self.base_url}/videos/{video_id}",
                        headers=self.headers,
                        timeout=30,
                    )
                    status_response.raise_for_status()
                    status = status_response.json()

                    # Reset error counter on successful request
                    consecutive_errors = 0

                    if status.get("status") == "completed":
                        return status
                    elif status.get("status") == "failed":
                        error = status.get("error", {})
                        error_msg = (
                            error.get("message", "Unknown error")
                            if error
                            else "Unknown error"
                        )
                        raise RuntimeError(f"Video generation failed: {error_msg}")
                except requests.exceptions.ConnectionError as e:
                    # Connection errors - likely server is down
                    consecutive_errors += 1
                    if consecutive_errors >= max_consecutive_errors:
                        raise RuntimeError(
                            f"Lost connection to server after {consecutive_errors} consecutive errors. "
                            f"Server may be unavailable: {str(e)}"
                        )
                except requests.exceptions.RequestException as e:
                    # Other network errors - continue polling but track errors
                    consecutive_errors += 1
                    if consecutive_errors >= max_consecutive_errors:
                        raise RuntimeError(
                            f"Network error after {consecutive_errors} consecutive failures: {str(e)}"
                        )

                time.sleep(poll_interval)

            raise TimeoutError(
                f"Video generation timed out after {max_wait_time} seconds"
            )
        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"Failed to generate video: {str(e)}")

    def _build_image_common_params(
        self,
        prompt: str,
        size: str,
        n: int,
        response_format: str,
        negative_prompt: Optional[str] = None,
        guidance_scale: Optional[float] = None,
        num_inference_steps: Optional[int] = None,
        seed: Optional[int] = None,
        enable_teacache: bool = False,
        background: Optional[str] = None,
        output_format: Optional[str] = None,
        generator_device: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Build common parameters for both image generation and editing.

        Returns:
            Dictionary containing common parameters
        """
        params: Dict[str, Any] = {
            "prompt": prompt,
            "size": size,
            "n": max(1, min(n, 10)),
            "response_format": response_format,
        }

        # Add optional parameters
        if negative_prompt:
            params["negative_prompt"] = negative_prompt
        if guidance_scale is not None:
            params["guidance_scale"] = guidance_scale
        if num_inference_steps is not None:
            params["num_inference_steps"] = num_inference_steps
        if seed is not None and seed >= 0:
            params["seed"] = seed
        if enable_teacache:
            params["enable_teacache"] = True
        if background:
            params["background"] = background
        if output_format:
            params["output_format"] = output_format
        if generator_device:
            params["generator_device"] = generator_device

        return params

    def _get_content_type(self, file_path: str) -> str:
        """Get content type based on file extension."""
        ext = os.path.splitext(file_path)[1].lower()
        content_types = {
            ".png": "image/png",
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".webp": "image/webp",
        }
        return content_types.get(ext, "image/png")

    def decode_image_from_response(
        self, response_data: Dict[str, Any], index: int = 0
    ) -> Image.Image:
        """
        Decode base64 image from API response.

        Args:
            response_data: API response dictionary
            index: Index of the image in the response (default: 0)

        Returns:
            PIL Image object
        """
        if "data" not in response_data or not response_data["data"]:
            raise ValueError("No image data in response")

        if index >= len(response_data["data"]):
            raise IndexError(f"Image index {index} out of range")

        image_data = response_data["data"][index]
        if "b64_json" not in image_data or not image_data["b64_json"]:
            raise ValueError("No base64 image data found")

        image_bytes = base64.b64decode(image_data["b64_json"])
        image = Image.open(io.BytesIO(image_bytes))

        # Convert to RGB if needed
        if image.mode != "RGB":
            image = image.convert("RGB")

        return image

    def set_lora(
        self,
        lora_nickname: str,
        lora_path: Optional[str] = None,
        target: str = "all",
    ) -> Dict[str, Any]:
        """
        Set a LoRA adapter for the specified transformer(s).

        Args:
            lora_nickname: The nickname of the adapter (required).
            lora_path: Path to the LoRA adapter (local path or HF repo id).
                      Required for the first load; optional if re-activating a cached nickname.
            target: Which transformer(s) to apply the LoRA to. One of:
                - "all": Apply to all transformers (default)
                - "transformer": Apply only to the primary transformer (high noise for Wan2.2)
                - "transformer_2": Apply only to transformer_2 (low noise for Wan2.2)
                - "critic": Apply only to the critic model

        Returns:
            Dictionary containing the API response with status and message
        """
        if not lora_nickname:
            raise ValueError("lora_nickname cannot be empty")

        # Prepare request payload
        payload: Dict[str, Any] = {
            "lora_nickname": lora_nickname,
            "target": target,
        }

        # Add optional lora_path if provided
        if lora_path:
            payload["lora_path"] = lora_path

        try:
            response = requests.post(
                f"{self.base_url}/set_lora",
                json=payload,
                headers=self.headers,
                timeout=30,
            )
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"Failed to set LoRA adapter: {str(e)}")

    def unset_lora(
        self,
        target: str = "all",
    ) -> Dict[str, Any]:
        """
        Unset (unmerge) LoRA weights from the base model.

        Args:
            target: same as set_lora

        Returns:
            Dictionary containing the API response with status and message
        """
        # Prepare request payload
        payload: Dict[str, Any] = {
            "target": target,
        }

        try:
            response = requests.post(
                f"{self.base_url}/unmerge_lora_weights",
                json=payload,
                headers=self.headers,
                timeout=30,
            )
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"Failed to unset LoRA adapter: {str(e)}")


if __name__ == "__main__":
    api = SGLDiffusionServerAPI(
        base_url="http://localhost:30010/v1", api_key="sk-proj-1234567890"
    )
    model_info = api.get_model_info()
    print(api.get_model_info())
    if model_info.get("task_type") == "T2V" or model_info.get("task_type") == "I2V":
        print(
            api.generate_video(
                prompt="A calico cat playing a piano on stage",
                num_inference_steps=1,
                size="480x480",
            )
        )
    else:
        print(
            api.generate_image(
                prompt="A calico cat playing a piano on stage", size="1024x1024"
            )
        )
