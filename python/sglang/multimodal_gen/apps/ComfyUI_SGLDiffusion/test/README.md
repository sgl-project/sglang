# ComfyUI SGLDiffusion Pipeline Tests

This directory contains tests for each ComfyUI pipeline integration.

## Test Files

- `test_zimage_pipeline.py` - Tests for ComfyUIZImagePipeline
- `test_flux_pipeline.py` - Tests for ComfyUIFluxPipeline
- `test_qwen_image_pipeline.py` - Tests for ComfyUIQwenImagePipeline
- `test_qwen_image_edit_pipeline.py` - Tests for ComfyUIQwenImageEditPipeline (I2I/edit mode)

## Running Tests

### Run all tests

```bash
pytest python/sglang/multimodal_gen/apps/ComfyUI_SGLDiffusion/test/ -v -s
```

### Run a specific test file

```bash
pytest python/sglang/multimodal_gen/apps/ComfyUI_SGLDiffusion/test/test_zimage_pipeline.py -v -s
```

## Environment Variables

You can configure model paths via environment variables. Model paths support two formats:
- **Safetensors file**: Path to a single `.safetensors` file (e.g., `/path/to/model.safetensors`)
- **Diffusers format**: HuggingFace model ID or local diffusers directory (e.g., `Tongyi-MAI/Z-Image-Turbo`)

Environment variables:
- `SGLANG_TEST_ZIMAGE_MODEL_PATH` - Path to ZImage model (default: `Tongyi-MAI/Z-Image-Turbo`)
- `SGLANG_TEST_FLUX_MODEL_PATH` - Path to Flux model (default: `black-forest-labs/FLUX.1-dev`)
- `SGLANG_TEST_QWEN_IMAGE_MODEL_PATH` - Path to QwenImage model (default: `Qwen/Qwen-Image`)
- `SGLANG_TEST_QWEN_IMAGE_EDIT_MODEL_PATH` - Path to QwenImageEdit model (default: `Qwen/Qwen-Image-Edit-2511`)

Examples:

```bash
# Using HuggingFace model ID (diffusers format)
export SGLANG_TEST_ZIMAGE_MODEL_PATH="Tongyi-MAI/Z-Image-Turbo"
pytest python/sglang/multimodal_gen/apps/ComfyUI_SGLDiffusion/test/test_zimage_pipeline.py -v -s

# Using safetensors file
export SGLANG_TEST_ZIMAGE_MODEL_PATH="/path/to/z_image_turbo_bf16.safetensors"
pytest python/sglang/multimodal_gen/apps/ComfyUI_SGLDiffusion/test/test_zimage_pipeline.py -v -s
```

## Test Structure

Each test file follows a similar structure:

1. **Setup**: Creates a `DiffGenerator` with the appropriate pipeline class
2. **Input Preparation**: Creates dummy tensors for latents, timesteps, and embeddings
3. **Request Preparation**: Uses `prepare_request` to convert `SamplingParams` to `Req`
4. **ComfyUI Inputs**: Sets ComfyUI-specific inputs directly on the `Req` object
5. **Execution**: Sends request to scheduler and waits for response
6. **Validation**: Checks that `noise_pred` is retrieved from `OutputBatch`

## Notes

- These tests use `comfyui_mode=True` to enable ComfyUI-specific behavior
- Tests use pre-processed inputs (latents, timesteps, embeddings) as ComfyUI would provide
- The tests verify that `noise_pred` can be retrieved from the `OutputBatch` after processing
- All tests use dummy/ones tensors for simplicity - in production, these would be actual model outputs
