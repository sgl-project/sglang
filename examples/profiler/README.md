# SGLang Shape Profiler

Standalone debug tool to profile tensor shapes during model inference.

## Usage

```bash
python profile_shapes.py \
  --model-path Qwen/Qwen2.5-14B-Instruct \
  --tp-size 8 \
  --num-prompts 3 \
  --max-tokens 50
```

Takes the same parameters as `launch_server`.

## Files

- `profile_shapes.py` - Main profiler tool
- `torch_shape_logger.py` - Shape logging implementation  
- `analyze_shapes.py` - Analysis utilities

## Note

This is a **debug tool** that doesn't modify SGLang's core code.

With TP > 1, operations happen in worker processes which cannot be captured by this tool. For detailed TP profiling, use NVIDIA Nsight Systems or PyTorch Profiler.

For best results, use with TP=1 to capture all operations.
