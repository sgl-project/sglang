Quantization on Ascend.

To load already quantized models, simply load the model weights and config. Again, if the model has been quantized offline, there's no need to add `--quantization` argument when starting the engine. The quantization method will be automatically parsed from the downloaded `quant_model_description.json` or `config.json` config.

[ModelSlim on Ascend support](https://github.com/sgl-project/sglang/pull/14504):
- [x] W4A4 dynamic linear
- [x] W8A8 static linear
- [x] W8A8 dynamic linear
- [x] W4A8 dynamic MOE
- [x] W8A8 dynamic MOE

[AWQ on Ascend support](https://github.com/sgl-project/sglang/pull/10158):
- [x] W4A16 linear
- [x] W8A16 linear # Need to test
- [x] W4A16 MOE # Need to test

Compressed-tensors (LLM Compressor) on Ascend support:
- [x] [W4A8 dynamic MOE with/without activation clip](https://github.com/sgl-project/sglang/pull/14736) # Need to test
- [x] [W4A16 MOE](https://github.com/sgl-project/sglang/pull/12759)
- [x] [W8A8 dynamic linear](https://github.com/sgl-project/sglang/pull/14504)
- [x] [W8A8 dynamic MOE](https://github.com/sgl-project/sglang/pull/14504)
