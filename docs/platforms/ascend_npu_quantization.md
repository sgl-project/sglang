Quantization on Ascend.

To load already quantized models, simply load the model weights and config. Again, if the model has been quantized offline, there's no need to add `--quantization` argument when starting the engine. The quantization method will be automatically parsed from the downloaded `quant_model_description.json` or `config.json` config.

MsModelSlim on Ascend support:
- [x] W4A4 dynamic linear
- [x] W8A8 static linear
- [x] W8A8 dynamic linear
- [x] W4A8 dynamic MOE
- [x] W8A8 dynamic MOE

AWQ on Ascend support:
- [x] W4A16 linear
- [x] W8A16 linear # Test required
- [x] W4A16 MOE # Test required
- [x] W8A16 MOE # Test required

Compressed-tensors (LLM Compressor) on Ascend support:
- [x] W8A8 dynamic linear
- [x] W8A8 dynamic MOE
- [x] W4A16 MOE
