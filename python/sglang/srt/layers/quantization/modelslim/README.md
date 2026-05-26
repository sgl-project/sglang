Quantization [ModelSlim](https://gitcode.com/Ascend/msit) module.

`--quantization modelslim` flag introduced. To load already quantized models, simply load the model weights. For models quantized with ModelSlim, there's no need to add `--quantization modelslim` argument when starting the engine. The quantization method will be automatically parsed from the downloaded `quant_model_description.json` config.

ModelSlim was developed in the format of compressed_tensors and includes support for various quantization schemes, such as:
- [x] W4A4 dynamic linear
- [x] W8A8 static linear
- [x] W8A8 dynamic linear
- [x] W4A8 dynamic MOE
- [x] W8A8 dynamic MOE

Also ModelSlim module include:
- [x] Automated config detection for modelslim format (without the need to specify --quantization modelslim flag)
- [x] Unit-tests for w4a4 modelslim, w8a8 modelslim
