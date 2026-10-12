# Murmur32 high-bit regression

Run `pytest test/manual/murmur_uint32/test_murmur_uint32_npu.py` in the patched SGLang release environment on an Ascend NPU. This is the existing 16-column regression. It uses deterministic integer reference values and skips without torch_npu / an NPU; a skip is not a hardware pass.

`murmur_vector_results.json` is preserved evidence from a separate earlier 128-output check (8 seed/position pairs, 16 columns each). Original mismatches: 72; corrected mismatches: 0. All expected and observed integers are included. No fresh device run is claimed for packaging this evidence. No host identity, credentials, audio or model prompts are included.
