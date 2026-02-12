import unittest


def get_quant_str(quant_cfg):
    if not quant_cfg:
        return None

    quant_method = quant_cfg.get("quant_method", "quantized")
    quant_str = f"{quant_method}"

    # Append interesting fields if they exist
    if "bits" in quant_cfg:
        quant_str += f", bits={quant_cfg['bits']}"
    if "quant_algo" in quant_cfg:
        quant_str += f", quant_algo={quant_cfg['quant_algo']}"
    if "fmt" in quant_cfg:
        quant_str += f", fmt={quant_cfg['fmt']}"

    return quant_str


class TestQuantStringFormatting(unittest.TestCase):
    def test_qwen_fp8_config(self):
        # Example from Qwen/Qwen3-4B-Thinking-2507-FP8
        quant_config = {
            "activation_scheme": "dynamic",
            "modules_to_not_convert": ["lm_head"],
            "fmt": "e4m3",
            "quant_method": "fp8",
            "weight_block_size": [128, 128],
        }
        expected = "fp8, fmt=e4m3"
        print(f"\n[Test Qwen FP8] Result: {get_quant_str(quant_config)}")
        self.assertEqual(get_quant_str(quant_config), expected)

    def test_llama_gptq_int4_config(self):
        # Example from hugging-quants/Meta-Llama-3.1-8B-Instruct-GPTQ-INT4
        quant_config = {
            "bits": 4,
            "quant_method": "gptq",
            # ... other fields ignored by logic
            "group_size": 128,
        }
        expected = "gptq, bits=4"
        print(f"\n[Test Llama GPTQ] Result: {get_quant_str(quant_config)}")
        self.assertEqual(get_quant_str(quant_config), expected)

    def test_awq_config(self):
        # Standard AWQ config
        quant_config = {
            "quant_method": "awq",
            "bits": 4,
            "group_size": 128,
        }
        expected = "awq, bits=4"
        print(f"\n[Test AWQ] Result: {get_quant_str(quant_config)}")
        self.assertEqual(get_quant_str(quant_config), expected)

    def test_modelopt_nvfp4(self):
        # Parsed modelopt config (simulated output from _parse_modelopt_quant_config)
        quant_config = {"quant_method": "modelopt_fp4"}
        expected = "modelopt_fp4"
        print(f"\n[Test ModelOpt] Result: {get_quant_str(quant_config)}")
        self.assertEqual(get_quant_str(quant_config), expected)

    def test_no_quant_config(self):
        self.assertIsNone(get_quant_str(None))


if __name__ == "__main__":
    unittest.main()
