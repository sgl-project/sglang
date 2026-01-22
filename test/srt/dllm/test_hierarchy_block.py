import unittest

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

import sglang as sgl
from sglang.test.test_utils import CustomTestCase

DIVIDER_WIDTH = 80
SECTION_CHAR = "="
SUBSECTION_CHAR = "-"


def print_section_header(title: str):
    print("\n" + SECTION_CHAR * DIVIDER_WIDTH)
    print(title)
    print(SECTION_CHAR * DIVIDER_WIDTH)


def print_subsection_header(title: str):
    print(f"\n{SUBSECTION_CHAR * 40}")
    print(f"{title}")
    print(SUBSECTION_CHAR * 40)


class TestFastDLLMv2Accuracy(CustomTestCase):
    MODEL = "Efficient-Large-Model/Fast_dLLM_v2_7B"
    TEST_PROMPTS = [
        "Explain quantum computing in simple terms.",
        "Explain how Low Confidence decoding works.",
    ]
    
    @classmethod
    def setUpClass(cls):
        cls.device = torch.device("cuda")
        
        print(f"Loading HF model: {cls.MODEL}")
        cls.tokenizer = AutoTokenizer.from_pretrained(cls.MODEL, trust_remote_code=True)
        cls.hf_model = AutoModelForCausalLM.from_pretrained(
            cls.MODEL,
            dtype=torch.float16,
            trust_remote_code=True,
        ).to(cls.device)
        cls.hf_model.eval()
        
        print(f"Launching SGLang Engine: {cls.MODEL}")
        cls.sgl_engine = sgl.Engine(
            model_path=cls.MODEL,
            trust_remote_code=True,
            mem_fraction_static=0.85,
            max_running_requests=1,
            attention_backend="flashinfer",
            disable_cuda_graph=True,
            dllm_algorithm="HierarchyBlock",
        )
    
    def _format_prompt_with_chat_template(self, prompt: str) -> str:
        messages = [{"role": "user", "content": prompt}]
        formatted = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        return formatted

    @classmethod
    def tearDownClass(cls):
        cls.sgl_engine.shutdown()
        del cls.hf_model
        torch.cuda.empty_cache()

    def _print_hf_generation_info(self, prompt: str, formatted_prompt: str, 
                                   prompt_token_count: int, input_ids_list: list,
                                   token_ids: list, text: str):
        print_subsection_header("HuggingFace Generation")
        print(f"Raw prompt: '{prompt}'")
        print(f"Formatted prompt: '{formatted_prompt}'")
        print(f"Prompt tokens: {prompt_token_count}")
        print(f"Input token IDs: {input_ids_list}")
        print(f"Generated {len(token_ids)} NEW tokens")
        print(f"Output token IDs: {token_ids}")
        print(f"Generated text: '{text}'")

    def _print_sglang_generation_info(self, prompt: str, formatted_prompt: str,
                                      prompt_ids: list, token_ids: list, text: str):
        print_subsection_header("SGLang Generation")
        print(f"Raw prompt: '{prompt}'")
        print(f"Formatted prompt: '{formatted_prompt}'")
        print(f"Prompt tokens: {len(prompt_ids)}")
        print(f"Input token IDs: {prompt_ids}")
        print(f"Generated {len(token_ids)} NEW tokens")
        print(f"Output token IDs: {token_ids}")
        print(f"Generated text: '{text}'")

    def _get_hf_outputs(self, prompt: str, max_new_tokens: int):
        formatted_prompt = self._format_prompt_with_chat_template(prompt)
        inputs = self.tokenizer(formatted_prompt, return_tensors="pt").to(self.device)
        prompt_token_count = inputs["input_ids"].shape[1]
        input_ids_list = inputs["input_ids"][0].tolist()
        
        with torch.no_grad():
            # FIXME(Jinwei): HF Fast dLLM v2 doesn't generate full max_new_tokens, but max_new_tokens-prompt_length tokens.
            outputs = self.hf_model.generate(
                inputs["input_ids"],
                tokenizer=self.tokenizer,
                max_new_tokens=max_new_tokens,
                block_size=32,
                small_block_size=8,
                threshold=0.9,
                temperature=0.0,
            )
        
        if isinstance(outputs, torch.Tensor):
            full_ids = outputs[0].tolist()
            token_ids = full_ids[prompt_token_count:]
            text = self.tokenizer.decode(token_ids, skip_special_tokens=False)
            
            self._print_hf_generation_info(
                prompt, formatted_prompt, prompt_token_count, 
                input_ids_list, token_ids, text
            )
        else:
            # FIXME(Jinwei): HF Fast dLLM v2 doesn't return logits and hidden_states
            raise ValueError(f"Unexpected HF output type: {type(outputs)}")
        
        # FIXME(Jinwei): Extract logits and hidden_states from HF generate
        logits = None
        hidden_states = None
        
        return {
            "token_ids": token_ids,
            "text": text,
            "logits": logits,
            "hidden_states": hidden_states,
            "prompt_ids": input_ids_list,
            "formatted_prompt": formatted_prompt,
        }

    def _get_sglang_outputs(self, prompt: str, max_new_tokens: int):
        formatted_prompt = self._format_prompt_with_chat_template(prompt)
        prompt_ids = self.tokenizer(formatted_prompt, return_tensors="pt")["input_ids"][0].tolist()
        
        # FIXME(Jinwei): Add return_logprob and return_hidden_states support
        outputs = self.sgl_engine.generate(
            prompt=[formatted_prompt],
            sampling_params={
                "max_new_tokens": max_new_tokens,
                "temperature": 0.0,
            },
        )
        
        result = outputs[0] if isinstance(outputs, list) else outputs
        
        if "output_ids" in result:
            token_ids = result["output_ids"]
            full_text = result.get("text", "")
            
            if full_text.startswith(formatted_prompt):
                generated_text = full_text[len(formatted_prompt):]
            else:
                generated_text = full_text
            
            self._print_sglang_generation_info(
                prompt, formatted_prompt, prompt_ids, token_ids, generated_text
            )
        else:
            raise KeyError(f"No 'output_ids' in result. Keys: {result.keys()}")
        
        logits = None
        hidden_states = None
        
        return {
            "token_ids": token_ids,
            "text": generated_text,
            "logits": logits,
            "hidden_states": hidden_states,
            "prompt_ids": prompt_ids,
            "formatted_prompt": formatted_prompt,
        }

    def _print_prompt_verification(self, hf_prompt_ids: list, sgl_prompt_ids: list):
        print_subsection_header("Prompt Input Verification")
        print(f"HF prompt length:     {len(hf_prompt_ids)} tokens")
        print(f"SGLang prompt length: {len(sgl_prompt_ids)} tokens")
        
        if hf_prompt_ids == sgl_prompt_ids:
            print(f"Prompt inputs are IDENTICAL")
        else:
            print(f"WARNING: Prompt inputs DIFFER!")
            for i, (h, s) in enumerate(zip(hf_prompt_ids, sgl_prompt_ids)):
                if h != s:
                    print(f"  First diff at position {i}: HF={h}, SGLang={s}")
                    break

    def _print_token_comparison(self, hf_tokens: list, sgl_tokens: list):
        min_len = min(len(hf_tokens), len(sgl_tokens))
        
        if min_len > 0:
            matches = sum(h == s for h, s in zip(hf_tokens[:min_len], sgl_tokens[:min_len]))
            accuracy = matches / min_len
            
            print_subsection_header("Token ID Comparison")
            print(f"HF generated:     {len(hf_tokens)} tokens")
            print(f"SGLang generated: {len(sgl_tokens)} tokens")
            print(f"Token Accuracy: {accuracy:.2%} ({matches}/{min_len})")
            print(f"\nHF tokens:     {hf_tokens}")
            print(f"SGLang tokens: {sgl_tokens}")
            
            for i, (h, s) in enumerate(zip(hf_tokens[:min_len], sgl_tokens[:min_len])):
                if h != s:
                    print(f"\nFirst mismatch at position {i}:")
                    print(f"  HF:     token={h} ('{self.tokenizer.decode([h])}')")
                    print(f"  SGLang: token={s} ('{self.tokenizer.decode([s])}')")
                    break
            else:
                if len(hf_tokens) != len(sgl_tokens):
                    print(f"\nAll tokens match up to position {min_len}, but lengths differ")
                else:
                    print(f"\nAll tokens IDENTICAL!")
        else:
            accuracy = 0.0
            print(f"\nToken Accuracy: 0.00% (one or both outputs empty)")
        
        return accuracy

    def _print_text_comparison(self, hf_text: str, sgl_text: str):
        print_subsection_header("Generated Text Comparison")
        print(f"HF text:     '{hf_text[:200]}'")
        print(f"SGLang text: '{sgl_text[:200]}'")

    def _compare_token_ids(self, hf_out: dict, sgl_out: dict):
        hf_tokens = hf_out["token_ids"]
        sgl_tokens = sgl_out["token_ids"]
        hf_text = hf_out.get("text", "")
        sgl_text = sgl_out.get("text", "")
        
        hf_prompt_ids = hf_out.get("prompt_ids", [])
        sgl_prompt_ids = sgl_out.get("prompt_ids", [])
        
        self._print_prompt_verification(hf_prompt_ids, sgl_prompt_ids)
        accuracy = self._print_token_comparison(hf_tokens, sgl_tokens)
        self._print_text_comparison(hf_text, sgl_text)
        
        return accuracy

    def _compare_logits(self, hf_out: dict, sgl_out: dict):
        pass

    def _compare_hidden_states(self, hf_out: dict, sgl_out: dict):
        pass

    def _run_single_prompt_test(self, prompt: str, max_new_tokens: int, test_name: str):
        print_section_header(f"{test_name}: {prompt[:50]}...")
        
        hf_out = self._get_hf_outputs(prompt, max_new_tokens=max_new_tokens)
        sgl_out = self._get_sglang_outputs(prompt, max_new_tokens=max_new_tokens)
        
        token_acc = self._compare_token_ids(hf_out, sgl_out)
        self._compare_logits(hf_out, sgl_out)
        self._compare_hidden_states(hf_out, sgl_out)
        
        return token_acc

    def test_single_block_accuracy(self):
        print_section_header("Testing Single Block (32 tokens)")
        
        for prompt in self.TEST_PROMPTS:
            token_acc = self._run_single_prompt_test(
                prompt, 
                max_new_tokens=32,
                test_name="Single Block Test"
            )
            self.assertGreater(token_acc, 0.0, f"No tokens generated for prompt: {prompt}")

    def test_multi_block_accuracy(self):
        print_section_header("Testing Multi Block (64 tokens)")
        
        for prompt in self.TEST_PROMPTS:
            token_acc = self._run_single_prompt_test(
                prompt,
                max_new_tokens=64,
                test_name="Multi Block Test"
            )
            self.assertGreater(token_acc, 0.0, f"No tokens generated for prompt: {prompt}")


if __name__ == "__main__":
    unittest.main()
