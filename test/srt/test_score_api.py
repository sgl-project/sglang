import unittest
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from sglang.srt.entrypoints.engine import Engine
from sglang.test.test_utils import CustomTestCase, DEFAULT_SMALL_MODEL_NAME_FOR_TEST

TEST_MODEL_NAME = DEFAULT_SMALL_MODEL_NAME_FOR_TEST

class TestScoreAPI(CustomTestCase):
    """Test the scoring API functionality."""

    def setUp(self):
        """Set up each test case."""
        self.engine = Engine(model_path=TEST_MODEL_NAME)

    def tearDown(self):
        """Clean up after each test case."""
        if self.engine is not None:
            self.engine.shutdown()
            torch.cuda.empty_cache()

    def compute_hf_scores(self, prompt, texts, label_token_ids, apply_softmax=False):
        """Compute scores using direct HuggingFace model inference.
        Returns probabilities for each token ID, optionally normalized with softmax.
        """
        # Initialize HF model and tokenizer
        tokenizer = AutoTokenizer.from_pretrained(TEST_MODEL_NAME, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(TEST_MODEL_NAME, trust_remote_code=True)
        
        try:
            scores = []
            for text in texts:
                # Combine prompt and text directly without newline
                full_text = f"{prompt}{text}"
                inputs = tokenizer(full_text, return_tensors="pt").to(model.device)
                
                # Get logits for the last token
                with torch.no_grad():
                    outputs = model(**inputs)
                    last_token_logits = outputs.logits[0, -1]
                    
                # Get logits for just our target tokens
                target_logits = last_token_logits[label_token_ids]
                
                # Apply softmax over just the target tokens
                target_probs = torch.softmax(target_logits, dim=-1)
                
                # Create dictionary of token probabilities
                probs = {tid: target_probs[i].item() for i, tid in enumerate(label_token_ids)}
                
                scores.append(probs)
            
            return scores
        finally:
            # Clean up HF resources
            model.cpu()
            del model
            del tokenizer
            torch.cuda.empty_cache()

    def test_score_consistency(self):
        """Test that SGLang scoring matches direct HuggingFace model scoring.
        """
     
        # Single test case
        prompt = "I pledge allegiance"
        texts = ["", " to"] 
        tokens = [" to", " the", " and"]
        
        # Get token IDs using HF tokenizer temporarily
        tokenizer = AutoTokenizer.from_pretrained(TEST_MODEL_NAME, trust_remote_code=True)
        label_token_ids = []
        for token in tokens:
            encoding = tokenizer.encode_plus(token, add_special_tokens=False)
            token_ids = encoding['input_ids']
            label_token_ids.append(token_ids[0])
        del tokenizer 
        # Get scores from SGLang first
        sglang_scores = self.engine.score(
            query=prompt,
            items=texts,
            label_token_ids=label_token_ids,
            apply_softmax=True
        )

        print(f"\nSGLang scores: {sglang_scores}")
        
        # Get scores from HuggingFace
        hf_scores = self.compute_hf_scores(
            prompt, texts,
            label_token_ids
        )
        
        print(f"\nHF scores: {hf_scores}")
        
        # Compare scores
        self.assertEqual(len(hf_scores), len(sglang_scores), "Score lengths don't match")
        for hf_score_dict, sglang_score_dict in zip(hf_scores, sglang_scores):
            print(f"\nScore comparison:")
            for tid in label_token_ids:
                hf_score = hf_score_dict[tid]
                sglang_score = sglang_score_dict[tid]
                self.assertAlmostEqual(hf_score, sglang_score, places=3,
                    msg=f"Scores don't match for token {tid}: HF={hf_score:.6f}, SGLang={sglang_score:.6f}")
                self.assertGreaterEqual(sglang_score, 0, f"SGLang score {sglang_score:.6f} not in [0,1]")
                self.assertLessEqual(sglang_score, 1, f"SGLang score {sglang_score:.6f} not in [0,1]")
            
                self.assertAlmostEqual(sum(sglang_score_dict.values()), 1.0, places=6,
                    msg=f"SGLang scores don't sum to 1: {sum(sglang_score_dict.values()):.6f}")


    def test_score_batch_handling(self):
        """Test that batch scoring works correctly."""
        # Test with different batch sizes
        batch_sizes = [1, 2, 4, 8]
        label_token_ids = [1, 2, 3] 
        
        for batch_size in batch_sizes:
            texts = [f"test {i}" for i in range(batch_size)]
            scores = self.engine.score(
                query="The test was",
                items=texts,
                label_token_ids=label_token_ids,
                apply_softmax=True
            )
            
            print(f"Scores: {scores}")

            self.assertEqual(len(scores), batch_size,
                f"Expected {batch_size} scores, got {len(scores)}")
            
            # Verify each score dictionary has all token IDs
            for score_dict in scores:
                self.assertEqual(set(score_dict.keys()), set(label_token_ids),
                    f"Score dict missing some token IDs: {set(score_dict.keys())} vs {set(label_token_ids)}")



if __name__ == "__main__":
    unittest.main() 