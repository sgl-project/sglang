import os
import pytest
import torch
import math
from transformers import AutoModelForCausalLM, AutoTokenizer
from sglang.srt.entrypoints.engine import Engine
from sglang.srt.managers.tokenizer_manager import TokenizerManager

def get_test_model_name():
    """Get the model name from environment or use a default test model."""
    return os.getenv("TEST_MODEL_NAME", "Qwen/Qwen2.5-1.5B-Instruct")

@pytest.fixture
def model_name():
    return get_test_model_name()

@pytest.fixture
def hf_model_and_tokenizer(model_name):
    """Load the HuggingFace model and tokenizer for direct comparison."""
    if model_name.startswith("meta-llama/"):
        pytest.skip("Skipping test for gated repo (meta-llama) – please use a public model (e.g. Qwen/Qwen2.5-1.5B-Instruct).")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, trust_remote_code=True)
    return model, tokenizer

@pytest.fixture
def sglang_engine(model_name):
    """Initialize the SGLang engine for scoring."""
    if model_name.startswith("meta-llama/"):
        pytest.skip("Skipping test for gated repo (meta-llama) – please use a public model (e.g. Qwen/Qwen2.5-1.5B-Instruct).")
    try:
        engine = Engine(model_path=model_name)
    except TypeError:
        engine = Engine(served_model_name=model_name)
    return engine

def compute_hf_scores(model, tokenizer, prompt, texts, positive_token_id, negative_token_id):
    """Compute scores using direct HuggingFace model inference.
    Scores are computed as prob(positive) / (prob(positive) + prob(negative)).
    """
    scores = []
    for text in texts:
        # Combine prompt and text directly without newline
        full_text = f"{prompt}{text}"
        inputs = tokenizer(full_text, return_tensors="pt").to(model.device)
        
        # Get logits for the last token
        with torch.no_grad():
            outputs = model(**inputs)
            last_token_logits = outputs.logits[0, -1]
            
        # Get logprobs for the target tokens
        logprobs = torch.log_softmax(last_token_logits, dim=-1)
        pos_logprob = logprobs[positive_token_id].item()
        neg_logprob = logprobs[negative_token_id].item()
        
        # Convert logprobs to probabilities
        pos_prob = math.exp(pos_logprob)
        neg_prob = math.exp(neg_logprob)
        
        # Get top 5 tokens and their probabilities
        top_probs, top_indices = torch.topk(logprobs, 5)
        top_tokens = tokenizer.batch_decode(top_indices.unsqueeze(-1))
        
        # Print debug info
        print(f"\nHF Debug for text: {text}")
        print(f"Input length: {len(inputs['input_ids'][0])}")
        print(f"Full input text: {full_text}")
        print("\nTop 5 most likely next tokens:")
        for prob, token in zip(top_probs, top_tokens):
            print(f"  Token '{token}': {math.exp(prob.item()):.6f} (logprob: {prob.item():.6f})")
        print("\nTarget token probabilities:")
        print(f"  Token {positive_token_id} ('{tokenizer.decode([positive_token_id])}'): {pos_prob:.6f} (logprob: {pos_logprob:.6f})")
        print(f"  Token {negative_token_id} ('{tokenizer.decode([negative_token_id])}'): {neg_prob:.6f} (logprob: {neg_logprob:.6f})")
        
        # Compute score as prob(positive) / (prob(positive) + prob(negative))
        score = pos_prob / (pos_prob + neg_prob)
        scores.append(score)
    
    return scores

def test_score_consistency(hf_model_and_tokenizer, sglang_engine):
    """Test that SGLang scoring matches direct HuggingFace model scoring.
    Scores should be in range [0,1] where higher values indicate stronger positive sentiment.
    """
    model, tokenizer = hf_model_and_tokenizer
    
    # Test cases - using simpler cases first for debugging
    test_cases = [
        {
            "prompt": "I pledge allegiance",
            "texts": ["", " to"],  # Remove space since we're directly concatenating
            "positive_token": " to",  # Keep space in token since that's how it's encoded
            "negative_token": " the"
        }
    ]
    
    for case in test_cases:
        # Get token IDs
        positive_token_id = tokenizer.encode(case["positive_token"])[0]
        negative_token_id = tokenizer.encode(case["negative_token"])[0]
        
        print(f"\nTesting case:")
        print(f"Prompt: {case['prompt']}")
        print(f"Text: {case['texts'][0]}")
        print(f"Positive token ID: {positive_token_id}")
        print(f"Negative token ID: {negative_token_id}")
        
        # Get scores from HuggingFace
        hf_scores = compute_hf_scores(
            model, tokenizer,
            case["prompt"], case["texts"],
            positive_token_id, negative_token_id
        )
        
        # Get scores from SGLang
        sglang_result = sglang_engine.score(
            text_1=case["prompt"],
            text_2=case["texts"],
            positive_token_id=positive_token_id,
            negative_token_id=negative_token_id
        )
        sglang_scores = sglang_result["scores"]
        
        # Print SGLang debug info
        print("\nSGLang Debug:")
        print(f"Output structure: {sglang_result}")
        
        # Compare scores
        assert len(hf_scores) == len(sglang_scores), "Score lengths don't match"
        for hf_score, sglang_score in zip(hf_scores, sglang_scores):
            print(f"\nScore comparison:")
            print(f"HF score: {hf_score:.6f}")
            print(f"SGLang score: {sglang_score:.6f}")
            assert abs(hf_score - sglang_score) < 1e-4, \
                f"Scores don't match: HF={hf_score:.6f}, SGLang={sglang_score:.6f}"
            assert 0 <= hf_score <= 1, f"HF score {hf_score:.6f} not in [0,1]"
            assert 0 <= sglang_score <= 1, f"SGLang score {sglang_score:.6f} not in [0,1]"

def test_score_edge_cases(hf_model_and_tokenizer, sglang_engine):
    """Test scoring with edge cases."""
    model, tokenizer = hf_model_and_tokenizer
    
    # Test with empty text
    with pytest.raises(ValueError):
        sglang_engine.score(
            text_1="",
            text_2=["test"],
            positive_token_id=1,
            negative_token_id=2
        )
    
    # Test with invalid token IDs
    vocab_size = tokenizer.vocab_size
    with pytest.raises(ValueError):
        sglang_engine.score(
            text_1="test",
            text_2=["test"],
            positive_token_id=vocab_size + 1,  # Invalid token ID
            negative_token_id=1
        )
    
    # Test with very long prompt
    long_prompt = "test " * 1000  # Create a very long prompt
    with pytest.raises(ValueError):
        sglang_engine.score(
            text_1=long_prompt,
            text_2=["test"],
            positive_token_id=1,
            negative_token_id=2
        )

def test_score_batch_handling(sglang_engine):
    """Test that batch scoring works correctly."""
    # Test with different batch sizes
    batch_sizes = [1, 2, 4, 8]
    
    for batch_size in batch_sizes:
        texts = [f"test {i}" for i in range(batch_size)]
        result = sglang_engine.score(
            text_1="The test was",
            text_2=texts,
            positive_token_id=1,  # Using dummy token IDs for this test
            negative_token_id=2
        )
        
        assert len(result["scores"]) == batch_size, \
            f"Expected {batch_size} scores, got {len(result['scores'])}"
        
        # Verify usage statistics
        assert result["usage"]["prompt_tokens"] > 0
        assert result["usage"]["completion_tokens"] > 0
        assert result["usage"]["total_tokens"] == \
            result["usage"]["prompt_tokens"] + result["usage"]["completion_tokens"]

if __name__ == "__main__":
    pytest.main([__file__]) 