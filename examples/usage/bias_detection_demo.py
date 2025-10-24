"""
SGLang Bias Detection Demo

This example demonstrates how to use the BiasAuditor to detect circular
reasoning bias in LLM evaluation workflows.

Scenario: We're evaluating an LLM's sentiment analysis capabilities and
iteratively adjusting the temperature parameter based on performance.
This is a common pattern that can lead to circular bias.
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from python.sglang.lang.bias_audit import BiasAuditor, create_auditor
import numpy as np


def simulate_llm_response(prompt: str, temperature: float) -> str:
    """
    Simulate an LLM response (in real usage, this would be actual LLM generation).
    
    For demonstration, we simulate that higher temperature leads to longer,
    more varied responses.
    """
    base_response = f"Analyzing sentiment for: '{prompt}'. "
    
    # Simulate temperature effect
    if temperature < 0.6:
        response = base_response + "The sentiment is neutral."
    elif temperature < 0.8:
        response = base_response + "The sentiment appears to be positive with moderate confidence."
    else:
        response = base_response + "The sentiment is decidedly positive, showing strong emotional indicators and enthusiastic language patterns."
    
    # Add some randomness
    extra = " Additional analysis: " * int(temperature * 3)
    return response + extra


def demo_no_bias():
    """
    Scenario 1: Stable evaluation (NO BIAS)
    
    Using consistent constraints throughout evaluation.
    """
    print("=" * 70)
    print("SCENARIO 1: Stable Evaluation (Expected: NO BIAS)")
    print("=" * 70)
    print()
    
    auditor = BiasAuditor()
    
    # Test prompts
    prompts = [
        "The product is excellent!",
        "I love this service.",
        "Best purchase ever!",
        "Highly recommended.",
        "Amazing quality!"
    ]
    
    # Consistent constraints
    temperature = 0.7
    max_tokens = 100
    
    print(f"Evaluating with CONSISTENT constraints:")
    print(f"  Temperature: {temperature}")
    print(f"  Max tokens: {max_tokens}")
    print()
    
    # Run evaluation
    for round_num in range(3):
        print(f"Round {round_num + 1}:")
        for i, prompt in enumerate(prompts):
            response = simulate_llm_response(prompt, temperature)
            
            # Record generation
            auditor.record_generation(
                output=response,
                constraints={
                    'temperature': temperature,
                    'max_tokens': max_tokens
                }
            )
            print(f"  ✓ Evaluated prompt {i+1}")
        print()
    
    # Perform audit
    print("Performing bias audit...")
    print()
    result = auditor.audit(time_periods=3)
    
    print(result.summary())
    print()
    
    # Interpretation
    if result.overall_bias:
        print("⚠️  UNEXPECTED: Bias detected in stable evaluation!")
    else:
        print("✅ EXPECTED: No bias detected. Evaluation is consistent.")
    print()
    print()


def demo_with_bias():
    """
    Scenario 2: Iterative tuning (BIAS LIKELY)
    
    Adjusting temperature based on perceived performance.
    """
    print("=" * 70)
    print("SCENARIO 2: Iterative Tuning (Expected: BIAS DETECTED)")
    print("=" * 70)
    print()
    
    auditor = BiasAuditor(
        psi_threshold=0.05,  # More sensitive
        ccs_threshold=0.85,
        rho_pc_threshold=0.3
    )
    
    # Test prompts
    prompts = [
        "The product is excellent!",
        "I love this service.",
        "Best purchase ever!"
    ]
    
    # Iteratively adjust temperature
    temperatures = [0.5, 0.6, 0.7, 0.8, 0.9]
    
    print("Evaluating with ITERATIVELY ADJUSTED constraints:")
    print("(Simulating optimization based on performance)")
    print()
    
    for round_num, temp in enumerate(temperatures):
        print(f"Round {round_num + 1} - Temperature: {temp:.1f}")
        
        for i, prompt in enumerate(prompts):
            response = simulate_llm_response(prompt, temp)
            
            # Record generation
            auditor.record_generation(
                output=response,
                constraints={
                    'temperature': temp,
                    'max_tokens': 100
                }
            )
            print(f"  ✓ Evaluated prompt {i+1}")
        print()
    
    # Perform audit
    print("Performing bias audit...")
    print()
    result = auditor.audit(time_periods=5)
    
    print(result.summary())
    print()
    
    # Detailed analysis
    print("Detailed Analysis:")
    print("-" * 70)
    print(f"PSI (Parameter Stability):    {result.psi_score:.4f}")
    print(f"  → Interpretation: {'HIGH instability' if result.psi_bias else 'Acceptable stability'}")
    print()
    print(f"CCS (Constraint Consistency): {result.ccs_score:.4f}")
    print(f"  → Interpretation: {'LOW consistency' if result.ccs_bias else 'Good consistency'}")
    print()
    print(f"ρ_PC (Performance-Constraint): {result.rho_pc_score:+.4f}")
    print(f"  → Interpretation: {'STRONG correlation' if result.rho_pc_bias else 'Weak correlation'}")
    print("-" * 70)
    print()
    
    # Interpretation
    if result.overall_bias:
        print("⚠️  EXPECTED: Bias detected!")
        print(f"   Confidence: {result.confidence:.0%}")
        print(f"   {result.bias_votes}/3 indicators flagged")
        print()
        print("   Recommendation:")
        print("   - Use fixed constraints across all evaluations")
        print("   - Separate tuning phase from evaluation phase")
        print("   - Report constraint settings with results")
    else:
        print("✅ UNEXPECTED: No strong bias detected.")
        print("   (Statistical indicators within acceptable range)")
    print()
    print()


def demo_batch_recording():
    """
    Scenario 3: Batch recording
    
    Recording multiple generations at once for efficiency.
    """
    print("=" * 70)
    print("SCENARIO 3: Batch Recording")
    print("=" * 70)
    print()
    
    auditor = create_auditor(thresholds={'psi': 0.1, 'ccs': 0.8})
    
    # Simulate batch generation
    batch_outputs = [
        "Positive sentiment detected.",
        "Neutral tone observed.",
        "Highly positive response.",
        "Moderately positive sentiment.",
        "Very positive evaluation."
    ]
    
    batch_constraints = [
        {'temperature': 0.7, 'max_tokens': 50},
        {'temperature': 0.7, 'max_tokens': 50},
        {'temperature': 0.7, 'max_tokens': 50},
        {'temperature': 0.7, 'max_tokens': 50},
        {'temperature': 0.7, 'max_tokens': 50}
    ]
    
    batch_scores = [0.8, 0.75, 0.85, 0.78, 0.82]
    
    print("Recording batch of 5 generations...")
    auditor.record_batch(batch_outputs, batch_constraints, batch_scores)
    print(f"✓ Recorded {len(batch_outputs)} generations")
    print()
    
    # Record more batches to reach minimum
    for _ in range(2):
        auditor.record_batch(batch_outputs, batch_constraints, batch_scores)
    
    print(f"Total generations: {len(auditor.history)}")
    print()
    
    # Statistics
    stats = auditor.get_statistics()
    print("Auditor Statistics:")
    for key, value in stats.items():
        if key != 'thresholds':
            print(f"  {key}: {value}")
    print()
    print()


def demo_json_export():
    """
    Scenario 4: JSON export for logging/reporting
    """
    print("=" * 70)
    print("SCENARIO 4: JSON Export")
    print("=" * 70)
    print()
    
    auditor = BiasAuditor()
    
    # Quick generation
    for i in range(10):
        auditor.record_generation(
            output=f"Response {i}",
            constraints={'temperature': 0.7},
            performance_score=0.8 + np.random.rand() * 0.1
        )
    
    result = auditor.audit(time_periods=3)
    
    print("JSON Export:")
    print("-" * 70)
    print(result.to_json())
    print("-" * 70)
    print()
    print("This JSON can be:")
    print("  • Logged to monitoring systems")
    print("  • Stored in databases")
    print("  • Included in evaluation reports")
    print("  • Used for automated alerts")
    print()
    print()


def main():
    """Run all demo scenarios."""
    print("\n")
    print("╔════════════════════════════════════════════════════════════════════╗")
    print("║     SGLang Circular Bias Detection Demo                           ║")
    print("║                                                                    ║")
    print("║  Demonstrating detection of circular reasoning in LLM evaluation  ║")
    print("╚════════════════════════════════════════════════════════════════════╝")
    print("\n")
    
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Run scenarios
    demo_no_bias()
    input("Press Enter to continue to next scenario...")
    print("\n")
    
    demo_with_bias()
    input("Press Enter to continue to next scenario...")
    print("\n")
    
    demo_batch_recording()
    input("Press Enter to continue to next scenario...")
    print("\n")
    
    demo_json_export()
    
    print("=" * 70)
    print("Demo Complete!")
    print("=" * 70)
    print()
    print("Key Takeaways:")
    print("  1. BiasAuditor detects circular reasoning in LLM evaluation")
    print("  2. Three statistical indicators provide robust detection")
    print("  3. Easy integration with existing evaluation workflows")
    print("  4. Supports both real-time and batch processing")
    print()
    print("For more information:")
    print("  • Documentation: docs/bias_detection.md")
    print("  • Paper: Zhang et al. (2024), JOSS (under review)")
    print("  • Repository: https://github.com/hongping-zh/circular-bias-detection")
    print()


if __name__ == "__main__":
    main()
