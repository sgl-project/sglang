import sglang as sgl

"""
SOTA MULTI-STEP REASONING EXAMPLE
This script demonstrates SGLang's unique ability to chain 
reasoning steps together with zero-latency state sharing.
"""

@sgl.function
def sota_reasoning_chain(s, question):
    # Step 1: High-level brainstorming
    s += "Question: " + question + "\n"
    s += "Brainstorming: " + sgl.gen("thought_process", max_tokens=150, stop="\n")
    
    # Step 2: Final Verification (SOTA logic)
    s += "\nBased on the above, provide the final concise answer: "
    s += sgl.gen("final_answer", max_tokens=50)

def run_example():
    # Example usage: Solving a complex logic puzzle
    state = sota_reasoning_chain.run(
        question="If a robot moves 10m North and 5m East, what is its displacement?"
    )
    
    print(f"--- SOTA Thought Process ---\n{state['thought_process']}")
    print(f"--- Final Answer ---\n{state['final_answer']}")

if __name__ == "__main__":
    run_example()
