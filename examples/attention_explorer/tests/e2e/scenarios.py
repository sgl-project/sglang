"""
Diverse prompt scenarios for attention pattern exploration.

Each scenario is designed to elicit specific attention patterns:
- SYNTAX_FLOOR: Local, grammar-focused attention (offset < 8)
- SEMANTIC_BRIDGE: Mid-range semantic retrieval (offset 8-255)
- LONG_RANGE: Document-level reasoning (offset 256+)
- STRUCTURE_RIPPLE: Periodic patterns (code, lists)
- DIFFUSE: Exploratory, uncertain attention

These scenarios help build a comprehensive manifold map and
inform future sinq (semantic intelligence query) implementation.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from enum import Enum
import random


class ExpectedManifold(Enum):
    SYNTAX_FLOOR = "syntax_floor"
    SEMANTIC_BRIDGE = "semantic_bridge"
    LONG_RANGE = "long_range"
    STRUCTURE_RIPPLE = "structure_ripple"
    DIFFUSE = "diffuse"
    MIXED = "mixed"  # Transitions between zones


@dataclass
class Scenario:
    """A test scenario with expected attention patterns."""
    name: str
    category: str
    description: str
    messages: List[Dict[str, str]]
    expected_manifold: ExpectedManifold
    expected_characteristics: Dict[str, Any] = field(default_factory=dict)
    max_tokens: int = 512
    temperature: float = 0.7
    tags: List[str] = field(default_factory=list)


# =============================================================================
# SYNTAX FLOOR SCENARIOS (Local attention, grammar/completion)
# =============================================================================

SYNTAX_SCENARIOS = [
    Scenario(
        name="simple_completion",
        category="syntax",
        description="Simple sentence completion requiring local context",
        messages=[
            {"role": "user", "content": "Complete this sentence naturally: The quick brown fox"}
        ],
        expected_manifold=ExpectedManifold.SYNTAX_FLOOR,
        expected_characteristics={"local_mass": ">0.5", "entropy": "<0.4"},
        max_tokens=50,
        tags=["completion", "grammar"],
    ),
    Scenario(
        name="grammar_correction",
        category="syntax",
        description="Grammar correction focusing on local structure",
        messages=[
            {"role": "user", "content": "Fix the grammar: Him went to the store yesterday and buyed milk."}
        ],
        expected_manifold=ExpectedManifold.SYNTAX_FLOOR,
        expected_characteristics={"local_mass": ">0.4"},
        max_tokens=100,
        tags=["grammar", "correction"],
    ),
    Scenario(
        name="word_by_word_translation",
        category="syntax",
        description="Simple translation requiring local token attention",
        messages=[
            {"role": "user", "content": "Translate word by word to Spanish: The cat sleeps on the mat."}
        ],
        expected_manifold=ExpectedManifold.SYNTAX_FLOOR,
        max_tokens=100,
        tags=["translation", "local"],
    ),
    Scenario(
        name="rhyme_generation",
        category="syntax",
        description="Generate rhyming words (phonetic local attention)",
        messages=[
            {"role": "user", "content": "List 10 words that rhyme with 'cat':"}
        ],
        expected_manifold=ExpectedManifold.SYNTAX_FLOOR,
        expected_characteristics={"entropy": "<0.5"},
        max_tokens=100,
        tags=["rhyme", "phonetic"],
    ),
    Scenario(
        name="autocomplete_code_line",
        category="syntax",
        description="Complete a single line of code",
        messages=[
            {"role": "user", "content": "Complete this Python line: result = sum([x for x in range("}
        ],
        expected_manifold=ExpectedManifold.SYNTAX_FLOOR,
        max_tokens=30,
        tags=["code", "completion"],
    ),
]


# =============================================================================
# SEMANTIC BRIDGE SCENARIOS (Mid-range, paragraph-level reasoning)
# =============================================================================

SEMANTIC_SCENARIOS = [
    Scenario(
        name="paragraph_summary",
        category="semantic",
        description="Summarize a paragraph (requires sentence-level context)",
        messages=[
            {"role": "user", "content": """Summarize this paragraph in one sentence:

The development of artificial intelligence has progressed rapidly over the past decade.
Machine learning algorithms have become increasingly sophisticated, enabling applications
from image recognition to natural language processing. However, concerns about AI safety
and ethical implications have grown alongside these technological advances. Researchers
are now working to ensure AI systems are aligned with human values and remain beneficial."""}
        ],
        expected_manifold=ExpectedManifold.SEMANTIC_BRIDGE,
        expected_characteristics={"mid_mass": ">0.4"},
        max_tokens=100,
        tags=["summary", "comprehension"],
    ),
    Scenario(
        name="cause_effect_reasoning",
        category="semantic",
        description="Identify cause and effect relationships",
        messages=[
            {"role": "user", "content": """Read the following and identify the cause and effect:

The city experienced unprecedented rainfall for three consecutive weeks. Streets flooded,
basements filled with water, and the subway system was forced to shut down. Many residents
had to evacuate their homes, and local businesses reported significant losses.

What was the cause and what were the effects?"""}
        ],
        expected_manifold=ExpectedManifold.SEMANTIC_BRIDGE,
        expected_characteristics={"mid_mass": ">0.35"},
        max_tokens=200,
        tags=["reasoning", "causality"],
    ),
    Scenario(
        name="text_comparison",
        category="semantic",
        description="Compare two short texts",
        messages=[
            {"role": "user", "content": """Compare these two descriptions:

Description A: "The algorithm processes data in linear time, making it efficient for large datasets."
Description B: "The method requires quadratic time complexity, limiting its use to smaller inputs."

What are the key differences?"""}
        ],
        expected_manifold=ExpectedManifold.SEMANTIC_BRIDGE,
        max_tokens=150,
        tags=["comparison", "analysis"],
    ),
    Scenario(
        name="context_qa",
        category="semantic",
        description="Answer question based on provided context",
        messages=[
            {"role": "user", "content": """Based on the following information, answer the question:

Context: The Python programming language was created by Guido van Rossum and first released
in 1991. It emphasizes code readability and uses significant whitespace. Python supports
multiple programming paradigms, including procedural, object-oriented, and functional programming.

Question: What programming paradigms does Python support?"""}
        ],
        expected_manifold=ExpectedManifold.SEMANTIC_BRIDGE,
        expected_characteristics={"mid_mass": ">0.3", "entropy": "<0.6"},
        max_tokens=100,
        tags=["qa", "comprehension"],
    ),
    Scenario(
        name="sentiment_analysis",
        category="semantic",
        description="Analyze sentiment of a review",
        messages=[
            {"role": "user", "content": """Analyze the sentiment of this review:

"I was really excited about this product, but after using it for a week, I'm disappointed.
The build quality is decent, and it looks nice on my desk. However, the performance is
sluggish, the battery life is terrible, and customer support was unhelpful when I reached
out. I expected better for the price."

Is this positive, negative, or mixed? Explain your analysis."""}
        ],
        expected_manifold=ExpectedManifold.SEMANTIC_BRIDGE,
        max_tokens=200,
        tags=["sentiment", "analysis"],
    ),
]


# =============================================================================
# LONG-RANGE SCENARIOS (Document-level, cross-context reasoning)
# =============================================================================

LONG_DOCUMENT = """
# Introduction to Neural Networks

Neural networks are computational models inspired by biological neural networks in the brain.
They consist of interconnected nodes (neurons) organized in layers that process information.

## Basic Architecture

A typical neural network has three types of layers:
1. Input Layer: Receives the initial data
2. Hidden Layers: Process information through weighted connections
3. Output Layer: Produces the final result

The connections between neurons have associated weights that are adjusted during training
to minimize the difference between predicted and actual outputs.

## Training Process

Training a neural network involves:
- Forward propagation: Data flows through the network to produce predictions
- Loss calculation: The error between predictions and actual values is computed
- Backpropagation: Gradients are calculated to determine how to adjust weights
- Weight updates: Parameters are modified using optimization algorithms like SGD or Adam

## Applications

Neural networks are used in various domains:
- Computer Vision: Image classification, object detection, facial recognition
- Natural Language Processing: Machine translation, sentiment analysis, chatbots
- Speech Recognition: Voice assistants, transcription services
- Autonomous Systems: Self-driving cars, robotic control

## Challenges

Despite their success, neural networks face several challenges:
- Interpretability: Understanding why a network makes certain decisions
- Data requirements: Often need large amounts of labeled training data
- Computational cost: Training large models requires significant resources
- Overfitting: Models may memorize training data rather than generalizing

## Future Directions

Research continues in areas like:
- Efficient architectures that require less computation
- Few-shot learning techniques
- Improved interpretability methods
- Energy-efficient hardware for neural network inference
"""

LONG_RANGE_SCENARIOS = [
    Scenario(
        name="document_qa_distant",
        category="long_range",
        description="Answer question requiring information from document start",
        messages=[
            {"role": "user", "content": f"""{LONG_DOCUMENT}

Based on the document above, what are the three types of layers in a typical neural network architecture?"""}
        ],
        expected_manifold=ExpectedManifold.LONG_RANGE,
        expected_characteristics={"long_mass": ">0.3"},
        max_tokens=150,
        tags=["qa", "document", "retrieval"],
    ),
    Scenario(
        name="document_cross_reference",
        category="long_range",
        description="Connect information from different parts of document",
        messages=[
            {"role": "user", "content": f"""{LONG_DOCUMENT}

How do the challenges mentioned relate to the applications described earlier?
Give specific examples connecting them."""}
        ],
        expected_manifold=ExpectedManifold.LONG_RANGE,
        expected_characteristics={"long_mass": ">0.25", "mid_mass": ">0.2"},
        max_tokens=300,
        tags=["synthesis", "cross-reference"],
    ),
    Scenario(
        name="multi_turn_recall",
        category="long_range",
        description="Multi-turn conversation requiring recall of earlier context",
        messages=[
            {"role": "user", "content": "I'm building a machine learning system. It needs to recognize faces in photos."},
            {"role": "assistant", "content": "That's a computer vision task! You'll want to use a convolutional neural network (CNN) architecture. Popular choices include ResNet, VGGNet, or more modern architectures like EfficientNet. For face recognition specifically, you might want to look at FaceNet or ArcFace which are optimized for generating face embeddings."},
            {"role": "user", "content": "Great! Now I also need it to understand voice commands."},
            {"role": "assistant", "content": "For voice command recognition, you'll need a speech recognition system. Recurrent neural networks (RNNs), particularly LSTMs or GRUs, have been traditionally used. However, transformer-based models like Wav2Vec 2.0 and Whisper have shown excellent performance. You'll need to handle audio preprocessing, feature extraction (like mel spectrograms), and possibly speaker verification if you need to identify who is speaking."},
            {"role": "user", "content": "What challenges might I face combining both the systems I asked about initially?"}
        ],
        expected_manifold=ExpectedManifold.LONG_RANGE,
        expected_characteristics={"long_mass": ">0.2"},
        max_tokens=300,
        tags=["multi-turn", "recall", "synthesis"],
    ),
    Scenario(
        name="story_continuation",
        category="long_range",
        description="Continue a story maintaining consistency with earlier details",
        messages=[
            {"role": "user", "content": """Continue this story, making sure to reference the earlier details:

Chapter 1: The Discovery
Dr. Sarah Chen had spent fifteen years searching for evidence of extraterrestrial intelligence.
Working at the Arecibo Observatory, she had analyzed countless radio signals, each one turning
out to be natural phenomena or human interference. Her colleagues had long since moved on to
other projects, but Sarah remained dedicated. On the morning of March 15th, 2025, everything
changed. The signal was different - a repeating pattern of prime numbers that could only be
artificial. It came from Proxima Centauri, our nearest stellar neighbor.

Chapter 2: The Response
Within hours, Sarah's discovery had been verified by observatories around the world. The
United Nations hastily convened an emergency session. World leaders debated whether and how
to respond. Some argued for immediate transmission of a return message, others counseled
caution. Dr. Chen was invited to address the assembly.

Chapter 3: The Decision
Write what happens next, referencing details from both previous chapters."""}
        ],
        expected_manifold=ExpectedManifold.LONG_RANGE,
        max_tokens=400,
        tags=["creative", "narrative", "consistency"],
    ),
]


# =============================================================================
# STRUCTURE RIPPLE SCENARIOS (Periodic patterns, code, lists)
# =============================================================================

STRUCTURE_SCENARIOS = [
    Scenario(
        name="code_generation_function",
        category="structure",
        description="Generate a complete function with structured code",
        messages=[
            {"role": "user", "content": """Write a Python function that:
1. Takes a list of numbers
2. Filters out negative numbers
3. Squares the remaining numbers
4. Returns the sum

Include docstring and type hints."""}
        ],
        expected_manifold=ExpectedManifold.STRUCTURE_RIPPLE,
        expected_characteristics={"entropy": "<0.5"},
        max_tokens=300,
        tags=["code", "function", "structured"],
    ),
    Scenario(
        name="code_generation_class",
        category="structure",
        description="Generate a class with methods",
        messages=[
            {"role": "user", "content": """Create a Python class called 'BankAccount' with:
- Constructor taking owner name and initial balance
- deposit() method
- withdraw() method with overdraft protection
- get_balance() method
- transfer_to() method for transferring to another account

Include proper error handling."""}
        ],
        expected_manifold=ExpectedManifold.STRUCTURE_RIPPLE,
        max_tokens=500,
        tags=["code", "class", "oop"],
    ),
    Scenario(
        name="numbered_list_generation",
        category="structure",
        description="Generate a numbered list with consistent structure",
        messages=[
            {"role": "user", "content": "List 10 tips for effective public speaking, numbered 1-10, with a brief explanation for each."}
        ],
        expected_manifold=ExpectedManifold.STRUCTURE_RIPPLE,
        max_tokens=500,
        tags=["list", "structured", "repetitive"],
    ),
    Scenario(
        name="table_generation",
        category="structure",
        description="Generate a markdown table",
        messages=[
            {"role": "user", "content": """Create a markdown table comparing 5 programming languages:
- Include columns: Language, Year Created, Primary Use, Key Feature, Popularity Rank
- Languages: Python, JavaScript, Rust, Go, TypeScript"""}
        ],
        expected_manifold=ExpectedManifold.STRUCTURE_RIPPLE,
        max_tokens=300,
        tags=["table", "markdown", "structured"],
    ),
    Scenario(
        name="json_generation",
        category="structure",
        description="Generate structured JSON data",
        messages=[
            {"role": "user", "content": """Generate a JSON object representing a user profile with:
- Basic info (name, email, age)
- Address (street, city, country, postal code)
- Preferences (theme, language, notifications as nested object)
- Array of 3 recent orders (id, date, total, items array)

Use realistic sample data."""}
        ],
        expected_manifold=ExpectedManifold.STRUCTURE_RIPPLE,
        expected_characteristics={"entropy": "<0.4"},
        max_tokens=500,
        tags=["json", "data", "nested"],
    ),
    Scenario(
        name="algorithm_implementation",
        category="structure",
        description="Implement a sorting algorithm with clear structure",
        messages=[
            {"role": "user", "content": """Implement quicksort in Python with:
- Clear partition function
- Recursive sort function
- Example usage with test array
- Comments explaining each step"""}
        ],
        expected_manifold=ExpectedManifold.STRUCTURE_RIPPLE,
        max_tokens=400,
        tags=["code", "algorithm", "recursive"],
    ),
]


# =============================================================================
# DIFFUSE / MIXED SCENARIOS (Exploratory, creative, uncertain)
# =============================================================================

DIFFUSE_SCENARIOS = [
    Scenario(
        name="open_ended_brainstorm",
        category="diffuse",
        description="Open-ended brainstorming with many possibilities",
        messages=[
            {"role": "user", "content": "Brainstorm innovative uses for blockchain technology beyond cryptocurrency."}
        ],
        expected_manifold=ExpectedManifold.DIFFUSE,
        expected_characteristics={"entropy": ">0.6"},
        max_tokens=400,
        tags=["brainstorm", "creative", "open"],
    ),
    Scenario(
        name="philosophical_question",
        category="diffuse",
        description="Abstract philosophical reasoning",
        messages=[
            {"role": "user", "content": "What does it mean for an AI to 'understand' something? Is there a meaningful difference between processing and understanding?"}
        ],
        expected_manifold=ExpectedManifold.DIFFUSE,
        expected_characteristics={"entropy": ">0.5"},
        max_tokens=400,
        tags=["philosophy", "abstract", "reasoning"],
    ),
    Scenario(
        name="creative_writing_prompt",
        category="diffuse",
        description="Creative writing with freedom of interpretation",
        messages=[
            {"role": "user", "content": "Write a short poem about the relationship between humans and technology."}
        ],
        expected_manifold=ExpectedManifold.DIFFUSE,
        max_tokens=200,
        tags=["creative", "poetry", "art"],
    ),
    Scenario(
        name="hypothetical_scenario",
        category="diffuse",
        description="Explore a hypothetical with many branches",
        messages=[
            {"role": "user", "content": "If humans could photosynthesize like plants, how might society be different?"}
        ],
        expected_manifold=ExpectedManifold.DIFFUSE,
        expected_characteristics={"entropy": ">0.5"},
        max_tokens=400,
        tags=["hypothetical", "speculative", "creative"],
    ),
]


# =============================================================================
# MIXED/TRANSITION SCENARIOS (Pattern changes during generation)
# =============================================================================

MIXED_SCENARIOS = [
    Scenario(
        name="explain_then_code",
        category="mixed",
        description="Explanation followed by code (semantic -> structure)",
        messages=[
            {"role": "user", "content": """First explain what a binary search tree is and why it's useful.
Then implement one in Python with insert, search, and in-order traversal methods."""}
        ],
        expected_manifold=ExpectedManifold.MIXED,
        expected_characteristics={"transition": "semantic_to_structure"},
        max_tokens=600,
        tags=["transition", "explanation", "code"],
    ),
    Scenario(
        name="analyze_then_summarize",
        category="mixed",
        description="Detailed analysis followed by summary (long -> local)",
        messages=[
            {"role": "user", "content": f"""{LONG_DOCUMENT}

First, analyze the key themes and arguments in this document in detail.
Then provide a one-sentence TL;DR summary."""}
        ],
        expected_manifold=ExpectedManifold.MIXED,
        expected_characteristics={"transition": "long_to_local"},
        max_tokens=400,
        tags=["transition", "analysis", "summary"],
    ),
    Scenario(
        name="code_review_and_fix",
        category="mixed",
        description="Review code then fix it (structure -> structure with semantic bridge)",
        messages=[
            {"role": "user", "content": """Review this code for bugs and then provide the corrected version:

```python
def calculate_average(numbers):
    total = 0
    for num in numbers:
        total += num
    return total / len(numbers)

def find_median(numbers):
    sorted_nums = numbers.sort()
    mid = len(sorted_nums) // 2
    if len(sorted_nums) % 2 == 0:
        return (sorted_nums[mid-1] + sorted_nums[mid]) / 2
    return sorted_nums[mid]

def process_data(data):
    if data == None:
        return []
    return [x * 2 for x in data if x > 0]
```

First identify all bugs, then provide the corrected code."""}
        ],
        expected_manifold=ExpectedManifold.MIXED,
        max_tokens=500,
        tags=["code-review", "debugging", "mixed"],
    ),
]


# =============================================================================
# MATH & REASONING SCENARIOS (For MoE expert analysis)
# =============================================================================

MATH_SCENARIOS = [
    Scenario(
        name="step_by_step_math",
        category="math",
        description="Multi-step mathematical problem",
        messages=[
            {"role": "user", "content": """Solve this problem step by step:

A train leaves Station A at 9:00 AM traveling at 60 mph toward Station B.
Another train leaves Station B at 10:00 AM traveling at 80 mph toward Station A.
If the stations are 280 miles apart, at what time will the trains meet?

Show all your work."""}
        ],
        expected_manifold=ExpectedManifold.SEMANTIC_BRIDGE,
        max_tokens=400,
        tags=["math", "word-problem", "reasoning"],
    ),
    Scenario(
        name="proof_explanation",
        category="math",
        description="Mathematical proof requiring logical chain",
        messages=[
            {"role": "user", "content": "Prove that the sum of the first n positive integers is n(n+1)/2 using mathematical induction."}
        ],
        expected_manifold=ExpectedManifold.SEMANTIC_BRIDGE,
        expected_characteristics={"mid_mass": ">0.35"},
        max_tokens=400,
        tags=["math", "proof", "induction"],
    ),
    Scenario(
        name="logic_puzzle",
        category="math",
        description="Logic puzzle requiring deduction",
        messages=[
            {"role": "user", "content": """Solve this logic puzzle:

There are five houses in a row, each painted a different color.
In each house lives a person of a different nationality.
Each person drinks a different beverage, smokes a different brand, and keeps a different pet.

Clues:
1. The Brit lives in the red house.
2. The Swede keeps dogs.
3. The Dane drinks tea.
4. The green house is directly to the left of the white house.
5. The green house owner drinks coffee.
6. The person who smokes Pall Mall keeps birds.
7. The owner of the yellow house smokes Dunhill.
8. The person in the center house drinks milk.
9. The Norwegian lives in the first house.
10. The Blend smoker lives next to the one who keeps cats.

Who owns the fish?"""}
        ],
        expected_manifold=ExpectedManifold.LONG_RANGE,  # Needs to track many constraints
        max_tokens=800,
        tags=["logic", "puzzle", "deduction"],
    ),
]


# =============================================================================
# SCENARIO REGISTRY
# =============================================================================

ALL_SCENARIOS: List[Scenario] = (
    SYNTAX_SCENARIOS +
    SEMANTIC_SCENARIOS +
    LONG_RANGE_SCENARIOS +
    STRUCTURE_SCENARIOS +
    DIFFUSE_SCENARIOS +
    MIXED_SCENARIOS +
    MATH_SCENARIOS
)

SCENARIOS_BY_CATEGORY: Dict[str, List[Scenario]] = {
    "syntax": SYNTAX_SCENARIOS,
    "semantic": SEMANTIC_SCENARIOS,
    "long_range": LONG_RANGE_SCENARIOS,
    "structure": STRUCTURE_SCENARIOS,
    "diffuse": DIFFUSE_SCENARIOS,
    "mixed": MIXED_SCENARIOS,
    "math": MATH_SCENARIOS,
}

SCENARIOS_BY_MANIFOLD: Dict[ExpectedManifold, List[Scenario]] = {}
for scenario in ALL_SCENARIOS:
    if scenario.expected_manifold not in SCENARIOS_BY_MANIFOLD:
        SCENARIOS_BY_MANIFOLD[scenario.expected_manifold] = []
    SCENARIOS_BY_MANIFOLD[scenario.expected_manifold].append(scenario)


def get_random_scenarios(n: int, categories: Optional[List[str]] = None) -> List[Scenario]:
    """Get n random scenarios, optionally filtered by category."""
    pool = ALL_SCENARIOS
    if categories:
        pool = [s for s in ALL_SCENARIOS if s.category in categories]
    return random.sample(pool, min(n, len(pool)))


def get_balanced_scenarios(n_per_category: int = 2) -> List[Scenario]:
    """Get balanced representation from each category."""
    result = []
    for category, scenarios in SCENARIOS_BY_CATEGORY.items():
        result.extend(random.sample(scenarios, min(n_per_category, len(scenarios))))
    return result
