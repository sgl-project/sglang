#!/usr/bin/env python3
"""
Prompt Analyzer for Pre-Generation Attention Pattern Prediction

Analyzes prompts before generation to:
1. Estimate complexity (simple/medium/complex)
2. Predict likely attention patterns (zones)
3. Recommend model routing (4B vs 80B)
4. Flag potential safety concerns

This enables proactive query routing and resource allocation
before incurring generation costs.

Usage:
    from prompt_analyzer import PromptAnalyzer

    analyzer = PromptAnalyzer()
    analysis = analyzer.analyze("Explain quantum entanglement to a 5-year-old")

    print(f"Complexity: {analysis['complexity']}")
    print(f"Recommended model: {analysis['recommended_model']}")
    print(f"Expected zones: {analysis['predicted_zones']}")
"""

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Tuple


class ComplexityLevel(Enum):
    SIMPLE = "simple"
    MEDIUM = "medium"
    COMPLEX = "complex"


class QueryType(Enum):
    FACTUAL = "factual"
    REASONING = "reasoning"
    CREATIVE = "creative"
    CODE = "code"
    MATH = "math"
    CONVERSATIONAL = "conversational"
    INSTRUCTION = "instruction"


@dataclass
class PromptAnalysis:
    """Result of prompt analysis."""

    # Core metrics
    complexity: ComplexityLevel
    complexity_score: float  # 0.0 to 1.0
    query_type: QueryType

    # Predictions
    predicted_zones: List[str]
    expected_entropy: Tuple[float, float]  # (min, max)
    key_entities: List[str]

    # Recommendations
    recommended_model: str  # "4b", "80b", or "either"
    recommended_max_tokens: int
    recommended_temperature: float

    # Safety
    safety_flags: List[str] = field(default_factory=list)
    confidence: float = 0.5

    def to_dict(self) -> Dict:
        return {
            "complexity": self.complexity.value,
            "complexity_score": self.complexity_score,
            "query_type": self.query_type.value,
            "predicted_zones": self.predicted_zones,
            "expected_entropy": list(self.expected_entropy),
            "key_entities": self.key_entities,
            "recommended_model": self.recommended_model,
            "recommended_max_tokens": self.recommended_max_tokens,
            "recommended_temperature": self.recommended_temperature,
            "safety_flags": self.safety_flags,
            "confidence": self.confidence,
        }


class PromptAnalyzer:
    """
    Analyzes prompts to predict attention patterns and recommend routing.

    Uses heuristics based on:
    - Query structure and keywords
    - Domain detection (code, math, factual)
    - Historical attention pattern correlations
    """

    def __init__(self):
        # Patterns for query type detection
        self.code_patterns = [
            r"\b(write|implement|create|code|function|class|method|api|debug|fix|refactor)\b.*\b(python|javascript|java|c\+\+|rust|go|sql|html|css)\b",
            r"\b(python|javascript|java|c\+\+|rust|go|sql)\b.*\b(code|function|class|script)\b",
            r"```",
            r"\bdef\s+\w+\(",
            r"\bclass\s+\w+",
            r"\bfunction\s+\w+",
        ]

        self.math_patterns = [
            r"\b(calculate|compute|solve|evaluate|prove|derive)\b",
            r"\b(equation|formula|integral|derivative|matrix|vector)\b",
            r"\b\d+\s*[\+\-\*\/\^]\s*\d+",
            r"\b(sum|product|limit|factorial)\b",
        ]

        self.reasoning_patterns = [
            r"\b(if|then|therefore|because|since|hence|thus)\b.*\b(then|because|therefore)\b",
            r"\b(who|what|why|how|which)\b.*\b(taller|shorter|older|younger|better|worse|more|less)\b",
            r"\b(step by step|think through|reason|analyze|compare)\b",
            r"\b(first|second|third|finally|next|then)\b.*\b(first|second|third|finally|next|then)\b",
        ]

        self.factual_patterns = [
            r"\b(what is|who is|where is|when did|how many|how much)\b",
            r"\b(capital|population|president|ceo|founder|year)\b",
            r"\b(definition|meaning|explain what)\b",
        ]

        self.creative_patterns = [
            r"\b(write|create|compose|generate)\b.*\b(story|poem|haiku|song|essay|article)\b",
            r"\b(creative|imaginative|fictional|fantasy)\b",
            r"\b(once upon|in a world|imagine)\b",
        ]

        # Domain keywords for entity extraction
        self.technical_domains = {
            "quantum",
            "neural",
            "machine learning",
            "ai",
            "algorithm",
            "blockchain",
            "cryptocurrency",
            "physics",
            "chemistry",
            "biology",
        }

        # Safety patterns
        self.safety_patterns = [
            (r"\b(hack|exploit|attack|bypass|crack)\b", "potential_security"),
            (r"\b(weapon|bomb|explosive|poison)\b", "potential_harm"),
            (r"\b(illegal|fraud|scam|steal)\b", "potential_illegal"),
        ]

    def analyze(self, prompt: str) -> PromptAnalysis:
        """
        Analyze a prompt and return predictions.

        Args:
            prompt: The user prompt to analyze

        Returns:
            PromptAnalysis with complexity, type, and recommendations
        """
        prompt_lower = prompt.lower()

        # Detect query type
        query_type = self._detect_query_type(prompt_lower)

        # Calculate complexity
        complexity, complexity_score = self._calculate_complexity(
            prompt, prompt_lower, query_type
        )

        # Predict attention zones
        predicted_zones = self._predict_zones(query_type, complexity)

        # Extract key entities
        key_entities = self._extract_entities(prompt)

        # Generate recommendations
        recommended_model = self._recommend_model(complexity, query_type)
        recommended_max_tokens = self._recommend_max_tokens(query_type, complexity)
        recommended_temperature = self._recommend_temperature(query_type)

        # Check for safety flags
        safety_flags = self._check_safety(prompt_lower)

        # Calculate overall confidence
        confidence = self._calculate_confidence(prompt, query_type)

        # Predict entropy range
        expected_entropy = self._predict_entropy(query_type, complexity)

        return PromptAnalysis(
            complexity=complexity,
            complexity_score=complexity_score,
            query_type=query_type,
            predicted_zones=predicted_zones,
            expected_entropy=expected_entropy,
            key_entities=key_entities,
            recommended_model=recommended_model,
            recommended_max_tokens=recommended_max_tokens,
            recommended_temperature=recommended_temperature,
            safety_flags=safety_flags,
            confidence=confidence,
        )

    def _detect_query_type(self, prompt_lower: str) -> QueryType:
        """Detect the type of query based on patterns."""
        scores = {
            QueryType.CODE: self._pattern_score(prompt_lower, self.code_patterns),
            QueryType.MATH: self._pattern_score(prompt_lower, self.math_patterns),
            QueryType.REASONING: self._pattern_score(
                prompt_lower, self.reasoning_patterns
            ),
            QueryType.FACTUAL: self._pattern_score(prompt_lower, self.factual_patterns),
            QueryType.CREATIVE: self._pattern_score(
                prompt_lower, self.creative_patterns
            ),
        }

        # Default to conversational if no strong signals
        max_score = max(scores.values())
        if max_score < 0.3:
            return QueryType.CONVERSATIONAL

        return max(scores, key=scores.get)

    def _pattern_score(self, text: str, patterns: List[str]) -> float:
        """Calculate match score for a list of patterns."""
        matches = sum(1 for p in patterns if re.search(p, text, re.IGNORECASE))
        return min(1.0, matches / max(1, len(patterns) * 0.3))

    def _calculate_complexity(
        self, prompt: str, prompt_lower: str, query_type: QueryType
    ) -> Tuple[ComplexityLevel, float]:
        """Calculate complexity level and score."""
        score = 0.0

        # Length factor
        word_count = len(prompt.split())
        if word_count > 100:
            score += 0.3
        elif word_count > 50:
            score += 0.2
        elif word_count > 20:
            score += 0.1

        # Query type factor
        type_complexity = {
            QueryType.CONVERSATIONAL: 0.0,
            QueryType.FACTUAL: 0.1,
            QueryType.CREATIVE: 0.2,
            QueryType.MATH: 0.3,
            QueryType.CODE: 0.3,
            QueryType.REASONING: 0.4,
            QueryType.INSTRUCTION: 0.2,
        }
        score += type_complexity.get(query_type, 0.1)

        # Multi-step indicators
        if any(w in prompt_lower for w in ["step by step", "first", "then", "finally"]):
            score += 0.2

        # Technical domain factor
        if any(domain in prompt_lower for domain in self.technical_domains):
            score += 0.15

        # Nested conditions
        if re.search(r"if.*then.*if", prompt_lower):
            score += 0.2

        # Normalize
        score = min(1.0, score)

        # Map to level
        if score < 0.3:
            level = ComplexityLevel.SIMPLE
        elif score < 0.6:
            level = ComplexityLevel.MEDIUM
        else:
            level = ComplexityLevel.COMPLEX

        return level, score

    def _predict_zones(
        self, query_type: QueryType, complexity: ComplexityLevel
    ) -> List[str]:
        """Predict likely attention zones based on query type."""
        zone_map = {
            QueryType.FACTUAL: ["semantic_bridge"],
            QueryType.REASONING: ["semantic_bridge", "structure_ripple"],
            QueryType.CODE: ["semantic_bridge", "syntax_floor"],
            QueryType.MATH: ["semantic_bridge", "structure_ripple"],
            QueryType.CREATIVE: ["exploration", "semantic_bridge"],
            QueryType.CONVERSATIONAL: ["syntax_floor", "semantic_bridge"],
            QueryType.INSTRUCTION: ["semantic_bridge"],
        }

        zones = zone_map.get(query_type, ["semantic_bridge"])

        # Add long_range for complex queries
        if complexity == ComplexityLevel.COMPLEX:
            zones.append("long_range")

        return zones

    def _extract_entities(self, prompt: str) -> List[str]:
        """Extract key entities from the prompt."""
        entities = []

        # Extract capitalized words (potential named entities)
        capitalized = re.findall(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b", prompt)
        entities.extend(capitalized[:5])

        # Extract quoted strings
        quoted = re.findall(r'"([^"]+)"', prompt)
        entities.extend(quoted[:3])

        # Extract technical terms
        technical = re.findall(
            r"\b(API|JSON|HTTP|SQL|Python|JavaScript|algorithm|function|class)\b",
            prompt,
            re.IGNORECASE,
        )
        entities.extend(technical[:3])

        return list(set(entities))[:8]

    def _recommend_model(
        self, complexity: ComplexityLevel, query_type: QueryType
    ) -> str:
        """Recommend which model to use."""
        # Always recommend 80B for complex reasoning
        if complexity == ComplexityLevel.COMPLEX:
            return "80b"

        # Simple queries can use 4B
        if complexity == ComplexityLevel.SIMPLE:
            if query_type in [QueryType.CONVERSATIONAL, QueryType.FACTUAL]:
                return "4b"

        # Medium complexity
        if query_type in [QueryType.CODE, QueryType.MATH, QueryType.REASONING]:
            return "80b"

        return "either"

    def _recommend_max_tokens(
        self, query_type: QueryType, complexity: ComplexityLevel
    ) -> int:
        """Recommend max_tokens based on expected output length."""
        base_tokens = {
            QueryType.FACTUAL: 100,
            QueryType.CONVERSATIONAL: 150,
            QueryType.CREATIVE: 300,
            QueryType.CODE: 500,
            QueryType.MATH: 200,
            QueryType.REASONING: 400,
            QueryType.INSTRUCTION: 250,
        }

        tokens = base_tokens.get(query_type, 200)

        # Adjust for complexity
        if complexity == ComplexityLevel.COMPLEX:
            tokens = int(tokens * 1.5)
        elif complexity == ComplexityLevel.SIMPLE:
            tokens = int(tokens * 0.7)

        return tokens

    def _recommend_temperature(self, query_type: QueryType) -> float:
        """Recommend temperature based on query type."""
        temperatures = {
            QueryType.FACTUAL: 0.1,
            QueryType.MATH: 0.1,
            QueryType.CODE: 0.2,
            QueryType.REASONING: 0.3,
            QueryType.CONVERSATIONAL: 0.7,
            QueryType.CREATIVE: 0.8,
            QueryType.INSTRUCTION: 0.4,
        }
        return temperatures.get(query_type, 0.5)

    def _check_safety(self, prompt_lower: str) -> List[str]:
        """Check for potential safety concerns."""
        flags = []
        for pattern, flag in self.safety_patterns:
            if re.search(pattern, prompt_lower):
                flags.append(flag)
        return flags

    def _calculate_confidence(self, prompt: str, query_type: QueryType) -> float:
        """Calculate confidence in the analysis."""
        # Higher confidence for clear patterns
        base_confidence = 0.5

        # Longer prompts give more signal
        word_count = len(prompt.split())
        if word_count > 20:
            base_confidence += 0.1
        if word_count > 50:
            base_confidence += 0.1

        # Clear query types have higher confidence
        if query_type in [QueryType.CODE, QueryType.MATH]:
            base_confidence += 0.2
        elif query_type == QueryType.CONVERSATIONAL:
            base_confidence -= 0.1

        return min(0.95, max(0.3, base_confidence))

    def _predict_entropy(
        self, query_type: QueryType, complexity: ComplexityLevel
    ) -> Tuple[float, float]:
        """Predict expected attention entropy range."""
        # Base entropy ranges by query type
        entropy_ranges = {
            QueryType.FACTUAL: (0.1, 0.3),
            QueryType.CODE: (0.15, 0.35),
            QueryType.MATH: (0.2, 0.4),
            QueryType.REASONING: (0.25, 0.5),
            QueryType.CREATIVE: (0.3, 0.6),
            QueryType.CONVERSATIONAL: (0.2, 0.5),
            QueryType.INSTRUCTION: (0.15, 0.4),
        }

        base_min, base_max = entropy_ranges.get(query_type, (0.2, 0.5))

        # Adjust for complexity
        if complexity == ComplexityLevel.COMPLEX:
            base_min += 0.1
            base_max += 0.1
        elif complexity == ComplexityLevel.SIMPLE:
            base_min -= 0.05
            base_max -= 0.05

        return (max(0.05, base_min), min(0.95, base_max))


# Convenience function for quick analysis
def analyze_prompt(prompt: str) -> Dict:
    """
    Quick analysis of a prompt.

    Args:
        prompt: The prompt to analyze

    Returns:
        Dictionary with analysis results
    """
    analyzer = PromptAnalyzer()
    return analyzer.analyze(prompt).to_dict()


if __name__ == "__main__":
    # Demo
    test_prompts = [
        "What is the capital of France?",
        "Write a Python function to check if a number is prime",
        "If Alice is taller than Bob, and Bob is taller than Carol, who is shortest?",
        "Write a haiku about programming",
        "Explain quantum entanglement to a 5-year-old",
        "Hello, how are you?",
    ]

    analyzer = PromptAnalyzer()

    for prompt in test_prompts:
        print(f"\n{'='*60}")
        print(f"Prompt: {prompt}")
        print("-" * 60)

        analysis = analyzer.analyze(prompt)

        print(
            f"Complexity: {analysis.complexity.value} ({analysis.complexity_score:.2f})"
        )
        print(f"Query Type: {analysis.query_type.value}")
        print(f"Predicted Zones: {analysis.predicted_zones}")
        print(
            f"Expected Entropy: {analysis.expected_entropy[0]:.2f} - {analysis.expected_entropy[1]:.2f}"
        )
        print(f"Key Entities: {analysis.key_entities}")
        print(f"Recommended Model: {analysis.recommended_model}")
        print(f"Recommended Max Tokens: {analysis.recommended_max_tokens}")
        print(f"Recommended Temperature: {analysis.recommended_temperature}")
        print(f"Safety Flags: {analysis.safety_flags}")
        print(f"Confidence: {analysis.confidence:.2f}")
