#!/usr/bin/env python3
"""
Qwen3-Next-80B-A3B-Thinking-FP8 Exploration Script

4-hour exploration of the 80B MoE model to discover:
- Attention patterns across different task types
- MoE routing behavior (512 experts)
- Zone distribution for various prompts
- Think vs output phase differences

Usage:
    python scripts/qwen80b_exploration.py --duration 4

Requirements:
    - SGLang server running with Qwen3-Next-80B-A3B-Thinking-FP8
    - Sidecar running with SQLite storage
"""

import argparse
import json
import logging
import random
import sqlite3
import struct
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import requests

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(
            f'exploration_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
        ),
    ],
)
logger = logging.getLogger(__name__)

# Configuration
SGLANG_URL = "http://localhost:30000"
SIDECAR_URL = "http://localhost:9009"
DB_PATH = "./exploration_fingerprints.db"
OUTPUT_DIR = "./exploration_outputs"
FINGERPRINT_DIM = 20

# =============================================================================
# PROMPT CATEGORIES
# =============================================================================

PROMPT_CATEGORIES = {
    "reasoning": [
        "Think step by step: If all roses are flowers and some flowers fade quickly, can we conclude that some roses fade quickly?",
        "A farmer has 17 sheep. All but 9 die. How many sheep are left?",
        "If it takes 5 machines 5 minutes to make 5 widgets, how long would it take 100 machines to make 100 widgets?",
        "Three people check into a hotel room that costs $30. They each contribute $10. Later, the clerk realizes the room only costs $25 and gives $5 to the bellboy to return. The bellboy keeps $2 and gives $1 back to each person. So each person paid $9 (total $27) and the bellboy kept $2 (total $29). Where did the other dollar go?",
        "You're in a room with two doors. One leads to freedom, one to death. Two guards stand by the doors. One always lies, one always tells the truth. You can ask one question to one guard. What do you ask?",
        "A bat and ball cost $1.10 in total. The bat costs $1 more than the ball. How much does the ball cost?",
        "If you have a 3-gallon jug and a 5-gallon jug, how do you measure exactly 4 gallons of water?",
        "In a race, you overtake the person in second place. What place are you in now?",
    ],
    "math": [
        "Calculate: What is the derivative of f(x) = x^3 * ln(x)?",
        "Solve the system of equations: 2x + 3y = 7 and 4x - y = 1",
        "What is the integral of sin(x) * cos(x) dx?",
        "Find all prime numbers between 100 and 150.",
        "If a triangle has sides of length 5, 12, and 13, what is its area?",
        "What is 17! (17 factorial)?",
        "Solve: log_2(x) + log_2(x-2) = 3",
        "What is the sum of the first 100 positive integers?",
        "Calculate the eigenvalues of the matrix [[1, 2], [2, 1]]",
        "What is the probability of getting exactly 3 heads in 5 coin flips?",
    ],
    "coding": [
        "Write a Python function to check if a string is a palindrome.",
        "Implement a binary search algorithm in Python.",
        "Write a function to find the longest common subsequence of two strings.",
        "Implement a simple LRU cache in Python.",
        "Write code to detect a cycle in a linked list.",
        "Implement quicksort in Python.",
        "Write a function to serialize and deserialize a binary tree.",
        "Implement a trie data structure with insert and search methods.",
        "Write code to find all permutations of a string.",
        "Implement a thread-safe singleton pattern in Python.",
    ],
    "creative": [
        "Write a haiku about artificial intelligence.",
        "Create a short story (100 words) about a robot learning to feel emotions.",
        "Write a limerick about debugging code.",
        "Compose a sonnet about the beauty of mathematics.",
        "Write a dialogue between the Sun and the Moon.",
        "Create a recipe for 'Algorithm Soup' with creative ingredients.",
        "Write a letter from the perspective of the last tree on Earth.",
        "Compose a song about machine learning, with verses and chorus.",
    ],
    "factual": [
        "What is the speed of light in meters per second?",
        "Who wrote 'Pride and Prejudice'?",
        "What is the capital of Australia?",
        "When did World War II end?",
        "What is the chemical formula for water?",
        "Who painted the Mona Lisa?",
        "What is the largest planet in our solar system?",
        "What year did the first iPhone release?",
        "What is the atomic number of gold?",
        "Who discovered penicillin?",
    ],
    "analysis": [
        "Compare and contrast democracy and authoritarianism.",
        "Analyze the pros and cons of remote work.",
        "What are the ethical implications of AI in healthcare?",
        "Evaluate the impact of social media on mental health.",
        "Discuss the relationship between economic growth and environmental sustainability.",
        "Analyze the causes and effects of the 2008 financial crisis.",
        "Compare the philosophies of Plato and Aristotle.",
        "Evaluate the effectiveness of different approaches to education.",
    ],
    "instruction_following": [
        "List exactly 5 countries that start with the letter 'S'. Only list the names, nothing else.",
        "Translate 'Hello, how are you?' into French, Spanish, and German. Format as a bullet list.",
        "Count backwards from 10 to 1, with each number on a new line.",
        "Write a sentence using exactly 7 words about the ocean.",
        "Name 3 fruits, 3 vegetables, and 3 animals. Use commas to separate items within each category.",
        "Summarize the concept of photosynthesis in exactly 2 sentences.",
        "List the days of the week in reverse order.",
        "Write 'AI' using ASCII art (simple block letters).",
    ],
    "multi_turn_context": [
        # These will be sent as multi-turn conversations
        [
            {
                "role": "user",
                "content": "I'm planning a trip to Japan. What's the best time to visit?",
            },
            {
                "role": "assistant",
                "content": "The best times to visit Japan are spring (March-May) for cherry blossoms and fall (September-November) for autumn foliage. Both seasons offer mild weather and beautiful scenery.",
            },
            {
                "role": "user",
                "content": "What about visiting temples? Which ones should I prioritize?",
            },
        ],
        [
            {
                "role": "user",
                "content": "I'm learning to cook. Can you suggest a simple pasta recipe?",
            },
            {
                "role": "assistant",
                "content": "Here's a simple aglio e olio: Cook spaghetti, sauté sliced garlic in olive oil until golden, add red pepper flakes, toss with pasta, and finish with parsley and parmesan.",
            },
            {
                "role": "user",
                "content": "That sounds good! How can I make it more protein-rich?",
            },
        ],
        [
            {"role": "user", "content": "Explain what a neural network is."},
            {
                "role": "assistant",
                "content": "A neural network is a computing system inspired by biological neurons. It consists of layers of interconnected nodes that process information by passing signals through weighted connections, learning patterns from data.",
            },
            {
                "role": "user",
                "content": "How does backpropagation work in training these networks?",
            },
        ],
    ],
    "long_context": [
        """Read this passage and answer the question below:

The history of computing can be traced back to ancient civilizations. The abacus, developed around 2400 BCE in Babylon, was one of the first computing tools. In the 17th century, Blaise Pascal invented the Pascaline, a mechanical calculator. Charles Babbage designed the Analytical Engine in the 1830s, which is considered the first general-purpose computer concept. Ada Lovelace wrote algorithms for this machine, making her the first computer programmer.

The modern era of computing began with the development of electronic computers in the 1940s. ENIAC, completed in 1945, was one of the first general-purpose electronic computers. It weighed 30 tons and occupied 1,800 square feet. The invention of the transistor in 1947 revolutionized computing, leading to smaller and more efficient machines.

The personal computer revolution started in the 1970s with machines like the Apple II and IBM PC. The development of graphical user interfaces and the internet in the following decades transformed computing from a specialized tool into a ubiquitous part of daily life.

Question: What was the significance of the transistor's invention in 1947?""",
        """Analyze this code and explain what it does:

```python
def mystery_function(n, memo={}):
    if n in memo:
        return memo[n]
    if n <= 1:
        return n
    result = mystery_function(n-1, memo) + mystery_function(n-2, memo)
    memo[n] = result
    return result

def process_sequence(limit):
    sequence = []
    for i in range(limit):
        val = mystery_function(i)
        if val % 2 == 0:
            sequence.append(val)
    return sequence

# What does process_sequence(20) return?
```

Explain the algorithm, its time complexity, and what the output would be.""",
    ],
    "roleplay": [
        "You are a Shakespearean actor. Explain how a computer works in the style of Shakespeare.",
        "Pretend you're a detective from a noir film. Describe what you observe in a typical office.",
        "You are a medieval scholar who just discovered a smartphone. Describe your observations.",
        "Act as a sports commentator providing play-by-play of someone making a cup of coffee.",
    ],
    "edge_cases": [
        "What is the sound of one hand clapping?",
        "Can you write a sentence that is both true and false?",
        "Explain consciousness to someone who has never experienced it.",
        "What existed before the Big Bang?",
        "If I ask you to not think of a pink elephant, what happens?",
        "Describe the color blue to a person who has been blind from birth.",
    ],
}


# =============================================================================
# DATA STRUCTURES
# =============================================================================


@dataclass
class ExplorationResult:
    """Result from a single exploration prompt."""

    prompt_id: str
    category: str
    prompt: str
    response: str
    completion_tokens: int
    total_time_ms: float
    tokens_per_second: float
    attention_steps: int
    fingerprints: List[Dict]
    moe_routing: List[Dict]
    think_tokens: int
    output_tokens: int
    zone_distribution: Dict[str, int]
    timestamp: str


@dataclass
class ExplorationSession:
    """Tracking for an exploration session."""

    session_id: str
    start_time: datetime
    end_time: Optional[datetime] = None
    total_prompts: int = 0
    total_tokens: int = 0
    category_counts: Dict[str, int] = field(default_factory=dict)
    zone_distribution: Dict[str, int] = field(default_factory=dict)
    discovery_runs: List[str] = field(default_factory=list)
    results: List[ExplorationResult] = field(default_factory=list)


# =============================================================================
# HELPERS
# =============================================================================


def pack_fingerprint(arr: np.ndarray) -> bytes:
    """Pack numpy array to fingerprint blob."""
    return struct.pack(f"<{FINGERPRINT_DIM}f", *arr.astype(np.float32))


def unpack_fingerprint(blob: bytes) -> np.ndarray:
    """Unpack fingerprint from blob to numpy array."""
    return np.array(struct.unpack(f"<{FINGERPRINT_DIM}f", blob), dtype=np.float32)


def check_services():
    """Check if SGLang and sidecar are running."""
    try:
        resp = requests.get(f"{SGLANG_URL}/v1/models", timeout=5)
        if resp.status_code != 200:
            return False, "SGLang server not responding"
        models = resp.json()
        logger.info(f"SGLang server running with models: {models}")
    except Exception as e:
        return False, f"SGLang server error: {e}"

    try:
        resp = requests.get(f"{SIDECAR_URL}/health", timeout=5)
        if resp.status_code != 200:
            return False, "Sidecar not responding"
        logger.info("Sidecar is healthy")
    except Exception as e:
        return False, f"Sidecar error: {e}"

    return True, "Services running"


def init_database(db_path: str):
    """Initialize the exploration database."""
    schema_path = Path(__file__).parent.parent / "discovery" / "schema.sql"

    if Path(db_path).exists():
        logger.info(f"Using existing database: {db_path}")
        return

    logger.info(f"Initializing database: {db_path}")
    conn = sqlite3.connect(db_path)
    with open(schema_path) as f:
        conn.executescript(f.read())
    conn.close()


# =============================================================================
# EXPLORATION ENGINE
# =============================================================================


class ExplorationEngine:
    """Engine for running model exploration."""

    def __init__(
        self,
        sglang_url: str = SGLANG_URL,
        sidecar_url: str = SIDECAR_URL,
        db_path: str = DB_PATH,
        output_dir: str = OUTPUT_DIR,
    ):
        self.sglang_url = sglang_url
        self.sidecar_url = sidecar_url
        self.db_path = db_path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

        self.session = ExplorationSession(
            session_id=datetime.now().strftime("%Y%m%d_%H%M%S"),
            start_time=datetime.now(),
        )

        # Initialize database
        init_database(db_path)

        # Findings log
        self.findings_path = (
            self.output_dir / f"findings_{self.session.session_id}.jsonl"
        )

    def send_prompt(
        self,
        messages: List[Dict],
        category: str,
        prompt_id: str,
        max_tokens: int = 2048,
    ) -> Optional[ExplorationResult]:
        """Send a prompt and capture attention data."""
        try:
            start_time = time.time()

            # Make request with attention capture
            resp = requests.post(
                f"{self.sglang_url}/v1/chat/completions",
                json={
                    "model": "default",
                    "messages": messages,
                    "max_tokens": max_tokens,
                    "temperature": 0.7,
                    "stream": True,
                    "return_attention_tokens": True,
                },
                stream=True,
                timeout=300,
            )

            if resp.status_code != 200:
                logger.error(f"Request failed: {resp.status_code}")
                return None

            # Parse streaming response
            completion_text = ""
            attention_steps = []
            moe_routing = []
            completion_tokens = 0

            for line in resp.iter_lines():
                if not line:
                    continue
                line = line.decode("utf-8")
                if not line.startswith("data: "):
                    continue
                data = line[6:]
                if data == "[DONE]":
                    break

                try:
                    chunk = json.loads(data)
                    delta = chunk.get("choices", [{}])[0].get("delta", {})

                    # Extract content
                    content = delta.get("content") or ""
                    completion_text += content

                    # Extract attention tokens
                    attn_data = delta.get("attention_tokens") or chunk.get(
                        "attention_tokens"
                    )
                    if attn_data:
                        for step_data in attn_data:
                            attention_steps.append(step_data)

                    # Extract MoE routing
                    moe_data = delta.get("moe_routing") or chunk.get("moe_routing")
                    if moe_data:
                        moe_routing.extend(moe_data)

                    # Extract usage
                    usage = chunk.get("usage", {})
                    if usage:
                        completion_tokens = usage.get(
                            "completion_tokens", completion_tokens
                        )

                except json.JSONDecodeError:
                    continue

            end_time = time.time()
            total_time_ms = (end_time - start_time) * 1000

            # Compute metrics
            if completion_tokens == 0:
                completion_tokens = len(attention_steps)
            tokens_per_second = (
                completion_tokens / (total_time_ms / 1000) if total_time_ms > 0 else 0
            )

            # Compute zone distribution from fingerprints
            zone_distribution = {
                "syntax_floor": 0,
                "semantic_bridge": 0,
                "structure_ripple": 0,
                "unknown": 0,
            }
            fingerprints = []

            for step in attention_steps:
                fp_data = step.get("fingerprint", {})
                if fp_data:
                    fingerprints.append(fp_data)
                    # Classify zone
                    zone = self._classify_zone(fp_data)
                    zone_distribution[zone] = zone_distribution.get(zone, 0) + 1

            # Count think vs output tokens
            think_tokens = sum(1 for s in attention_steps if s.get("phase") == "think")
            output_tokens = sum(
                1 for s in attention_steps if s.get("phase") == "output"
            )

            # Store fingerprints in sidecar
            self._store_fingerprints(prompt_id, fingerprints)

            result = ExplorationResult(
                prompt_id=prompt_id,
                category=category,
                prompt=messages[-1]["content"] if messages else "",
                response=completion_text[:500],  # Truncate for storage
                completion_tokens=completion_tokens,
                total_time_ms=total_time_ms,
                tokens_per_second=tokens_per_second,
                attention_steps=len(attention_steps),
                fingerprints=fingerprints[:10],  # Store sample
                moe_routing=moe_routing[:10],  # Store sample
                think_tokens=think_tokens,
                output_tokens=output_tokens,
                zone_distribution=zone_distribution,
                timestamp=datetime.now().isoformat(),
            )

            return result

        except Exception as e:
            logger.error(f"Error sending prompt: {e}")
            return None

    def _classify_zone(self, fp_data: Dict) -> str:
        """Classify fingerprint into zone."""
        local_mass = fp_data.get("local_mass", 0)
        mid_mass = fp_data.get("mid_mass", 0)
        long_mass = fp_data.get("long_mass", 0)
        entropy = fp_data.get("entropy", 0)

        # Simple heuristics (should match classifier.py)
        if local_mass > 0.5 and entropy < 2.5:
            return "syntax_floor"
        if long_mass > 0.25:
            return "structure_ripple"
        return "semantic_bridge"

    def _store_fingerprints(self, request_id: str, fingerprints: List[Dict]):
        """Store fingerprints in sidecar."""
        for i, fp in enumerate(fingerprints):
            try:
                # Build vector from fingerprint data
                vector = (
                    [
                        fp.get("local_mass", 0),
                        fp.get("mid_mass", 0),
                        fp.get("long_mass", 0),
                        fp.get("entropy", 0),
                    ]
                    + fp.get("histogram", [0] * 8)
                    + fp.get("layer_entropy", [0] * 8)
                )

                if len(vector) < FINGERPRINT_DIM:
                    vector.extend([0] * (FINGERPRINT_DIM - len(vector)))
                vector = vector[:FINGERPRINT_DIM]

                requests.post(
                    f"{self.sidecar_url}/fingerprint",
                    json={
                        "request_id": request_id,
                        "vector": vector,
                        "step": i,
                    },
                    timeout=5,
                )
            except Exception as e:
                logger.warning(f"Failed to store fingerprint: {e}")

    def log_finding(self, finding: Dict):
        """Log a finding to the findings file."""
        with open(self.findings_path, "a") as f:
            f.write(json.dumps(finding) + "\n")

    def run_discovery(self) -> Optional[str]:
        """Run discovery job and return run_id."""
        logger.info("Running discovery job...")

        try:
            discovery_script = (
                Path(__file__).parent.parent / "discovery" / "discovery_job.py"
            )

            import subprocess

            result = subprocess.run(
                [
                    sys.executable,
                    str(discovery_script),
                    "--db",
                    self.db_path,
                    "--output",
                    str(self.output_dir / "discovery"),
                    "--hours",
                    "24",
                    "--min-cluster-size",
                    "10",
                ],
                capture_output=True,
                text=True,
                timeout=300,
            )

            if result.returncode == 0:
                # Parse run_id from output
                for line in result.stdout.split("\n"):
                    if "Run ID:" in line:
                        run_id = line.split("Run ID:")[1].strip()
                        logger.info(f"Discovery completed: {run_id}")
                        return run_id
            else:
                logger.error(f"Discovery failed: {result.stderr}")

        except Exception as e:
            logger.error(f"Discovery error: {e}")

        return None

    def analyze_moe_routing(self, moe_data: List[Dict]) -> Dict:
        """Analyze MoE routing patterns."""
        if not moe_data:
            return {}

        # Count expert usage
        expert_counts = {}
        for routing in moe_data:
            for expert_id in routing.get("expert_ids", []):
                expert_counts[expert_id] = expert_counts.get(expert_id, 0) + 1

        # Find most/least used experts
        if expert_counts:
            sorted_experts = sorted(
                expert_counts.items(), key=lambda x: x[1], reverse=True
            )
            return {
                "total_routings": len(moe_data),
                "unique_experts": len(expert_counts),
                "top_5_experts": sorted_experts[:5],
                "bottom_5_experts": sorted_experts[-5:],
                "expert_entropy": -sum(
                    (c / len(moe_data)) * np.log(c / len(moe_data) + 1e-10)
                    for c in expert_counts.values()
                ),
            }
        return {}

    def run_exploration(
        self,
        duration_hours: float = 4.0,
        discovery_interval_minutes: float = 30.0,
    ):
        """Run the full exploration for the specified duration."""
        logger.info(f"Starting {duration_hours}-hour exploration of Qwen3-80B")
        logger.info(f"Session ID: {self.session.session_id}")
        logger.info(f"Discovery interval: {discovery_interval_minutes} minutes")

        end_time = datetime.now() + timedelta(hours=duration_hours)
        last_discovery = datetime.now()
        prompt_count = 0

        # Flatten all prompts with their categories
        all_prompts = []
        for category, prompts in PROMPT_CATEGORIES.items():
            for prompt in prompts:
                all_prompts.append((category, prompt))

        logger.info(f"Total unique prompts: {len(all_prompts)}")

        try:
            while datetime.now() < end_time:
                # Select a random prompt
                category, prompt = random.choice(all_prompts)

                # Handle multi-turn prompts
                if isinstance(prompt, list):
                    messages = prompt
                else:
                    messages = [{"role": "user", "content": prompt}]

                prompt_id = f"{self.session.session_id}_{prompt_count:05d}"
                logger.info(f"[{prompt_count}] Category: {category}")
                logger.info(f"    Prompt: {messages[-1]['content'][:80]}...")

                # Send prompt
                result = self.send_prompt(messages, category, prompt_id)

                if result:
                    self.session.results.append(result)
                    self.session.total_prompts += 1
                    self.session.total_tokens += result.completion_tokens
                    self.session.category_counts[category] = (
                        self.session.category_counts.get(category, 0) + 1
                    )

                    # Update zone distribution
                    for zone, count in result.zone_distribution.items():
                        self.session.zone_distribution[zone] = (
                            self.session.zone_distribution.get(zone, 0) + count
                        )

                    # Log key metrics
                    logger.info(
                        f"    Response: {result.completion_tokens} tokens, {result.tokens_per_second:.1f} tok/s"
                    )
                    logger.info(f"    Zones: {result.zone_distribution}")
                    logger.info(
                        f"    Think/Output: {result.think_tokens}/{result.output_tokens}"
                    )

                    # Log interesting findings
                    if result.think_tokens > result.output_tokens * 2:
                        self.log_finding(
                            {
                                "type": "heavy_thinking",
                                "prompt_id": prompt_id,
                                "category": category,
                                "think_tokens": result.think_tokens,
                                "output_tokens": result.output_tokens,
                                "prompt": messages[-1]["content"][:200],
                            }
                        )

                    if (
                        result.zone_distribution.get("syntax_floor", 0)
                        > result.completion_tokens * 0.5
                    ):
                        self.log_finding(
                            {
                                "type": "high_syntax_floor",
                                "prompt_id": prompt_id,
                                "category": category,
                                "zone_distribution": result.zone_distribution,
                                "prompt": messages[-1]["content"][:200],
                            }
                        )

                    # Analyze MoE routing
                    if result.moe_routing:
                        moe_analysis = self.analyze_moe_routing(result.moe_routing)
                        if moe_analysis:
                            logger.info(
                                f"    MoE: {moe_analysis.get('unique_experts', 0)} unique experts"
                            )
                            if moe_analysis.get("unique_experts", 0) < 10:
                                self.log_finding(
                                    {
                                        "type": "low_expert_diversity",
                                        "prompt_id": prompt_id,
                                        "category": category,
                                        "moe_analysis": moe_analysis,
                                    }
                                )

                prompt_count += 1

                # Check if time for discovery
                if datetime.now() - last_discovery > timedelta(
                    minutes=discovery_interval_minutes
                ):
                    # Flush sidecar
                    try:
                        requests.post(f"{self.sidecar_url}/storage/flush", timeout=10)
                    except:
                        pass

                    run_id = self.run_discovery()
                    if run_id:
                        self.session.discovery_runs.append(run_id)
                    last_discovery = datetime.now()

                    # Log session stats
                    elapsed = datetime.now() - self.session.start_time
                    logger.info("=" * 60)
                    logger.info(f"Session Stats (elapsed: {elapsed})")
                    logger.info(f"  Total prompts: {self.session.total_prompts}")
                    logger.info(f"  Total tokens: {self.session.total_tokens}")
                    logger.info(
                        f"  Category distribution: {self.session.category_counts}"
                    )
                    logger.info(
                        f"  Zone distribution: {self.session.zone_distribution}"
                    )
                    logger.info(f"  Discovery runs: {len(self.session.discovery_runs)}")
                    logger.info("=" * 60)

                # Small delay between prompts
                time.sleep(1)

        except KeyboardInterrupt:
            logger.info("Exploration interrupted by user")

        finally:
            self.session.end_time = datetime.now()

            # Final discovery run
            logger.info("Running final discovery...")
            try:
                requests.post(f"{self.sidecar_url}/storage/flush", timeout=10)
            except:
                pass
            run_id = self.run_discovery()
            if run_id:
                self.session.discovery_runs.append(run_id)

            # Save session summary
            self.save_session_summary()

    def save_session_summary(self):
        """Save the session summary to a file."""
        summary_path = (
            self.output_dir / f"session_{self.session.session_id}_summary.json"
        )

        summary = {
            "session_id": self.session.session_id,
            "start_time": self.session.start_time.isoformat(),
            "end_time": (
                self.session.end_time.isoformat() if self.session.end_time else None
            ),
            "duration_hours": (
                (self.session.end_time - self.session.start_time).total_seconds() / 3600
                if self.session.end_time
                else None
            ),
            "total_prompts": self.session.total_prompts,
            "total_tokens": self.session.total_tokens,
            "category_counts": self.session.category_counts,
            "zone_distribution": self.session.zone_distribution,
            "discovery_runs": self.session.discovery_runs,
            "findings_file": str(self.findings_path),
        }

        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)

        logger.info(f"Session summary saved to: {summary_path}")

        # Print final summary
        print("\n" + "=" * 60)
        print("EXPLORATION COMPLETE")
        print("=" * 60)
        print(f"Duration: {summary['duration_hours']:.2f} hours")
        print(f"Total prompts: {summary['total_prompts']}")
        print(f"Total tokens: {summary['total_tokens']}")
        print(f"\nCategory distribution:")
        for cat, count in sorted(
            summary["category_counts"].items(), key=lambda x: -x[1]
        ):
            print(f"  {cat}: {count}")
        print(f"\nZone distribution:")
        for zone, count in sorted(
            summary["zone_distribution"].items(), key=lambda x: -x[1]
        ):
            print(f"  {zone}: {count}")
        print(f"\nDiscovery runs: {len(summary['discovery_runs'])}")
        print(f"\nFindings logged to: {summary['findings_file']}")
        print("=" * 60)


# =============================================================================
# MAIN
# =============================================================================


def main():
    parser = argparse.ArgumentParser(
        description="Qwen3-Next-80B-A3B-Thinking-FP8 Exploration",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=4.0,
        help="Duration in hours",
    )
    parser.add_argument(
        "--discovery-interval",
        type=float,
        default=30.0,
        help="Discovery interval in minutes",
    )
    parser.add_argument(
        "--sglang-url",
        default=SGLANG_URL,
        help="SGLang server URL",
    )
    parser.add_argument(
        "--sidecar-url",
        default=SIDECAR_URL,
        help="Sidecar URL",
    )
    parser.add_argument(
        "--db",
        default=DB_PATH,
        help="Database path",
    )
    parser.add_argument(
        "--output",
        default=OUTPUT_DIR,
        help="Output directory",
    )

    args = parser.parse_args()

    print("=" * 60)
    print("Qwen3-Next-80B-A3B-Thinking-FP8 Exploration")
    print("=" * 60)
    print(f"Duration: {args.duration} hours")
    print(f"Discovery interval: {args.discovery_interval} minutes")
    print(f"SGLang URL: {args.sglang_url}")
    print(f"Sidecar URL: {args.sidecar_url}")
    print()

    # Check services
    ok, msg = check_services()
    if not ok:
        print(f"ERROR: {msg}")
        print("\nPlease ensure:")
        print("1. SGLang server is running with Qwen3-Next-80B-A3B-Thinking-FP8")
        print("2. Sidecar is running with --db and --discovery-dir")
        sys.exit(1)

    print("Services are running. Starting exploration...")
    print()

    engine = ExplorationEngine(
        sglang_url=args.sglang_url,
        sidecar_url=args.sidecar_url,
        db_path=args.db,
        output_dir=args.output,
    )

    engine.run_exploration(
        duration_hours=args.duration,
        discovery_interval_minutes=args.discovery_interval,
    )


if __name__ == "__main__":
    main()
