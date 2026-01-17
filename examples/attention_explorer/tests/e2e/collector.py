"""
Data collection module for E2E attention exploration tests.

Collects attention patterns, fingerprints, MoE routing data, and
performance metrics for analysis and insight generation.
"""

import json
import logging
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import httpx
from scenarios import Scenario

logger = logging.getLogger(__name__)


@dataclass
class AttentionToken:
    """A single attention token with metadata."""

    token_id: int
    token_str: str
    offset: int
    weight: float
    position: int


@dataclass
class AttentionStep:
    """Attention data for a single generation step."""

    step_index: int
    output_token_id: int
    output_token_str: str
    top_k_tokens: List[AttentionToken]
    entropy: Optional[float] = None
    local_mass: Optional[float] = None  # offset < 8
    mid_mass: Optional[float] = None  # 8 <= offset < 256
    long_mass: Optional[float] = None  # offset >= 256

    def compute_mass_distribution(self):
        """Compute local/mid/long mass from top-k tokens."""
        if not self.top_k_tokens:
            return
        total = sum(t.weight for t in self.top_k_tokens)
        if total == 0:
            return

        local = sum(t.weight for t in self.top_k_tokens if t.offset < 8)
        mid = sum(t.weight for t in self.top_k_tokens if 8 <= t.offset < 256)
        long = sum(t.weight for t in self.top_k_tokens if t.offset >= 256)

        self.local_mass = local / total
        self.mid_mass = mid / total
        self.long_mass = long / total


@dataclass
class Fingerprint:
    """20D attention fingerprint vector."""

    local_mass: float
    mid_mass: float
    long_mass: float
    entropy: float
    histogram: List[float]  # 16 bins

    def to_vector(self) -> List[float]:
        return [
            self.local_mass,
            self.mid_mass,
            self.long_mass,
            self.entropy,
        ] + self.histogram


@dataclass
class MoERouting:
    """MoE routing data for a generation step."""

    step_index: int
    layer_idx: int
    expert_ids: List[int]
    expert_weights: List[float]
    token_position: int


@dataclass
class TraceData:
    """Complete trace data for a scenario run."""

    scenario_name: str
    scenario_category: str
    expected_manifold: str

    # Generation data
    prompt: str
    completion: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int

    # Attention data
    attention_steps: List[AttentionStep] = field(default_factory=list)
    fingerprints: List[Fingerprint] = field(default_factory=list)

    # MoE data (if available)
    moe_routing: List[MoERouting] = field(default_factory=list)

    # Performance metrics
    time_to_first_token_ms: Optional[float] = None
    total_generation_time_ms: Optional[float] = None
    tokens_per_second: Optional[float] = None

    # Timestamps
    start_time: str = ""
    end_time: str = ""

    # Raw response (for debugging)
    raw_response: Optional[Dict] = None


@dataclass
class CollectionRun:
    """A complete collection run with multiple traces."""

    run_id: str
    start_time: str
    end_time: Optional[str] = None
    model_name: str = ""
    server_url: str = ""
    traces: List[TraceData] = field(default_factory=list)
    errors: List[Dict[str, Any]] = field(default_factory=list)

    def add_trace(self, trace: TraceData):
        self.traces.append(trace)

    def add_error(self, scenario_name: str, error: str):
        self.errors.append(
            {
                "scenario": scenario_name,
                "error": error,
                "timestamp": datetime.now().isoformat(),
            }
        )

    def summary(self) -> Dict[str, Any]:
        """Generate summary statistics."""
        by_category = {}
        by_manifold = {}

        for trace in self.traces:
            # By category
            if trace.scenario_category not in by_category:
                by_category[trace.scenario_category] = []
            by_category[trace.scenario_category].append(trace)

            # By expected manifold
            if trace.expected_manifold not in by_manifold:
                by_manifold[trace.expected_manifold] = []
            by_manifold[trace.expected_manifold].append(trace)

        return {
            "run_id": self.run_id,
            "total_traces": len(self.traces),
            "total_errors": len(self.errors),
            "by_category": {k: len(v) for k, v in by_category.items()},
            "by_manifold": {k: len(v) for k, v in by_manifold.items()},
            "avg_tokens_per_second": sum(
                t.tokens_per_second for t in self.traces if t.tokens_per_second
            )
            / max(1, len([t for t in self.traces if t.tokens_per_second])),
        }


class AttentionCollector:
    """
    Collects attention data from SGLang server.

    Usage:
        collector = AttentionCollector("http://localhost:30000")
        await collector.connect()
        trace = await collector.run_scenario(scenario)
    """

    def __init__(
        self,
        server_url: str = "http://localhost:30000",
        timeout: float = 120.0,
        attention_top_k: int = 32,
        include_prompt_attention: bool = True,
    ):
        self.server_url = server_url.rstrip("/")
        self.timeout = timeout
        self.attention_top_k = attention_top_k
        self.include_prompt_attention = include_prompt_attention
        self.client: Optional[httpx.AsyncClient] = None
        self.model_name: Optional[str] = None

    async def connect(self) -> bool:
        """Connect to the server and verify capabilities."""
        self.client = httpx.AsyncClient(timeout=self.timeout)

        try:
            # Get model info
            response = await self.client.get(f"{self.server_url}/v1/models")
            if response.status_code == 200:
                data = response.json()
                if data.get("data"):
                    self.model_name = data["data"][0]["id"]
                    logger.info(f"Connected to model: {self.model_name}")

            # Check capabilities
            response = await self.client.get(f"{self.server_url}/v1/capabilities")
            if response.status_code == 200:
                caps = response.json()
                if not caps.get("attention_tokens", {}).get("supported"):
                    logger.warning("Attention tokens not supported by server")
                    return False

            return True
        except Exception as e:
            logger.error(f"Failed to connect: {e}")
            return False

    async def close(self):
        """Close the HTTP client."""
        if self.client:
            await self.client.aclose()

    async def run_scenario(self, scenario: Scenario) -> TraceData:
        """Run a single scenario and collect attention data."""
        if not self.client:
            raise RuntimeError("Not connected. Call connect() first.")

        trace = TraceData(
            scenario_name=scenario.name,
            scenario_category=scenario.category,
            expected_manifold=scenario.expected_manifold.value,
            prompt=json.dumps(scenario.messages),
            completion="",
            prompt_tokens=0,
            completion_tokens=0,
            total_tokens=0,
            start_time=datetime.now().isoformat(),
        )

        start_time = time.perf_counter()
        first_token_time = None

        try:
            # Build request
            request_body = {
                "model": self.model_name or "default",
                "messages": scenario.messages,
                "max_tokens": scenario.max_tokens,
                "temperature": scenario.temperature,
                "stream": True,
                "return_attention_tokens": True,
                "attention_top_k": self.attention_top_k,
                "include_prompt_attention": self.include_prompt_attention,
            }

            # Stream response
            completion_text = ""
            step_index = 0

            async with self.client.stream(
                "POST",
                f"{self.server_url}/v1/chat/completions",
                json=request_body,
            ) as response:
                async for line in response.aiter_lines():
                    if not line.strip() or not line.startswith("data: "):
                        continue

                    if line == "data: [DONE]":
                        break

                    try:
                        chunk = json.loads(line[6:])  # Skip "data: "
                    except json.JSONDecodeError:
                        continue

                    if first_token_time is None:
                        first_token_time = time.perf_counter()

                    # Extract content
                    delta = chunk.get("choices", [{}])[0].get("delta", {})
                    content = delta.get("content") or ""  # Handle None values
                    completion_text += content

                    # Extract attention tokens - in streaming mode they're in delta
                    attention_data = delta.get("attention_tokens") or chunk.get(
                        "attention_tokens"
                    )
                    if attention_data and isinstance(attention_data, list):
                        for step_data in attention_data:
                            step = self._parse_attention_step(step_data, step_index)
                            if step:
                                step.compute_mass_distribution()
                                trace.attention_steps.append(step)

                                # Compute fingerprint for each step
                                fp = self._compute_fingerprint(step)
                                if fp:
                                    trace.fingerprints.append(fp)
                            step_index += 1

                    # Extract MoE routing if present - check delta first for streaming
                    moe_data = delta.get("moe_routing") or chunk.get("moe_routing")
                    if moe_data:
                        for routing in moe_data:
                            moe = self._parse_moe_routing(routing, step_index)
                            if moe:
                                trace.moe_routing.append(moe)

                    # Store usage data
                    usage = chunk.get("usage")
                    if usage:
                        trace.prompt_tokens = usage.get("prompt_tokens", 0)
                        trace.completion_tokens = usage.get("completion_tokens", 0)
                        trace.total_tokens = usage.get("total_tokens", 0)

            end_time = time.perf_counter()

            # Compute timing
            trace.completion = completion_text
            trace.end_time = datetime.now().isoformat()
            trace.total_generation_time_ms = (end_time - start_time) * 1000

            if first_token_time:
                trace.time_to_first_token_ms = (first_token_time - start_time) * 1000

            # Compute tokens per second - use completion_tokens if available,
            # otherwise estimate from attention steps
            token_count = (
                trace.completion_tokens
                if trace.completion_tokens > 0
                else len(trace.attention_steps)
            )
            if token_count > 0 and trace.total_generation_time_ms > 0:
                trace.tokens_per_second = token_count / (
                    trace.total_generation_time_ms / 1000
                )

            logger.info(
                f"Completed {scenario.name}: {trace.completion_tokens} tokens, "
                f"{len(trace.attention_steps)} attention steps"
            )

        except Exception as e:
            logger.error(f"Error running scenario {scenario.name}: {e}")
            trace.end_time = datetime.now().isoformat()
            raise

        return trace

    def _parse_attention_step(
        self, data: Dict[str, Any], step_index: int
    ) -> Optional[AttentionStep]:
        """Parse attention step from response data.

        Handles both old format (top_k list) and new format (token_positions + attention_scores arrays).
        """
        try:
            top_k = []

            # Try new format first: token_positions + attention_scores arrays
            token_positions = data.get("token_positions", [])
            attention_scores = data.get("attention_scores", [])

            if token_positions and attention_scores:
                # New format from server - build AttentionToken list from parallel arrays
                for i, (pos, score) in enumerate(
                    zip(token_positions, attention_scores)
                ):
                    # In new format, position is the source position, offset is distance from current
                    # We use position directly and compute offset later if needed
                    top_k.append(
                        AttentionToken(
                            token_id=0,  # Not available in new format
                            token_str="",  # Not available in new format
                            offset=pos,  # Position from start (can compute relative offset if needed)
                            weight=score,
                            position=pos,
                        )
                    )
            else:
                # Old format: top_k list with full token data
                tokens_data = data.get("top_k", [])
                for t in tokens_data:
                    top_k.append(
                        AttentionToken(
                            token_id=t.get("token_id", 0),
                            token_str=t.get("token", ""),
                            offset=t.get("offset", 0),
                            weight=t.get("weight", 0.0),
                            position=t.get("position", 0),
                        )
                    )

            # Get fingerprint data if available (new format)
            fingerprint = data.get("fingerprint", {})
            entropy = fingerprint.get("entropy") if fingerprint else data.get("entropy")
            local_mass = fingerprint.get("local_mass") if fingerprint else None
            mid_mass = fingerprint.get("mid_mass") if fingerprint else None
            long_mass = fingerprint.get("long_mass") if fingerprint else None

            # Use decode_step from data if available, otherwise use passed step_index
            actual_step_index = data.get("decode_step", step_index)

            return AttentionStep(
                step_index=actual_step_index,
                output_token_id=data.get("output_token_id", 0),
                output_token_str=data.get("output_token", ""),
                top_k_tokens=top_k,
                entropy=entropy,
                local_mass=local_mass,
                mid_mass=mid_mass,
                long_mass=long_mass,
            )
        except Exception as e:
            logger.warning(f"Failed to parse attention step: {e}")
            return None

    def _parse_moe_routing(
        self, data: Dict[str, Any], step_index: int
    ) -> Optional[MoERouting]:
        """Parse MoE routing from response data."""
        try:
            return MoERouting(
                step_index=step_index,
                layer_idx=data.get("layer", 0),
                expert_ids=data.get("expert_ids", []),
                expert_weights=data.get("expert_weights", []),
                token_position=data.get("position", 0),
            )
        except Exception as e:
            logger.warning(f"Failed to parse MoE routing: {e}")
            return None

    def _compute_fingerprint(self, step: AttentionStep) -> Optional[Fingerprint]:
        """Compute fingerprint from attention step."""
        if not step.top_k_tokens:
            return None

        # Get mass distribution
        local_mass = step.local_mass or 0.0
        mid_mass = step.mid_mass or 0.0
        long_mass = step.long_mass or 0.0

        # Compute entropy
        weights = [t.weight for t in step.top_k_tokens if t.weight > 0]
        if weights:
            total = sum(weights)
            probs = [w / total for w in weights]
            import math

            entropy = -sum(p * math.log2(p) for p in probs if p > 0)
            # Normalize to [0, 1] range (assuming max entropy = log2(top_k))
            max_entropy = math.log2(len(weights)) if len(weights) > 1 else 1
            entropy = entropy / max_entropy if max_entropy > 0 else 0
        else:
            entropy = 0.0

        # Compute histogram (16 bins, exponential spacing)
        # Bins: [0-1], [2-3], [4-7], [8-15], ..., [2^14-2^15]
        histogram = [0.0] * 16
        total_weight = sum(t.weight for t in step.top_k_tokens)

        if total_weight > 0:
            for token in step.top_k_tokens:
                offset = token.offset
                # Find bin using log2
                if offset == 0:
                    bin_idx = 0
                else:
                    import math

                    bin_idx = min(15, int(math.log2(offset + 1)))
                histogram[bin_idx] += token.weight / total_weight

        return Fingerprint(
            local_mass=local_mass,
            mid_mass=mid_mass,
            long_mass=long_mass,
            entropy=entropy,
            histogram=histogram,
        )


class CollectionRunner:
    """
    Manages complete collection runs across multiple scenarios.
    """

    def __init__(
        self,
        collector: AttentionCollector,
        output_dir: Path,
    ):
        self.collector = collector
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    async def run_scenarios(
        self,
        scenarios: List[Scenario],
        run_id: Optional[str] = None,
        progress_callback: Optional[callable] = None,
    ) -> CollectionRun:
        """Run multiple scenarios and collect data."""
        if run_id is None:
            run_id = datetime.now().strftime("%Y%m%d_%H%M%S")

        run = CollectionRun(
            run_id=run_id,
            start_time=datetime.now().isoformat(),
            model_name=self.collector.model_name or "",
            server_url=self.collector.server_url,
        )

        total = len(scenarios)
        for i, scenario in enumerate(scenarios):
            try:
                logger.info(f"Running scenario {i+1}/{total}: {scenario.name}")
                trace = await self.collector.run_scenario(scenario)
                run.add_trace(trace)

                if progress_callback:
                    progress_callback(i + 1, total, scenario.name, trace)

            except Exception as e:
                logger.error(f"Failed scenario {scenario.name}: {e}")
                run.add_error(scenario.name, str(e))

        run.end_time = datetime.now().isoformat()

        # Save results
        self._save_run(run)

        return run

    def _save_run(self, run: CollectionRun):
        """Save run data to disk."""
        output_file = self.output_dir / f"run_{run.run_id}.json"

        # Convert to serializable format
        data = {
            "run_id": run.run_id,
            "start_time": run.start_time,
            "end_time": run.end_time,
            "model_name": run.model_name,
            "server_url": run.server_url,
            "summary": run.summary(),
            "traces": [asdict(t) for t in run.traces],
            "errors": run.errors,
        }

        with open(output_file, "w") as f:
            json.dump(data, f, indent=2)

        logger.info(f"Saved run data to {output_file}")


def load_run(path: Path) -> CollectionRun:
    """Load a collection run from disk."""
    with open(path) as f:
        data = json.load(f)

    run = CollectionRun(
        run_id=data["run_id"],
        start_time=data["start_time"],
        end_time=data.get("end_time"),
        model_name=data.get("model_name", ""),
        server_url=data.get("server_url", ""),
        errors=data.get("errors", []),
    )

    for trace_data in data.get("traces", []):
        trace = TraceData(
            scenario_name=trace_data["scenario_name"],
            scenario_category=trace_data["scenario_category"],
            expected_manifold=trace_data["expected_manifold"],
            prompt=trace_data["prompt"],
            completion=trace_data["completion"],
            prompt_tokens=trace_data["prompt_tokens"],
            completion_tokens=trace_data["completion_tokens"],
            total_tokens=trace_data["total_tokens"],
            time_to_first_token_ms=trace_data.get("time_to_first_token_ms"),
            total_generation_time_ms=trace_data.get("total_generation_time_ms"),
            tokens_per_second=trace_data.get("tokens_per_second"),
            start_time=trace_data.get("start_time", ""),
            end_time=trace_data.get("end_time", ""),
        )

        # Parse attention steps
        for step_data in trace_data.get("attention_steps", []):
            step = AttentionStep(
                step_index=step_data["step_index"],
                output_token_id=step_data["output_token_id"],
                output_token_str=step_data["output_token_str"],
                top_k_tokens=[
                    AttentionToken(**t) for t in step_data.get("top_k_tokens", [])
                ],
                entropy=step_data.get("entropy"),
                local_mass=step_data.get("local_mass"),
                mid_mass=step_data.get("mid_mass"),
                long_mass=step_data.get("long_mass"),
            )
            trace.attention_steps.append(step)

        # Parse fingerprints
        for fp_data in trace_data.get("fingerprints", []):
            fp = Fingerprint(
                local_mass=fp_data["local_mass"],
                mid_mass=fp_data["mid_mass"],
                long_mass=fp_data["long_mass"],
                entropy=fp_data["entropy"],
                histogram=fp_data["histogram"],
            )
            trace.fingerprints.append(fp)

        # Parse MoE routing
        for moe_data in trace_data.get("moe_routing", []):
            moe = MoERouting(**moe_data)
            trace.moe_routing.append(moe)

        run.add_trace(trace)

    return run
