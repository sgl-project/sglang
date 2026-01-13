"""
Experiment Manifest - Reproducibility Layer for Attention Explorer

Captures all metadata needed to reproduce an evaluation or discovery run:
- Git SHAs (server, sidecar, UI components)
- Model ID and revision
- Quantization configuration
- Decoding parameters (temperature, top_p, seed)
- Probe-pack mix and seed
- Hardware information (GPU, CUDA, memory)
- Timestamps (start, end, duration)
- Artifact paths (database, exports, plots)

Usage:
    manifest = ExperimentManifest.create(
        run_type="discovery",
        run_id="2024-01-08T15-23-45",
        model_id="Qwen/Qwen3-8B",
    )
    manifest.set_decoding(temperature=0.7, top_p=0.9, seed=42)
    manifest.add_artifact("embeddings", "/path/to/embeddings.parquet")
    manifest.finalize()
    manifest.save("/path/to/manifest.json")
"""

import json
import logging
import os
import platform
import subprocess
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

logger = logging.getLogger(__name__)


# =============================================================================
# Enums
# =============================================================================


class RunType(str, Enum):
    DISCOVERY = "discovery"
    COMPARISON = "comparison"
    EVALUATION = "evaluation"
    E2E_TEST = "e2e_test"
    PROBE_HARNESS = "probe_harness"


class ArtifactType(str, Enum):
    DATABASE = "database"
    PARQUET = "parquet"
    JSON = "json"
    JOBLIB = "joblib"
    PLOT = "plot"
    LOG = "log"
    REPORT = "report"


# =============================================================================
# Git Utilities
# =============================================================================


def get_git_sha(repo_path: Optional[str] = None) -> Optional[str]:
    """
    Get the current git SHA for a repository.

    Args:
        repo_path: Path to the repository. If None, uses current directory.

    Returns:
        Short git SHA (7 chars) or None if not a git repo.
    """
    try:
        cmd = ["git", "rev-parse", "--short=7", "HEAD"]
        result = subprocess.run(
            cmd,
            cwd=repo_path,
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
        pass
    return None


def get_git_branch(repo_path: Optional[str] = None) -> Optional[str]:
    """Get the current git branch name."""
    try:
        cmd = ["git", "rev-parse", "--abbrev-ref", "HEAD"]
        result = subprocess.run(
            cmd,
            cwd=repo_path,
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
        pass
    return None


def get_git_dirty(repo_path: Optional[str] = None) -> bool:
    """Check if the git repository has uncommitted changes."""
    try:
        cmd = ["git", "status", "--porcelain"]
        result = subprocess.run(
            cmd,
            cwd=repo_path,
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            return len(result.stdout.strip()) > 0
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
        pass
    return False


def get_git_info(repo_path: Optional[str] = None) -> Dict[str, Any]:
    """Get comprehensive git information for a repository."""
    return {
        "sha": get_git_sha(repo_path),
        "branch": get_git_branch(repo_path),
        "dirty": get_git_dirty(repo_path),
    }


# =============================================================================
# Hardware Utilities
# =============================================================================


def get_gpu_info() -> List[Dict[str, Any]]:
    """
    Get GPU information using nvidia-smi.

    Returns:
        List of GPU info dicts with name, memory, driver version.
    """
    gpus = []
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,name,memory.total,driver_version,cuda_version",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode == 0:
            for line in result.stdout.strip().split("\n"):
                if line.strip():
                    parts = [p.strip() for p in line.split(",")]
                    if len(parts) >= 5:
                        gpus.append(
                            {
                                "index": int(parts[0]),
                                "name": parts[1],
                                "memory_mb": int(float(parts[2])),
                                "driver_version": parts[3],
                                "cuda_version": parts[4],
                            }
                        )
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError, ValueError):
        pass
    return gpus


def get_hardware_info() -> Dict[str, Any]:
    """
    Get comprehensive hardware information.

    Returns:
        Dict with CPU, memory, GPU, and system info.
    """

    # CPU info
    cpu_info = {
        "processor": platform.processor() or platform.machine(),
        "physical_cores": os.cpu_count(),
    }

    # Memory info
    memory_info = {}
    try:
        with open("/proc/meminfo", "r") as f:
            for line in f:
                if line.startswith("MemTotal:"):
                    # Convert KB to GB
                    kb = int(line.split()[1])
                    memory_info["total_gb"] = round(kb / 1024 / 1024, 1)
                    break
    except (FileNotFoundError, OSError, ValueError):
        pass

    # GPU info
    gpus = get_gpu_info()

    # System info
    system_info = {
        "platform": platform.system(),
        "platform_release": platform.release(),
        "python_version": platform.python_version(),
    }

    return {
        "cpu": cpu_info,
        "memory": memory_info,
        "gpus": gpus,
        "gpu_count": len(gpus),
        "system": system_info,
    }


# =============================================================================
# Dataclasses
# =============================================================================


@dataclass
class GitContext:
    """Git repository context."""

    sha: Optional[str] = None
    branch: Optional[str] = None
    dirty: bool = False

    @classmethod
    def from_path(cls, repo_path: Optional[str] = None) -> "GitContext":
        info = get_git_info(repo_path)
        return cls(**info)


@dataclass
class ModelConfig:
    """Model configuration."""

    model_id: str
    revision: Optional[str] = None
    dtype: Optional[str] = None
    tensor_parallel: int = 1
    pipeline_parallel: int = 1
    attention_backend: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {k: v for k, v in asdict(self).items() if v is not None}


@dataclass
class QuantizationConfig:
    """Quantization configuration."""

    method: str  # none, sinq, asinq, awq, gptq, fp8
    nbits: Optional[int] = None
    group_size: Optional[int] = None
    tiling_mode: Optional[str] = None  # 1D, 2D
    symmetric: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return {k: v for k, v in asdict(self).items() if v is not None}


@dataclass
class DecodingConfig:
    """Decoding/sampling configuration."""

    temperature: float = 1.0
    top_p: float = 1.0
    top_k: int = -1
    seed: Optional[int] = None
    max_tokens: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        return {k: v for k, v in asdict(self).items() if v is not None}


@dataclass
class ProbeConfig:
    """Probe harness configuration."""

    packs: List[str] = field(default_factory=list)  # json_repair, coreference, etc.
    mix_weights: Optional[Dict[str, float]] = None
    duration_minutes: Optional[int] = None
    seed: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        result = {"packs": self.packs}
        if self.mix_weights:
            result["mix_weights"] = self.mix_weights
        if self.duration_minutes:
            result["duration_minutes"] = self.duration_minutes
        if self.seed is not None:
            result["seed"] = self.seed
        return result


@dataclass
class Artifact:
    """Reference to an output artifact."""

    name: str
    path: str
    artifact_type: str
    size_bytes: Optional[int] = None
    checksum: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {k: v for k, v in asdict(self).items() if v is not None}


@dataclass
class ExperimentManifest:
    """
    Complete experiment manifest for reproducibility.

    Captures all metadata needed to reproduce an evaluation or discovery run.
    """

    # Core identification
    schema_version: str = "1.0.0"
    run_type: str = "discovery"
    run_id: str = ""

    # Timing
    created_at: str = ""
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    duration_seconds: Optional[float] = None

    # Git context
    git: Dict[str, Any] = field(default_factory=dict)

    # Model configuration
    model: Optional[Dict[str, Any]] = None

    # Quantization (for comparison runs)
    quantization: Optional[Dict[str, Any]] = None

    # Decoding parameters
    decoding: Optional[Dict[str, Any]] = None

    # Probe configuration (for harness runs)
    probes: Optional[Dict[str, Any]] = None

    # Hardware
    hardware: Dict[str, Any] = field(default_factory=dict)

    # Run-specific parameters
    parameters: Dict[str, Any] = field(default_factory=dict)

    # Artifacts produced
    artifacts: List[Dict[str, Any]] = field(default_factory=list)

    # Data sources
    sources: Dict[str, Any] = field(default_factory=dict)

    # Summary statistics
    statistics: Dict[str, Any] = field(default_factory=dict)

    # Free-form metadata
    metadata: Dict[str, Any] = field(default_factory=dict)

    # Internal timing
    _start_time: float = field(default=0.0, repr=False)

    @classmethod
    def create(
        cls,
        run_type: Union[str, RunType],
        run_id: Optional[str] = None,
        model_id: Optional[str] = None,
        capture_git: bool = True,
        capture_hardware: bool = True,
    ) -> "ExperimentManifest":
        """
        Create a new experiment manifest.

        Args:
            run_type: Type of run (discovery, comparison, evaluation, etc.)
            run_id: Unique identifier for this run. Defaults to ISO timestamp.
            model_id: Model identifier (e.g., "Qwen/Qwen3-8B")
            capture_git: Whether to capture git SHA automatically
            capture_hardware: Whether to capture hardware info

        Returns:
            Initialized ExperimentManifest
        """
        if isinstance(run_type, RunType):
            run_type = run_type.value

        if run_id is None:
            run_id = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%S")

        manifest = cls(
            run_type=run_type,
            run_id=run_id,
            created_at=datetime.now(timezone.utc).isoformat(),
            _start_time=time.time(),
        )

        if capture_git:
            manifest.git = {
                "server": get_git_info(),  # Main sglang repo
            }

        if capture_hardware:
            manifest.hardware = get_hardware_info()

        if model_id:
            manifest.model = {"model_id": model_id}

        manifest.started_at = datetime.now(timezone.utc).isoformat()

        return manifest

    def set_model(
        self,
        model_id: str,
        revision: Optional[str] = None,
        dtype: Optional[str] = None,
        tensor_parallel: int = 1,
        pipeline_parallel: int = 1,
        attention_backend: Optional[str] = None,
    ) -> "ExperimentManifest":
        """Set model configuration."""
        config = ModelConfig(
            model_id=model_id,
            revision=revision,
            dtype=dtype,
            tensor_parallel=tensor_parallel,
            pipeline_parallel=pipeline_parallel,
            attention_backend=attention_backend,
        )
        self.model = config.to_dict()
        return self

    def set_quantization(
        self,
        method: str,
        nbits: Optional[int] = None,
        group_size: Optional[int] = None,
        tiling_mode: Optional[str] = None,
        symmetric: bool = True,
    ) -> "ExperimentManifest":
        """Set quantization configuration."""
        config = QuantizationConfig(
            method=method,
            nbits=nbits,
            group_size=group_size,
            tiling_mode=tiling_mode,
            symmetric=symmetric,
        )
        self.quantization = config.to_dict()
        return self

    def set_decoding(
        self,
        temperature: float = 1.0,
        top_p: float = 1.0,
        top_k: int = -1,
        seed: Optional[int] = None,
        max_tokens: Optional[int] = None,
    ) -> "ExperimentManifest":
        """Set decoding/sampling parameters."""
        config = DecodingConfig(
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            seed=seed,
            max_tokens=max_tokens,
        )
        self.decoding = config.to_dict()
        return self

    def set_probes(
        self,
        packs: List[str],
        mix_weights: Optional[Dict[str, float]] = None,
        duration_minutes: Optional[int] = None,
        seed: Optional[int] = None,
    ) -> "ExperimentManifest":
        """Set probe harness configuration."""
        config = ProbeConfig(
            packs=packs,
            mix_weights=mix_weights,
            duration_minutes=duration_minutes,
            seed=seed,
        )
        self.probes = config.to_dict()
        return self

    def set_source(
        self,
        database_path: Optional[str] = None,
        time_window_hours: Optional[int] = None,
        fingerprint_count: Optional[int] = None,
        request_count: Optional[int] = None,
    ) -> "ExperimentManifest":
        """Set data source information."""
        if database_path:
            self.sources["database_path"] = database_path
        if time_window_hours:
            self.sources["time_window_hours"] = time_window_hours
        if fingerprint_count:
            self.sources["fingerprint_count"] = fingerprint_count
        if request_count:
            self.sources["request_count"] = request_count
        return self

    def add_parameter(self, key: str, value: Any) -> "ExperimentManifest":
        """Add a run-specific parameter."""
        self.parameters[key] = value
        return self

    def set_parameters(self, params: Dict[str, Any]) -> "ExperimentManifest":
        """Set multiple run-specific parameters."""
        self.parameters.update(params)
        return self

    def add_artifact(
        self,
        name: str,
        path: str,
        artifact_type: Optional[Union[str, ArtifactType]] = None,
        compute_size: bool = True,
    ) -> "ExperimentManifest":
        """
        Add an output artifact reference.

        Args:
            name: Descriptive name (e.g., "embeddings", "clusters")
            path: File path (absolute or relative to run directory)
            artifact_type: Type of artifact. Auto-detected if None.
            compute_size: Whether to compute file size
        """
        if artifact_type is None:
            # Auto-detect from extension
            ext = Path(path).suffix.lower()
            type_map = {
                ".parquet": "parquet",
                ".json": "json",
                ".joblib": "joblib",
                ".db": "database",
                ".sqlite": "database",
                ".png": "plot",
                ".svg": "plot",
                ".log": "log",
                ".md": "report",
                ".csv": "report",
            }
            artifact_type = type_map.get(ext, "unknown")
        elif isinstance(artifact_type, ArtifactType):
            artifact_type = artifact_type.value

        artifact = Artifact(
            name=name,
            path=path,
            artifact_type=artifact_type,
        )

        if compute_size and os.path.exists(path):
            try:
                artifact.size_bytes = os.path.getsize(path)
            except OSError:
                pass

        self.artifacts.append(artifact.to_dict())
        return self

    def set_statistics(self, stats: Dict[str, Any]) -> "ExperimentManifest":
        """Set summary statistics."""
        self.statistics.update(stats)
        return self

    def add_metadata(self, key: str, value: Any) -> "ExperimentManifest":
        """Add free-form metadata."""
        self.metadata[key] = value
        return self

    def finalize(self) -> "ExperimentManifest":
        """
        Finalize the manifest by setting completion time and duration.

        Call this when the run is complete.
        """
        self.completed_at = datetime.now(timezone.utc).isoformat()
        if self._start_time > 0:
            self.duration_seconds = round(time.time() - self._start_time, 2)
        return self

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary, excluding internal fields."""
        result = {}
        for key, value in asdict(self).items():
            if key.startswith("_"):
                continue
            if value is None:
                continue
            if isinstance(value, dict) and not value:
                continue
            if isinstance(value, list) and not value:
                continue
            result[key] = value
        return result

    def save(self, path: str) -> None:
        """Save manifest to JSON file."""
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2, default=str)
        logger.info(f"Saved manifest to {path}")

    @classmethod
    def load(cls, path: str) -> "ExperimentManifest":
        """Load manifest from JSON file."""
        with open(path, "r") as f:
            data = json.load(f)

        manifest = cls()
        for key, value in data.items():
            if hasattr(manifest, key):
                setattr(manifest, key, value)

        return manifest


# =============================================================================
# Convenience Functions
# =============================================================================


def create_discovery_manifest(
    run_id: str,
    db_path: str,
    config: Any,  # DiscoveryConfig
    output_dir: str,
    model_id: Optional[str] = None,
) -> ExperimentManifest:
    """
    Create a manifest for a discovery job.

    Args:
        run_id: Unique run identifier
        db_path: Path to fingerprint database
        config: DiscoveryConfig object
        output_dir: Output directory for artifacts
        model_id: Optional model identifier
    """
    manifest = ExperimentManifest.create(
        run_type=RunType.DISCOVERY,
        run_id=run_id,
        model_id=model_id,
    )

    manifest.set_source(
        database_path=db_path,
        time_window_hours=getattr(config, "time_window_hours", None),
    )

    # Add discovery parameters
    if hasattr(config, "__dict__"):
        manifest.set_parameters(
            {
                k: v
                for k, v in config.__dict__.items()
                if not k.startswith("_")
                and isinstance(v, (int, float, str, bool, list))
            }
        )

    return manifest


def create_comparison_manifest(
    comparison_id: str,
    baseline_model: str,
    candidate_model: str,
    quantization_method: Optional[str] = None,
    nbits: Optional[int] = None,
) -> ExperimentManifest:
    """
    Create a manifest for a quantization comparison run.

    Args:
        comparison_id: Unique comparison identifier
        baseline_model: Baseline model ID
        candidate_model: Candidate model ID
        quantization_method: Quantization method (sinq, asinq, etc.)
        nbits: Bit width
    """
    manifest = ExperimentManifest.create(
        run_type=RunType.COMPARISON,
        run_id=comparison_id,
        model_id=baseline_model,
    )

    manifest.add_metadata("baseline_model", baseline_model)
    manifest.add_metadata("candidate_model", candidate_model)

    if quantization_method:
        manifest.set_quantization(
            method=quantization_method,
            nbits=nbits,
        )

    return manifest


def create_probe_harness_manifest(
    run_id: str,
    packs: List[str],
    duration_minutes: int,
    model_id: Optional[str] = None,
    seed: Optional[int] = None,
) -> ExperimentManifest:
    """
    Create a manifest for a probe harness run.

    Args:
        run_id: Unique run identifier
        packs: List of probe pack names
        duration_minutes: Run duration
        model_id: Model being probed
        seed: Random seed for reproducibility
    """
    manifest = ExperimentManifest.create(
        run_type=RunType.PROBE_HARNESS,
        run_id=run_id,
        model_id=model_id,
    )

    manifest.set_probes(
        packs=packs,
        duration_minutes=duration_minutes,
        seed=seed,
    )

    return manifest
