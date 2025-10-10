"""
Profile merger for SGLang traces.

Merges Chrome trace files from multiple ranks (TP, DP, PP, EP) into a single
consolidated trace file. Based on torch_utils/sglang_profiler_trace_merger.py
but extended to support all parallelism types.
"""

import glob
import gzip
import json
import logging
import os
import re
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class ProfileMerger:
    """Merges SGLang profile traces from different ranks into a single consolidated trace.

    Extends torch_utils/sglang_profiler_trace_merger.py to support all parallelism types:
    - TP (Tensor Parallel)
    - DP (Data Parallel)
    - PP (Pipeline Parallel)
    - EP (Expert Parallel)
    """

    def __init__(self, output_dir: str, profile_id: str):
        """Initialize ProfileMerger.

        Args:
            output_dir: Directory containing trace files
            profile_id: Profile ID to match trace files
        """
        self.output_dir = output_dir
        self.profile_id = profile_id
        self.merged_trace_path = os.path.join(
            output_dir, f"merged-{profile_id}.trace.json.gz"
        )

    def merge_chrome_traces(self) -> str:
        """Merge Chrome traces from all ranks into a single trace.

        Returns:
            Path to the merged trace file

        Raises:
            ValueError: If no trace files found
        """
        trace_files = self._discover_trace_files()
        if not trace_files:
            raise ValueError(f"No trace files found for profile_id: {self.profile_id}")

        logger.info(f"Found {len(trace_files)} trace files to merge")

        merged_trace = {"traceEvents": []}

        # Track deviceProperties from all files
        all_device_properties = []

        for trace_file in sorted(trace_files, key=self._get_rank_sort_key):
            rank_info = self._extract_rank_info(trace_file)
            logger.info(f"Processing {trace_file} with rank info: {rank_info}")

            output = self._handle_file(trace_file, rank_info)

            # Merge traceEvents
            merged_trace["traceEvents"].extend(output["traceEvents"])

            # Collect deviceProperties from ALL files (not just first)
            if "deviceProperties" in output:
                all_device_properties.extend(output["deviceProperties"])
                del output["deviceProperties"]  # Don't add to merged_trace yet

            # Preserve all other keys (first occurrence wins)
            for key, value in output.items():
                if key != "traceEvents" and key not in merged_trace:
                    merged_trace[key] = value

        # Add collected deviceProperties from all ranks
        if all_device_properties:
            merged_trace["deviceProperties"] = all_device_properties

        # Save merged trace
        with gzip.open(self.merged_trace_path, "wb") as f:
            f.write(json.dumps(merged_trace).encode("utf-8"))

        logger.info(f"Merged profile saved to: {self.merged_trace_path}")
        logger.info(f"Total events merged: {len(merged_trace['traceEvents'])}")

        return self.merged_trace_path

    def _discover_trace_files(self) -> List[str]:
        """Discover all trace files for this profile_id.

        Supports both old format (TP only) and new format (TP/DP/PP/EP).

        Returns:
            List of trace file paths
        """

        # Search for all possible patterns
        patterns = [
            f"{self.profile_id}*.trace.json.gz",  # Catch all files with profile_id
        ]

        trace_files = []
        for pattern in patterns:
            search_pattern = os.path.join(self.output_dir, pattern)
            trace_files.extend(glob.glob(search_pattern))

        # Filter out merged files and other non-trace files
        trace_files = [
            f
            for f in trace_files
            if not f.endswith(f"merged-{self.profile_id}.trace.json.gz")
            and not f.endswith("-memory.pickle")
            and "TP-" in f  # Must have at least TP rank
        ]

        # Remove duplicates (sorting will be done in merge_chrome_traces with rank-based key)
        trace_files = list(set(trace_files))
        return trace_files

    def _extract_rank_info(self, filename: str) -> Dict[str, int]:
        """Extract ALL rank types from filename.

        EXTENDS torch_utils which only extracts TP rank.

        Args:
            filename: Trace file path

        Returns:
            Dictionary with rank information
        """
        basename = os.path.basename(filename)
        rank_info = {}

        # Extract TP rank
        tp_match = re.search(r"TP-(\d+)", basename)
        if tp_match:
            rank_info["tp_rank"] = int(tp_match.group(1))

        # Extract DP rank
        dp_match = re.search(r"DP-(\d+)", basename)
        if dp_match:
            rank_info["dp_rank"] = int(dp_match.group(1))

        # Extract PP rank
        pp_match = re.search(r"PP-(\d+)", basename)
        if pp_match:
            rank_info["pp_rank"] = int(pp_match.group(1))

        # Extract EP rank
        ep_match = re.search(r"EP-(\d+)", basename)
        if ep_match:
            rank_info["ep_rank"] = int(ep_match.group(1))

        return rank_info

    def _create_rank_label(self, rank_info: Dict[str, int]) -> str:
        """Create rank label for ALL parallelism types.

        EXTENDS torch_utils which formats as [TP{rank:02d}].
        SGLang formats as [TP{tp:02d}-DP{dp:02d}-PP{pp:02d}-EP{ep:02d}].

        Args:
            rank_info: Dictionary with rank information

        Returns:
            Formatted rank label
        """
        parts = []
        if "tp_rank" in rank_info:
            parts.append(f"TP{rank_info['tp_rank']:02d}")
        if "dp_rank" in rank_info:
            parts.append(f"DP{rank_info['dp_rank']:02d}")
        if "pp_rank" in rank_info:
            parts.append(f"PP{rank_info['pp_rank']:02d}")
        if "ep_rank" in rank_info:
            parts.append(f"EP{rank_info['ep_rank']:02d}")

        return f"[{'-'.join(parts)}]" if parts else "[Unknown]"

    def _handle_file(self, path: str, rank_info: Dict[str, int]) -> Dict[str, Any]:
        """Handle a single trace file.

        Args:
            path: Path to trace file
            rank_info: Rank information for this file

        Returns:
            Processed trace data
        """
        logger.info(f"Processing file: {path}")

        try:
            with gzip.open(path, "rt", encoding="utf-8") as f:
                trace = json.load(f)

            # Process events
            output = {
                key: value for key, value in trace.items() if key != "traceEvents"
            }
            output["traceEvents"] = self._process_events(
                trace.get("traceEvents", []), rank_info
            )

            return output

        except Exception as e:
            logger.error(f"Failed to process trace file {path}: {e}")
            # Return empty structure to continue with other files
            return {"traceEvents": []}

    def _process_events(
        self, events: List[Dict], rank_info: Dict[str, int]
    ) -> List[Dict]:
        """Process events following torch_utils logic.

        Args:
            events: List of trace events
            rank_info: Rank information

        Returns:
            Processed events
        """
        rank_label = self._create_rank_label(rank_info)

        for event in events:
            # Update process_sort_index for proper ordering
            if event.get("name") == "process_sort_index":
                pid = self._maybe_cast_int(event.get("pid"))
                if pid is not None and pid < 1000:
                    # Calculate sort index considering all rank dimensions
                    event["args"]["sort_index"] = self._calculate_sort_index(
                        rank_info, pid
                    )

            # Modify pid to include rank label (following torch_utils pattern)
            event["pid"] = f"{rank_label} {event['pid']}"

        return events

    def _calculate_sort_index(self, rank_info: Dict[str, int], pid: int) -> int:
        """Calculate sort index from all rank dimensions.

        EXTENDS torch_utils which only uses TP rank.
        Formula: (((TP * dp_size + DP) * pp_size + PP) * ep_size + EP) * 100 + pid

        Args:
            rank_info: Rank information
            pid: Process ID

        Returns:
            Calculated sort index
        """
        # Use simple multipliers for now (can be made configurable)
        sort_idx = 0
        multiplier = 1

        # Build sort index from least to most significant rank
        if "ep_rank" in rank_info:
            sort_idx += rank_info["ep_rank"] * multiplier
            multiplier *= 100  # Assume max 100 EP ranks

        if "pp_rank" in rank_info:
            sort_idx += rank_info["pp_rank"] * multiplier
            multiplier *= 100

        if "dp_rank" in rank_info:
            sort_idx += rank_info["dp_rank"] * multiplier
            multiplier *= 100

        if "tp_rank" in rank_info:
            sort_idx += rank_info["tp_rank"] * multiplier
            multiplier *= 100

        # Add pid for fine-grained ordering
        sort_idx = sort_idx * 100 + pid

        return sort_idx

    def _get_rank_sort_key(self, path: str) -> Tuple[int, int, int, int]:
        """Get sort key for ranking files.

        EXTENDS torch_utils which only sorts by TP rank.

        Args:
            path: Path to trace file

        Returns:
            Tuple for sorting (TP, DP, PP, EP)
        """
        rank_info = self._extract_rank_info(path)
        return (
            rank_info.get("tp_rank", 0),
            rank_info.get("dp_rank", 0),
            rank_info.get("pp_rank", 0),
            rank_info.get("ep_rank", 0),
        )

    def _maybe_cast_int(self, x) -> Optional[int]:
        """Safely cast to int.

        Args:
            x: Value to cast

        Returns:
            Integer value or None if cast fails
        """
        try:
            return int(x)
        except (ValueError, TypeError):
            return None

    def get_merge_summary(self) -> Dict[str, Any]:
        """Get summary information about the merge operation.

        Returns:
            Dictionary with merge statistics
        """
        if not os.path.exists(self.merged_trace_path):
            return {"error": "Merged trace file not found"}

        try:
            with gzip.open(self.merged_trace_path, "rt") as f:
                merged_data = json.load(f)

            # Count source files
            trace_files = self._discover_trace_files()

            return {
                "merged_file": self.merged_trace_path,
                "total_events": len(merged_data.get("traceEvents", [])),
                "total_files": len(trace_files),
                "source_files": [os.path.basename(f) for f in trace_files],
                "profile_id": self.profile_id,
                "device_properties_count": len(merged_data.get("deviceProperties", [])),
            }
        except Exception as e:
            return {"error": f"Failed to read merged trace: {str(e)}"}
