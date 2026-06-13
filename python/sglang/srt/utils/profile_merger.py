"""Merge Chrome trace files from multiple ranks (TP, DP, PP, EP) into a single trace."""

import glob
import gzip
import json
import logging
import os
import re
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class ProfileMerger:
    """Merge profile traces from all parallelism types: TP, DP, PP, EP."""

    def __init__(self, output_dir: str, profile_id: str):
        self.output_dir = output_dir
        self.profile_id = profile_id
        self.merged_trace_path = os.path.join(
            output_dir, f"merged-{profile_id}.trace.json.gz"
        )

        # Rank types in priority order (used for sorting and labeling)
        self.rank_types = ["tp", "dp", "pp", "ep"]

        # Sort index multipliers: DP (highest) > EP > PP > TP (lowest)
        # These ensure proper visual ordering in trace viewer
        self.sort_index_multipliers = {
            "dp_rank": 100_000_000,
            "ep_rank": 1_000_000,
            "pp_rank": 10_000,
            "tp_rank": 100,
        }

        # PID threshold for sort_index updates (only update for system PIDs < 1000)
        self.pid_sort_index_threshold = 1000

    def merge_chrome_traces(self) -> str:
        """Merge Chrome traces from all ranks into a single trace.

        Returns:
            Path to merged trace file.

        Raises:
            ValueError: If no trace files found.
        """
        trace_files = self._discover_trace_files()
        if not trace_files:
            raise ValueError(f"No trace files found for profile_id: {self.profile_id}")

        logger.info(f"Found {len(trace_files)} trace files to merge")

        merged_trace = {"traceEvents": []}
        all_device_properties = []

        for trace_file in sorted(trace_files, key=self._get_rank_sort_key):
            rank_info = self._extract_rank_info(trace_file)
            logger.info(f"Processing {trace_file} with rank info: {rank_info}")

            output = self._handle_file(trace_file, rank_info)

            merged_trace["traceEvents"].extend(output["traceEvents"])

            if "deviceProperties" in output:
                all_device_properties.extend(output["deviceProperties"])
                del output["deviceProperties"]

            for key, value in output.items():
                if key != "traceEvents" and key not in merged_trace:
                    merged_trace[key] = value

        if all_device_properties:
            merged_trace["deviceProperties"] = all_device_properties

        with gzip.open(self.merged_trace_path, "wb") as f:
            f.write(json.dumps(merged_trace).encode("utf-8"))

        logger.info(f"Merged profile saved to: {self.merged_trace_path}")
        logger.info(f"Total events merged: {len(merged_trace['traceEvents'])}")

        return self.merged_trace_path

    def _discover_trace_files(self) -> List[str]:
        """Discover trace files matching profile_id (supports TP/DP/PP/EP formats)."""
        patterns = [f"{self.profile_id}*.trace.json.gz"]

        trace_files = []
        for pattern in patterns:
            search_pattern = os.path.join(self.output_dir, pattern)
            trace_files.extend(glob.glob(search_pattern))

        trace_files = [
            f
            for f in trace_files
            if not f.endswith(f"merged-{self.profile_id}.trace.json.gz")
            and not f.endswith("-memory.pickle")
            and "TP-" in f
        ]
        trace_files = list(set(trace_files))
        return trace_files

    def _extract_rank_info(self, filename: str) -> Dict[str, int]:
        """Extract rank info (TP/DP/PP/EP) from filename."""
        basename = os.path.basename(filename)
        rank_info = {}

        for rank_type in self.rank_types:
            match = re.search(rf"{rank_type.upper()}-(\d+)", basename)
            if match:
                rank_info[f"{rank_type}_rank"] = int(match.group(1))

        return rank_info

    def _create_rank_label(self, rank_info: Dict[str, int]) -> str:
        parts = []
        for rank_type in self.rank_types:
            rank_key = f"{rank_type}_rank"
            if rank_key in rank_info:
                parts.append(f"{rank_type.upper()}{rank_info[rank_key]:02d}")

        return f"[{'-'.join(parts)}]" if parts else "[Unknown]"

    def _handle_file(self, path: str, rank_info: Dict[str, int]) -> Dict[str, Any]:
        logger.info(f"Processing file: {path}")

        try:
            with gzip.open(path, "rt", encoding="utf-8") as f:
                trace = json.load(f)

            output = {
                key: value for key, value in trace.items() if key != "traceEvents"
            }
            output["traceEvents"] = self._process_events(
                trace.get("traceEvents", []), rank_info
            )
            return output

        except Exception as e:
            logger.error(f"Failed to process trace file {path}: {e}")
            return {"traceEvents": []}

    def _process_events(
        self, events: List[Dict], rank_info: Dict[str, int]
    ) -> List[Dict]:
        """Process events: update sort_index and add rank labels to PIDs."""
        rank_label = self._create_rank_label(rank_info)

        for event in events:
            if event.get("name") == "process_sort_index":
                pid = self._maybe_cast_int(event.get("pid"))
                if pid is not None and pid < self.pid_sort_index_threshold:
                    event["args"]["sort_index"] = self._calculate_sort_index(
                        rank_info, pid
                    )

            event["pid"] = f"{rank_label} {event['pid']}"

        return events

    def _calculate_sort_index(self, rank_info: Dict[str, int], pid: int) -> int:
        sort_index = pid
        for rank_type, multiplier in self.sort_index_multipliers.items():
            sort_index += rank_info.get(rank_type, 0) * multiplier
        return sort_index

    def _get_rank_sort_key(self, path: str) -> Tuple[int, int, int, int]:
        rank_info = self._extract_rank_info(path)
        return tuple(
            rank_info.get(f"{rank_type}_rank", 0)
            for rank_type in ["dp", "ep", "pp", "tp"]
        )

    def _maybe_cast_int(self, x) -> Optional[int]:
        try:
            return int(x)
        except (ValueError, TypeError):
            return None

    def get_merge_summary(self) -> Dict[str, Any]:
        if not os.path.exists(self.merged_trace_path):
            return {"error": "Merged trace file not found"}

        try:
            with gzip.open(self.merged_trace_path, "rt") as f:
                merged_data = json.load(f)

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
