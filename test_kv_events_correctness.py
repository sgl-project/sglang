#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
KV Events Correctness Validator with Visualization

Validates that SGLang KV events follow correct semantics:
1. No duplicate BlockStored for same (hash, medium) without intervening BlockRemoved
2. No BlockRemoved for blocks not currently stored in that medium
3. All events have required fields (block_hashes, medium)
4. Block lifecycle is consistent across tiers

Saves events to JSON and generates tier transition visualizations.

Usage:
    # Start SGLang server with KV events enabled, then run:
    python test_kv_events_correctness.py --endpoint tcp://127.0.0.1:5557 --timeout 60 -v

    # Save events and generate visualization:
    python test_kv_events_correctness.py --endpoint tcp://127.0.0.1:5557 --timeout 60 --save-events events.json --visualize
"""

import argparse
import json
import signal
import sys
import time
from collections import defaultdict
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Set, Tuple

import msgspec
import zmq

# Import event types directly from SGLang
from sglang.srt.disaggregation.kv_events import (
    KVEventBatch,
    BlockStored,
    BlockRemoved,
    AllBlocksCleared,
)


# Tier display names and colors for visualization
TIER_NAMES = {
    "GPU": "L1 (GPU)",
    "CPU_TIER1": "L2 (Host)",
    "CPU_TIER2": "L3 (Storage)",
}

TIER_ORDER = ["GPU", "CPU_TIER1", "CPU_TIER2"]


@dataclass
class TierTransition:
    """Record of a block's tier transition."""
    timestamp: float
    seq: int
    block_hash: int
    event_type: str  # "store" or "remove"
    medium: str
    tiers_after: List[str]  # Tiers block is in after this event


@dataclass
class BlockHistory:
    """Complete history of a single block."""
    hash: int
    first_seen: float
    transitions: List[TierTransition] = field(default_factory=list)
    current_tiers: Set[str] = field(default_factory=set)

    def to_dict(self):
        return {
            "hash": self.hash,
            "first_seen": self.first_seen,
            "transitions": [asdict(t) for t in self.transitions],
            "current_tiers": list(self.current_tiers),
        }


@dataclass
class BlockState:
    """Track state of a single block across tiers."""
    hash: int
    tiers: Set[str] = field(default_factory=set)
    store_count: Dict[str, int] = field(default_factory=lambda: defaultdict(int))
    remove_count: Dict[str, int] = field(default_factory=lambda: defaultdict(int))


@dataclass
class ValidationError:
    """Record of a validation error."""
    seq: int
    event_type: str
    block_hash: int
    medium: str
    error: str


class KvEventValidator:
    """Validate KV event stream for correctness with history tracking."""

    def __init__(self):
        self.blocks: Dict[int, BlockState] = {}
        self.block_histories: Dict[int, BlockHistory] = {}
        self.errors: List[ValidationError] = []
        self.warnings: List[str] = []
        self.raw_events: List[dict] = []  # Store raw events for export

        self.total_events = 0
        self.total_batches = 0
        self.store_events = 0
        self.remove_events = 0
        self.clear_events = 0
        self.events_by_medium: Dict[str, int] = defaultdict(int)
        self.start_time = time.time()

    def get_or_create_block(self, block_hash: int) -> BlockState:
        if block_hash not in self.blocks:
            self.blocks[block_hash] = BlockState(hash=block_hash)
        return self.blocks[block_hash]

    def get_or_create_history(self, block_hash: int) -> BlockHistory:
        if block_hash not in self.block_histories:
            self.block_histories[block_hash] = BlockHistory(
                hash=block_hash,
                first_seen=time.time() - self.start_time,
            )
        return self.block_histories[block_hash]

    def validate_batch(self, batch: KVEventBatch, seq: int, verbose: bool = False) -> bool:
        """Validate a batch of events."""
        self.total_batches += 1
        all_valid = True
        batch_time = time.time() - self.start_time

        for event in batch.events:
            self.total_events += 1

            if isinstance(event, BlockStored):
                block_hashes = event.block_hashes
                medium = event.medium or "UNKNOWN"

                if verbose:
                    print(f"  [seq={seq}] BlockStored: medium={medium}, hashes={block_hashes[:2]}{'...' if len(block_hashes) > 2 else ''}")

                # Store raw event
                self.raw_events.append({
                    "type": "BlockStored",
                    "seq": seq,
                    "time": batch_time,
                    "medium": medium,
                    "block_hashes": block_hashes,
                    "token_ids": event.token_ids[:10] if event.token_ids else None,
                })

                for block_hash in block_hashes:
                    if not self._validate_store(seq, block_hash, medium, batch_time):
                        all_valid = False

            elif isinstance(event, BlockRemoved):
                block_hashes = event.block_hashes
                medium = event.medium or "UNKNOWN"

                if verbose:
                    print(f"  [seq={seq}] BlockRemoved: medium={medium}, hashes={block_hashes[:2]}{'...' if len(block_hashes) > 2 else ''}")

                # Store raw event
                self.raw_events.append({
                    "type": "BlockRemoved",
                    "seq": seq,
                    "time": batch_time,
                    "medium": medium,
                    "block_hashes": block_hashes,
                })

                for block_hash in block_hashes:
                    if not self._validate_remove(seq, block_hash, medium, batch_time):
                        all_valid = False

            elif isinstance(event, AllBlocksCleared):
                self.clear_events += 1
                if verbose:
                    print(f"  [seq={seq}] AllBlocksCleared")

                self.raw_events.append({
                    "type": "AllBlocksCleared",
                    "seq": seq,
                    "time": batch_time,
                })

                # Clear all tracked state
                self.blocks.clear()
                self.block_histories.clear()

            else:
                self.warnings.append(f"[seq={seq}] Unknown event type: {type(event)}")

        return all_valid

    def _validate_store(self, seq: int, block_hash: int, medium: str, timestamp: float) -> bool:
        """Validate BlockStored event."""
        self.store_events += 1
        self.events_by_medium[f"store_{medium}"] += 1

        block = self.get_or_create_block(block_hash)
        history = self.get_or_create_history(block_hash)

        # Check for duplicate store
        if medium in block.tiers:
            self.errors.append(ValidationError(
                seq=seq, event_type="BlockStored", block_hash=block_hash, medium=medium,
                error=f"Duplicate store: block already in {medium} tier"
            ))
            return False

        block.tiers.add(medium)
        block.store_count[medium] += 1

        # Record transition
        history.current_tiers.add(medium)
        history.transitions.append(TierTransition(
            timestamp=timestamp,
            seq=seq,
            block_hash=block_hash,
            event_type="store",
            medium=medium,
            tiers_after=list(history.current_tiers),
        ))

        return True

    def _validate_remove(self, seq: int, block_hash: int, medium: str, timestamp: float) -> bool:
        """Validate BlockRemoved event."""
        self.remove_events += 1
        self.events_by_medium[f"remove_{medium}"] += 1

        block = self.get_or_create_block(block_hash)
        history = self.get_or_create_history(block_hash)

        # Check for orphan remove
        if medium not in block.tiers:
            self.errors.append(ValidationError(
                seq=seq, event_type="BlockRemoved", block_hash=block_hash, medium=medium,
                error=f"Orphan remove: block not in {medium} tier (current tiers: {block.tiers or 'none'})"
            ))
            return False

        block.tiers.remove(medium)
        block.remove_count[medium] += 1

        # Record transition
        history.current_tiers.discard(medium)
        history.transitions.append(TierTransition(
            timestamp=timestamp,
            seq=seq,
            block_hash=block_hash,
            event_type="remove",
            medium=medium,
            tiers_after=list(history.current_tiers),
        ))

        return True

    def get_summary(self) -> dict:
        blocks_in_gpu = sum(1 for b in self.blocks.values() if "GPU" in b.tiers)
        blocks_in_cpu1 = sum(1 for b in self.blocks.values() if "CPU_TIER1" in b.tiers)
        blocks_in_cpu2 = sum(1 for b in self.blocks.values() if "CPU_TIER2" in b.tiers)
        blocks_in_multiple = sum(1 for b in self.blocks.values() if len(b.tiers) > 1)
        blocks_orphaned = sum(1 for b in self.blocks.values() if len(b.tiers) == 0)

        return {
            "total_batches": self.total_batches,
            "total_events": self.total_events,
            "store_events": self.store_events,
            "remove_events": self.remove_events,
            "clear_events": self.clear_events,
            "events_by_medium": dict(self.events_by_medium),
            "unique_blocks_seen": len(self.blocks),
            "current_state": {
                "blocks_in_GPU": blocks_in_gpu,
                "blocks_in_CPU_TIER1": blocks_in_cpu1,
                "blocks_in_CPU_TIER2": blocks_in_cpu2,
                "blocks_in_multiple_tiers": blocks_in_multiple,
                "blocks_fully_evicted": blocks_orphaned,
            },
            "errors": len(self.errors),
            "warnings": len(self.warnings),
        }

    def save_events(self, filepath: str):
        """Save all events and block histories to JSON."""
        data = {
            "summary": self.get_summary(),
            "raw_events": self.raw_events,
            "block_histories": {
                str(h): hist.to_dict()
                for h, hist in self.block_histories.items()
            },
            "errors": [asdict(e) for e in self.errors],
        }
        with open(filepath, "w") as f:
            json.dump(data, f, indent=2)
        print(f"Events saved to {filepath}")

    def print_report(self):
        summary = self.get_summary()

        print("\n" + "=" * 60)
        print("KV EVENTS VALIDATION REPORT")
        print("=" * 60)

        print(f"\n Event Statistics:")
        print(f"   Total batches:    {summary['total_batches']}")
        print(f"   Total events:     {summary['total_events']}")
        print(f"   BlockStored:      {summary['store_events']}")
        print(f"   BlockRemoved:     {summary['remove_events']}")
        print(f"   AllBlocksCleared: {summary['clear_events']}")
        print(f"   Unique blocks:    {summary['unique_blocks_seen']}")

        print(f"\n Events by Medium:")
        for key, count in sorted(summary['events_by_medium'].items()):
            print(f"   {key}: {count}")

        print(f"\n Current Block State:")
        state = summary['current_state']
        print(f"   In GPU:           {state['blocks_in_GPU']}")
        print(f"   In CPU_TIER1:     {state['blocks_in_CPU_TIER1']}")
        print(f"   In CPU_TIER2:     {state['blocks_in_CPU_TIER2']}")
        print(f"   In multiple tiers: {state['blocks_in_multiple_tiers']}")
        print(f"   Fully evicted:    {state['blocks_fully_evicted']}")

        if self.errors:
            print(f"\n ERRORS ({len(self.errors)}):")
            for i, err in enumerate(self.errors[:10]):
                print(f"   [{err.seq}] {err.event_type} hash={err.block_hash} medium={err.medium}")
                print(f"       {err.error}")
            if len(self.errors) > 10:
                print(f"   ... and {len(self.errors) - 10} more errors")

        if self.warnings:
            print(f"\n Warnings ({len(self.warnings)}):")
            for w in self.warnings[:5]:
                print(f"   {w}")
            if len(self.warnings) > 5:
                print(f"   ... and {len(self.warnings) - 5} more warnings")

        print("\n" + "=" * 60)
        if self.errors:
            print("VALIDATION FAILED")
        else:
            print("VALIDATION PASSED")
        print("=" * 60)

    def visualize_tier_transitions(self, max_blocks: int = 20):
        """Generate ASCII visualization of block tier transitions."""
        print("\n" + "=" * 60)
        print("BLOCK TIER TRANSITION VISUALIZATION")
        print("=" * 60)

        # Get blocks with interesting histories (multiple transitions)
        interesting_blocks = sorted(
            [(h, hist) for h, hist in self.block_histories.items() if len(hist.transitions) > 1],
            key=lambda x: len(x[1].transitions),
            reverse=True
        )[:max_blocks]

        if not interesting_blocks:
            print("\nNo blocks with multiple transitions to visualize.")
            return

        print(f"\nShowing {len(interesting_blocks)} blocks with most transitions:\n")
        print("Legend: [G]=GPU  [H]=Host(L2)  [S]=Storage(L3)  .=not present")
        print("        + = stored,  - = removed")
        print()

        for block_hash, history in interesting_blocks:
            short_hash = f"{block_hash & 0xFFFF:04x}"
            print(f"Block ...{short_hash}:")

            # Build timeline
            timeline = []
            current_tiers = set()

            for t in history.transitions:
                if t.event_type == "store":
                    current_tiers.add(t.medium)
                    symbol = "+"
                else:
                    current_tiers.discard(t.medium)
                    symbol = "-"

                tier_state = ""
                for tier in TIER_ORDER:
                    if tier in current_tiers:
                        tier_state += {"GPU": "[G]", "CPU_TIER1": "[H]", "CPU_TIER2": "[S]"}[tier]
                    else:
                        tier_state += " . "

                tier_changed = {"GPU": "G", "CPU_TIER1": "H", "CPU_TIER2": "S"}[t.medium]
                timeline.append(f"  t={t.timestamp:5.2f}s seq={t.seq:3d}: {symbol}{tier_changed} -> {tier_state}")

            for line in timeline:
                print(line)
            print()

    def generate_mermaid_diagram(self, max_blocks: int = 10) -> str:
        """Generate Mermaid sequence diagram for tier transitions."""
        lines = [
            "```mermaid",
            "sequenceDiagram",
            "    participant GPU as L1 (GPU)",
            "    participant Host as L2 (Host)",
            "    participant Storage as L3 (Storage)",
            "",
        ]

        # Get blocks with interesting histories
        interesting_blocks = sorted(
            [(h, hist) for h, hist in self.block_histories.items() if len(hist.transitions) > 1],
            key=lambda x: len(x[1].transitions),
            reverse=True
        )[:max_blocks]

        participant_map = {
            "GPU": "GPU",
            "CPU_TIER1": "Host",
            "CPU_TIER2": "Storage",
        }

        for block_hash, history in interesting_blocks:
            short_hash = f"{block_hash & 0xFFFF:04x}"
            lines.append(f"    Note over GPU,Storage: Block ...{short_hash}")

            prev_tier = None
            for t in history.transitions:
                participant = participant_map[t.medium]
                if t.event_type == "store":
                    if prev_tier:
                        # Show as transfer from previous tier
                        lines.append(f"    {participant_map.get(prev_tier, 'GPU')}->>{participant}: store")
                    else:
                        lines.append(f"    Note right of {participant}: stored")
                    prev_tier = t.medium
                else:
                    lines.append(f"    Note right of {participant}: removed")
                    if t.tiers_after:
                        prev_tier = t.tiers_after[0]
                    else:
                        prev_tier = None

            lines.append("")

        lines.append("```")
        return "\n".join(lines)

    def print_tier_flow_summary(self):
        """Print summary of common tier transition patterns."""
        print("\n" + "=" * 60)
        print("TIER FLOW PATTERNS")
        print("=" * 60)

        # Analyze transition patterns
        patterns: Dict[str, int] = defaultdict(int)

        for history in self.block_histories.values():
            if len(history.transitions) < 2:
                continue

            # Extract pattern as sequence of (event_type, medium)
            pattern_parts = []
            for t in history.transitions:
                pattern_parts.append(f"{t.event_type[0].upper()}{t.medium[0]}")

            pattern = " -> ".join(pattern_parts)
            patterns[pattern] += 1

        if patterns:
            print("\nCommon transition patterns (SG=store GPU, RG=remove GPU, etc.):")
            for pattern, count in sorted(patterns.items(), key=lambda x: -x[1])[:15]:
                print(f"  {count:4d}x: {pattern}")
        else:
            print("\nNo multi-transition patterns found.")

        # Tier occupancy over time
        print("\nTier state evolution:")
        time_points = sorted(set(t.timestamp for h in self.block_histories.values() for t in h.transitions))

        if len(time_points) > 10:
            # Sample at most 10 time points
            step = len(time_points) // 10
            time_points = time_points[::step]

        print(f"  {'Time':>8s}  {'GPU':>6s}  {'Host':>6s}  {'Storage':>8s}")
        print("  " + "-" * 35)

        for tp in time_points:
            gpu_count = 0
            host_count = 0
            storage_count = 0

            for history in self.block_histories.values():
                current = set()
                for t in history.transitions:
                    if t.timestamp > tp:
                        break
                    if t.event_type == "store":
                        current.add(t.medium)
                    else:
                        current.discard(t.medium)

                if "GPU" in current:
                    gpu_count += 1
                if "CPU_TIER1" in current:
                    host_count += 1
                if "CPU_TIER2" in current:
                    storage_count += 1

            print(f"  {tp:7.2f}s  {gpu_count:6d}  {host_count:6d}  {storage_count:8d}")


def main():
    parser = argparse.ArgumentParser(description="Validate KV event stream correctness")
    parser.add_argument("--endpoint", default="tcp://127.0.0.1:5557",
                        help="ZMQ endpoint to subscribe to")
    parser.add_argument("--timeout", type=int, default=30,
                        help="Seconds to collect events (0 = run until Ctrl+C)")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Print each event as received")
    parser.add_argument("--save-events", type=str, default=None,
                        help="Save events to JSON file")
    parser.add_argument("--visualize", action="store_true",
                        help="Show tier transition visualization")
    parser.add_argument("--mermaid", action="store_true",
                        help="Generate Mermaid diagram")
    args = parser.parse_args()

    validator = KvEventValidator()
    running = True

    def signal_handler(sig, frame):
        nonlocal running
        print("\nStopping...")
        running = False

    signal.signal(signal.SIGINT, signal_handler)

    # Set up ZMQ subscriber
    ctx = zmq.Context()
    sock = ctx.socket(zmq.SUB)
    sock.connect(args.endpoint)
    sock.setsockopt_string(zmq.SUBSCRIBE, "")
    sock.setsockopt(zmq.RCVTIMEO, 1000)

    # Decoder for KVEventBatch (with properly typed events)
    decoder = msgspec.msgpack.Decoder(KVEventBatch)

    print(f"Subscribing to {args.endpoint}")
    print(f"Collecting events for {args.timeout}s..." if args.timeout else "Collecting events (Ctrl+C to stop)...")

    start_time = time.time()

    while running:
        if args.timeout and (time.time() - start_time) >= args.timeout:
            break

        try:
            # Receive multipart message: [topic, seq_bytes, payload]
            parts = sock.recv_multipart()

            if len(parts) == 3:
                topic, seq_bytes, payload = parts
                seq = int.from_bytes(seq_bytes, "big")
            elif len(parts) == 1:
                payload = parts[0]
                seq = validator.total_batches
            else:
                print(f"Unexpected message parts: {len(parts)}")
                continue

            # Decode the batch
            batch = decoder.decode(payload)
            validator.validate_batch(batch, seq, verbose=args.verbose)

        except zmq.Again:
            continue
        except msgspec.DecodeError as e:
            print(f"Decode error: {e}")
        except Exception as e:
            print(f"Error processing event: {e}")
            import traceback
            traceback.print_exc()

    sock.close()
    ctx.term()

    # Print main report
    validator.print_report()

    # Save events if requested
    if args.save_events:
        validator.save_events(args.save_events)

    # Visualizations
    if args.visualize:
        validator.visualize_tier_transitions()
        validator.print_tier_flow_summary()

    if args.mermaid:
        print("\n" + "=" * 60)
        print("MERMAID DIAGRAM")
        print("=" * 60)
        print(validator.generate_mermaid_diagram())

    sys.exit(1 if validator.errors else 0)


if __name__ == "__main__":
    main()
