#!/usr/bin/env python3
"""Fix subagent timestamps in AgentX trace files.

This script converts subagent timestamps from parent-relative (starting from 0)
to session-relative (cumulative from session start) so they can be properly
sorted in chronological order.

Usage:
    python fix_subagent_timestamps.py input.jsonl output.jsonl
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def fix_timestamps(requests: list, parent_time: float = 0.0) -> None:
    """Recursively fix subagent timestamps to be session-relative.

    Args:
        requests: List of request objects (may contain nested subagent requests)
        parent_time: The parent request's timestamp (0.0 for root level)
    """
    for request in requests:
        if request.get("type") == "subagent":
            # This is a subagent request - its time is the anchor for its children
            subagent_start = float(request.get("t", 0.0))

            # Recursively fix nested requests, using this subagent's time as the base
            if "requests" in request and isinstance(request["requests"], list):
                fix_timestamps(request["requests"], subagent_start)

        elif "t" in request:
            # This is a regular request - add parent time to make it session-relative
            request["t"] = float(request["t"]) + parent_time


def fix_trace_file(input_path: Path, output_path: Path, dry_run: bool = False) -> dict:
    """Process trace file and fix all subagent timestamps.

    Args:
        input_path: Input JSONL file path
        output_path: Output JSONL file path
        dry_run: If True, only analyze without writing

    Returns:
        Statistics dictionary
    """
    stats = {
        "sessions_processed": 0,
        "total_requests": 0,
        "subagent_requests": 0,
        "requests_fixed": 0,
    }

    output_lines = []

    with input_path.open() as f:
        for line_num, line in enumerate(f, 1):
            try:
                session = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"[ERROR] Line {line_num}: Invalid JSON - {e}", file=sys.stderr)
                continue

            if "requests" not in session:
                print(f"[WARNING] Line {line_num}: No 'requests' field, skipping", file=sys.stderr)
                output_lines.append(line)
                continue

            # Count requests before fixing
            def count_requests(items, is_subagent=False):
                count = subagent_count = 0
                for req in items:
                    if req.get("type") == "subagent":
                        subagent_count += 1
                        if "requests" in req:
                            sub_cnt, sub_agent_cnt = count_requests(req["requests"], True)
                            count += sub_cnt
                            subagent_count += sub_agent_cnt
                    else:
                        count += 1
                        if is_subagent:
                            subagent_count += 1
                return count, subagent_count

            before_count, subagent_count = count_requests(session["requests"])

            # Fix timestamps (modifies session in place)
            fix_timestamps(session["requests"], parent_time=0.0)

            # Count after fixing (should be the same)
            after_count, _ = count_requests(session["requests"])

            stats["sessions_processed"] += 1
            stats["total_requests"] += after_count
            stats["subagent_requests"] += subagent_count
            stats["requests_fixed"] += subagent_count

            if before_count != after_count:
                print(f"[WARNING] Line {line_num}: Request count changed from {before_count} to {after_count}",
                      file=sys.stderr)

            # Store the fixed session
            output_lines.append(json.dumps(session, ensure_ascii=False) + "\n")

    # Write output
    if not dry_run:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w") as f:
            f.writelines(output_lines)
        print(f"[INFO] Fixed trace written to: {output_path}")
    else:
        print(f"[INFO] Dry run - no output written")

    return stats


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("input", type=Path, help="Input trace JSONL file")
    parser.add_argument("output", type=Path, help="Output trace JSONL file")
    parser.add_argument("--dry-run", action="store_true",
                       help="Analyze without writing output")
    parser.add_argument("--verbose", "-v", action="store_true",
                       help="Print detailed progress")

    args = parser.parse_args()

    if not args.input.exists():
        print(f"[ERROR] Input file not found: {args.input}", file=sys.stderr)
        sys.exit(1)

    if args.output.exists() and not args.dry_run:
        response = input(f"Output file exists: {args.output}\nOverwrite? (y/N) ")
        if not response.strip().lower().startswith('y'):
            print("Aborted.")
            sys.exit(0)

    print(f"[INFO] Processing: {args.input}")
    stats = fix_trace_file(args.input, args.output, args.dry_run)

    print("\n=== Statistics ===")
    print(f"Sessions processed:      {stats['sessions_processed']}")
    print(f"Total requests:          {stats['total_requests']}")
    print(f"Subagent requests:       {stats['subagent_requests']}")
    print(f"Timestamps fixed:        {stats['requests_fixed']}")

    if not args.dry_run:
        print(f"\n[SUCCESS] Fixed trace saved to: {args.output}")


if __name__ == "__main__":
    main()
