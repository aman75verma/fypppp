#!/usr/bin/env python3
"""
submit_batches.py

Simple helper to submit a `batch_requests.jsonl` file to Anthropic's
Message Batches API. Respects any `cache_control` entries already present
in the requests (e.g., ephemeral system prompt caching).

Usage:
  python submit_batches.py --file batch_requests.jsonl

The script reads each JSONL line as a dict with keys `custom_id` and
`params` and forwards them to `client.messages.batches.create`.
"""
import argparse
import json
import logging
import math
import os
import sys
import time
from pathlib import Path

try:
    from anthropic import Anthropic
except Exception as e:
    print("Unable to import `anthropic`. Install requirements: pip install -r requirements.txt")
    raise


def load_jsonl(path):
    reqs = []
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as ex:
                raise ValueError(f"Invalid JSON on line {i}: {ex}")
            if "custom_id" not in obj or "params" not in obj:
                raise ValueError(f"Line {i} missing required keys 'custom_id' and 'params'")
            reqs.append(obj)
    return reqs


def build_requests(objs):
    # Return plain dicts for compatibility with different Anthropic SDKs.
    return [{"custom_id": o["custom_id"], "params": o["params"]} for o in objs]


def split_into_chunks(objs, size: int):
    for i in range(0, len(objs), size):
        yield objs[i : i + size]


def write_split_files(objs, size: int, out_stem: str, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for idx, chunk in enumerate(split_into_chunks(objs, size), start=1):
        name = f"{out_stem}_part_{idx:03d}.jsonl"
        path = out_dir / name
        with open(path, "w", encoding="utf-8") as f:
            for obj in chunk:
                f.write(json.dumps(obj, ensure_ascii=False) + "\n")
        written.append(path)
    return written


def main():
    parser = argparse.ArgumentParser(description="Submit JSONL batch requests to Anthropic batches API")
    parser.add_argument("--file", "-f", default="batch_requests.jsonl", help="Path to JSONL file")
    parser.add_argument("--api-key", help="Anthropic API key (or set ANTHROPIC_API_KEY env var)")
    parser.add_argument("--preview", type=int, default=0, help="Preview first N requests and exit")
    parser.add_argument("--dry-run", action="store_true", help="Validate file but do not submit")
    parser.add_argument("--split-size", type=int, default=0, help="If >0, split the input into files of this many requests")
    parser.add_argument("--split-only", action="store_true", help="Only write split files and validate, do not submit")
    parser.add_argument("--out-dir", default="./split_batches", help="Directory to write split files into")
    parser.add_argument("--out-stem", default=None, help="Stem for split files (default: derived from input file)")
    parser.add_argument("--pilot", type=int, default=0, help="If >0, submit the first N requests as a pilot before the full batch")
    parser.add_argument("--pilot-wait", action="store_true", default=False, help="Wait for pilot batch to complete and evaluate before continuing")
    parser.add_argument("--pilot-failure-threshold", type=float, default=0.1, help="Max allowed failure fraction for pilot (e.g., 0.1 = 10%%)")
    parser.add_argument("--pilot-poll-interval", type=int, default=10, help="Seconds between pilot status polls")
    parser.add_argument("--pilot-poll-timeout", type=int, default=600, help="Seconds to wait for pilot completion before giving up")
    args = parser.parse_args()

    api_key = args.api_key or os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("Error: Anthropic API key not found. Set ANTHROPIC_API_KEY or pass --api-key.")
        sys.exit(2)

    objs = load_jsonl(args.file)
    print(f"Loaded {len(objs)} requests from {args.file}")

    if args.preview > 0:
        print(json.dumps(objs[: args.preview], indent=2))
        return

    if args.dry_run:
        print("Dry run OK — file validated. Exiting.")
        return

    # If splitting requested, write split files and optionally exit
    if args.split_size and args.split_size > 0:
        out_stem = args.out_stem or Path(args.file).stem
        out_dir = Path(args.out_dir)
        written = write_split_files(objs, args.split_size, out_stem, out_dir)
        print(f"Wrote {len(written)} files to {out_dir.resolve()}")
        for p in written:
            print(f" - {p}")
        expected = math.ceil(len(objs) / args.split_size)
        if len(written) != expected:
            print(f"Warning: expected {expected} parts but wrote {len(written)}")
        else:
            print("Split validation OK — all parts generated.")
        if args.split_only:
            return

    # Pilot mode: submit first N requests and validate before full submission
    client = Anthropic(api_key=api_key)

    if args.pilot and args.pilot > 0:
        pilot_n = min(args.pilot, len(objs))
        pilot_objs = objs[:pilot_n]
        print(f"Submitting pilot of {pilot_n} requests...")
        pilot_requests = build_requests(pilot_objs)
        try:
            pilot_batch = client.messages.batches.create(requests=pilot_requests)
        except Exception as exc:
            logging.exception("Pilot batch submission failed")
            print("Pilot submission failed:", exc)
            sys.exit(1)

        pilot_batch_id = getattr(pilot_batch, "id", None) or (pilot_batch.get("id") if isinstance(pilot_batch, dict) else None)
        if not pilot_batch_id:
            print("Pilot submitted but no batch id found in response. Inspect pilot_batch:")
            print(pilot_batch)
            print("Aborting to avoid accidental full submission.")
            sys.exit(1)

        print(f"Pilot Batch ID: {pilot_batch_id}")

        if args.pilot_wait:
            start = time.time()
            pilot_ok = None
            while True:
                try:
                    status_resp = client.messages.batches.get(pilot_batch_id)
                except Exception:
                    try:
                        status_resp = client.messages.batches.retrieve(pilot_batch_id)
                    except Exception:
                        status_resp = None

                if status_resp is None:
                    print("Unable to fetch pilot status via SDK. Please check Anthropic console for Batch ID and confirm manually.")
                    break

                failed = None
                total = None
                if isinstance(status_resp, dict):
                    for k in ("failed_count", "num_failed", "failed", "errors"):
                        if k in status_resp:
                            try:
                                failed = int(status_resp.get(k) or 0)
                            except Exception:
                                failed = 0
                            break
                    for k in ("total_count", "num_requests", "num_total", "requests_count"):
                        if k in status_resp:
                            try:
                                total = int(status_resp.get(k) or 0)
                            except Exception:
                                total = 0
                            break
                    if total is None and "summary" in status_resp and isinstance(status_resp["summary"], dict):
                        s = status_resp["summary"]
                        failed = failed or int(s.get("failed", s.get("num_failed", 0)) or 0)
                        total = total or int(s.get("total", s.get("num_requests", 0)) or 0)
                    if total is None and "requests" in status_resp and isinstance(status_resp["requests"], list):
                        total = len(status_resp["requests"])
                        failed = sum(1 for r in status_resp["requests"] if r.get("status") in ("failed", "error"))

                if total and failed is not None:
                    fail_rate = failed / total if total > 0 else 0.0
                    print(f"Pilot results: {failed}/{total} failed ({fail_rate:.1%})")
                    if fail_rate > args.pilot_failure_threshold:
                        print(f"Failure rate {fail_rate:.1%} exceeds threshold {args.pilot_failure_threshold:.1%}. Aborting full submission.")
                        sys.exit(1)
                    else:
                        pilot_ok = True
                        print("Pilot passed threshold checks — continuing to full submission.")
                        break

                status_val = None
                if isinstance(status_resp, dict):
                    for k in ("status", "state", "life_cycle", "run_state"):
                        if k in status_resp:
                            status_val = status_resp.get(k)
                            break
                if status_val and str(status_val).lower() in ("completed", "succeeded", "finished"):
                    print("Pilot batch marked completed by API. (counts not available)")
                    pilot_ok = True
                    break

                elapsed = time.time() - start
                if elapsed > args.pilot_poll_timeout:
                    print("Pilot poll timed out. Please inspect the pilot batch in the Anthropic console. Aborting.")
                    sys.exit(1)

                print(f"Pilot still running... waiting {args.pilot_poll_interval}s (elapsed {int(elapsed)}s)")
                time.sleep(args.pilot_poll_interval)

            if pilot_ok is not True:
                print("Pilot check inconclusive — aborting full submission to avoid accidental charges.")
                sys.exit(1)

    # create full request list and submit
    requests = build_requests(objs)

    try:
        print("Submitting batch to Anthropic...")
        batch = client.messages.batches.create(requests=requests)
    except Exception as exc:
        logging.exception("Batch submission failed")
        print("Batch submission failed:", exc)
        sys.exit(1)

    # Print summary (batch id and basic metadata when available)
    try:
        batch_id = getattr(batch, "id", None) or (batch.get("id") if isinstance(batch, dict) else None)
    except Exception:
        batch_id = None

    if batch_id:
        print(f"Batch submitted. Batch ID: {batch_id}")
    else:
        print("Batch submitted. Response:")
        print(batch)

if __name__ == "__main__":
    main()
