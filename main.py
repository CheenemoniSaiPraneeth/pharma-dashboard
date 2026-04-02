"""
main.py — Full Pharma Intelligence Pipeline

Runs all four stages in sequence with one command:

  Stage 1 │ harvester.py  → scrapes press pages    → total_list.json
  Stage 2 │ today_info.py → deduplicates            → today.json  +  master_storage.json
  Stage 3 │ llm.py        → fetches + extracts text → text.json
  Stage 4 │ summary.py    → LLM extraction          → briefs.json  (or --output)

Usage:
  python main.py --query "PROTAC"
  python main.py --query "CAR-T" --output results.json
  python main.py --query "PROTAC" --skip-harvest               # Stage 4 only, reads text.json
  python main.py --query "PROTAC" --skip-harvest --input my_text.json  # custom text file
  python main.py --query "PROTAC" --from-stage 3               # resume from Stage 3
"""

import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

# ── STAGE REGISTRY ────────────────────────────────────────────────────────────

STAGES = [
    {
        "id":     1,
        "name":   "Harvester",
        "script": "harvester.py",
        "output": "total_list.json",
        "desc":   "Scraping press pages",
    },
    {
        "id":     2,
        "name":   "Delta Engine",
        "script": "today_info.py",
        "output": "today.json",
        "desc":   "Deduplicating against master storage",
    },
    {
        "id":     3,
        "name":   "Text Extractor",
        "script": "llm.py",
        "output": "text.json",
        "desc":   "Fetching and extracting article text",
    },
    {
        "id":     4,
        "name":   "Summariser",
        "script": "summary.py",
        "output": "briefs.json",  # overridden at runtime by --output
        "desc":   "Extracting structured intelligence via LLM",
    },
]

# ── HELPERS ───────────────────────────────────────────────────────────────────

def banner(text: str) -> None:
    line = "─" * 64
    print(f"\n{line}\n  {text}\n{line}")


def run_stage(stage: dict, extra_args: list[str] | None = None) -> bool:
    """
    Run a pipeline stage as a subprocess.
    Returns True on success, False on failure.
    """
    extra_args = extra_args or []
    script     = stage["script"]

    banner(f"Stage {stage['id']}: {stage['name']} — {stage['desc']}")

    if not Path(script).exists():
        print(f"[ERROR] Script not found: {script}")
        return False

    cmd = [sys.executable, script] + extra_args
    print(f"[CMD] {' '.join(cmd)}\n")

    result = subprocess.run(cmd)

    if result.returncode != 0:
        print(f"\n[ERROR] Stage {stage['id']} exited with code {result.returncode}")
        return False

    out = stage.get("output")
    if out and Path(out).exists():
        size = Path(out).stat().st_size
        print(f"\n[OK] {out}  ({size:,} bytes)")
    else:
        print(f"\n[WARN] Expected output '{out}' not found after stage.")

    return True


def json_is_empty(filepath: str) -> bool:
    """Return True if the file is missing, unreadable, or its top-level JSON is empty."""
    p = Path(filepath)
    if not p.exists():
        return True
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
        return len(data) == 0
    except Exception:
        return True


def print_summary(output_file: str) -> None:
    p = Path(output_file)
    if not p.exists():
        print("[WARN] Output file not found.")
        return

    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        print("[WARN] Could not read output file.")
        return

    meta = data.get("meta", {})
    news = data.get("news", [])

    print("\n" + "═" * 64)
    print("  PIPELINE COMPLETE")
    print("═" * 64)
    print(f"  Query           : {meta.get('query', '?')}")
    print(f"  Articles used   : {meta.get('articles_used', '?')}")
    print(f"  Items extracted : {meta.get('items_extracted', len(news))}")
    print(f"  Items dropped   : {meta.get('items_dropped', 0)}")
    print(f"  Generated at    : {meta.get('generated_at', '?')}")
    print(f"  Output file     : {output_file}")
    print("═" * 64)

    if news:
        print("\n  TOP RESULTS:\n")
        for i, item in enumerate(news[:5], 1):
            print(f"  {i}. [{item.get('modality','?')}] {item.get('company','?')}")
            print(f"     {item.get('news','')[:120]}")
            if item.get("url"):
                print(f"     {item['url']}")
            print()

# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Pharma Intelligence Pipeline — runs all four stages with one command.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --query "PROTAC"
  python main.py --query "CAR-T" --output results.json
  python main.py --query "PROTAC" --skip-harvest
  python main.py --query "PROTAC" --skip-harvest --input my_articles.json
  python main.py --query "PROTAC" --from-stage 3
        """,
    )
    p.add_argument(
        "--query", "-q", required=True,
        help="Therapeutic modality focus e.g. 'PROTAC', 'bispecific antibodies'",
    )
    p.add_argument(
        "--output", "-o", default="briefs.json",
        help="Final output JSON file (default: briefs.json)",
    )
    p.add_argument(
        "--input", "-i", default="text.json",
        help=(
            "Extracted text file fed into Stage 4. "
            "Only used when --skip-harvest or --from-stage 4 is set. "
            "(default: text.json)"
        ),
    )
    p.add_argument(
        "--skip-harvest", action="store_true",
        help="Skip Stages 1-3 and run only Stage 4 (summariser).",
    )
    p.add_argument(
        "--from-stage", type=int, default=1, choices=[1, 2, 3, 4],
        help="Start from a specific stage number (default: 1).",
    )
    return p.parse_args()

# ── MAIN ──────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    start_time = datetime.now()
    print("\n" + "═" * 64)
    print("  PHARMA INTELLIGENCE PIPELINE")
    print(f"  Query   : {args.query}")
    print(f"  Output  : {args.output}")
    print(f"  Started : {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("═" * 64)

    # ── Resolve start stage ───────────────────────────────────────────────────
    if args.skip_harvest:
        start_from = 4
        print("[INFO] --skip-harvest: jumping to Stage 4.")
    else:
        start_from = args.from_stage

    stages_to_run = [s for s in STAGES if s["id"] >= start_from]

    # ── Run each stage ────────────────────────────────────────────────────────
    for stage in stages_to_run:
        extra: list[str] = []

        if stage["id"] == 4:
            # Resolve which text file Stage 4 should read:
            #   - If we started at Stage 4 (skip-harvest or from-stage 4) → use --input
            #   - If Stage 3 ran as part of this session → always use text.json
            text_input = args.input if start_from >= 4 else "text.json"

            stage["output"] = args.output  # propagate --output into stage record
            extra = [
                "--query",  args.query,
                "--input",  text_input,
                "--output", args.output,
            ]

        ok = run_stage(stage, extra_args=extra)

        if not ok:
            print(f"\n[FATAL] Pipeline aborted at Stage {stage['id']}.")
            sys.exit(1)

        # ── Early-exit guard after Stage 2 ───────────────────────────────────
        if stage["id"] == 2 and json_is_empty("today.json"):
            print("\n[INFO] No new articles found today — nothing to process.")
            sys.exit(0)

    # ── Final summary ─────────────────────────────────────────────────────────
    elapsed = int((datetime.now() - start_time).total_seconds())
    print(f"\n[INFO] Total elapsed: {elapsed}s")
    print_summary(args.output)


if __name__ == "__main__":
    main()
