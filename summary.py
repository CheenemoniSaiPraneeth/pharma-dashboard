"""
summary.py — Pharma Intelligence Brief Generator

Reads extracted article text (text.json), calls the LLM with guaranteed
JSON output, validates the schema, and writes structured news items to
a clean JSON file.

After each run, results are also appended to briefs_history.json:

  {
    "2026-04-02": {
      "bispecific antibodies": [ { company, modality, news, url }, ... ],
      "CAR-T":                 [ ... ]
    },
    "2026-04-01": { ... }
  }

Usage:
  python summary.py --query "PROTAC"
  python summary.py --query "CAR-T" --input text.json --output briefs.json
"""

import argparse
import json
import re
import sys
import time
import requests
from datetime import datetime
from pathlib import Path

# ── CONFIG ────────────────────────────────────────────────────────────────────
import os

INVOKE_URL      = "https://integrate.api.nvidia.com/v1/chat/completions"
API_KEY         = "Bearer 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"
MODEL           = "qwen/qwen3.5-122b-a10b"
MAX_RETRIES     = 3
BACKOFF_BASE    = 1      # seconds — attempt 1: no wait, 2: 1s, 3: 2s, 4: 4s
REQUEST_TIMEOUT = 180    # seconds per attempt

BRIEFS_HISTORY_FILE = "briefs_history.json"   # ← datewise store

HEADERS = {
    "Authorization": API_KEY,
    "Content-Type":  "application/json",
    "Accept":        "application/json",
}

VALID_MODALITIES = {
    "bispecific antibodies",
    "monoclonal antibodies",
    "gene editing",
    "molecular glues",
}

# ── SYSTEM PROMPT ─────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """
You are a pharmaceutical news extraction assistant.

Your job is to scan pharmaceutical articles and extract structured news events
related to specific therapeutic modalities:
  - bispecific antibodies
  - monoclonal antibodies
  - gene editing
  - molecular glues

RULES:
- Extract ONLY real news events: clinical trials, approvals, partnerships,
  acquisitions, funding rounds, or scientific breakthroughs.
- Ignore background/explainer content and unrelated topics.
- Do NOT hallucinate company names, drug names, or URLs.
- If multiple companies are involved, choose the primary one.
- If URL is unavailable, set it to null.
- Avoid duplicate entries.
- Return ONLY valid JSON — no explanations, no markdown fences.

OUTPUT FORMAT (strict JSON, nothing else):
{
  "news": [
    {
      "company": "string",
      "modality": "one of the four predefined modalities",
      "news": "2-3 line description of the event",
      "url": "source article URL or null"
    }
  ]
}
"""

# ── PROMPT BUILDER ────────────────────────────────────────────────────────────

def build_prompt(articles: list, query: str) -> str | None:
    sections = []

    for i, art in enumerate(articles, 1):
        title  = art.get("title", "Untitled")
        source = art.get("url", "Unknown")
        date   = art.get("date") or art.get("period", "")
        body   = (art.get("body") or "").strip()

        if not body:
            continue

        sections.append(
            f"--- ARTICLE {i} ---\n"
            f"Source : {source}\n"
            f"Date   : {date}\n"
            f"Title  : {title}\n\n"
            f"{body}"
        )

    if not sections:
        return None

    return (
        f"Query focus: {query}\n\n"
        f"Below are {len(sections)} pharmaceutical news articles.\n"
        f"Extract all relevant news events matching the query focus.\n\n"
        + "\n\n".join(sections)
    )

# ── LLM CALL ─────────────────────────────────────────────────────────────────

def call_llm(user_prompt: str) -> str:
    payload = {
        "model":   MODEL,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": user_prompt},
        ],
        "max_tokens":      16384,
        "temperature":     0.3,
        "top_p":           0.95,
        "stream":          False,
        "response_format": {"type": "json_object"},
    }

    last_error: str = "unknown"

    for attempt in range(1, MAX_RETRIES + 1):

        if attempt > 1:
            wait = BACKOFF_BASE * (2 ** (attempt - 2))
            print(f"[LLM] Retry {attempt}/{MAX_RETRIES} — backing off {wait}s...")
            time.sleep(wait)

        print(f"[LLM] Attempt {attempt}/{MAX_RETRIES} — sending request...", flush=True)

        try:
            resp = requests.post(
                INVOKE_URL,
                headers=HEADERS,
                json=payload,
                timeout=REQUEST_TIMEOUT,
            )
            resp.raise_for_status()

        except requests.exceptions.Timeout:
            last_error = f"timed out after {REQUEST_TIMEOUT}s"
            print(f"[LLM] ✗ Attempt {attempt} failed: {last_error}")
            continue

        except requests.exceptions.HTTPError as exc:
            code = exc.response.status_code if exc.response is not None else 0
            last_error = f"HTTP {code}"
            if 400 <= code < 500:
                raise RuntimeError(
                    f"HTTP {code} — aborting retries (client error): {exc}"
                ) from exc
            print(f"[LLM] ✗ Attempt {attempt} failed: {last_error}")
            continue

        except requests.exceptions.RequestException as exc:
            last_error = str(exc)
            print(f"[LLM] ✗ Attempt {attempt} network error: {last_error}")
            continue

        try:
            envelope = resp.json()
        except json.JSONDecodeError as exc:
            last_error = f"response body not valid JSON: {exc}"
            print(f"[LLM] ✗ Attempt {attempt} failed: {last_error}")
            continue

        content = (
            envelope
            .get("choices", [{}])[0]
            .get("message", {})
            .get("content", "")
            .strip()
        )

        if not content:
            last_error = "empty content in API response"
            print(f"[LLM] ✗ Attempt {attempt} failed: {last_error}")
            continue

        print(f"[LLM] ✓ Response received ({len(content):,} chars)")
        return content

    raise RuntimeError(
        f"LLM call failed after {MAX_RETRIES} attempts. Last error: {last_error}"
    )

# ── JSON PARSER ───────────────────────────────────────────────────────────────

def parse_llm_response(raw: str) -> list:
    cleaned = re.sub(r"```(?:json)?|```", "", raw).strip()

    try:
        obj = json.loads(cleaned)
        if isinstance(obj, dict):
            return obj.get("news", [])
        if isinstance(obj, list):
            return obj
    except json.JSONDecodeError:
        pass

    match = re.search(r"\{.*\}", cleaned, re.DOTALL)
    if match:
        try:
            obj = json.loads(match.group())
            return obj.get("news", [])
        except json.JSONDecodeError:
            pass

    print("[WARN] parse_llm_response: could not extract JSON — returning []")
    return []

# ── SCHEMA VALIDATOR ──────────────────────────────────────────────────────────

REQUIRED_KEYS = {"company", "modality", "news", "url"}

def validate_items(raw_items: list) -> tuple[list, int]:
    valid   = []
    dropped = 0

    for idx, item in enumerate(raw_items, 1):
        if not isinstance(item, dict):
            print(f"[VALIDATE] Item {idx}: not a dict — dropped")
            dropped += 1
            continue

        missing = REQUIRED_KEYS - item.keys()
        if missing:
            print(f"[VALIDATE] Item {idx}: missing keys {missing} — dropped")
            dropped += 1
            continue

        if item["modality"] not in VALID_MODALITIES:
            print(f"[VALIDATE] Item {idx}: invalid modality '{item['modality']}' — dropped")
            dropped += 1
            continue

        if not isinstance(item["company"], str) or not item["company"].strip():
            print(f"[VALIDATE] Item {idx}: empty company — dropped")
            dropped += 1
            continue

        if not isinstance(item["news"], str) or not item["news"].strip():
            print(f"[VALIDATE] Item {idx}: empty news text — dropped")
            dropped += 1
            continue

        if item["url"] is not None and not isinstance(item["url"], str):
            item["url"] = str(item["url"])

        valid.append(item)

    return valid, dropped

# ── OUTPUT BUILDER ────────────────────────────────────────────────────────────

def build_output(news_items: list, query: str, article_count: int, dropped: int) -> dict:
    return {
        "meta": {
            "query":           query,
            "articles_used":   article_count,
            "items_extracted": len(news_items),
            "items_dropped":   dropped,
            "generated_at":    datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        },
        "news": news_items,
    }

# ── DATEWISE HISTORY ──────────────────────────────────────────────────────────

def append_to_history(
    news_items: list,
    query: str,
    history_file: str = BRIEFS_HISTORY_FILE,
) -> None:
    """
    Append today's validated news items into briefs_history.json.

    Structure:
      {
        "2026-04-02": {
          "bispecific antibodies": [ {company, modality, news, url}, ... ],
          "CAR-T": [ ... ]
        },
        "2026-04-01": { ... }
      }

    - Items are stored under their actual modality key (not the query),
      so you can look up "what bispecific news ran on 2026-04-02" directly.
    - Duplicate URLs within the same date+modality are skipped.
    - Dates are kept sorted newest-first.
    """
    today = datetime.now().strftime("%Y-%m-%d")

    # ── Load existing history ─────────────────────────────────────────────────
    p = Path(history_file)
    if p.exists():
        try:
            history = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            history = {}
    else:
        history = {}

    # ── Ensure today's slot exists ────────────────────────────────────────────
    if today not in history:
        history[today] = {}

    # ── Append items, grouped by modality ─────────────────────────────────────
    added = 0
    for item in news_items:
        modality = item.get("modality", "unknown")
        url      = item.get("url")

        if modality not in history[today]:
            history[today][modality] = []

        # Skip exact URL duplicates within the same date+modality
        existing_sigs = {
            (r.get("company"), r.get("news")) for r in history[today][modality]
        }
        if (item.get("company"), item.get("news")) in existing_sigs:
            continue

        history[today][modality].append({
            "company":  item.get("company"),
            "modality": modality,
            "news":     item.get("news"),
            "url":      url,
            "query":    query,           # which query surfaced this item
        })
        added += 1

    # ── Sort dates newest-first ───────────────────────────────────────────────
    history = dict(sorted(history.items(), reverse=True))

    # ── Save ──────────────────────────────────────────────────────────────────
    p.write_text(json.dumps(history, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"[HISTORY] Appended {added} items → {history_file}")
    print(f"[HISTORY] Dates stored: {list(history.keys())[:5]} ...")


def print_history_summary(history_file: str = BRIEFS_HISTORY_FILE) -> None:
    """Pretty-print a summary table of briefs_history.json."""
    p = Path(history_file)
    if not p.exists():
        return

    try:
        history = json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return

    print("\n" + "═" * 64)
    print("  BRIEFS HISTORY SUMMARY")
    print("═" * 64)
    print(f"  {'Date':<14}  {'Modality':<28}  {'Items':>5}")
    print("  " + "─" * 52)

    for date in sorted(history.keys(), reverse=True):
        modalities = history[date]
        for modality, items in modalities.items():
            print(f"  {date:<14}  {modality:<28}  {len(items):>5}")

    total = sum(
        len(items)
        for day in history.values()
        for items in day.values()
    )
    print("═" * 64)
    print(f"  Total items stored : {total}")
    print(f"  Total dates        : {len(history)}")
    print("═" * 64 + "\n")

# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Pharma Intelligence Brief Generator")
    p.add_argument("--input",  "-i", default="text.json",   help="Extracted articles JSON")
    p.add_argument("--output", "-o", default="briefs.json", help="Output JSON file")
    p.add_argument("--query",  "-q", required=True,         help="Therapeutic focus e.g. 'PROTAC'")
    p.add_argument(
        "--no-history", action="store_true",
        help="Skip writing to briefs_history.json"
    )
    return p.parse_args()

# ── MAIN ──────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    # ── Load input ────────────────────────────────────────────────────────────
    src = Path(args.input)
    if not src.exists():
        sys.exit(f"[ERROR] Input file not found: {src}")

    try:
        with open(src, encoding="utf-8") as f:
            raw_data = json.load(f)
    except Exception as e:
        sys.exit(f"[ERROR] Failed to load JSON: {e}")

    if isinstance(raw_data, dict) and "articles" in raw_data:
        articles = raw_data["articles"]
    elif isinstance(raw_data, list):
        articles = raw_data
    else:
        sys.exit("[ERROR] Unexpected JSON format in input file.")

    valid_articles = [a for a in articles if (a.get("body") or "").strip()]
    skipped        = len(articles) - len(valid_articles)

    print(f"[INFO] Loaded   : {len(articles)} articles")
    print(f"[INFO] Valid    : {len(valid_articles)}")
    if skipped:
        print(f"[INFO] Skipped  : {skipped} (no body text)")
    print(f"[INFO] Query    : {args.query}")

    if not valid_articles:
        sys.exit("[WARN] No valid articles found.")

    # ── Build prompt ──────────────────────────────────────────────────────────
    prompt = build_prompt(valid_articles, args.query)
    if not prompt:
        sys.exit("[WARN] Prompt generation failed.")

    # ── Call LLM ─────────────────────────────────────────────────────────────
    print(f"[INFO] Sending {len(valid_articles)} articles to LLM...")
    try:
        raw_response = call_llm(prompt)
    except RuntimeError as e:
        sys.exit(f"[FATAL] {e}")

    # ── Parse + validate ──────────────────────────────────────────────────────
    raw_items            = parse_llm_response(raw_response)
    news_items, dropped  = validate_items(raw_items)

    print(f"[INFO] Raw items parsed   : {len(raw_items)}")
    print(f"[INFO] Passed validation  : {len(news_items)}")
    if dropped:
        print(f"[WARN] Dropped bad items  : {dropped}")

    # ── Write briefs.json (unchanged behaviour) ───────────────────────────────
    output_data = build_output(news_items, args.query, len(valid_articles), dropped)
    out_path    = Path(args.output)

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)

    print(f"[INFO] Saved → {out_path}")

    # ── Append to datewise history ← NEW ─────────────────────────────────────
    if not args.no_history and news_items:
        append_to_history(news_items, args.query)
        print_history_summary()

    # ── Preview ───────────────────────────────────────────────────────────────
    if news_items:
        print("\n── PREVIEW (first 3 items) ──────────────────────────────────")
        for item in news_items[:3]:
            print(f"  [{item.get('modality','?')}] {item.get('company','?')}")
            print(f"  {item.get('news','')[:110]}...")
        print("─────────────────────────────────────────────────────────────\n")


if __name__ == "__main__":
    main()
