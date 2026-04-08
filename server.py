"""
server.py — DrugParadigm Unified Intelligence Platform
Fetches data from all 4 source repos via GitHub raw URLs and serves unified dashboard.
"""

import json
import os
import urllib.request
import urllib.error
from pathlib import Path
from datetime import datetime, timedelta

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

# ─── CONFIG ───────────────────────────────────────────────────────────────────

GITHUB_USER = "CheenemoniSaiPraneeth"

STATIC_DIR    = Path("static")
FRONTEND_FILE = STATIC_DIR / "index.html"

MODALITIES = [
    "bispecific_antibodies",
    "monoclonal_antibodies",
    "gene_editing",
    "molecular_glues",
]

MODALITY_LABELS = {
    "bispecific_antibodies": "Bispecific Antibodies",
    "monoclonal_antibodies": "Monoclonal Antibodies",
    "gene_editing":          "Gene Editing",
    "molecular_glues":       "Molecular Glues",
}

MODALITY_KEYS_PHARMA = {
    "bispecific_antibodies": "bispecific antibodies",
    "monoclonal_antibodies": "monoclonal antibodies",
    "gene_editing":          "gene editing",
    "molecular_glues":       "molecular glues",
}

MODALITY_KEYS_MARKET = {
    "bispecific_antibodies": "Bispecific Antibodies",
    "monoclonal_antibodies": "Monoclonal Antibodies",
    "gene_editing":          "Gene Editing",
    "molecular_glues":       "Molecular Glue",
}

MODALITY_KEYS_NEWSAM = {
    "bispecific_antibodies": "bispecific_antibodies",
    "monoclonal_antibodies": "monoclonal_antibodies",
    "gene_editing":          "gene_editing",
    "molecular_glues":       "molecular_glues",
}

MODALITY_COLORS = {
    "bispecific_antibodies": "#4af4b0",
    "monoclonal_antibodies": "#6e8cff",
    "gene_editing":          "#f4a24a",
    "molecular_glues":       "#f46a6a",
}

# GitHub raw base URLs
PREPRINT_BASE = f"https://raw.githubusercontent.com/{GITHUB_USER}/preprint-dashboard/main"
PHARMA_BASE   = f"https://raw.githubusercontent.com/{GITHUB_USER}/pharma-dashboard/main"
NEWSAM_BASE   = f"https://raw.githubusercontent.com/{GITHUB_USER}/new-sam/main"
MARKET_BASE   = f"https://raw.githubusercontent.com/{GITHUB_USER}/market/main"

# ─── HELPERS ──────────────────────────────────────────────────────────────────

def fetch_json(url: str):
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "DrugParadigm/1.0"})
        with urllib.request.urlopen(req, timeout=15) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except Exception as e:
        print(f"[FETCH ERROR] {url}: {e}")
        return None


def get_latest_preprint_url():
    """Find most recent ranked_results_DATE.json from preprint repo."""
    today = datetime.today().date()
    for i in range(14):  # check last 14 days
        d = today - timedelta(days=i)
        url = f"{PREPRINT_BASE}/ranked_results_{d.isoformat()}.json"
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "DrugParadigm/1.0"})
            with urllib.request.urlopen(req, timeout=8) as resp:
                if resp.status == 200:
                    return url, d.isoformat()
        except:
            continue
    return None, None


# ─── APP ──────────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("✅ DrugParadigm server started")
    yield

app = FastAPI(title="DrugParadigm", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

# ─── API: MODALITIES OVERVIEW ─────────────────────────────────────────────────

@app.get("/api/modalities")
def get_modalities():
    pharma_data = fetch_json(f"{PHARMA_BASE}/briefs_history.json") or {}
    preprint_url, preprint_date = get_latest_preprint_url()
    preprint_data = fetch_json(preprint_url) if preprint_url else {}
    market_data = fetch_json(f"{MARKET_BASE}/briefs.json") or {}

    result = []
    for key in MODALITIES:
        pharma_key = MODALITY_KEYS_PHARMA[key]
        market_key = MODALITY_KEYS_MARKET[key]

        # Count pharma briefs
        pharma_items = []
        if isinstance(pharma_data, dict):
            for date_group in pharma_data.values():
                if isinstance(date_group, dict):
                    items = date_group.get(pharma_key, [])
                    pharma_items.extend(items)

        # Count preprints
        preprint_items = []
        if preprint_data and isinstance(preprint_data, dict):
            preprint_items = preprint_data.get(key, [])

        # Count market items
        market_items = []
        mi = market_data.get("modality_intelligence", [])
        for m in mi:
            if m.get("modality_name", "").lower() == market_key.lower():
                market_items = [m]
                break

        result.append({
            "key":            key,
            "label":          MODALITY_LABELS[key],
            "color":          MODALITY_COLORS[key],
            "pharma_count":   len(pharma_items),
            "preprint_count": len(preprint_items),
            "preprint_date":  preprint_date,
            "market_ready":   len(market_items) > 0,
        })

    return result


# ─── API: COMPANY NEWS (pharma-dashboard) ────────────────────────────────────

@app.get("/api/modality/{modality}/company-news")
def get_company_news(modality: str):
    if modality not in MODALITIES:
        raise HTTPException(404, "Unknown modality")

    pharma_key = MODALITY_KEYS_PHARMA[modality]

    # briefs_history.json has the correct {date: {modality: [{company, news, url}]}} format
    raw = fetch_json(f"{PHARMA_BASE}/briefs_history.json")
    if not raw:
        return {"modality": modality, "items": []}

    items = []
    if isinstance(raw, dict):
        for date_key in sorted(raw.keys(), reverse=True):
            date_group = raw[date_key]
            if isinstance(date_group, dict):
                entries = date_group.get(pharma_key, [])
                for e in entries:
                    items.append({
                        "company":  e.get("company", ""),
                        "modality": e.get("modality", pharma_key),
                        "news":     e.get("news", ""),
                        "url":      e.get("url", ""),
                        "date":     date_key,
                    })

    return {"modality": modality, "label": MODALITY_LABELS[modality], "items": items}


# ─── API: WEBSITE NEWS (new-sam) ─────────────────────────────────────────────

@app.get("/api/modality/{modality}/website-news")
def get_website_news(modality: str):
    if modality not in MODALITIES:
        raise HTTPException(404, "Unknown modality")

    raw = fetch_json(f"{NEWSAM_BASE}/allinone.json")
    if not raw:
        return {"modality": modality, "items": []}

    items = []
    # allinone.json structure: {"articles": [...], ...} or list
    articles = []
    if isinstance(raw, dict):
        articles = raw.get("articles", [])
    elif isinstance(raw, list):
        articles = raw

    newsam_key = MODALITY_KEYS_NEWSAM[modality]
    pharma_key = MODALITY_KEYS_PHARMA[modality]  # "bispecific antibodies"

    for a in articles:
        tag = a.get("tag", "").lower()
        title = a.get("title", "")
        text = a.get("text", "")[:400]  # truncate for display
        url = a.get("url", "")
        date = a.get("date", "")

        # Match by tag or keyword in title
        label_lower = MODALITY_LABELS[modality].lower()
        if (tag == "news" or tag == modality or
            any(k in (title + text).lower() for k in pharma_key.split()[:2])):
            items.append({
                "title": title,
                "text":  text,
                "url":   url,
                "date":  date,
                "tag":   tag,
            })

    return {"modality": modality, "label": MODALITY_LABELS[modality], "items": items[:30]}


# ─── API: PREPRINTS (preprint-dashboard) ─────────────────────────────────────

@app.get("/api/modality/{modality}/preprints")
def get_preprints(modality: str):
    if modality not in MODALITIES:
        raise HTTPException(404, "Unknown modality")

    url, date = get_latest_preprint_url()
    if not url:
        return {"modality": modality, "items": [], "date": None}

    raw = fetch_json(url)
    if not raw:
        return {"modality": modality, "items": [], "date": date}

    articles = raw.get(modality, [])
    items = []
    for a in articles:
        kws = a.get("primary_abstract_matched_keywords", [])
        if isinstance(kws, str):
            try: kws = json.loads(kws)
            except: kws = []
        items.append({
            "abstract":  a.get("abstract", "")[:300],
            "url":       a.get("url", ""),
            "date":      a.get("date", ""),
            "website":   a.get("website", ""),
            "score":     a.get("score", 0),
            "keywords":  kws[:4],
        })

    return {"modality": modality, "label": MODALITY_LABELS[modality], "items": items, "date": date}


# ─── API: MARKET (market repo) ───────────────────────────────────────────────

@app.get("/api/modality/{modality}/market")
def get_market(modality: str):
    if modality not in MODALITIES:
        raise HTTPException(404, "Unknown modality")

    market_key = MODALITY_KEYS_MARKET[modality]
    raw = fetch_json(f"{MARKET_BASE}/briefs.json")
    if not raw:
        return {"modality": modality, "data": None}

    meta = raw.get("meta", {})
    mi = raw.get("modality_intelligence", [])

    for m in mi:
        if m.get("modality_name", "").lower() == market_key.lower():
            return {
                "modality": modality,
                "label":    MODALITY_LABELS[modality],
                "meta":     meta,
                "data":     m,
            }

    return {"modality": modality, "data": None, "meta": meta}


# ─── API: MARKET GRAPH ───────────────────────────────────────────────────────

@app.get("/api/modality/{modality}/graph")
def get_graph(modality: str):
    if modality not in MODALITIES:
        raise HTTPException(404, "Unknown modality")
    raw = fetch_json(f"{MARKET_BASE}/briefs.json")
    if not raw:
        return {"nodes": [], "edges": []}
    # Return full data so frontend can build graph
    return raw


# ─── SPA CATCH-ALL ───────────────────────────────────────────────────────────

@app.get("/{full_path:path}")
async def serve_spa(full_path: str):
    if FRONTEND_FILE.exists():
        return FileResponse(str(FRONTEND_FILE))
    return HTMLResponse("<h1>Frontend not found</h1>", 503)
