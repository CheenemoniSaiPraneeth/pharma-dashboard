import asyncio
import json
import re
import random
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup
from playwright.async_api import async_playwright


INPUT_FILE = "companies.json"
OUTPUT_FILE = "total_list.json"

HEADLESS = True
TIMEOUT = 60000
MAX_LINKS_PER_SITE = 300
CONCURRENT_WORKERS = 8

USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)",
    "Mozilla/5.0 (X11; Linux x86_64)"
]


# ======================================================
# LOAD
# ======================================================

def load_companies():
    print(f"[LOAD] Reading input file: {INPUT_FILE}")
    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)

    companies = []
    for company, details in data.items():
        if isinstance(details, dict) and "press_url" in details:
            companies.append({
                "company": company,
                "press_url": details["press_url"]
            })

    print(f"[LOAD] Total companies loaded: {len(companies)}")
    return companies


# ======================================================
# SCORING
# ======================================================

def score_url(url):
    score = 0
    path = urlparse(url).path

    score += path.count("/") * 2
    score += len(path)

    if re.search(r"/20\d{2}/", path):
        score += 25

    if "-" in path:
        score += 10

    return score


def filter_links(base_url, links):
    print(f"[FILTER] Filtering {len(links)} raw links from: {base_url}")
    base_domain = urlparse(base_url).netloc
    clean = {}

    for url, text in links:
        if not url:
            continue

        if url.lower().endswith(
            (".pdf", ".doc", ".docx", ".png", ".jpg", ".zip")
        ):
            continue

        if urlparse(url).netloc != base_domain:
            continue

        clean[url] = {
            "url": url,
            "anchor_text": text,
            "score": score_url(url)
        }

    results = list(clean.values())
    results.sort(key=lambda x: x["score"], reverse=True)

    print(f"[FILTER] Kept {len(results[:MAX_LINKS_PER_SITE])} candidate links for: {base_url}")
    return results[:MAX_LINKS_PER_SITE]


# ======================================================
# PLAYWRIGHT HARVEST
# ======================================================

async def playwright_harvest(context, company, url, wait_mode="domcontentloaded"):

    print(f"[PLAYWRIGHT] START | Company: {company} | URL: {url} | wait_mode: {wait_mode}")
    collected_links = []
    page = await context.new_page()

    try:
        print(f"[PLAYWRIGHT] GOTO | Company: {company}")
        await page.goto(url, timeout=TIMEOUT, wait_until=wait_mode)
        print(f"[PLAYWRIGHT] LOADED | Company: {company}")

        await page.wait_for_timeout(3000)
        print(f"[PLAYWRIGHT] WAITED 3s | Company: {company}")

        for i in range(4):
            print(f"[PLAYWRIGHT] SCROLL {i+1}/4 | Company: {company}")
            await page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
            await page.wait_for_timeout(1200)

        print(f"[PLAYWRIGHT] EXTRACT HTML | Company: {company}")
        html = await page.content()
        soup = BeautifulSoup(html, "lxml")

        for a in soup.find_all("a", href=True):
            full = urljoin(url, a["href"])
            text = a.get_text(strip=True)
            collected_links.append((full, text))

        print(f"[PLAYWRIGHT] RAW LINKS FOUND: {len(collected_links)} | Company: {company}")

        await page.close()
        print(f"[PLAYWRIGHT] PAGE CLOSED | Company: {company}")

        filtered = filter_links(url, collected_links)
        print(f"[PLAYWRIGHT] FINAL LINKS: {len(filtered)} | Company: {company}")

        return filtered

    except Exception as e:
        print(f"[PLAYWRIGHT] ERROR | Company: {company} | URL: {url} | Error: {e}")
        await page.close()
        print(f"[PLAYWRIGHT] PAGE CLOSED AFTER ERROR | Company: {company}")
        raise e


# ======================================================
# STATIC FALLBACK
# ======================================================

def static_harvest(url):
    print(f"[STATIC] START | URL: {url}")
    try:
        r = requests.get(url, headers={"User-Agent": random.choice(USER_AGENTS)}, timeout=20)
        print(f"[STATIC] STATUS: {r.status_code} | URL: {url}")

        soup = BeautifulSoup(r.text, "lxml")

        collected_links = []
        for a in soup.find_all("a", href=True):
            full = urljoin(url, a["href"])
            text = a.get_text(strip=True)
            collected_links.append((full, text))

        print(f"[STATIC] RAW LINKS FOUND: {len(collected_links)} | URL: {url}")
        filtered = filter_links(url, collected_links)
        print(f"[STATIC] FINAL LINKS: {len(filtered)} | URL: {url}")

        return filtered
    except Exception as e:
        print(f"[STATIC] ERROR | URL: {url} | Error: {e}")
        return []


# ======================================================
# MAIN ADAPTIVE PIPELINE
# ======================================================

async def main():

    print("[MAIN] Starting adaptive harvesting pipeline")
    companies = load_companies()

    parallel_success = []
    retry_queue = []
    final_results = []

    semaphore = asyncio.Semaphore(CONCURRENT_WORKERS)

    async with async_playwright() as p:

        print("[MAIN] Launching browser")
        browser = await p.chromium.launch(headless=HEADLESS)

        # ==============================
        # PHASE 1: PARALLEL
        # ==============================

        async def worker(entry):
            async with semaphore:
                print(f"[PHASE 1] START | {entry['company']} | {entry['press_url']}")
                context = await browser.new_context(
                    user_agent=random.choice(USER_AGENTS)
                )
                try:
                    links = await playwright_harvest(
                        context,
                        entry["company"],
                        entry["press_url"],
                        wait_mode="domcontentloaded"
                    )

                    await context.close()
                    print(f"[PHASE 1] CONTEXT CLOSED | {entry['company']}")

                    if links:
                        print(f"[PHASE 1] SUCCESS | {entry['company']} | Links: {len(links)}")
                        return {"status": "success", "data": links}
                    else:
                        print(f"[PHASE 1] EMPTY -> RETRY | {entry['company']}")
                        return {"status": "retry"}

                except Exception as e:
                    await context.close()
                    print(f"[PHASE 1] ERROR -> RETRY | {entry['company']} | Error: {e}")
                    print(f"[PHASE 1] CONTEXT CLOSED AFTER ERROR | {entry['company']}")
                    return {"status": "retry"}

        print(f"[PHASE 1] Creating tasks for {len(companies)} companies")
        tasks = [worker(entry) for entry in companies]
        results = await asyncio.gather(*tasks)
        print("[PHASE 1] All parallel tasks completed")

        # Split results
        for entry, result in zip(companies, results):
            if result["status"] == "success":
                final_results.append({
                    "company": entry["company"],
                    "press_url": entry["press_url"],
                    "candidate_links": result["data"]
                })
                print(f"[RESULT] ADDED SUCCESS | {entry['company']}")
            else:
                retry_queue.append(entry)
                print(f"[RESULT] ADDED TO RETRY QUEUE | {entry['company']}")

        print(f"[RESULT] Phase 1 success count: {len(final_results)}")
        print(f"[RESULT] Retry queue count: {len(retry_queue)}")

        # ==============================
        # PHASE 2: SEQUENTIAL RETRY
        # ==============================

        print("\n[PHASE 2] Retrying failed sites sequentially...\n")

        for idx, entry in enumerate(retry_queue, 1):
            print(f"[PHASE 2] RETRY {idx}/{len(retry_queue)} | {entry['company']} | {entry['press_url']}")

            context = await browser.new_context(
                user_agent=random.choice(USER_AGENTS)
            )

            try:
                links = await playwright_harvest(
                    context,
                    entry["company"],
                    entry["press_url"],
                    wait_mode="networkidle"
                )

                await context.close()
                print(f"[PHASE 2] CONTEXT CLOSED | {entry['company']}")

                if links:
                    final_results.append({
                        "company": entry["company"],
                        "press_url": entry["press_url"],
                        "candidate_links": links
                    })
                    print(f"[PHASE 2] SUCCESS | {entry['company']} | Links: {len(links)}")
                else:
                    print(f"[PHASE 2] EMPTY PLAYWRIGHT -> STATIC FALLBACK | {entry['company']}")
                    links = static_harvest(entry["press_url"])
                    final_results.append({
                        "company": entry["company"],
                        "press_url": entry["press_url"],
                        "candidate_links": links
                    })
                    print(f"[PHASE 2] STATIC SAVED | {entry['company']} | Links: {len(links)}")

            except Exception as e:
                print(f"[PHASE 2] ERROR | {entry['company']} | Error: {e}")
                await context.close()
                print(f"[PHASE 2] CONTEXT CLOSED AFTER ERROR | {entry['company']}")
                links = static_harvest(entry["press_url"])
                final_results.append({
                    "company": entry["company"],
                    "press_url": entry["press_url"],
                    "candidate_links": links
                })
                print(f"[PHASE 2] STATIC SAVED AFTER ERROR | {entry['company']} | Links: {len(links)}")

        print("[MAIN] Closing browser")
        await browser.close()

    print(f"[MAIN] Writing output to {OUTPUT_FILE}")
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(final_results, f, indent=2, ensure_ascii=False)

    print("\nAdaptive harvesting complete.")
    print("Saved to:", OUTPUT_FILE)
    print(f"[MAIN] Total final results: {len(final_results)}")


if __name__ == "__main__":
    asyncio.run(main())