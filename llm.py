import asyncio
import json
import random
from datetime import datetime
from bs4 import BeautifulSoup
import aiohttp
from playwright.async_api import async_playwright

# ==============================
# CONFIG
# ==============================

INPUT_FILE = "today.json"
OUTPUT_FILE = "text.json"

CONCURRENCY = 12
HEADLESS = True

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"
}

TODAY_DATE = datetime.today().strftime("%Y-%m-%d")

# ==============================
# ACCESS DENIED CHECK
# ==============================

def is_access_denied(text):
    if not text:
        return True

    patterns = ["Access Denied", "Request blocked", "akamai"]
    text_lower = text.lower()
    return any(p.lower() in text_lower for p in patterns)

# ==============================
# TEXT + TITLE EXTRACTION
# ==============================

def extract_content(html, url):
    soup = BeautifulSoup(html, "html.parser")

    # remove junk
    for tag in soup(["script", "style", "noscript", "header", "footer", "nav"]):
        tag.decompose()

    # title
    title = None
    if soup.title:
        title = soup.title.get_text(strip=True)

    # body text
    article = soup.find("article")
    if article:
        body = article.get_text(" ", strip=True)
    else:
        body = soup.get_text(" ", strip=True)

    if not body or len(body) < 300:
        return None

    return {
        "title": title,
        "url": url,
        "date": TODAY_DATE,
        "body": body
    }

# ==============================
# FETCHERS
# ==============================

async def fetch_aiohttp(session, url):
    try:
        async with session.get(url, timeout=aiohttp.ClientTimeout(total=20)) as r:
            if r.status == 200:
                return await r.text()
    except:
        return None

async def fetch_playwright(browser, url):
    context = await browser.new_context(user_agent=HEADERS["User-Agent"])
    page = await context.new_page()

    try:
        await page.goto(url, timeout=60000)
        await page.wait_for_timeout(4000)
        content = await page.content()
        await context.close()
        return content
    except:
        await context.close()
        return None

# ==============================
# PROCESS LINK
# ==============================

async def process_link(browser, session, company, url, semaphore):
    async with semaphore:
        print(f"\nProcessing: {company}")
        print("URL:", url)

        html = await fetch_aiohttp(session, url)

        if not html or is_access_denied(html):
            html = await fetch_playwright(browser, url)

        if not html or is_access_denied(html):
            return None

        result = extract_content(html, url)

        await asyncio.sleep(1 + random.uniform(0, 1))  # light throttle

        return result

# ==============================
# BUILD TASKS
# ==============================

def build_tasks(data, browser, session, semaphore):
    tasks = []

    for item in data:
        company = item.get("company")

        if "candidate_links" in item:
            for link in item.get("candidate_links", []):
                url = link.get("url")
                if url:
                    tasks.append(
                        process_link(browser, session, company, url, semaphore)
                    )

        elif "url" in item:
            tasks.append(
                process_link(browser, session, company, item["url"], semaphore)
            )

    return tasks

# ==============================
# MAIN
# ==============================

async def main():
    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)

    semaphore = asyncio.Semaphore(CONCURRENCY)
    results = []

    async with aiohttp.ClientSession(headers=HEADERS) as session:
        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=HEADLESS)

            tasks = build_tasks(data, browser, session, semaphore)
            responses = await asyncio.gather(*tasks)

            for r in responses:
                if r:
                    results.append(r)

            await browser.close()

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print("\nDone.")
    print("Saved:", OUTPUT_FILE)
    print("Total extracted:", len(results))


if __name__ == "__main__":
    asyncio.run(main())

