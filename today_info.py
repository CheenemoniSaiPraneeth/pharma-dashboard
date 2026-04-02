import json
import os
import re
from datetime import datetime

INPUT_FILE = "total_list.json"
MASTER_FILE = "master_storage.json"
TODAY_FILE = "today.json"

TODAY_DATE = datetime.now().strftime("%Y-%m-%d")

# ---------------------------------
# ARTICLE STRUCTURE CHECK
# ---------------------------------

def is_article_url(url):
    url_lower = url.lower()

    # Reject obvious pagination patterns
    if "?page=" in url_lower:
        return False

    # Reject URLs ending with page numbers like /18 or /0 or /1
    if re.search(r"/\d+/?$", url_lower):
        return False

    # Reject PDF links
    if url_lower.endswith(".pdf"):
        return False

    # Reject index-style URLs
    bad_patterns = [
        "/press-releases",
        "/news",
        "/events",
        "/about",
        "/esg",
        "/announcements",
        "/presentations"
    ]

    # If URL ends exactly with these (no slug), reject
    for pattern in bad_patterns:
        if url_lower.rstrip("/").endswith(pattern):
            return False

    # Require decent slug length
    last_part = url_lower.rstrip("/").split("/")[-1]
    if len(last_part) < 15:
        return False

    return True

# ---------------------------------
# LOAD HARVESTED
# ---------------------------------

with open(INPUT_FILE, "r", encoding="utf-8") as f:
    harvested = json.load(f)

# ---------------------------------
# LOAD MASTER
# ---------------------------------

if os.path.exists(MASTER_FILE):
    with open(MASTER_FILE, "r", encoding="utf-8") as f:
        master_data = json.load(f)
else:
    master_data = []

master_urls = {item["url"] for item in master_data}

today_new = []

# ---------------------------------
# PROCESS
# ---------------------------------

for company_block in harvested:

    company = company_block.get("company")

    for link in company_block.get("candidate_links", []):

        url = link.get("url")
        score = link.get("score")

        if not url:
            continue

        if url in master_urls:
            continue

        article_flag = is_article_url(url)

        record = {
            "company": company,
            "url": url,
            "score": score,
            "date_added": TODAY_DATE,
            "article_like": article_flag
        }

        # Always store in master
        master_data.append(record)
        master_urls.add(url)

        # Only article-like go to today's delta
        if article_flag:
            today_new.append(record)

# ---------------------------------
# SAVE FILES
# ---------------------------------

with open(MASTER_FILE, "w", encoding="utf-8") as f:
    json.dump(master_data, f, indent=4)

with open(TODAY_FILE, "w", encoding="utf-8") as f:
    json.dump(today_new, f, indent=4)

print("\nDelta Engine Complete")
print("New total URLs added:", len(today_new))
print("Total master URLs:", len(master_data))