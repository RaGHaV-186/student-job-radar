import time
import requests
from bs4 import BeautifulSoup
import re
from datetime import datetime
import json
from pathlib import Path

BASE_URL = "https://arbeitnow.com/api/job-board-api"
NUM_PAGES = 3
FOOTER = "Find more English Speaking Jobs in Germany on Arbeitnow"
DATA_PATH = Path(__file__).parent.parent / "data" / "postings.json"

all_jobs = []

def cleanhtml(raw_html):
    soup = BeautifulSoup(raw_html,"lxml")
    text = soup.get_text(separator=" ")
    text = re.sub(r"\s+", " ", text).strip()
    text = text.replace(FOOTER, "").strip()
    return text

def build_record(job, index):
    return {
        "id": index,
        "slug": job["slug"],
        "title": job["title"],
        "company": job["company_name"],
        "description": cleanhtml(job["description"]),
        "tags": " ".join(job["tags"]),
        "location": job["location"],
        "remote": job["remote"],
        "url": job["url"],
        "posted_at": datetime.fromtimestamp(job["created_at"]).strftime("%Y-%m-%d"),
        "date_collected": datetime.now().strftime("%Y-%m-%d"),
    }

for page in range(1,NUM_PAGES+1):
    response = requests.get(BASE_URL,params={"page":page})

    if response.status_code != 200:
        print(f"Page {page} failed with status {response.status_code}")
        break

    jobs = response.json()["data"]
    all_jobs.extend(jobs)

    print(f"Page {page}: fetched {len(jobs)} jobs (total so far: {len(all_jobs)})")

    time.sleep(1)

print(f"\nDone. Collected {len(all_jobs)} jobs.")

records = [build_record(job, i) for i, job in enumerate(all_jobs)]

with open(DATA_PATH, "w", encoding="utf-8") as f:
    json.dump(records, f, ensure_ascii=False, indent=2)

print(f"Saved {len(records)} records to data/postings.json")