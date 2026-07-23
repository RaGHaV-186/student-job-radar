import json
import shutil
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "postings.json"
BACKUP_PATH = PROJECT_ROOT / "data" / "postings_raw.json"

FOOTER = "Find Jobs in Germany on Arbeitnow"

if not BACKUP_PATH.exists():
    shutil.copy(DATA_PATH, BACKUP_PATH)
    print(f"Backed up original to {BACKUP_PATH.name}")

with open(DATA_PATH, encoding="utf-8") as f:
    postings = json.load(f)

affected = 0
for posting in postings:
    description = posting["description"]
    if FOOTER in description:
        posting["description"] = description.replace(FOOTER, "").strip()
        affected += 1

with open(DATA_PATH, "w", encoding="utf-8") as f:
    json.dump(postings, f, ensure_ascii=False, indent=2)

print(f"Removed footer from {affected} of {len(postings)} postings")