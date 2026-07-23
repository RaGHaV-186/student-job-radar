import requests
import json

URL = "https://arbeitnow.com/api/job-board-api"

response = requests.get(URL)
print("Status code:", response.status_code)

data = response.json()

print("Type:", type(data))
if isinstance(data, dict):
    print("Top-level keys:", list(data.keys()))

jobs = data["data"] if isinstance(data, dict) else data
print("Number of jobs returned:", len(jobs))
first_job = jobs[0]
print("\nFields on one job:", list(first_job.keys()))
print("\nFull first job:")
print(json.dumps(first_job, indent=2))

print(json.dumps(data["links"], indent=2))
print(json.dumps(data["meta"], indent=2))
