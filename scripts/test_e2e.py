import threading
import time
import requests
import uvicorn
import sys
import os
sys.path.insert(0, '.')

from backend.main import app

def run_server():
    uvicorn.run(app, host='127.0.0.1', port=7860, log_level='error')

t = threading.Thread(target=run_server, daemon=True)
t.start()
time.sleep(3) # Wait for startup + model load

print("Testing /health/ready...")
try:
    r = requests.get('http://127.0.0.1:7860/health/ready')
    print("Status:", r.status_code, r.json())
except Exception as e:
    print("Error:", e)
    sys.exit(1)

print("\nTesting /api/upload with real file...")
test_file = "dummy.tif"
if not os.path.exists(test_file):
    print("Test file not found!")
    sys.exit(1)

with open(test_file, 'rb') as f:
    r = requests.post('http://127.0.0.1:7860/api/upload', files={'file': ('test.tif', f, 'image/tiff')})
    
print("Status:", r.status_code, r.text)
if r.status_code != 200:
    sys.exit(1)
    
job_id = r.json()['job_id']
print(f"Job ID: {job_id}")

print("\nPolling job status...")
while True:
    r = requests.get(f'http://127.0.0.1:7860/api/jobs/{job_id}')
    status = r.json().get('status')
    pct = r.json().get('progress_pct')
    print(f"Status: {status} ({pct}%)")
    if status in ['COMPLETED', 'FAILED']:
        break
    time.sleep(2)

if status == 'COMPLETED':
    print("\nFetching result...")
    r = requests.get(f'http://127.0.0.1:7860/api/jobs/{job_id}/result')
    print("Result Status:", r.status_code)
    if r.status_code == 200:
        with open(f"test_out_{job_id}.tif", 'wb') as f:
            f.write(r.content)
        print("Saved result!")
else:
    print(f"Job Failed: {r.json().get('error_message')}")
