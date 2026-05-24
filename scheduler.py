"""
scheduler.py
Auto-refreshes sentiment data every 6 hours.
Run:  python3 scheduler.py
Requires: pip install schedule
"""
import schedule, time, subprocess, sys
from datetime import datetime

def run_fetch():
    print(f"\n[{datetime.now().strftime('%Y-%m-%d %H:%M')}] Running fetch_news.py...")
    result = subprocess.run([sys.executable, "fetch_news.py"], capture_output=True, text=True)
    print(result.stdout)
    if result.returncode != 0:
        print("Error:", result.stderr[:200])

schedule.every(1).hours.do(run_fetch)
print("Scheduler running — fetching every 6 hours. Press Ctrl+C to stop.")
run_fetch()  # run immediately on start
while True:
    schedule.run_pending()
    time.sleep(60)
