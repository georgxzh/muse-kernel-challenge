"""Read an existing OKC job until completion; never starts a check or submission."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time

root = Path(__file__).resolve().parents[1]
job, filename = sys.argv[1:3]
last = None
while True:
    try:
        p = subprocess.run(["/home/georg/.okc/bin/okc", "status", job, "--json"],
                           capture_output=True, text=True, timeout=45, cwd=root,
                           env=os.environ | {"GODEBUG": "http2client=0"})
        if p.returncode:
            raise RuntimeError((p.stderr or p.stdout).strip()[:300])
        data = json.loads(p.stdout)
        state = data.get("status")
        if state != last:
            print(job, state, flush=True)
            last = state
        if data.get("result") is not None:
            (root / filename).write_text(json.dumps(data, indent=2) + "\n")
            result = data["result"]
            print(json.dumps({k: result.get(k) for k in
                  ("passed", "status", "error", "speedup", "speedup_ci", "flags")}), flush=True)
            for rec in result.get("records", []):
                if rec.get("status") == "fail":
                    print(json.dumps(rec), flush=True)
            break
    except (RuntimeError, subprocess.TimeoutExpired, json.JSONDecodeError) as exc:
        print("Status read retry:", str(exc)[:300], flush=True)
    time.sleep(30)
