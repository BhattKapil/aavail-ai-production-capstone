"""Application logging kept separate from model artifacts and test logs."""
from __future__ import annotations
import json
from datetime import datetime, timezone
from pathlib import Path

def write_prediction_log(log_dir, payload):
    path=Path(log_dir); path.mkdir(parents=True, exist_ok=True)
    record={"timestamp":datetime.now(timezone.utc).isoformat(), **payload}
    with (path/"predictions.jsonl").open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, default=str)+"\n")
    return record

def read_prediction_logs(log_dir):
    file=Path(log_dir)/"predictions.jsonl"
    if not file.exists():
        return []
    return [json.loads(line) for line in file.read_text(encoding="utf-8").splitlines() if line.strip()]
