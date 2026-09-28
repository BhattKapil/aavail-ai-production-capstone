import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pathlib import Path
from src.ingest import ingest_directory, aggregate_daily
base=Path(__file__).resolve().parents[1]
tx=ingest_directory(base/"data/input", base/"data/processed/transactions.csv")
aggregate_daily(tx, base/"data/processed/daily_revenue.csv")
print("Data ingestion complete.")
