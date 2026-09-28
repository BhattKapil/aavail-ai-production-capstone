"""Automated ingestion and normalization for AAVAIL-style transaction JSON files."""
from __future__ import annotations
import json
from pathlib import Path
from typing import Iterable
import pandas as pd

COLUMN_ALIASES = {
    "invoice": ["invoice", "Invoice", "InvoiceNo", "Invoice Number"],
    "stock_code": ["StockCode", "stock_code", "Stock Code"],
    "description": ["Description", "description"],
    "quantity": ["Quantity", "quantity", "Qty"],
    "invoice_date": ["InvoiceDate", "invoice_date", "Invoice Date"],
    "price": ["Price", "price", "UnitPrice", "unit_price"],
    "customer_id": ["Customer ID", "CustomerID", "customer_id"],
    "country": ["Country", "country"],
}

def _first_present(row: dict, aliases: Iterable[str]):
    for key in aliases:
        if key in row:
            return row[key]
    return None

def read_json_records(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    if isinstance(payload, dict):
        payload = payload.get("data", [payload])
    if not isinstance(payload, list):
        raise ValueError(f"{path} does not contain a JSON list of records")
    return payload

def ingest_directory(input_dir: str | Path, output_csv: str | Path | None = None) -> pd.DataFrame:
    """Read all JSON sources, normalize field names, validate values and return transactions."""
    input_dir = Path(input_dir)
    files = sorted(input_dir.glob("*.json"))
    if not files:
        raise FileNotFoundError(f"No JSON files found in {input_dir}")

    normalized = []
    for path in files:
        for row in read_json_records(path):
            item = {canonical: _first_present(row, aliases)
                    for canonical, aliases in COLUMN_ALIASES.items()}
            normalized.append(item)

    df = pd.DataFrame(normalized)
    required = ["invoice", "quantity", "invoice_date", "price", "country"]
    missing = [c for c in required if c not in df or df[c].isna().all()]
    if missing:
        raise ValueError(f"Missing required fields: {missing}")

    df["invoice"] = df["invoice"].astype(str).str.replace(r"\D", "", regex=True)
    df["quantity"] = pd.to_numeric(df["quantity"], errors="coerce")
    df["price"] = pd.to_numeric(df["price"], errors="coerce")
    df["invoice_date"] = pd.to_datetime(df["invoice_date"], errors="coerce")
    df["country"] = df["country"].astype("string").str.strip()
    df = df.dropna(subset=["invoice_date", "quantity", "price", "country"])
    df = df[df["quantity"] != 0].copy()
    df["revenue"] = df["quantity"] * df["price"]
    df = df.sort_values("invoice_date").reset_index(drop=True)

    if output_csv:
        output_csv = Path(output_csv)
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_csv, index=False)
    return df

def aggregate_daily(df: pd.DataFrame, output_csv: str | Path | None = None) -> pd.DataFrame:
    """Aggregate transaction revenue by date and country."""
    daily = (df.assign(date=df["invoice_date"].dt.normalize())
             .groupby(["date", "country"], as_index=False)["revenue"].sum()
             .sort_values(["country", "date"]))
    if output_csv:
        output_csv = Path(output_csv)
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        daily.to_csv(output_csv, index=False)
    return daily

if __name__ == "__main__":
    base = Path(__file__).resolve().parents[1]
    tx = ingest_directory(base / "data/input", base / "data/processed/transactions.csv")
    aggregate_daily(tx, base / "data/processed/daily_revenue.csv")
    print(f"Ingested {len(tx):,} transactions.")
