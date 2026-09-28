import json
from pathlib import Path
from src.ingest import ingest_directory, aggregate_daily

def test_ingestion_normalizes_multiple_schemas(tmp_path):
    p=tmp_path/"input"; p.mkdir()
    rows=[{"invoice":"INV1","Quantity":2,"InvoiceDate":"2024-01-01","Price":10,"Country":"Australia"},
          {"InvoiceNo":"INV2","quantity":1,"Invoice Date":"2024-01-02","UnitPrice":12,"country":"Canada"}]
    (p/"a.json").write_text(json.dumps(rows))
    df=ingest_directory(p)
    assert len(df)==2 and set(["invoice","quantity","price","country","revenue"]).issubset(df.columns)

def test_daily_aggregation():
    import pandas as pd
    df=pd.DataFrame({"invoice_date":pd.to_datetime(["2024-01-01","2024-01-01"]),
                     "country":["Australia","Australia"],"revenue":[10,15]})
    out=aggregate_daily(df)
    assert out.iloc[0].revenue==25
