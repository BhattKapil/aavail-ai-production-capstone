import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
"""Create a post-production monitoring comparison on a held-out window."""
from pathlib import Path
import pandas as pd
from src.model import train_country_model, recursive_forecast
from src.monitor import evaluate_predictions
base=Path(__file__).resolve().parents[1]
daily=pd.read_csv(base/"data/processed/daily_revenue.csv",parse_dates=["date"])
rows=[]
for country,g in daily.groupby("country"):
    s=g.set_index("date")["revenue"].asfreq("D",fill_value=0)
    train=s.iloc[:-30]; actual=s.iloc[-30:]
    # Use the deployed model family selected by metadata.
    import json, joblib
    meta=json.loads((base/"models/metadata.json").read_text())
    model=train_country_model(train,meta["selected_model"])
    pred=recursive_forecast(model,train,30)
    m=evaluate_predictions(actual,pred)
    rows.append({"country":country,**m})
out=pd.DataFrame(rows)
out.to_csv(base/"reports/post_production_metrics.csv",index=False)
print(out)
