import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
"""Generate EDA visualizations required by the capstone."""
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
base=Path(__file__).resolve().parents[1]
out=base/"reports"; out.mkdir(exist_ok=True)
df=pd.read_csv(base/"data/processed/daily_revenue.csv",parse_dates=["date"])
# Overall time series
overall=df.groupby("date",as_index=False).revenue.sum()
plt.figure(figsize=(12,5)); plt.plot(overall.date,overall.revenue); plt.title("AAVAIL Daily Revenue"); plt.xlabel("Date"); plt.ylabel("Revenue"); plt.tight_layout(); plt.savefig(out/"eda_daily_revenue.png",dpi=160); plt.close()
# Country comparison
pivot=df.pivot(index="date",columns="country",values="revenue")
plt.figure(figsize=(12,5)); pivot.plot(ax=plt.gca()); plt.title("Daily Revenue by Country"); plt.xlabel("Date"); plt.ylabel("Revenue"); plt.tight_layout(); plt.savefig(out/"eda_country_revenue.png",dpi=160); plt.close()
# Monthly seasonality
monthly=df.assign(month=df.date.dt.month).groupby(["month","country"],as_index=False).revenue.mean()
plt.figure(figsize=(10,5))
for c,g in monthly.groupby("country"): plt.plot(g.month,g.revenue,marker="o",label=c)
plt.title("Average Revenue by Month"); plt.xlabel("Month"); plt.ylabel("Average daily revenue"); plt.legend(); plt.tight_layout(); plt.savefig(out/"eda_monthly_seasonality.png",dpi=160); plt.close()
print("EDA visualizations written to reports/.")
