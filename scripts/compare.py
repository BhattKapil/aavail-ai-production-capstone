import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
base=Path(__file__).resolve().parents[1]
r=pd.read_csv(base/"models/model_comparison.csv")
summary=r.groupby("model",as_index=False)["mae"].mean().sort_values("mae")
print(summary)
plt.figure(figsize=(9,5))
plt.bar(summary.model,summary.mae)
plt.title("Model comparison by mean validation MAE")
plt.xlabel("Model"); plt.ylabel("Mean MAE")
plt.tight_layout(); plt.savefig(base/"reports/model_comparison.png",dpi=160); plt.close()
