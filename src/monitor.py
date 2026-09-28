"""Post-production monitoring metrics."""
from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error

def wasserstein_1d(a, b):
    """Exact 1-D empirical Wasserstein distance without a SciPy dependency."""
    a=np.sort(np.asarray(a, dtype=float)); b=np.sort(np.asarray(b, dtype=float))
    if len(a)==0 or len(b)==0: return float("nan")
    # quantile integration over the larger empirical grid
    q=np.linspace(0,1,max(len(a),len(b)))
    aq=np.quantile(a,q); bq=np.quantile(b,q)
    return float(np.mean(np.abs(aq-bq)))

def evaluate_predictions(actual, predicted):
    actual=np.asarray(actual,dtype=float); predicted=np.asarray(predicted,dtype=float)
    return {
        "mae": float(mean_absolute_error(actual,predicted)),
        "rmse": float(mean_squared_error(actual,predicted)**0.5),
        "wasserstein": wasserstein_1d(actual,predicted),
        "n": int(len(actual))
    }

def monitor_csv(actual_csv, prediction_csv):
    actual=pd.read_csv(actual_csv, parse_dates=["date"])
    pred=pd.read_csv(prediction_csv, parse_dates=["date"])
    merged=actual.merge(pred,on=["date","country"],suffixes=("_actual","_predicted")).dropna()
    return evaluate_predictions(merged.revenue, merged.prediction)
