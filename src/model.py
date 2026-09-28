"""Forecasting models, comparison and artifact management."""
from __future__ import annotations
from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

LAGS = [1, 7, 14, 28]
ROLLS = [7, 30]

def make_features(series: pd.Series) -> pd.DataFrame:
    s = series.sort_index().astype(float)
    x = pd.DataFrame(index=s.index)
    for lag in LAGS:
        x[f"lag_{lag}"] = s.shift(lag)
    for window in ROLLS:
        x[f"roll_mean_{window}"] = s.shift(1).rolling(window).mean()
        x[f"roll_std_{window}"] = s.shift(1).rolling(window).std()
    x["dow"] = x.index.dayofweek
    x["dom"] = x.index.day
    x["month"] = x.index.month
    x["doy"] = x.index.dayofyear
    x["trend"] = np.arange(len(x))
    return x

def supervised_frame(series: pd.Series) -> tuple[pd.DataFrame, pd.Series]:
    x = make_features(series)
    y = series.astype(float)
    mask = x.notna().all(axis=1)
    return x.loc[mask], y.loc[mask]

def _baseline_forecast(history: pd.Series, horizon: int) -> np.ndarray:
    window = history.tail(7)
    return np.repeat(float(window.mean()), horizon)

def recursive_forecast(model, history: pd.Series, horizon: int) -> np.ndarray:
    history = history.copy().astype(float)
    preds=[]
    for _ in range(horizon):
        idx = history.index[-1] + pd.Timedelta(days=1)
        x = make_features(history)
        row = x.iloc[[-1]].copy()
        # make_features uses the current index; append a placeholder by rebuilding
        future = pd.concat([history, pd.Series([np.nan], index=[idx])])
        fx = make_features(future).iloc[[-1]]
        pred=float(model.predict(fx)[0])
        pred=max(0.0, pred)
        preds.append(pred)
        history.loc[idx]=pred
    return np.array(preds)

def train_country_model(series: pd.Series, kind: str):
    x, y = supervised_frame(series)
    if kind == "random_forest":
        model = RandomForestRegressor(n_estimators=80, random_state=42, min_samples_leaf=2, n_jobs=-1)
    elif kind == "extra_trees":
        model = ExtraTreesRegressor(n_estimators=80, random_state=42, min_samples_leaf=2, n_jobs=-1)
    else:
        raise ValueError(kind)
    model.fit(x, y)
    return model

def compare_models(daily: pd.DataFrame, validation_days: int = 60) -> tuple[pd.DataFrame, dict]:
    rows=[]
    models={}
    for country, group in daily.groupby("country"):
        s=group.set_index("date")["revenue"].asfreq("D", fill_value=0.0)
        if len(s) < validation_days + max(LAGS) + 10:
            continue
        train=s.iloc[:-validation_days]
        test=s.iloc[-validation_days:]
        baseline=np.repeat(float(train.tail(7).mean()), len(test))
        base_mae=mean_absolute_error(test, baseline)
        rows.append({"country":country,"model":"baseline_7day_mean","mae":base_mae,
                     "rmse":mean_squared_error(test,baseline)**0.5})
        for kind in ["random_forest","extra_trees"]:
            model=train_country_model(train, kind)
            pred=recursive_forecast(model, train, len(test))
            rows.append({"country":country,"model":kind,"mae":mean_absolute_error(test,pred),
                         "rmse":mean_squared_error(test,pred)**0.5})
            models[(country,kind)] = model
    results=pd.DataFrame(rows)
    # Select the model with lowest mean MAE across countries, then retrain it on all data.
    means=results[results.model!="baseline_7day_mean"].groupby("model")["mae"].mean()
    selected=str(means.idxmin())
    return results, {"selected_model": selected, "validation_days": validation_days}

def train_and_save(daily_path: str | Path, model_dir: str | Path) -> dict:
    daily=pd.read_csv(daily_path, parse_dates=["date"])
    model_dir=Path(model_dir); model_dir.mkdir(parents=True, exist_ok=True)
    results, meta=compare_models(daily)
    selected=meta["selected_model"]
    for country, group in daily.groupby("country"):
        s=group.set_index("date")["revenue"].asfreq("D", fill_value=0.0)
        model=train_country_model(s, selected)
        joblib.dump({"model":model, "last_date":str(s.index.max().date()), "country":country},
                    model_dir / f"{country.replace(' ','_')}.joblib")
    results.to_csv(model_dir/"model_comparison.csv", index=False)
    meta.update({"countries": sorted(daily.country.unique().tolist()),
                 "trained_through": str(daily.date.max().date())})
    (model_dir/"metadata.json").write_text(json.dumps(meta, indent=2))
    return meta

def load_metadata(model_dir: str | Path) -> dict:
    return json.loads((Path(model_dir)/"metadata.json").read_text())

def forecast_from_date(daily_path: str | Path, model_dir: str | Path, date: str,
                       country: str | None = None, duration: int = 30) -> pd.DataFrame:
    if duration < 1 or duration > 365:
        raise ValueError("duration must be between 1 and 365")
    daily=pd.read_csv(daily_path, parse_dates=["date"])
    targets=[country] if country else sorted(daily.country.unique())
    unknown=set(targets)-set(daily.country.unique())
    if unknown:
        raise ValueError(f"Unknown country/countries: {sorted(unknown)}")
    cutoff=pd.Timestamp(date)
    frames=[]
    for c in targets:
        s=(daily[daily.country==c].set_index("date")["revenue"]
           .asfreq("D", fill_value=0.0))
        history=s[s.index <= cutoff]
        if len(history) < max(LAGS)+2:
            raise ValueError("Not enough historical data for requested date")
        artifact=joblib.load(Path(model_dir)/f"{c.replace(' ','_')}.joblib")
        pred=recursive_forecast(artifact["model"], history, duration)
        dates=pd.date_range(cutoff+pd.Timedelta(days=1), periods=duration, freq="D")
        frames.append(pd.DataFrame({"date":dates,"country":c,"prediction":pred}))
    out=pd.concat(frames, ignore_index=True)
    if country is None:
        out=out.groupby("date", as_index=False)["prediction"].sum()
        out["country"]="ALL"
        out=out[["date","country","prediction"]]
    return out
