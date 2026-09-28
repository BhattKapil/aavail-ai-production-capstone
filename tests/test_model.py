import pandas as pd
import numpy as np
from src.model import make_features, train_country_model, recursive_forecast

def test_features_have_expected_lags():
    s=pd.Series(np.arange(100,dtype=float),index=pd.date_range("2024-01-01",periods=100))
    x=make_features(s)
    assert {"lag_1","lag_7","lag_28","roll_mean_7","roll_mean_30"}.issubset(x.columns)

def test_model_predicts_nonnegative_forecast():
    s=pd.Series(100+np.arange(100)*.5,index=pd.date_range("2024-01-01",periods=100))
    m=train_country_model(s,"extra_trees")
    pred=recursive_forecast(m,s,10)
    assert len(pred)==10 and np.all(pred>=0)
