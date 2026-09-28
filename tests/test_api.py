import json
from pathlib import Path
import pytest
flask=pytest.importorskip("flask")
from src.api import create_app

def test_health():
    app=create_app()
    client=app.test_client()
    response=client.get("/health")
    assert response.status_code==200
    assert response.json["status"]=="ok"

def test_predict_requires_date():
    app=create_app()
    response=app.test_client().post("/predict")
    assert response.status_code==400
