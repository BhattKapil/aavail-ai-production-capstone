"""Flask API for AAVAIL revenue forecasting."""
from __future__ import annotations
import os
from pathlib import Path
from flask import Flask, jsonify, request
from .model import forecast_from_date, train_and_save, load_metadata
from .logging_utils import write_prediction_log, read_prediction_logs

def create_app(config=None):
    app=Flask(__name__)
    base=Path(__file__).resolve().parents[1]
    cfg={
        "DATA_PATH": base/"data/processed/daily_revenue.csv",
        "MODEL_DIR": base/"models",
        "LOG_DIR": base/"logs",
    }
    if config: cfg.update(config)
    app.config.update(cfg)

    @app.get("/health")
    def health():
        return jsonify({"status":"ok"})

    @app.post("/train")
    def train():
        try:
            meta=train_and_save(app.config["DATA_PATH"], app.config["MODEL_DIR"])
            return jsonify({"status":"trained", **meta})
        except Exception as exc:
            return jsonify({"error":str(exc)}),500

    @app.post("/predict")
    def predict():
        date=request.args.get("date")
        duration=int(request.args.get("duration",30))
        country=request.args.get("country")
        if not date:
            return jsonify({"error":"date query parameter is required"}),400
        try:
            out=forecast_from_date(app.config["DATA_PATH"],app.config["MODEL_DIR"],date,country,duration)
            payload={
                "date":date,"duration":duration,"country":country or "ALL",
                "predictions":[{"date":str(r.date()),"prediction":float(p)}
                               for r,p in zip(out.date,out.prediction)],
                "total_prediction":float(out.prediction.sum())
            }
            write_prediction_log(app.config["LOG_DIR"], payload)
            return jsonify(payload)
        except ValueError as exc:
            return jsonify({"error":str(exc)}),400
        except Exception as exc:
            return jsonify({"error":str(exc)}),500

    @app.get("/logs")
    def logs():
        return jsonify(read_prediction_logs(app.config["LOG_DIR"]))

    return app

app=create_app()
if __name__=="__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT","8080")))
