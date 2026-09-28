FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
RUN python scripts/ingest.py && python scripts/train.py && python scripts/eda.py && python scripts/compare.py
EXPOSE 8080
CMD ["python","-m","src.api"]
