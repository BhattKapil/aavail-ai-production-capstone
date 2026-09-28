import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pathlib import Path
from src.model import train_and_save
base=Path(__file__).resolve().parents[1]
print(train_and_save(base/"data/processed/daily_revenue.csv",base/"models"))
