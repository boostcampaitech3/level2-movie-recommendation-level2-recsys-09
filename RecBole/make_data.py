import pandas as pd
import os

FILE = "/opt/ml/input/data/train/train_ratings.csv"
TARGET_DIR = os.path.join(os.getcwd(), "data/boostcamp")
TARGET_NAME = "boostcamp.inter"

os.makedirs(TARGET_DIR, exist_ok=True)

df = pd.read_csv(FILE)
df = df.rename(
    columns={
        "user": "user_id:token",
        "item": "item_id:token",
        "time": "timestamp:float",
    }
)
df.to_csv(os.path.join(TARGET_DIR, TARGET_NAME), index=False, sep="\t")
print("Done!")
