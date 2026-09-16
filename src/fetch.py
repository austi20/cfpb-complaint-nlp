"""Cache CFPB complaint narratives locally as a parquet file.

The live CFPB API and bulk CSV no longer include narrative text (checked
2026-09-16 - see PLAN.md Milestone 0). This pulls a historical mirror of the
same CFPB data from Kaggle instead, which still has the narrative column.
"""
import os

import pandas as pd

RAW_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "raw")
ZIP_PATH = os.path.join(RAW_DIR, "complaints.csv.zip")
PARQUET_PATH = os.path.join(RAW_DIR, "complaints.parquet")

DATASET = "namigabbasov/consumer-complaint-dataset"
FILE_NAME = "complaints.csv"
SAMPLE_SIZE = 75000

# rename to match the plan's field list (company_response isn't in this dataset)
COLUMN_MAP = {
    "narrative": "narrative",
    "Product": "product",
    "Sub-product": "sub_product",
    "Issue": "issue",
    "Company": "company",
    "Date received": "date_received",
    "State": "state",
    "Timely response?": "timely",
}


def download():
    if os.path.exists(ZIP_PATH):
        print(f"using cached {ZIP_PATH}")
        return
    import kaggle

    os.makedirs(RAW_DIR, exist_ok=True)
    kaggle.api.authenticate()
    kaggle.api.dataset_download_file(DATASET, FILE_NAME, path=RAW_DIR)
    print(f"downloaded {ZIP_PATH}")


def build_parquet():
    if os.path.exists(PARQUET_PATH):
        print(f"using cached {PARQUET_PATH}")
        return
    df = pd.read_csv(ZIP_PATH, usecols=list(COLUMN_MAP), dtype=str)
    df = df.rename(columns=COLUMN_MAP)
    df = df.dropna(subset=["narrative"])
    df = df.sample(n=SAMPLE_SIZE, random_state=42).sort_values("date_received")
    df.to_parquet(PARQUET_PATH, index=False)
    print(f"wrote {len(df)} rows to {PARQUET_PATH}")


if __name__ == "__main__":
    download()
    build_parquet()
