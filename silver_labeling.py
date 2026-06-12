import pandas as pd
from pathlib import Path
from config import LABELS, KEYWORDS
from preprocess import clean_text

ROOT_DIR = Path(__file__).resolve().parents[1]
RAW_DATA_PATH = ROOT_DIR / "data" / "fashion_ads.csv"
OUTPUT_PATH = ROOT_DIR / "data" / "silver_labeled_fashion_ads.csv"

def assign_silver_labels(text: str) -> dict:
    cleaned = clean_text(text)
    labels = {}

    for label in LABELS:
        labels[label] = 0
        for keyword in KEYWORDS[label]:
            if keyword in cleaned:
                labels[label] = 1
                break

    return labels

def main():
    df = pd.read_csv(RAW_DATA_PATH)
    label_rows = df["ad_text"].apply(assign_silver_labels)
    label_df = pd.DataFrame(label_rows.tolist())
    final_df = pd.concat([df, label_df], axis=1)
    final_df.to_csv(OUTPUT_PATH, index=False)
    print(f"Silver-labelled dataset saved to: {OUTPUT_PATH}")
    print(final_df.head())

if __name__ == "__main__":
    main()
