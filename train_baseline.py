import pandas as pd
import joblib
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from config import LABELS
from preprocess import clean_text

ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT_DIR / "data" / "silver_labeled_fashion_ads.csv"
MODEL_PATH = ROOT_DIR / "models" / "greenwashing_tfidf_model.pkl"

def main():
    df = pd.read_csv(DATA_PATH)
    df["clean_text"] = df["ad_text"].apply(clean_text)

    X = df["clean_text"]
    y = df[LABELS]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42
    )

    model = Pipeline([
        ("tfidf", TfidfVectorizer(max_features=5000, ngram_range=(1, 2))),
        ("classifier", OneVsRestClassifier(LogisticRegression(max_iter=1000)))
    ])

    model.fit(X_train, y_train)

    joblib.dump(
        {"model": model, "labels": LABELS, "X_test": X_test, "y_test": y_test},
        MODEL_PATH
    )

    print(f"Model saved to: {MODEL_PATH}")

if __name__ == "__main__":
    main()
