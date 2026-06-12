import joblib
from pathlib import Path
from preprocess import clean_text
from config import LABEL_DISPLAY_NAMES

ROOT_DIR = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT_DIR / "models" / "greenwashing_tfidf_model.pkl"

def predict_greenwashing(text: str):
    saved = joblib.load(MODEL_PATH)
    model = saved["model"]
    labels = saved["labels"]

    cleaned = clean_text(text)
    prediction = model.predict([cleaned])[0]

    results = {}
    for label, value in zip(labels, prediction):
        results[LABEL_DISPLAY_NAMES[label]] = bool(value)

    return results

if __name__ == "__main__":
    sample = "Our new collection uses organic cotton and recycled polyester while supporting local artisans."
    print(predict_greenwashing(sample))
