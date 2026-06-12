import joblib
from pathlib import Path
from sklearn.metrics import classification_report, hamming_loss, f1_score

ROOT_DIR = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT_DIR / "models" / "greenwashing_tfidf_model.pkl"

def main():
    saved = joblib.load(MODEL_PATH)
    model = saved["model"]
    labels = saved["labels"]
    X_test = saved["X_test"]
    y_test = saved["y_test"]

    y_pred = model.predict(X_test)

    print("Classification Report")
    print(classification_report(y_test, y_pred, target_names=labels, zero_division=0))
    print("Hamming Loss:", round(hamming_loss(y_test, y_pred), 4))
    print("Micro F1 Score:", round(f1_score(y_test, y_pred, average="micro", zero_division=0), 4))
    print("Macro F1 Score:", round(f1_score(y_test, y_pred, average="macro", zero_division=0), 4))

if __name__ == "__main__":
    main()
