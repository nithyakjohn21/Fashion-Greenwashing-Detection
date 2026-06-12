import streamlit as st
import joblib
from pathlib import Path
from src.preprocess import clean_text
from src.config import LABEL_DISPLAY_NAMES

ROOT_DIR = Path(__file__).resolve().parent
MODEL_PATH = ROOT_DIR / "models" / "greenwashing_tfidf_model.pkl"

st.set_page_config(page_title="Fashion Greenwashing Detection", layout="wide")

st.title("Fashion Greenwashing Detection")
st.write("Detect greenwashing-related claims in fashion advertisement text using NLP.")

sample_text = "Our new collection uses organic cotton, recycled polyester, and supports local artisans while reducing carbon emissions."

ad_text = st.text_area(
    "Enter fashion advertisement text",
    value=sample_text,
    height=180
)

if st.button("Analyze Advertisement"):
    if not MODEL_PATH.exists():
        st.error("Model not found. Please run: python src/silver_labeling.py and python src/train_baseline.py")
    elif not ad_text.strip():
        st.warning("Please enter advertisement text.")
    else:
        saved = joblib.load(MODEL_PATH)
        model = saved["model"]
        labels = saved["labels"]

        cleaned = clean_text(ad_text)
        prediction = model.predict([cleaned])[0]

        st.subheader("Detected Greenwashing Narratives")

        detected_count = 0

        for label, value in zip(labels, prediction):
            display_name = LABEL_DISPLAY_NAMES[label]
            if value == 1:
                detected_count += 1
                st.success(f"{display_name}: Detected")
            else:
                st.info(f"{display_name}: Not Detected")

        st.metric("Total Detected Categories", detected_count)
