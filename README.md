# Fashion Greenwashing Detection

## Overview
This project detects greenwashing narratives in fashion advertisement text using Natural Language Processing and multi-label classification.

Greenwashing happens when brands exaggerate or misrepresent sustainability-related claims. This project classifies fashion advertisements into seven possible greenwashing-related categories.

## Labels
1. Clean Materials
2. Local Sourcing
3. Ethical Production
4. Carbon Reduction
5. Recycling Claims
6. Community Benefit
7. Sustainability Commitment

## Features
- Multi-label text classification
- Silver-label generation using keyword rules
- TF-IDF baseline model
- Model evaluation using Precision, Recall, F1 Score and Hamming Loss
- Streamlit app for testing fashion advertisement text

## Tech Stack
Python, NLP, Scikit-learn, Pandas, NumPy, Streamlit

## How to Run

```bash
pip install -r requirements.txt
python src/silver_labeling.py
python src/train_baseline.py
python src/evaluate.py
streamlit run app.py
```

