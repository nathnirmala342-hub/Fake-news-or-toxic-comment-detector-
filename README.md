# Toxic Comment Detector

A beginner-friendly NLP project that classifies a comment as **toxic** or **not toxic**, with a small web app to try it out.

**How it works:** comments are converted to numbers with TF-IDF, then a Logistic Regression model learns which words and phrases signal toxicity.

## Project structure

```
toxic-detector/
├── data/comments.csv     # small starter dataset (text,label)
├── train.py              # trains the model and saves model.joblib
├── app.py                # Streamlit web app
├── requirements.txt
└── README.md
```

## Run it

```bash
pip install -r requirements.txt
python train.py
streamlit run app.py
```

Then open the link Streamlit prints (usually http://localhost:8501).

## Use real data (recommended next step)

The starter dataset has only ~60 comments, so the model is limited. For a much better one:

1. Download the **Jigsaw Toxic Comment Classification** dataset from Kaggle (`train.csv`).
2. Convert it to the `text,label` format:

```python
import pandas as pd
df = pd.read_csv("train.csv")
df["label"] = (df[["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]].sum(axis=1) > 0).astype(int)
df[["comment_text", "label"]].rename(columns={"comment_text": "text"}).to_csv("data/jigsaw.csv", index=False)
```

3. Train on it: `python train.py --data data/jigsaw.csv`

## Ideas to extend it

- Try other models: Naive Bayes, Linear SVM, Random Forest, and compare accuracy.
- Show the top words that push a comment toward "toxic" (look at `model.named_steps["clf"].coef_`).
- Handle all 6 Jigsaw categories (multi-label classification).
- Switch to a pretrained transformer such as `unitary/toxic-bert` from Hugging Face.
- Deploy it free on Streamlit Community Cloud.

## Limitations

Toxicity detectors can be biased and make mistakes, especially with sarcasm, slang, or quoted text. Treat this as a learning project, not a replacement for human moderators.
