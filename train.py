"""Train a toxic comment detector.

Usage:
    python train.py                          # uses the small starter dataset
    python train.py --data my_data.csv       # your own CSV with columns: text,label (1 = toxic, 0 = not)

For the real Kaggle "Jigsaw Toxic Comment" data, see the README.
"""
import argparse

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/comments.csv")
    parser.add_argument("--out", default="model.joblib")
    args = parser.parse_args()

    df = pd.read_csv(args.data).dropna(subset=["text", "label"])
    print(f"Loaded {len(df)} comments ({df['label'].sum()} toxic, {(df['label'] == 0).sum()} clean)")

    X_train, X_test, y_train, y_test = train_test_split(
        df["text"], df["label"], test_size=0.25, random_state=42, stratify=df["label"]
    )

    # TF-IDF turns text into numbers, Logistic Regression learns which words signal toxicity.
    model = Pipeline([
        ("tfidf", TfidfVectorizer(lowercase=True, ngram_range=(1, 2), stop_words="english")),
        ("clf", LogisticRegression(C=10, max_iter=1000, class_weight="balanced")),
    ])
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    print("\nClassification report:")
    print(classification_report(y_test, preds, target_names=["clean", "toxic"]))
    print("Confusion matrix:")
    print(confusion_matrix(y_test, preds))

    joblib.dump(model, args.out)
    print(f"\nModel saved to {args.out}")


if __name__ == "__main__":
    main()
