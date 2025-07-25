from pathlib import Path

# Project root =  .../sentiment_mlops/src/..
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR   = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"

DATA_CONFIG = {
    "input_file":    "data.csv",
    "text_column":   "Sentence",
    "target_column": "Sentiment",
    "test_size":     0.20,
    "random_state":  42,
}

MODEL_CONFIG = {
    "models": {
        "RandomForest": {
            "class":  "sklearn.ensemble.RandomForestClassifier",
            "params": {"n_estimators": 100, "random_state": 42}
        },
        "GaussianNB": {
            "class":  "sklearn.naive_bayes.GaussianNB",
            "params": {}
        },
        "LogisticRegression": {
            "class":  "sklearn.linear_model.LogisticRegression",
            "params": {"max_iter": 1000}
        },
        "LinearSVC": {
            "class":  "sklearn.svm.LinearSVC",
            "params": {"max_iter": 10_000}
        },
    },
    # Models that need dense (non-sparse) input
    "dense_models": {"GaussianNB", "MultinomialNB"},
}
