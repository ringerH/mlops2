import importlib
from typing import Dict, Any, Tuple
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

from ..config.settings import DATA_CONFIG, MODEL_CONFIG
from ..preprocessing.pipeline import create_preprocessing_pipeline


class ModelTrainer:
    """Train multiple models defined in MODEL_CONFIG and keep metrics."""
    def __init__(self, config: Dict[str, Any] = MODEL_CONFIG):
        self.config   = config
        self.pipeline = create_preprocessing_pipeline()

    # ---------- helpers ----------
    def _build_model(self, class_path: str, params: Dict[str, Any]):
        module_name, cls_name = class_path.rsplit(".", 1)
        cls = getattr(importlib.import_module(module_name), cls_name)
        return cls(**params)

    # ---------- public API ----------
    def train(self, df) -> Tuple[Dict[str, Any], Dict[str, str]]:
        X = df[DATA_CONFIG["text_column"]]
        y = df[DATA_CONFIG["target_column"]]
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size   = DATA_CONFIG["test_size"],
            stratify    = y,
            random_state= DATA_CONFIG["random_state"],
        )

        # Fit vectorizer once
        X_train_vec = self.pipeline.fit_transform(X_train)
        X_test_vec  = self.pipeline.transform(X_test)

        reports = {}
        trained = {}

        for name, meta in self.config["models"].items():
            clf = self._build_model(meta["class"], meta["params"])
            if name in self.config["dense_models"]:
                clf.fit(X_train_vec.toarray(), y_train)
                preds = clf.predict(X_test_vec.toarray())
            else:
                clf.fit(X_train_vec, y_train)
                preds = clf.predict(X_test_vec)

            trained[name] = clf
            reports[name] = classification_report(y_test, preds, zero_division=0)
        return trained, reports
