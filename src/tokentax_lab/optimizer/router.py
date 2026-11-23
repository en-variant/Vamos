from __future__ import annotations

from pathlib import Path
import joblib
import numpy as np

from .features import extract_features


class Router:
    """Max-depth-4 decision-tree router, trainable on passage-level splits."""

    FEATURE_NAMES = ["length", "word_count", "script_mix", "digit_share", "named_entity_density", "code_mixing"]

    def __init__(self, max_depth: int = 4, random_state: int = 42):
        if not 1 <= max_depth <= 4:
            raise ValueError("Router max depth must be between 1 and 4")
        self.max_depth = max_depth
        self.random_state = random_state
        self.model = None

    def vectorize(self, text: str):
        features = extract_features(text)
        return np.asarray([[float(features[k]) for k in self.FEATURE_NAMES]], dtype=float)

    def fit(self, texts: list[str], labels: list[str]):
        from sklearn.tree import DecisionTreeClassifier
        if len(texts) != len(labels):
            raise ValueError("texts and labels must have equal length")
        if not texts:
            raise ValueError("Cannot fit the router without training examples")
        X = np.vstack([self.vectorize(t)[0] for t in texts])
        self.model = DecisionTreeClassifier(max_depth=self.max_depth, random_state=self.random_state)
        self.model.fit(X, labels)
        return self

    def predict(self, text: str) -> str:
        if self.model is None:
            raise RuntimeError("Router is not fitted")
        return str(self.model.predict(self.vectorize(text))[0])

    def save(self, path: str | Path):
        if self.model is None:
            raise RuntimeError("Cannot save an unfitted router")
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.model, path)

    def load(self, path: str | Path):
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Router artifact not found: {path}")
        self.model = joblib.load(path)
        self.max_depth = int(self.model.get_depth())
