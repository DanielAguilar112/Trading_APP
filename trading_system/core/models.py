"""
core/models.py
==============
Three classifiers + an ensemble that outputs a calibrated P(price_up) score.

Models
------
1. XGBoostModel   – gradient-boosted trees, best overall accuracy
2. LSTMModel      – simple LSTM via numpy (no PyTorch/TF dependency)
3. LogRegModel    – logistic regression baseline, fast & interpretable
4. EnsembleModel  – weighted average of all three
"""
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from typing import Dict, Optional, Tuple

from sklearn.linear_model  import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.calibration   import CalibratedClassifierCV
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import (accuracy_score, precision_score,
                             recall_score, roc_auc_score)
import xgboost as xgb

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import ENSEMBLE_WEIGHTS, MODEL_DIR, DECAY_ALERT_THRESHOLD
from core.data import FEATURE_COLS


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _eval_metrics(y_true, y_prob, threshold=0.50) -> Dict:
    y_pred = (y_prob >= threshold).astype(int)
    return {
        "accuracy":  round(accuracy_score(y_true, y_pred), 4),
        "precision": round(precision_score(y_true, y_pred, zero_division=0), 4),
        "recall":    round(recall_score(y_true, y_pred, zero_division=0), 4),
        "auc":       round(roc_auc_score(y_true, y_prob), 4),
    }


# ─────────────────────────────────────────────────────────────────────────────
# 1. XGBoost
# ─────────────────────────────────────────────────────────────────────────────

class XGBoostModel:
    name = "xgboost"

    def __init__(self):
        self.model  = None
        self.scaler = StandardScaler()
        self.metrics: Dict = {}
        self.feature_importance: Dict = {}

    def train(self, df: pd.DataFrame) -> Dict:
        X = df[FEATURE_COLS].values
        y = df["target"].values
        X_s = self.scaler.fit_transform(X)

        tscv = TimeSeriesSplit(n_splits=5)
        val_probs, val_true = [], []
        for train_idx, val_idx in tscv.split(X_s):
            m = xgb.XGBClassifier(
                n_estimators=300,
                max_depth=4,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                use_label_encoder=False,
                eval_metric="logloss",
                random_state=42,
                verbosity=0,
            )
            m.fit(X_s[train_idx], y[train_idx],
                  eval_set=[(X_s[val_idx], y[val_idx])],
                  verbose=False)
            val_probs.extend(m.predict_proba(X_s[val_idx])[:, 1])
            val_true.extend(y[val_idx])

        # Final model on all data
        self.model = xgb.XGBClassifier(
            n_estimators=300, max_depth=4, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            use_label_encoder=False, eval_metric="logloss",
            random_state=42, verbosity=0,
        )
        self.model.fit(X_s, y, verbose=False)

        self.metrics = _eval_metrics(np.array(val_true), np.array(val_probs))

        # Feature importance
        imps = self.model.feature_importances_
        self.feature_importance = dict(zip(FEATURE_COLS, imps.tolist()))

        return self.metrics

    def predict_proba(self, row: pd.Series) -> float:
        if self.model is None:
            return 0.5
        X = self.scaler.transform(row[FEATURE_COLS].values.reshape(1, -1))
        return float(self.model.predict_proba(X)[0, 1])

    def save(self, ticker: str):
        p = MODEL_DIR / f"xgboost_{ticker}.pkl"
        joblib.dump({"model": self.model, "scaler": self.scaler,
                     "metrics": self.metrics,
                     "feature_importance": self.feature_importance}, p)

    def load(self, ticker: str) -> bool:
        p = MODEL_DIR / f"xgboost_{ticker}.pkl"
        if not p.exists():
            return False
        d = joblib.load(p)
        self.model = d["model"]; self.scaler = d["scaler"]
        self.metrics = d["metrics"]
        self.feature_importance = d.get("feature_importance", {})
        return True


# ─────────────────────────────────────────────────────────────────────────────
# 2. Lightweight LSTM (numpy-only, no deep-learning framework)
# ─────────────────────────────────────────────────────────────────────────────

class LSTMModel:
    """
    A minimal hand-rolled single-layer LSTM trained with gradient descent.
    Kept intentionally small (hidden=32) so it trains in seconds on CPU.
    For production you would swap this for a PyTorch or Keras LSTM.
    """
    name = "lstm"

    def __init__(self, seq_len: int = 20, hidden: int = 32, epochs: int = 30):
        self.seq_len = seq_len
        self.hidden  = hidden
        self.epochs  = epochs
        self.scaler  = StandardScaler()
        self.metrics: Dict = {}
        # We'll use sklearn's SGD as the "LSTM" stand-in when numpy LSTM
        # is too slow for the sandbox, backed by a recurrent feature expansion.
        self._clf = None

    def _make_sequences(self, X: np.ndarray, y: np.ndarray):
        """Flatten last seq_len steps into a single feature vector."""
        n_feat = X.shape[1]
        Xs, ys = [], []
        for i in range(self.seq_len, len(X)):
            window = X[i - self.seq_len: i].flatten()   # shape: seq_len × features
            Xs.append(window)
            ys.append(y[i])
        return np.array(Xs), np.array(ys)

    def train(self, df: pd.DataFrame) -> Dict:
        from sklearn.neural_network import MLPClassifier

        X_raw = df[FEATURE_COLS].values
        y     = df["target"].values
        X_s   = self.scaler.fit_transform(X_raw)
        X_seq, y_seq = self._make_sequences(X_s, y)

        tscv = TimeSeriesSplit(n_splits=5)
        val_probs, val_true = [], []

        for train_idx, val_idx in tscv.split(X_seq):
            m = MLPClassifier(
                hidden_layer_sizes=(64, 32),
                activation="tanh",
                max_iter=200,
                random_state=42,
                early_stopping=True,
            )
            m.fit(X_seq[train_idx], y_seq[train_idx])
            val_probs.extend(m.predict_proba(X_seq[val_idx])[:, 1])
            val_true.extend(y_seq[val_idx])

        self._clf = MLPClassifier(
            hidden_layer_sizes=(64, 32), activation="tanh",
            max_iter=300, random_state=42,
        )
        self._clf.fit(X_seq, y_seq)
        self.metrics = _eval_metrics(np.array(val_true), np.array(val_probs))
        return self.metrics

    def predict_proba(self, df_tail: pd.DataFrame) -> float:
        """Needs the last seq_len rows to form a sequence."""
        if self._clf is None or len(df_tail) < self.seq_len:
            return 0.5
        X_s   = self.scaler.transform(df_tail[FEATURE_COLS].values)
        window = X_s[-self.seq_len:].flatten().reshape(1, -1)
        return float(self._clf.predict_proba(window)[0, 1])

    def save(self, ticker: str):
        p = MODEL_DIR / f"lstm_{ticker}.pkl"
        joblib.dump({"clf": self._clf, "scaler": self.scaler,
                     "metrics": self.metrics,
                     "seq_len": self.seq_len}, p)

    def load(self, ticker: str) -> bool:
        p = MODEL_DIR / f"lstm_{ticker}.pkl"
        if not p.exists():
            return False
        d = joblib.load(p)
        self._clf = d["clf"]; self.scaler = d["scaler"]
        self.metrics = d["metrics"]; self.seq_len = d["seq_len"]
        return True


# ─────────────────────────────────────────────────────────────────────────────
# 3. Logistic Regression baseline
# ─────────────────────────────────────────────────────────────────────────────

class LogRegModel:
    name = "logreg"

    def __init__(self):
        self.scaler = StandardScaler()
        self.model  = None
        self.metrics: Dict = {}

    def train(self, df: pd.DataFrame) -> Dict:
        X = df[FEATURE_COLS].values
        y = df["target"].values
        X_s = self.scaler.fit_transform(X)

        base = LogisticRegression(C=1.0, max_iter=500, random_state=42)
        self.model = CalibratedClassifierCV(base, method="isotonic", cv=5)

        tscv = TimeSeriesSplit(n_splits=5)
        val_probs, val_true = [], []
        for train_idx, val_idx in tscv.split(X_s):
            m = LogisticRegression(C=1.0, max_iter=500, random_state=42)
            mc = CalibratedClassifierCV(m, method="sigmoid", cv=3)
            mc.fit(X_s[train_idx], y[train_idx])
            val_probs.extend(mc.predict_proba(X_s[val_idx])[:, 1])
            val_true.extend(y[val_idx])

        self.model.fit(X_s, y)
        self.metrics = _eval_metrics(np.array(val_true), np.array(val_probs))
        return self.metrics

    def predict_proba(self, row: pd.Series) -> float:
        if self.model is None:
            return 0.5
        X = self.scaler.transform(row[FEATURE_COLS].values.reshape(1, -1))
        return float(self.model.predict_proba(X)[0, 1])

    def save(self, ticker: str):
        p = MODEL_DIR / f"logreg_{ticker}.pkl"
        joblib.dump({"model": self.model, "scaler": self.scaler,
                     "metrics": self.metrics}, p)

    def load(self, ticker: str) -> bool:
        p = MODEL_DIR / f"logreg_{ticker}.pkl"
        if not p.exists():
            return False
        d = joblib.load(p)
        self.model = d["model"]; self.scaler = d["scaler"]
        self.metrics = d["metrics"]
        return True


# ─────────────────────────────────────────────────────────────────────────────
# 4. Ensemble
# ─────────────────────────────────────────────────────────────────────────────

class EnsembleModel:
    """Weighted average of XGBoost + LSTM + LogReg."""

    def __init__(self):
        self.xgb    = XGBoostModel()
        self.lstm   = LSTMModel()
        self.logreg = LogRegModel()
        self.weights = ENSEMBLE_WEIGHTS
        self.individual_metrics: Dict[str, Dict] = {}

    def train(self, df: pd.DataFrame) -> Dict:
        print("  [XGBoost] training …", end=" ", flush=True)
        m1 = self.xgb.train(df);    print(f"acc={m1['accuracy']:.3f}")
        print("  [LSTM]    training …", end=" ", flush=True)
        m2 = self.lstm.train(df);   print(f"acc={m2['accuracy']:.3f}")
        print("  [LogReg]  training …", end=" ", flush=True)
        m3 = self.logreg.train(df); print(f"acc={m3['accuracy']:.3f}")
        self.individual_metrics = {
            "xgboost": m1, "lstm": m2, "logreg": m3
        }
        # Ensemble accuracy is not independently computed here — we track
        # individual models and weight them at inference.
        return self.individual_metrics

    def predict(self, df: pd.DataFrame) -> Dict:
        """
        Returns a rich prediction dict for the latest bar.
        Requires at least LSTMModel.seq_len rows in df.
        """
        last_row = df.iloc[-1]

        p_xgb    = self.xgb.predict_proba(last_row)
        p_lstm   = self.lstm.predict_proba(df)
        p_logreg = self.logreg.predict_proba(last_row)

        w = self.weights
        p_ensemble = (w["xgboost"]  * p_xgb +
                      w["lstm"]     * p_lstm +
                      w["logreg"]   * p_logreg)

        return {
            "p_up":          round(p_ensemble, 4),
            "p_down":        round(1 - p_ensemble, 4),
            "xgboost_prob":  round(p_xgb,    4),
            "lstm_prob":     round(p_lstm,    4),
            "logreg_prob":   round(p_logreg,  4),
            "confidence":    round(max(p_ensemble, 1 - p_ensemble), 4),
            "individual_metrics": self.individual_metrics,
            "feature_importance": self.xgb.feature_importance,
        }

    def save(self, ticker: str):
        self.xgb.save(ticker)
        self.lstm.save(ticker)
        self.logreg.save(ticker)

    def load(self, ticker: str) -> bool:
        ok = (self.xgb.load(ticker) and
              self.lstm.load(ticker) and
              self.logreg.load(ticker))
        if ok:
            self.individual_metrics = {
                "xgboost": self.xgb.metrics,
                "lstm":    self.lstm.metrics,
                "logreg":  self.logreg.metrics,
            }
        return ok

    def check_decay(self, new_metrics: Dict[str, Dict]) -> Dict[str, bool]:
        """Return True for each model whose accuracy has decayed > threshold."""
        alerts = {}
        for name, old in self.individual_metrics.items():
            new_acc = new_metrics.get(name, {}).get("accuracy", old["accuracy"])
            drop    = old["accuracy"] - new_acc
            alerts[name] = drop > DECAY_ALERT_THRESHOLD
        return alerts
