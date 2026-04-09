"""
core/models.py  v2
==================
Upgraded ensemble:
- XGBoost (tuned hyperparameters)
- Random Forest (new)
- LSTM/MLP (deeper network)
- Logistic Regression (calibrated baseline)
- Dynamic ensemble weighting based on recent accuracy
"""
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from typing import Dict, Optional
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
from sklearn.neural_network import MLPClassifier
import xgboost as xgb
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import MODEL_DIR, DECAY_ALERT_THRESHOLD
from core.data import FEATURE_COLS

def _metrics(y_true, y_prob, thresh=0.50):
    y_pred = (y_prob >= thresh).astype(int)
    return {
        "accuracy":  round(accuracy_score(y_true, y_pred), 4),
        "precision": round(precision_score(y_true, y_pred, zero_division=0), 4),
        "recall":    round(recall_score(y_true, y_pred, zero_division=0), 4),
        "auc":       round(roc_auc_score(y_true, y_prob), 4),
    }

class XGBoostModel:
    name = "xgboost"
    def __init__(self):
        self.model = None
        self.scaler = RobustScaler()
        self.metrics = {}
        self.feature_importance = {}

    def train(self, df):
        X = self.scaler.fit_transform(df[FEATURE_COLS].values)
        y = df["target"].values
        tscv = TimeSeriesSplit(n_splits=5)
        vp, vt = [], []
        for ti, vi in tscv.split(X):
            m = xgb.XGBClassifier(
                n_estimators=500, max_depth=5, learning_rate=0.03,
                subsample=0.8, colsample_bytree=0.7, min_child_weight=3,
                gamma=0.1, reg_alpha=0.1, reg_lambda=1.0,
                use_label_encoder=False, eval_metric="logloss",
                random_state=42, verbosity=0,
            )
            m.fit(X[ti], y[ti], eval_set=[(X[vi], y[vi])], verbose=False)
            vp.extend(m.predict_proba(X[vi])[:,1]); vt.extend(y[vi])
        self.model = xgb.XGBClassifier(
            n_estimators=500, max_depth=5, learning_rate=0.03,
            subsample=0.8, colsample_bytree=0.7, min_child_weight=3,
            gamma=0.1, reg_alpha=0.1, reg_lambda=1.0,
            use_label_encoder=False, eval_metric="logloss",
            random_state=42, verbosity=0,
        )
        self.model.fit(X, y, verbose=False)
        self.metrics = _metrics(np.array(vt), np.array(vp))
        self.feature_importance = dict(zip(FEATURE_COLS, self.model.feature_importances_.tolist()))
        return self.metrics

    def predict_proba(self, row):
        if self.model is None: return 0.5
        X = self.scaler.transform(row[FEATURE_COLS].values.reshape(1,-1))
        return float(self.model.predict_proba(X)[0,1])

    def save(self, ticker):
        joblib.dump({"model":self.model,"scaler":self.scaler,
                     "metrics":self.metrics,"fi":self.feature_importance},
                    MODEL_DIR/f"xgboost_{ticker}.pkl")

    def load(self, ticker):
        p = MODEL_DIR/f"xgboost_{ticker}.pkl"
        if not p.exists(): return False
        d = joblib.load(p)
        self.model=d["model"]; self.scaler=d["scaler"]
        self.metrics=d["metrics"]; self.feature_importance=d.get("fi",{})
        return True


class RandomForestModel:
    name = "random_forest"
    def __init__(self):
        self.model = None
        self.scaler = RobustScaler()
        self.metrics = {}

    def train(self, df):
        X = self.scaler.fit_transform(df[FEATURE_COLS].values)
        y = df["target"].values
        tscv = TimeSeriesSplit(n_splits=5)
        vp, vt = [], []
        for ti, vi in tscv.split(X):
            m = RandomForestClassifier(
                n_estimators=300, max_depth=8, min_samples_split=10,
                min_samples_leaf=5, max_features="sqrt",
                random_state=42, n_jobs=-1,
            )
            mc = CalibratedClassifierCV(m, method="isotonic", cv=3)
            mc.fit(X[ti], y[ti])
            vp.extend(mc.predict_proba(X[vi])[:,1]); vt.extend(y[vi])
        self.model = CalibratedClassifierCV(
            RandomForestClassifier(n_estimators=300, max_depth=8,
                min_samples_split=10, min_samples_leaf=5,
                max_features="sqrt", random_state=42, n_jobs=-1),
            method="isotonic", cv=5
        )
        self.model.fit(X, y)
        self.metrics = _metrics(np.array(vt), np.array(vp))
        return self.metrics

    def predict_proba(self, row):
        if self.model is None: return 0.5
        X = self.scaler.transform(row[FEATURE_COLS].values.reshape(1,-1))
        return float(self.model.predict_proba(X)[0,1])

    def save(self, ticker):
        joblib.dump({"model":self.model,"scaler":self.scaler,"metrics":self.metrics},
                    MODEL_DIR/f"rf_{ticker}.pkl")

    def load(self, ticker):
        p = MODEL_DIR/f"rf_{ticker}.pkl"
        if not p.exists(): return False
        d = joblib.load(p)
        self.model=d["model"]; self.scaler=d["scaler"]; self.metrics=d["metrics"]
        return True


class LSTMModel:
    name = "lstm"
    def __init__(self, seq_len=20):
        self.seq_len = seq_len
        self.scaler = RobustScaler()
        self._clf = None
        self.metrics = {}

    def _make_seq(self, X, y):
        Xs, ys = [], []
        for i in range(self.seq_len, len(X)):
            Xs.append(X[i-self.seq_len:i].flatten())
            ys.append(y[i])
        return np.array(Xs), np.array(ys)

    def train(self, df):
        X = self.scaler.fit_transform(df[FEATURE_COLS].values)
        y = df["target"].values
        Xs, ys = self._make_seq(X, y)
        tscv = TimeSeriesSplit(n_splits=5)
        vp, vt = [], []
        for ti, vi in tscv.split(Xs):
            m = MLPClassifier(
                hidden_layer_sizes=(128, 64, 32),
                activation="relu", max_iter=300,
                random_state=42, early_stopping=True,
                validation_fraction=0.1, learning_rate_init=0.001,
            )
            m.fit(Xs[ti], ys[ti])
            vp.extend(m.predict_proba(Xs[vi])[:,1]); vt.extend(ys[vi])
        self._clf = MLPClassifier(
            hidden_layer_sizes=(128, 64, 32), activation="relu",
            max_iter=400, random_state=42,
        )
        self._clf.fit(Xs, ys)
        self.metrics = _metrics(np.array(vt), np.array(vp))
        return self.metrics

    def predict_proba(self, df_tail):
        if self._clf is None or len(df_tail) < self.seq_len: return 0.5
        X = self.scaler.transform(df_tail[FEATURE_COLS].values)
        window = X[-self.seq_len:].flatten().reshape(1,-1)
        return float(self._clf.predict_proba(window)[0,1])

    def save(self, ticker):
        joblib.dump({"clf":self._clf,"scaler":self.scaler,
                     "metrics":self.metrics,"seq_len":self.seq_len},
                    MODEL_DIR/f"lstm_{ticker}.pkl")

    def load(self, ticker):
        p = MODEL_DIR/f"lstm_{ticker}.pkl"
        if not p.exists(): return False
        d = joblib.load(p)
        self._clf=d["clf"]; self.scaler=d["scaler"]
        self.metrics=d["metrics"]; self.seq_len=d["seq_len"]
        return True


class LogRegModel:
    name = "logreg"
    def __init__(self):
        self.scaler = RobustScaler()
        self.model = None
        self.metrics = {}

    def train(self, df):
        X = self.scaler.fit_transform(df[FEATURE_COLS].values)
        y = df["target"].values
        tscv = TimeSeriesSplit(n_splits=5)
        vp, vt = [], []
        for ti, vi in tscv.split(X):
            m = CalibratedClassifierCV(
                LogisticRegression(C=0.5, max_iter=1000, random_state=42),
                method="isotonic", cv=3
            )
            m.fit(X[ti], y[ti])
            vp.extend(m.predict_proba(X[vi])[:,1]); vt.extend(y[vi])
        self.model = CalibratedClassifierCV(
            LogisticRegression(C=0.5, max_iter=1000, random_state=42),
            method="isotonic", cv=5
        )
        self.model.fit(X, y)
        self.metrics = _metrics(np.array(vt), np.array(vp))
        return self.metrics

    def predict_proba(self, row):
        if self.model is None: return 0.5
        X = self.scaler.transform(row[FEATURE_COLS].values.reshape(1,-1))
        return float(self.model.predict_proba(X)[0,1])

    def save(self, ticker):
        joblib.dump({"model":self.model,"scaler":self.scaler,"metrics":self.metrics},
                    MODEL_DIR/f"logreg_{ticker}.pkl")

    def load(self, ticker):
        p = MODEL_DIR/f"logreg_{ticker}.pkl"
        if not p.exists(): return False
        d = joblib.load(p)
        self.model=d["model"]; self.scaler=d["scaler"]; self.metrics=d["metrics"]
        return True


class EnsembleModel:
    """
    Dynamic weighted ensemble.
    Weights are adjusted based on each model's recent AUC score.
    Better models automatically get more influence.
    """
    def __init__(self):
        self.xgb    = XGBoostModel()
        self.rf     = RandomForestModel()
        self.lstm   = LSTMModel()
        self.logreg = LogRegModel()
        self.individual_metrics = {}
        self._weights = {"xgboost":0.35,"random_forest":0.30,"lstm":0.20,"logreg":0.15}

    def _compute_weights(self):
        """Weight each model by its AUC score — better models get more say."""
        aucs = {}
        for name, m in [("xgboost",self.xgb),("random_forest",self.rf),
                         ("lstm",self.lstm),("logreg",self.logreg)]:
            aucs[name] = self.individual_metrics.get(name,{}).get("auc", 0.5)
        total = sum(aucs.values())
        if total > 0:
            self._weights = {k: v/total for k,v in aucs.items()}

    def train(self, df):
        print("  [XGBoost]      training ...", end=" ", flush=True)
        m1 = self.xgb.train(df);    print(f"acc={m1['accuracy']:.3f} auc={m1['auc']:.3f}")
        print("  [RandomForest] training ...", end=" ", flush=True)
        m2 = self.rf.train(df);     print(f"acc={m2['accuracy']:.3f} auc={m2['auc']:.3f}")
        print("  [LSTM/MLP]     training ...", end=" ", flush=True)
        m3 = self.lstm.train(df);   print(f"acc={m3['accuracy']:.3f} auc={m3['auc']:.3f}")
        print("  [LogReg]       training ...", end=" ", flush=True)
        m4 = self.logreg.train(df); print(f"acc={m4['accuracy']:.3f} auc={m4['auc']:.3f}")
        self.individual_metrics = {
            "xgboost":m1,"random_forest":m2,"lstm":m3,"logreg":m4
        }
        self._compute_weights()
        print(f"  [Weights] xgb={self._weights['xgboost']:.2f} rf={self._weights['random_forest']:.2f} "
              f"lstm={self._weights['lstm']:.2f} lr={self._weights['logreg']:.2f}")
        return self.individual_metrics

    def predict(self, df):
        last = df.iloc[-1]
        p_xgb    = self.xgb.predict_proba(last)
        p_rf     = self.rf.predict_proba(last)
        p_lstm   = self.lstm.predict_proba(df)
        p_logreg = self.logreg.predict_proba(last)
        w = self._weights
        p = (w["xgboost"]*p_xgb + w["random_forest"]*p_rf +
             w["lstm"]*p_lstm + w["logreg"]*p_logreg)
        # Agreement bonus — if all models agree, boost confidence
        probs = [p_xgb, p_rf, p_lstm, p_logreg]
        all_bull = all(x > 0.55 for x in probs)
        all_bear = all(x < 0.45 for x in probs)
        if all_bull: p = min(p * 1.05, 0.95)
        if all_bear: p = max(p * 0.95, 0.05)
        return {
            "p_up": round(p, 4),
            "p_down": round(1-p, 4),
            "xgboost_prob": round(p_xgb, 4),
            "rf_prob": round(p_rf, 4),
            "lstm_prob": round(p_lstm, 4),
            "logreg_prob": round(p_logreg, 4),
            "confidence": round(max(p, 1-p), 4),
            "model_agreement": all_bull or all_bear,
            "individual_metrics": self.individual_metrics,
            "feature_importance": self.xgb.feature_importance,
            "weights": self._weights,
        }

    def save(self, ticker):
        self.xgb.save(ticker); self.rf.save(ticker)
        self.lstm.save(ticker); self.logreg.save(ticker)

    def load(self, ticker):
        ok = (self.xgb.load(ticker) and self.rf.load(ticker) and
              self.lstm.load(ticker) and self.logreg.load(ticker))
        if ok:
            self.individual_metrics = {
                "xgboost":self.xgb.metrics,"random_forest":self.rf.metrics,
                "lstm":self.lstm.metrics,"logreg":self.logreg.metrics,
            }
            self._compute_weights()
        return ok