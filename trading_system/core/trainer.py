"""
core/trainer.py
===============
Manages model lifecycle:
  - Initial training
  - Scheduled retraining
  - Performance decay detection
  - Model versioning
"""
import json
import time
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import (RETRAIN_EVERY_N_DAYS, DECAY_ALERT_THRESHOLD,
                    MODEL_DIR, LOG_DIR)
from core.data   import prepare_dataset
from core.models import EnsembleModel

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(LOG_DIR / "trainer.log"),
    ],
)
log = logging.getLogger(__name__)

REGISTRY_FILE = MODEL_DIR / "registry.json"


# ─────────────────────────────────────────────────────────────────────────────
# Registry helpers
# ─────────────────────────────────────────────────────────────────────────────

def _load_registry() -> Dict:
    if REGISTRY_FILE.exists():
        return json.loads(REGISTRY_FILE.read_text())
    return {}

def _save_registry(reg: Dict):
    REGISTRY_FILE.write_text(json.dumps(reg, indent=2, default=str))


# ─────────────────────────────────────────────────────────────────────────────
# Trainer
# ─────────────────────────────────────────────────────────────────────────────

class ModelTrainer:

    def __init__(self):
        self.registry = _load_registry()

    def needs_retrain(self, ticker: str) -> bool:
        """True if the model hasn't been trained yet or is past its schedule."""
        rec = self.registry.get(ticker)
        if rec is None:
            return True
        last = datetime.fromisoformat(rec["last_trained"])
        return datetime.now() - last > timedelta(days=RETRAIN_EVERY_N_DAYS)

    def train(self, ticker: str, force: bool = False) -> Optional[EnsembleModel]:
        if not force and not self.needs_retrain(ticker):
            log.info(f"[{ticker}] Model is fresh — skipping retrain")
            return self._load_existing(ticker)

        log.info(f"[{ticker}] Fetching data …")
        df = prepare_dataset(ticker)
        if df is None:
            log.error(f"[{ticker}] Not enough data — aborting")
            return None

        log.info(f"[{ticker}] Training ensemble on {len(df)} bars …")
        t0 = time.time()

        model = EnsembleModel()
        metrics = model.train(df)
        model.save(ticker)

        elapsed = round(time.time() - t0, 1)
        log.info(f"[{ticker}] Training done in {elapsed}s")

        # ── Decay check ───────────────────────────────────────────────────
        old_rec = self.registry.get(ticker)
        decay_alerts = {}
        if old_rec and "metrics" in old_rec:
            old_metrics = old_rec["metrics"]
            for m_name, m_metrics in metrics.items():
                old_acc = old_metrics.get(m_name, {}).get("accuracy", 0)
                new_acc = m_metrics.get("accuracy", 0)
                drop = old_acc - new_acc
                if drop > DECAY_ALERT_THRESHOLD:
                    log.warning(f"[{ticker}] {m_name} accuracy dropped {drop:.1%}!")
                    decay_alerts[m_name] = round(drop, 4)

        # ── Update registry ───────────────────────────────────────────────
        self.registry[ticker] = {
            "last_trained": datetime.now().isoformat(),
            "num_bars":     len(df),
            "metrics":      metrics,
            "decay_alerts": decay_alerts,
        }
        _save_registry(self.registry)

        return model

    def train_all(self, tickers: List[str], force: bool = False) -> Dict[str, EnsembleModel]:
        models = {}
        for ticker in tickers:
            log.info(f"\n{'-'*50}\nTraining {ticker}")
            m = self.train(ticker, force=force)
            if m:
                models[ticker] = m
        return models

    def _load_existing(self, ticker: str) -> Optional[EnsembleModel]:
        model = EnsembleModel()
        if model.load(ticker):
            return model
        return None

    def get_status(self) -> Dict:
        """Return a status summary for the dashboard."""
        status = {}
        for ticker, rec in self.registry.items():
            last  = datetime.fromisoformat(rec["last_trained"])
            age_d = (datetime.now() - last).days
            status[ticker] = {
                "last_trained":  rec["last_trained"],
                "age_days":      age_d,
                "needs_retrain": age_d >= RETRAIN_EVERY_N_DAYS,
                "metrics":       rec.get("metrics", {}),
                "decay_alerts":  rec.get("decay_alerts", {}),
            }
        return status

    def get_feature_importance(self, ticker: str) -> Dict:
        model = self._load_existing(ticker)
        if model is None:
            return {}
        return model.xgb.feature_importance
