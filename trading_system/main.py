"""
main.py  —  AI Trading System — Master Entry Point
===================================================
Commands
--------
  python main.py --mode train                       # train all models
  python main.py --mode signals                     # generate signals only
  python main.py --mode backtest --ticker AAPL      # backtest
  python main.py --mode status                      # model health

  python main.py --mode execute --broker dry_run    # signal + execute (simulated)
  python main.py --mode execute --broker paper \    # signal + execute (Alpaca paper)
      --api-key PKxxx --secret-key XXX

  python main.py --mode loop --broker dry_run       # continuous loop (dry run)
  python main.py --mode loop --broker paper \       # continuous loop (Alpaca paper)
      --api-key PKxxx --secret-key XXX --once

  python main.py --mode loop --broker live --live \ # LIVE TRADING
      --api-key AKxxx --secret-key XXX
"""
import argparse, json, os, warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from datetime import datetime
from pathlib  import Path
import sys
sys.path.insert(0, str(Path(__file__).parent))

from config          import WATCHLIST, CAPITAL, CONFIDENCE_THRESHOLD, OUTPUT_DIR
from core.data       import prepare_dataset
from core.models     import EnsembleModel
from core.strategy   import SignalGenerator
from core.trainer    import ModelTrainer
from backtest.engine import Backtester


def run_train(tickers=None, force=False):
    tickers = tickers or WATCHLIST
    print(f"\n{'='*60}\n  TRAINING  —  {', '.join(tickers)}\n{'='*60}\n")
    models = ModelTrainer().train_all(tickers, force=force)
    print(f"\n✓ Trained {len(models)}/{len(tickers)} models.")
    return models


def run_signals(tickers=None, capital=CAPITAL):
    tickers = tickers or WATCHLIST
    generator = SignalGenerator(capital=capital)
    signals, skipped, open_heat = [], [], 0.0
    print(f"\n{'='*60}\n  SIGNALS  —  {datetime.now():%Y-%m-%d %H:%M:%S}\n{'='*60}\n")
    for ticker in tickers:
        print(f"  {ticker} ...", end=" ", flush=True)
        model = EnsembleModel()
        if not model.load(ticker):
            df = prepare_dataset(ticker)
            if df is None:
                print("SKIP"); skipped.append(ticker); continue
            model.train(df); model.save(ticker)
        df = prepare_dataset(ticker)
        if df is None:
            print("SKIP"); skipped.append(ticker); continue
        try:
            sig = generator.generate(ticker, df, model.predict(df), open_heat)
            if sig.action != "HOLD":
                open_heat += sig.position_size_pct / 100
            signals.append(sig)
            print(f"{sig.action}  conf={sig.confidence_pct}%  p_up={sig.p_up:.2f}")
        except Exception as e:
            print(f"ERROR: {e}"); skipped.append(ticker)
    print(f"\n{'─'*60}\n  RECOMMENDATIONS\n")
    for s in sorted([s for s in signals if s.action != "HOLD"], key=lambda x: -x.confidence):
        print(s)
    hold = [s for s in signals if s.action == "HOLD"]
    if hold:
        print(f"\n  HOLD: {', '.join(s.ticker for s in hold)}")
    p = OUTPUT_DIR / "signals_latest.json"
    p.write_text(json.dumps({"generated_at": datetime.now().isoformat(),
        "signals": [s.to_dict() for s in signals]}, indent=2, default=str))
    print(f"\n  Saved -> {p}")
    return signals


def run_backtest(ticker="AAPL", days=365):
    print(f"\n{'='*60}\n  BACKTEST  {ticker}  {days}d\n{'='*60}\n")
    df = prepare_dataset(ticker)
    if df is None:
        print("ERROR: no data"); return
    df = df.iloc[-days:].copy()
    split = int(len(df)*0.7)
    df_train, df_test = df.iloc[:split], df.iloc[split:]
    print(f"  Train {len(df_train)} | Test {len(df_test)}")
    model = EnsembleModel()
    model.train(df_train)
    probs = []
    for i in range(len(df_test)):
        window = pd.concat([df_train, df_test.iloc[:i+1]])
        try: probs.append(model.predict(window)["p_up"])
        except: probs.append(np.nan)
    bt = Backtester(capital=CAPITAL)
    res = bt.run(ticker, df_test, pd.Series(probs, index=df_test.index))
    bt.print_report(res)
    bh = (df_test["close"].iloc[-1] / df_test["close"].iloc[0]) - 1
    print(f"\n  Buy&hold {bh:.2%}  |  Strategy {res['total_return']:.2%}  |  Alpha {res['total_return']-bh:.2%}")
    eq = res["equity"].to_frame("strategy")
    eq["buy_hold"] = CAPITAL * (df_test["close"] / df_test["close"].iloc[0])
    p = OUTPUT_DIR / f"backtest_{ticker}.csv"
    eq.to_csv(p); print(f"  Equity curve -> {p}")
    return res


def run_status():
    status = ModelTrainer().get_status()
    print(f"\n{'='*60}\n  MODEL STATUS\n{'='*60}")
    if not status:
        print("  No models. Run: python main.py --mode train"); return
    for ticker, rec in status.items():
        flag = "RETRAIN" if rec["needs_retrain"] else "Fresh"
        print(f"\n  {ticker:10} [{flag}]  age={rec['age_days']}d")
        for m, mt in rec["metrics"].items():
            print(f"    {m:12}  acc={mt.get('accuracy',0):.3f}  auc={mt.get('auc',0):.3f}")


def run_execute(tickers=None, capital=CAPITAL,
                broker_mode="dry_run", api_key="", secret_key=""):
    from broker.client   import BrokerClient, BrokerMode
    from broker.executor import Executor
    broker = BrokerClient(mode=BrokerMode(broker_mode),
                          api_key=api_key, secret_key=secret_key)
    if not broker.connect():
        print("ERROR: broker connection failed"); return
    signals = run_signals(tickers or WATCHLIST, capital)
    price_map = {}
    for t in (tickers or WATCHLIST):
        df = prepare_dataset(t)
        if df is not None:
            price_map[t] = float(df.iloc[-1]["close"])
    Executor(broker, dry_run_price_map=price_map).execute_signals(signals)
    broker.print_summary()
    if hasattr(broker, "save_state"):
        broker.save_state()


def run_live_loop(tickers=None, capital=CAPITAL,
                  broker_mode="dry_run", api_key="", secret_key="",
                  live=False, once=False):
    from broker.client    import BrokerMode
    from broker.live_loop import run_loop
    run_loop(mode=BrokerMode(broker_mode), tickers=tickers or WATCHLIST,
             api_key=api_key, secret_key=secret_key,
             live=live, once=once, capital=capital)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="AI Trading System")
    ap.add_argument("--mode", choices=["train","signals","backtest","status",
                                        "execute","loop","full"], default="signals")
    ap.add_argument("--ticker");  ap.add_argument("--tickers", nargs="+")
    ap.add_argument("--days",    type=int,   default=365)
    ap.add_argument("--capital", type=float, default=CAPITAL)
    ap.add_argument("--force",   action="store_true")
    ap.add_argument("--broker",  default="dry_run",
                    choices=["dry_run","paper","live"])
    ap.add_argument("--api-key",    default=os.getenv("ALPACA_API_KEY",""))
    ap.add_argument("--secret-key", default=os.getenv("ALPACA_SECRET_KEY",""))
    ap.add_argument("--live",  action="store_true")
    ap.add_argument("--once",  action="store_true")
    args = ap.parse_args()
    tickers = args.tickers or ([args.ticker] if args.ticker else None)

    if   args.mode == "train":    run_train(tickers, force=args.force)
    elif args.mode == "signals":  run_signals(tickers, args.capital)
    elif args.mode == "backtest": run_backtest(args.ticker or "AAPL", args.days)
    elif args.mode == "status":   run_status()
    elif args.mode == "execute":
        run_execute(tickers, args.capital, args.broker,
                    getattr(args, 'api_key', ''), getattr(args, 'secret_key', ''))
    elif args.mode == "loop":
        run_live_loop(tickers, args.capital, args.broker,
                      getattr(args, 'api_key', ''), getattr(args, 'secret_key', ''),
                      args.live, args.once)
    elif args.mode == "full":
        run_train(tickers, force=args.force)
        run_execute(tickers, args.capital, args.broker,
                    getattr(args, 'api_key', ''), getattr(args, 'secret_key', ''))
