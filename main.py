"""
main.py — Entry point for EUR/USD ML Signal Backtesting
Usage:
    python main.py
    python main.py --accuracy 0.55 --stop-loss 0.003 --capital 50000
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

from src.data_loader import load_eurusd
from src.signal_simulator import simulate_ml_signal
from src.backtest import BacktestConfig, run_backtest
from src.plot import plot_full_report


def main():
    parser = argparse.ArgumentParser(description='EUR/USD ML Signal Backtest')
    parser.add_argument('--data',       default='data/EUR_USD.csv')
    parser.add_argument('--accuracy',   type=float, default=0.60,    help='ML model accuracy (0-1)')
    parser.add_argument('--stop-loss',  type=float, default=0.005,   help='Stop-loss pct (e.g. 0.005 = 0.5%)')
    parser.add_argument('--capital',    type=float, default=100_000, help='Initial capital (USD)')
    parser.add_argument('--position',   type=float, default=0.10,    help='Position size as fraction of capital')
    parser.add_argument('--leverage',   type=float, default=1.0)
    parser.add_argument('--seed',       type=int,   default=42)
    parser.add_argument('--output',     default='results/backtest_report.png')
    args = parser.parse_args()

    os.makedirs('results', exist_ok=True)

    print('━' * 55)
    print('  EUR/USD · ML SIGNAL BACKTESTING ENGINE')
    print('━' * 55)

    # 1. Load data
    print('\n[1/4] Loading data...')
    df = load_eurusd(args.data)
    print(f'  ✓ {len(df)} trading days loaded ({df["Date"].min().date()} → {df["Date"].max().date()})')

    # 2. Simulate ML signal
    print(f'\n[2/4] Simulating ML signal (accuracy={args.accuracy:.0%}, seed={args.seed})...')
    df = simulate_ml_signal(df, accuracy=args.accuracy, random_seed=args.seed)
    empirical_acc = df['Signal_Correct'].mean()
    print(f'  ✓ Empirical accuracy: {empirical_acc:.2%}')
    print(f'  ✓ LONG signals: {(df["ML_Signal"] == 1).sum()} | SHORT signals: {(df["ML_Signal"] == -1).sum()}')

    # 3. Run backtest
    print(f'\n[3/4] Running backtest...')
    config = BacktestConfig(
        initial_capital=args.capital,
        position_size_pct=args.position,
        stop_loss_pct=args.stop_loss,
        leverage=args.leverage,
    )
    result = run_backtest(df, config)

    # 4. Print metrics
    print(f'\n[4/4] Performance Metrics:')
    print('─' * 40)
    for k, v in result['metrics'].items():
        print(f'  {k:<25} {v}')
    print('─' * 40)

    # 5. Plot
    print(f'\n  Generating report chart...')
    plot_full_report(result, save_path=args.output)

    # 6. Save trades CSV
    csv_path = args.output.replace('.png', '_trades.csv')
    result['trades'].to_csv(csv_path, index=False)
    print(f'  ✓ Trades CSV saved → {csv_path}')

    equity_path = args.output.replace('.png', '_equity.csv')
    result['equity'].to_csv(equity_path)
    print(f'  ✓ Equity curve saved → {equity_path}')

    print('\n  Done! ✓')
    print('━' * 55)


if __name__ == '__main__':
    main()
