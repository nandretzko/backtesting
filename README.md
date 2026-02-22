# EUR/USD ML Signal Backtesting Engine

A complete quantitative backtesting framework for a machine learning-based EUR/USD trading strategy.

## Project Structure

```
eurusd_backtest/
├── data/
│   └── EUR_USD.csv          # Historical EUR/USD rates
├── src/
│   ├── data_loader.py        # CSV ingestion & preprocessing
│   ├── signal_simulator.py   # ML signal simulation (~60% accuracy)
│   ├── backtest.py           # Backtesting engine + metrics
│   └── plot.py               # PnL charts & reporting
├── results/                  # Generated outputs
├── main.py                   # CLI entry point
├── requirements.txt
└── README.md
```

## Strategy

| Parameter | Default |
|-----------|---------|
| Initial capital | $100,000 |
| Position size | 10% of capital |
| Stop-loss | 0.5% of entry price |
| Leverage | 1x |
| Entry | Next-day open |
| Exit | Next-day close (or stop-loss intraday) |

### Signal Logic
- **+1 (LONG)**: Model predicts EUR/USD will rise → Buy at open T+1, exit at close T+1
- **-1 (SHORT)**: Model predicts EUR/USD will fall → Sell at open T+1, exit at close T+1
- **Stop-loss**: If price hits stop level intraday (based on High/Low), position closes early

## Installation

```bash
pip install -r requirements.txt
```

## Usage

```bash
# Run with defaults (60% accuracy, 0.5% stop-loss, $100k capital)
python main.py

# Custom parameters
python main.py --accuracy 0.55 --stop-loss 0.003 --capital 50000 --position 0.15

# Different random seed for signal simulation
python main.py --seed 123
```

## CLI Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--stop-loss` | 0.005 | Stop-loss % (0.5%) |
| `--capital` | 100000 | Initial capital (USD) |
| `--position` | 0.10 | Position size as fraction of capital |
| `--leverage` | 1.0 | Leverage multiplier |
| `--seed` | 42 | Random seed for reproducibility |
| `--output` | results/backtest_report.png | Output chart path |

## Outputs

- `results/backtest_report.png` — 6-panel performance dashboard
- `results/backtest_report_trades.csv` — All trade details
- `results/backtest_report_equity.csv` — Daily equity curve

## Metrics Computed

- Win rate, Total PnL, Total & Annualized Return
- Maximum Drawdown, Sharpe Ratio, Calmar Ratio
- Profit Factor, Avg Win/Loss, Stop-out rate
