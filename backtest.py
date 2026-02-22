"""
Backtesting Engine
Implements a daily signal-based strategy with stop-loss,
position sizing, and PnL tracking for EUR/USD.
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import List


@dataclass
class BacktestConfig:
    """Strategy configuration."""
    initial_capital: float = 100_000.0   # USD
    position_size_pct: float = 0.10      # 10% of capital per trade
    stop_loss_pct: float = 0.005         # 0.5% stop loss on position (in price terms)
    take_profit_pct: float = None        # None = no take-profit, just EOD exit
    transaction_cost_pips: float = 1.0   # spread in pips (1 pip = 0.0001)
    leverage: float = 1.0               # leverage multiplier


@dataclass
class Trade:
    """Records a single trade."""
    date: pd.Timestamp
    direction: int          # +1 long, -1 short
    entry_price: float
    exit_price: float
    stop_loss_price: float
    pnl_usd: float
    stopped_out: bool
    position_size: float    # in base currency (EUR)


def run_backtest(df: pd.DataFrame, config: BacktestConfig = None) -> dict:
    """
    Run the backtest on signal data.

    Assumes:
      - Signal is known at close of day T
      - Trade is entered at open of day T+1
      - Stop-loss is monitored intraday using High/Low of T+1
      - Exit at close of T+1 if not stopped out

    Returns a dict with:
      - trades: list of Trade objects
      - equity_curve: pd.Series
      - metrics: dict of performance metrics
    """
    if config is None:
        config = BacktestConfig()

    trades: List[Trade] = []
    capital = config.initial_capital
    equity_curve = []
    pip = 0.0001
    transaction_cost = config.transaction_cost_pips * pip

    # We need next-day OHLC to simulate stop-loss
    # Signal day is T, trade day is T+1
    for i in range(len(df) - 1):
        row = df.iloc[i]
        next_row = df.iloc[i + 1]

        signal = row['ML_Signal']

        # Entry at next day open
        entry_price = next_row['Open']
        high = next_row['High']
        low = next_row['Low']
        exit_price = next_row['Close']  # default: EOD exit

        # Position size in EUR
        position_value = capital * config.position_size_pct * config.leverage
        position_size = position_value / entry_price

        # Stop-loss levels
        if signal == 1:  # LONG
            stop_price = entry_price * (1 - config.stop_loss_pct)
            stopped_out = low <= stop_price
            actual_exit = stop_price if stopped_out else exit_price
            price_move = actual_exit - entry_price - transaction_cost
        else:  # SHORT
            stop_price = entry_price * (1 + config.stop_loss_pct)
            stopped_out = high >= stop_price
            actual_exit = stop_price if stopped_out else exit_price
            price_move = entry_price - actual_exit - transaction_cost

        pnl_usd = position_size * price_move
        capital += pnl_usd

        trades.append(Trade(
            date=next_row['Date'],
            direction=signal,
            entry_price=entry_price,
            exit_price=actual_exit,
            stop_loss_price=stop_price,
            pnl_usd=pnl_usd,
            stopped_out=stopped_out,
            position_size=position_size,
        ))

        equity_curve.append({'Date': next_row['Date'], 'Equity': capital})

    equity_df = pd.DataFrame(equity_curve).set_index('Date')

    trades_df = pd.DataFrame([
        {
            'Date': t.date,
            'Direction': 'LONG' if t.direction == 1 else 'SHORT',
            'Entry': t.entry_price,
            'Exit': t.exit_price,
            'Stop_Loss': t.stop_loss_price,
            'PnL_USD': t.pnl_usd,
            'Stopped_Out': t.stopped_out,
            'Position_EUR': t.position_size,
        }
        for t in trades
    ])

    metrics = compute_metrics(trades_df, equity_df, config)

    return {
        'trades': trades_df,
        'equity': equity_df,
        'metrics': metrics,
        'config': config,
    }


def compute_metrics(trades_df: pd.DataFrame, equity_df: pd.DataFrame, config: BacktestConfig) -> dict:
    """Compute standard performance metrics."""
    pnl = trades_df['PnL_USD']
    equity = equity_df['Equity']

    total_trades = len(trades_df)
    winners = (pnl > 0).sum()
    losers = (pnl <= 0).sum()
    win_rate = winners / total_trades if total_trades > 0 else 0

    total_pnl = pnl.sum()
    avg_win = pnl[pnl > 0].mean() if winners > 0 else 0
    avg_loss = pnl[pnl <= 0].mean() if losers > 0 else 0
    profit_factor = (pnl[pnl > 0].sum() / abs(pnl[pnl <= 0].sum())) if losers > 0 else np.inf

    # Drawdown
    rolling_max = equity.cummax()
    drawdown = (equity - rolling_max) / rolling_max
    max_drawdown = drawdown.min()

    # Sharpe ratio (annualized, assuming ~252 trading days)
    daily_returns = equity.pct_change().dropna()
    sharpe = (daily_returns.mean() / daily_returns.std()) * np.sqrt(252) if daily_returns.std() > 0 else 0

    # Calmar ratio
    annual_return = (equity.iloc[-1] / equity.iloc[0]) ** (252 / len(equity)) - 1
    calmar = annual_return / abs(max_drawdown) if max_drawdown != 0 else np.inf

    stop_outs = trades_df['Stopped_Out'].sum()

    return {
        'Total Trades': total_trades,
        'Win Rate': f'{win_rate:.1%}',
        'Total PnL (USD)': f'{total_pnl:,.2f}',
        'Total Return': f'{(equity.iloc[-1] / config.initial_capital - 1):.2%}',
        'Annualized Return': f'{annual_return:.2%}',
        'Max Drawdown': f'{max_drawdown:.2%}',
        'Sharpe Ratio': f'{sharpe:.2f}',
        'Calmar Ratio': f'{calmar:.2f}',
        'Profit Factor': f'{profit_factor:.2f}',
        'Avg Win (USD)': f'{avg_win:,.2f}',
        'Avg Loss (USD)': f'{avg_loss:,.2f}',
        'Stop-outs': stop_outs,
        'Stop-out Rate': f'{stop_outs / total_trades:.1%}',
    }
