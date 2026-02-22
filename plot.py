"""
Visualization Module
Generates comprehensive PnL and backtest analysis charts.
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.dates as mdates
import numpy as np
import pandas as pd
from matplotlib.patches import FancyBboxPatch
from matplotlib.ticker import FuncFormatter


# ── Palette ──────────────────────────────────────────────────────────────────
DARK_BG   = '#0d1117'
PANEL_BG  = '#161b22'
ACCENT    = '#58a6ff'
GREEN     = '#3fb950'
RED       = '#f85149'
AMBER     = '#e3b341'
MUTED     = '#8b949e'
WHITE     = '#e6edf3'

def _apply_dark_theme():
    plt.rcParams.update({
        'figure.facecolor':  DARK_BG,
        'axes.facecolor':    PANEL_BG,
        'axes.edgecolor':    '#30363d',
        'axes.labelcolor':   WHITE,
        'axes.titlecolor':   WHITE,
        'xtick.color':       MUTED,
        'ytick.color':       MUTED,
        'grid.color':        '#21262d',
        'grid.linewidth':    0.6,
        'text.color':        WHITE,
        'legend.facecolor':  PANEL_BG,
        'legend.edgecolor':  '#30363d',
        'font.family':       'monospace',
    })

def usd_fmt(x, pos):
    return f'${x:,.0f}'

def pct_fmt(x, pos):
    return f'{x:.1%}'


def plot_full_report(result: dict, save_path: str = 'results/backtest_report.png'):
    """Generate a full 6-panel backtest report."""
    _apply_dark_theme()

    trades  = result['trades']
    equity  = result['equity']['Equity']
    metrics = result['metrics']
    config  = result['config']

    fig = plt.figure(figsize=(20, 24), facecolor=DARK_BG)
    fig.suptitle(
        'EUR/USD · ML Signal Backtest Report',
        fontsize=22, fontweight='bold', color=WHITE, y=0.98,
        fontfamily='monospace',
    )

    gs = gridspec.GridSpec(4, 2, figure=fig, hspace=0.45, wspace=0.30,
                           top=0.95, bottom=0.04, left=0.07, right=0.97)

    # ── 1. Equity Curve ───────────────────────────────────────────────────────
    ax1 = fig.add_subplot(gs[0, :])
    ax1.set_title('Equity Curve', fontsize=13, fontweight='bold', pad=10)

    initial = config.initial_capital
    ax1.axhline(initial, color=MUTED, lw=0.8, ls='--', alpha=0.5, label='Initial capital')
    ax1.fill_between(equity.index, initial, equity.values,
                     where=equity.values >= initial, alpha=0.15, color=GREEN)
    ax1.fill_between(equity.index, initial, equity.values,
                     where=equity.values < initial, alpha=0.15, color=RED)
    ax1.plot(equity.index, equity.values, color=ACCENT, lw=1.6, label='Portfolio equity')

    # Drawdown shading
    rolling_max = equity.cummax()
    in_dd = equity < rolling_max
    ax1.fill_between(equity.index, rolling_max, equity,
                     where=in_dd, alpha=0.08, color=RED, label='Drawdown')

    ax1.yaxis.set_major_formatter(FuncFormatter(usd_fmt))
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
    ax1.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
    ax1.tick_params(axis='x', rotation=30)
    ax1.grid(True, alpha=0.4)
    ax1.legend(loc='upper left', fontsize=9)

    # ── 2. Daily PnL Bar Chart ─────────────────────────────────────────────────
    ax2 = fig.add_subplot(gs[1, :])
    ax2.set_title('Daily PnL per Trade', fontsize=13, fontweight='bold', pad=10)

    pnl = trades['PnL_USD']
    colors = [GREEN if v >= 0 else RED for v in pnl]
    ax2.bar(trades['Date'], pnl, color=colors, alpha=0.85, width=1.0)
    ax2.axhline(0, color=MUTED, lw=0.8)
    ax2.yaxis.set_major_formatter(FuncFormatter(usd_fmt))
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
    ax2.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
    ax2.tick_params(axis='x', rotation=30)
    ax2.grid(True, alpha=0.4, axis='y')

    # ── 3. Drawdown ────────────────────────────────────────────────────────────
    ax3 = fig.add_subplot(gs[2, 0])
    ax3.set_title('Drawdown Over Time', fontsize=12, fontweight='bold', pad=10)

    dd = (equity - equity.cummax()) / equity.cummax()
    ax3.fill_between(dd.index, dd.values, 0, color=RED, alpha=0.6)
    ax3.plot(dd.index, dd.values, color=RED, lw=1.0)
    ax3.yaxis.set_major_formatter(FuncFormatter(pct_fmt))
    ax3.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    ax3.xaxis.set_major_locator(mdates.YearLocator())
    ax3.tick_params(axis='x', rotation=30)
    ax3.grid(True, alpha=0.4)

    # ── 4. PnL Distribution ───────────────────────────────────────────────────
    ax4 = fig.add_subplot(gs[2, 1])
    ax4.set_title('PnL Distribution', fontsize=12, fontweight='bold', pad=10)

    wins  = pnl[pnl > 0]
    loss  = pnl[pnl <= 0]
    bins  = np.linspace(pnl.min(), pnl.max(), 40)
    ax4.hist(loss,  bins=bins, color=RED,   alpha=0.75, label='Losses',  edgecolor='none')
    ax4.hist(wins,  bins=bins, color=GREEN, alpha=0.75, label='Wins',    edgecolor='none')
    ax4.axvline(pnl.mean(), color=AMBER, lw=1.5, ls='--', label=f'Mean: ${pnl.mean():,.0f}')
    ax4.xaxis.set_major_formatter(FuncFormatter(usd_fmt))
    ax4.grid(True, alpha=0.4, axis='y')
    ax4.legend(fontsize=8)

    # ── 5. Cumulative win/loss count ──────────────────────────────────────────
    ax5 = fig.add_subplot(gs[3, 0])
    ax5.set_title('Cumulative Wins vs Losses', fontsize=12, fontweight='bold', pad=10)

    cum_wins  = (pnl > 0).cumsum()
    cum_loss  = (pnl <= 0).cumsum()
    ax5.plot(trades['Date'], cum_wins, color=GREEN, lw=1.5, label='Wins')
    ax5.plot(trades['Date'], cum_loss, color=RED,   lw=1.5, label='Losses')
    ax5.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    ax5.xaxis.set_major_locator(mdates.YearLocator())
    ax5.tick_params(axis='x', rotation=30)
    ax5.grid(True, alpha=0.4)
    ax5.legend(fontsize=8)

    # ── 6. Metrics Table ──────────────────────────────────────────────────────
    ax6 = fig.add_subplot(gs[3, 1])
    ax6.set_title('Performance Metrics', fontsize=12, fontweight='bold', pad=10)
    ax6.axis('off')

    rows = list(metrics.items())
    mid = len(rows) // 2
    col1, col2 = rows[:mid], rows[mid:]

    def _draw_kv(ax, items, x_start):
        y = 0.95
        for k, v in items:
            color = WHITE
            if 'PnL' in k or 'Return' in k or 'Win' in k or 'Profit' in k:
                try:
                    val = float(v.replace('$','').replace(',','').replace('%','').replace('USD',''))
                    color = GREEN if val > 0 else RED
                except:
                    pass
            ax.text(x_start,      y, k + ':', transform=ax.transAxes,
                    fontsize=9, color=MUTED, ha='left', va='top')
            ax.text(x_start+0.45, y, v, transform=ax.transAxes,
                    fontsize=9, color=color, ha='left', va='top', fontweight='bold')
            y -= 0.13

    _draw_kv(ax6, col1, 0.0)
    _draw_kv(ax6, col2, 0.5)

    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor=DARK_BG)
    plt.close(fig)
    print(f'  ✓ Report saved → {save_path}')
