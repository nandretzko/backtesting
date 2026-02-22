"""
Data Loader Module
Handles loading and preprocessing EUR/USD historical data.
"""

import pandas as pd
import numpy as np


def load_eurusd(filepath: str) -> pd.DataFrame:
    """Load EUR/USD CSV with French locale formatting."""
    df = pd.read_csv(filepath, sep=',', encoding='utf-8-sig')

    # Rename columns
    df.columns = ['Date', 'Close', 'Open', 'High', 'Low', 'Change_pct']

    # Parse date (format: DD/MM/YYYY)
    df['Date'] = pd.to_datetime(df['Date'], format='%d/%m/%Y')
    df = df.sort_values('Date').reset_index(drop=True)

    # Convert French number format (comma decimal) to float
    for col in ['Close', 'Open', 'High', 'Low']:
        df[col] = df[col].astype(str).str.replace(',', '.').str.replace('"', '').astype(float)

    # Parse Change_pct
    df['Change_pct'] = (
        df['Change_pct'].astype(str)
        .str.replace('%', '')
        .str.replace(',', '.')
        .str.replace('"', '')
        .astype(float)
    )

    # Compute log returns
    df['Return'] = df['Close'].pct_change()
    df['Log_Return'] = np.log(df['Close'] / df['Close'].shift(1))

    df = df.dropna().reset_index(drop=True)
    return df
