# application_logic/analytics_service.py

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pandas.tseries.offsets import DateOffset
from typing import Dict, Any, List
# Assuming the data_layer is accessible from the application_logic directory
from data_layer.db_manager import DBManager


class AnalyticsService:
    """
    Calculates all financial performance metrics and prepares charting data.
    """
    def __init__(self, db_manager: DBManager):
        self.db = db_manager

    def get_cumulative_returns(self, start_date: str = None):
        """
        EVOLVED: Now accepts a start_date to support timeframe filtering (1M, 6M, ALL).
        """
        # 1. Fallback logic: If no start_date provided, default to current year
        if not start_date:
            start_date = datetime.now().replace(month=1, day=1).strftime('%Y-%m-%d')

        # 2. Fetch data using the existing calculation logic
        # This calls self.calculate_cumulative_returns which filters snapshots by start_date
        result = self.calculate_cumulative_returns(start_date)

        # 3. Reformat for Frontend: [{"date":..., "equity":...}]
        formatted_data = []
        for i in range(len(result['timestamps'])):
            formatted_data.append({
                "date": result['timestamps'][i],
                "equity": result['cumulative_returns'][i]
            })

        return formatted_data

    # --- Utility Method to Get Returns Series ---
    def _get_daily_returns_series(self, start_date) -> pd.Series:
        """Helper to retrieve and calculate the core daily returns series."""
        data = self.db.get_daily_nav_snapshots(start_date)

        if not data:
            return pd.Series([0.0]) # Return minimal series if no data

        df = pd.DataFrame(data).set_index('snapshot_date').sort_index()

        # Calculate daily returns (adjusted for cash flow)
        # Formula: (Today_Value - Prev_Value - Cash_Flow) / Prev_Value
        df['prev_value'] = df['total_value'].shift(1).fillna(df['total_value'].iloc[0])
        df['daily_return'] = (df['total_value'] - df['prev_value'] - df['cash_flow'].fillna(0)) / df['prev_value']

        # Set the first return to 0
        df.iloc[0, df.columns.get_loc('daily_return')] = 0.0

        return df['daily_return']

    def _get_nav_dataframe(self, start_date='2000-01-01') -> pd.DataFrame:
        """Helper to get the raw Total Value (NAV) series indexed by date."""
        data = self.db.get_daily_nav_snapshots(start_date)
        if not data:
            return pd.DataFrame()

        df = pd.DataFrame(data).set_index('snapshot_date').sort_index()
        return df[['total_value']]

    # --- 1. Cumulative Returns (Revised for modularity) ---
    def calculate_cumulative_returns(self, start_date):
        """
        Calculates the cumulative portfolio returns and drawdown series.
        """
        daily_returns = self._get_daily_returns_series(start_date)

        if daily_returns.empty or len(daily_returns) <= 1:
            return {"timestamps": [], "cumulative_returns": [], "drawdown_series": []}

        # 1. Calculate Wealth Index (Cumulative Returns)
        wealth_index = (1 + daily_returns).cumprod()
        cumulative_returns = wealth_index - 1

        # 2. Calculate Drawdown Series
        # We calculate this here so the backend provides the source of truth
        running_peak = wealth_index.cummax()
        # We multiply by -100 to provide values like -5.5 for a 5.5% drop, matching chart expectations
        drawdown_series = (wealth_index / running_peak - 1) * 100

        # --- DEBUG TRACE: Verify Drawdown Calculation ---
        print(f"DEBUG: Sample Wealth Index: {wealth_index.tail(5).values}")
        print(f"DEBUG: Sample Running Peak: {running_peak.tail(5).values}")
        print(f"DEBUG: Sample Drawdown %: {drawdown_series.tail(5).values}")

        return {
            'timestamps': cumulative_returns.index.strftime('%Y-%m-%d').tolist(),
            'cumulative_returns': cumulative_returns.tolist(),
            'drawdown_series': drawdown_series.tolist()
        }

    # --- 2. Risk Metrics Implementation ---
    def calculate_risk_metrics(self, start_date, risk_free_rate_annual=0.02):
        """
        Computes Sharpe Ratio and Maximum Drawdown.
        UPDATED: Added keys for ReportsEngine without changing Dashboard logic.
        """
        daily_returns = self._get_daily_returns_series(start_date)

        if len(daily_returns) < 2:
            return {
                "sharpe_ratio": 0.0, "max_drawdown": 0.0, "volatility_annualized": 0.0,
                "mdd": "0%", "mdd_dates": "N/A", "pf": "0.0", "wr": "0%", "rf": "0.0"
            }

        # --- A. Maximum Drawdown (MDD) ---
        wealth_index = (1 + daily_returns).cumprod()
        running_peak = wealth_index.cummax()
        drawdown_series = 1 - (wealth_index / running_peak)
        max_drawdown = drawdown_series.max()

        # --- DEBUG TRACE 1: Inspect the Series ---
        print(f"DEBUG: daily_returns length: {len(daily_returns)}")
        print(f"DEBUG: drawdown_series max value: {max_drawdown}")

        # NEW: Find MDD Dates for Reports
        try:
            trough_date = drawdown_series.idxmax()
            # --- DEBUG TRACE 2: Verify Trough Date ---
            print(f"DEBUG: trough_date found: {trough_date}")

            peak_date = wealth_index.loc[:trough_date].idxmax()
            # --- DEBUG TRACE 3: Verify Peak Date ---
            print(f"DEBUG: peak_date found: {peak_date}")

            mdd_dates = f"{peak_date.strftime('%Y-%m-%d')} to {trough_date.strftime('%Y-%m-%d')}"
        except Exception as e:
            # --- DEBUG TRACE 4: Catch Failures ---
            print(f"DEBUG: Error calculating dates: {e}")
            mdd_dates = "N/A to N/A"

        print(f"DEBUG: Final formatted string: {mdd_dates}")

        # --- B. Sharpe Ratio (Annualized) ---
        R_f_daily = (1 + risk_free_rate_annual) ** (1/252) - 1
        avg_excess_return = daily_returns.mean() - R_f_daily
        daily_volatility = daily_returns.std()

        if daily_volatility == 0:
            sharpe_ratio = 0.0
        else:
            sharpe_ratio = (avg_excess_return / daily_volatility) * np.sqrt(252)

        # NEW: Profit Factor & Win Rate Calculation for Reports
        gains = daily_returns[daily_returns > 0]
        losses = daily_returns[daily_returns < 0]
        win_rate = len(gains) / len(daily_returns[daily_returns != 0]) if not daily_returns[daily_returns != 0].empty else 0
        profit_factor = abs(gains.sum() / losses.sum()) if not losses.empty and losses.sum() != 0 else 0
        recovery_factor = abs(daily_returns.sum() / max_drawdown) if max_drawdown != 0 else 0

        # Return combined dictionary to satisfy both Dashboard and Reports
        return {
            # ORIGINAL KEYS (Keeps Dashboard working)
            "max_drawdown": round(float(max_drawdown), 4),
            "sharpe_ratio": round(float(sharpe_ratio), 4),
            "volatility_annualized": round(float(daily_volatility * np.sqrt(252)), 4),

            # NEW KEYS (Fixes ReportsEngine stats)
            "mdd": f"{round(float(max_drawdown * 100), 2)}%",
            "mdd_dates": mdd_dates,
            "pf": str(round(float(profit_factor), 2)),
            "wr": f"{round(float(win_rate * 100), 1)}%",
            "rf": str(round(float(recovery_factor), 2))
        }

    # --- 3. Periodic Returns (Implemented) ---
    def calculate_periodic_returns(self, start_date=None):
        """
        Computes trailing returns for standard time periods (1W, 1M, YTD, etc.)
        based on the latest available portfolio snapshot.
        """
        # Fetch all available history to ensure we can look back far enough
        # We fetch full history, and index by date is already set in _get_nav_dataframe
        df = self._get_nav_dataframe(start_date='2000-01-01')

        if df.empty:
            return {"1W": 0.0, "1M": 0.0, "3M": 0.0, "6M": 0.0, "1Y": 0.0, "YTD": 0.0}

        latest_date = df.index[-1]
        latest_val = df['total_value'].iloc[-1]

        # Use Pandas DateOffset for accurate financial periods (respects month/year lengths)
        from pandas.tseries.offsets import DateOffset

        periods = {
            '1W': DateOffset(weeks=1),
            '1M': DateOffset(months=1),
            '3M': DateOffset(months=3),
            '6M': DateOffset(months=6),
            '1Y': DateOffset(years=1)
        }

        results = {}

        # 1. Calculate Trailing Periods
        for label, offset in periods.items():
            # Calculate the accurate target start date using DateOffset
            target_date = latest_date - offset

            try:
                # Use .asof() to find the last index value at or before the target date.
                # This is the most robust way to get the historical value for a trailing return.
                hist_date = df.index.asof(target_date)

                if pd.isna(hist_date) or hist_date == latest_date: # Check if data was found
                    results[label] = None
                    continue

                hist_val = df.loc[hist_date, 'total_value']

                if hist_val == 0:
                    results[label] = 0.0
                else:
                    # Return formula: (Latest Value - Historical Value) / Historical Value
                    results[label] = round((latest_val - hist_val) / hist_val, 4)
            except Exception:
                results[label] = None

        # 2. Calculate YTD (Year-To-Date)
        current_year = latest_date.year
        # Find the start of the year as a datetime object
        start_of_year = latest_date.replace(month=1, day=1, hour=0, minute=0, second=0, microsecond=0)

        try:
            # Use .asof() to find the last data point in the previous year or the first in the current year.
            ytd_date = df.index.asof(start_of_year)

            if pd.isna(ytd_date):
                 results['YTD'] = None
            else:
                ytd_val = df.loc[ytd_date, 'total_value']

                if ytd_val == 0:
                    results['YTD'] = 0.0
                else:
                    results['YTD'] = round((latest_val - ytd_val) / ytd_val, 4)
        except Exception:
            results['YTD'] = None

        return results

    # --- Charting Data Prep Implementation ---
    def get_signal_chart_data(self, ticker: str, start_date: str):
        query = """
            SELECT timestamp, close_price, signal_A_value, signal_B_value, position
            FROM market_signal
            WHERE ticker = ? AND timestamp >= ?
            ORDER BY timestamp ASC
        """
        # Use self.db as defined in your __init__
        df = pd.read_sql(query, self.db.engine, params=(ticker, start_date))

        # Ensure timestamp is a datetime object
        df['timestamp'] = pd.to_datetime(df['timestamp'])

        # --- NO OVERWRITE HERE ---
        # We deleted the line: df['position'] = (sig_A + sig_B) / 2
        # This keeps the DB truth (0.0 or 1.0)

        return {
            # FIX: map(str, ...) or .astype(str) is safer than .dt.isoformat() on a Series
            "timestamps": df['timestamp'].dt.strftime('%Y-%m-%dT%H:%M:%S').tolist(),
            "price": df['close_price'].tolist(),
            "position": df['position'].tolist(),
            "component_signals": [
                {"name": "TMOM", "data": df['signal_A_value'].tolist()},
                {"name": "SMA", "data": df['signal_B_value'].tolist()}
            ]
        }

    def get_current_position_state(self, ticker: str = "LQQ.PA"):
        latest = self.db.get_latest_market_signal(ticker)
        if not latest:
            return 0.0

        val_a = latest.get('signal_A_value', 0) or 0
        val_b = latest.get('signal_B_value', 0) or 0
        return (val_a + val_b) / 2

    def get_monthly_returns_heatmap(self, year="ALL"):
        """
        New: Transforms daily returns into a Year/Month matrix for the Heatmap.
        """
        # Fetch full history of daily returns
        daily_returns = self._get_daily_returns_series(start_date="2000-01-01")
        if daily_returns.empty:
            return []

        # Convert to DataFrame for easy Resampling
        df = daily_returns.to_frame(name='returns')
        df['year'] = df.index.year
        df['month'] = df.index.month

        # Group by Year and Month to get monthly geometric returns
        # Formula: product(1 + daily_returns) - 1
        monthly_grp = df.groupby(['year', 'month'])['returns'].apply(lambda x: (1 + x).prod() - 1)

        heatmap_data = []
        for y in sorted(monthly_grp.index.get_level_values('year').unique(), reverse=True):
            if year != "ALL" and int(year) != y:
                continue

            # Create a list of 12 months, fill with 0.0 if data is missing
            months = []
            year_total = 0.0
            for m in range(1, 13):
                val = monthly_grp.get((y, m), 0.0)
                months.append(round(float(val) * 100, 2)) # Convert to %
                year_total += val

            heatmap_data.append({
                "year": y,
                "months": months,
                "total": round(float((1 + year_total) - 1) * 100, 2) # Annualized total
            })

        return heatmap_data

    def get_trade_journal_data(self, ticker: str = "LQQ.PA") -> Dict[str, Any]:
        """
        Fetches paired trades from DB and calculates high-level summary metrics
        for the Trade Journal UI (Summary Ribbon).
        """

        # 1. Fetch the paired trades from the DB Manager
        trades = self.db.get_trade_journal(ticker)

        if not trades:
            return {"summary": {}, "trades": []}

        # 2. Filter for 'Closed' trades to calculate performance metrics
        closed_trades = [t for t in trades if t['status'] == 'Closed']
        win_rate = 0
        total_pl = 0
        avg_profit = 0.0

        if closed_trades:
            wins = [t for t in closed_trades if t['pl_pct'] > 0]
            win_rate = (len(wins) / len(closed_trades)) * 100
            total_pl = sum(t['pl_pct'] for t in closed_trades)
            # Calculate average profit per trade
            avg_profit = total_pl / len(closed_trades)

        return {
            "summary": {
                "total_trades": len(closed_trades),
                "win_rate": f"{round(win_rate, 1)}%",
                "avg_profit": f"{round(avg_profit, 2)}%", # Match JS
                "net_pl": f"{round(total_pl, 2)}%",      # Match JS
                "open_positions": len(trades) - len(closed_trades)
            },
            "trades": trades # The list of paired dictionaries
        }
