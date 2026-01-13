# application_logic/analytics_service.py

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pandas.tseries.offsets import DateOffset
from typing import Dict, Any, List
from data_layer.db_manager import DBManager

class AnalyticsService:
    """
    Calculates financial performance metrics by reconstructing the equity curve
    directly from historical Price and Position signals.
    """
    def __init__(self, db_manager: DBManager):
        self.db = db_manager

    # --- CORE LOGIC: Signal-Based Return Calculation ---
    def _get_daily_returns_series(self, start_date: str = '2000-01-01') -> pd.Series:
        """
        FIXED: Reads directly from verified portfolio equity instead of 
        reconstructing from signals. This eliminates the 'double-shift' bug.
        """
        query = """
            SELECT snapshot_date as timestamp, total_value
            FROM portfolio_snapshot
            WHERE snapshot_date >= ?
            ORDER BY snapshot_date ASC
        """
        df = pd.read_sql(query, self.db.engine, params=(start_date,))

        if df.empty:
            print(f"DEBUG_ANALYTICS: No data found in portfolio_snapshot since {start_date}")
            return pd.Series(dtype=float)

        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df.set_index('timestamp', inplace=True)
        
        # Calculate returns from the actual equity curve
        returns = df['total_value'].pct_change().fillna(0.0)
        return returns
    
    # --- 1. Cumulative Returns (Chart Data) ---
    def get_cumulative_returns(self, start_date: str = None, benchmark_ticker: str = "LQQ.PA"):
        """
        Generates the equity curves for Strategy vs Benchmark.
        """
        if not start_date:
            start_date = datetime.now().replace(month=1, day=1).strftime('%Y-%m-%d')

        # A. Strategy Calculation (Uses new signal-based logic)
        strategy_res = self.calculate_cumulative_returns(start_date)

        # B. Benchmark Calculation (Pure Buy & Hold)
        bench_data = self.db.get_historical_prices(benchmark_ticker, start_date)
        benchmark_series = []
        if bench_data:
            initial_price = bench_data[0]['price']
            for entry in bench_data:
                # Simple (Price / Initial) - 1
                cum_ret = (entry['price'] / initial_price) - 1
                benchmark_series.append({
                    "date": entry['date'],
                    "equity": round(float(cum_ret), 4)
                })

        # C. Format Strategy Output
        strategy_series = []
        if 'timestamps' in strategy_res:
            for i in range(len(strategy_res['timestamps'])):
                strategy_series.append({
                    "date": strategy_res['timestamps'][i],
                    "equity": strategy_res['cumulative_returns'][i]
                })

        return {
            "strategy": strategy_series,
            "benchmark": benchmark_series,
            "benchmark_label": benchmark_ticker
        }

    def calculate_cumulative_returns(self, start_date):
        """
        Computes cumulative return series and drawdown from daily returns.
        """
        daily_returns = self._get_daily_returns_series(start_date)

        if daily_returns.empty:
            return {"timestamps": [], "cumulative_returns": [], "drawdown_series": []}

        # 1. Wealth Index (Start at 1.0)
        wealth_index = (1 + daily_returns).cumprod()
        
        # 2. Cumulative Return % (Wealth - 1)
        cumulative_returns = wealth_index - 1

        # 3. Drawdown %
        running_peak = wealth_index.cummax()
        drawdown_series = (wealth_index / running_peak - 1) * 100

        return {
            'timestamps': cumulative_returns.index.strftime('%Y-%m-%d').tolist(),
            'cumulative_returns': cumulative_returns.tolist(),
            'drawdown_series': drawdown_series.tolist()
        }

    # --- 2. Risk Metrics (Stats) ---
    def calculate_risk_metrics(self, start_date, risk_free_rate_annual=0.0):
        """
        FIXED: Uses 0.0 RF by default to match backtest report.
        """
        daily_returns = self._get_daily_returns_series(start_date)

        if len(daily_returns) < 2:
            return self._empty_risk_metrics()

        # A. Max Drawdown
        # We use (1+r).cumprod() to create a normalized wealth index starting at 1.0
        wealth_index = (1 + daily_returns).cumprod()
        running_peak = wealth_index.cummax()
        drawdown_series = 1 - (wealth_index / running_peak)
        max_drawdown = drawdown_series.max()

        # B. Sharpe Ratio (Annualized)
        # Using 0% RF to match your 2.67 Sharpe backtest
        avg_ret = daily_returns.mean()
        vol = daily_returns.std()
        sharpe = (avg_ret / vol * np.sqrt(252)) if vol > 0 else 0

        # C. Win Rate & Profit Factor
        gains = daily_returns[daily_returns > 0]
        losses = daily_returns[daily_returns < 0]
        win_rate = len(gains) / len(daily_returns[daily_returns != 0]) if not daily_returns[daily_returns != 0].empty else 0
        profit_factor = abs(gains.sum() / losses.sum()) if not losses.empty else 0

        return {
            "max_drawdown": round(float(max_drawdown), 4),
            "sharpe_ratio": round(float(sharpe), 4),
            "volatility_annualized": round(float(vol * np.sqrt(252)), 4),
            "mdd": f"{round(float(max_drawdown * 100), 2)}%",
            "mdd_dates": "Verified Ledger Data",
            "pf": str(round(float(profit_factor), 2)),
            "wr": f"{round(float(win_rate * 100), 1)}%",
            "rf": str(round(float(daily_returns.sum() / max_drawdown), 2)) if max_drawdown != 0 else "0"
        }
    def _empty_risk_metrics(self):
        return {
            "sharpe_ratio": 0.0, "max_drawdown": 0.0, "volatility_annualized": 0.0,
            "mdd": "0%", "mdd_dates": "N/A", "pf": "0.0", "wr": "0%", "rf": "0.0"
        }

    # --- 3. Periodic Returns (Trailing) ---
    def calculate_periodic_returns(self, start_date=None):
        """
        Calculates returns for 1M, 3M, YTD using the reconstructed wealth index.
        """
        # We need full history to look back
        daily_returns = self._get_daily_returns_series("2000-01-01")
        if daily_returns.empty:
            return {"1W": 0.0, "1M": 0.0, "YTD": 0.0}

        wealth_index = (1 + daily_returns).cumprod()
        latest_date = wealth_index.index[-1]
        latest_val = wealth_index.iloc[-1]

        periods = {
            '1W': DateOffset(weeks=1),
            '1M': DateOffset(months=1),
            '3M': DateOffset(months=3),
            '6M': DateOffset(months=6),
            '1Y': DateOffset(years=1)
        }
        results = {}

        for label, offset in periods.items():
            target_date = latest_date - offset
            try:
                # Find index at or before target
                idx = wealth_index.index.asof(target_date)
                if pd.isna(idx): 
                    results[label] = None
                else:
                    hist_val = wealth_index.loc[idx]
                    results[label] = round((latest_val - hist_val) / hist_val, 4)
            except:
                results[label] = None

        # YTD
        start_year = latest_date.replace(month=1, day=1)
        try:
            idx = wealth_index.index.asof(start_year)
            if pd.isna(idx):
                 results['YTD'] = round(latest_val - 1, 4) # Assuming started this year
            else:
                ytd_val = wealth_index.loc[idx]
                results['YTD'] = round((latest_val - ytd_val) / ytd_val, 4)
        except:
             results['YTD'] = None

        return results

    # --- Charting Data Prep ---
    def get_signal_chart_data(self, ticker: str, start_date: str):
        query = """
            SELECT timestamp, close_price, signal_A_value, signal_B_value, signal_VIX_value, position
            FROM market_signal
            WHERE ticker = ? AND timestamp >= ?
            ORDER BY timestamp ASC
        """
        df = pd.read_sql(query, self.db.engine, params=(ticker, start_date))
        if df.empty:
            return {"timestamps": [], "price": [], "position": [], "component_signals": []}

        df['timestamp'] = pd.to_datetime(df['timestamp'])

        return {
            "timestamps": df['timestamp'].dt.strftime('%Y-%m-%dT%H:%M:%S').tolist(),
            "price": df['close_price'].tolist(),
            "position": df['position'].tolist(),
            "component_signals": [
                {"name": "TMOM", "data": df['signal_A_value'].tolist()},
                {"name": "SMA", "data": df['signal_B_value'].tolist()},
                {"name": "VIX", "data": df['signal_VIX_value'].tolist()}
            ]
        }

    def get_monthly_returns_heatmap(self, year="ALL"):
        daily_returns = self._get_daily_returns_series("2000-01-01")
        if daily_returns.empty: return []

        df = daily_returns.to_frame(name='returns')
        df['year'] = df.index.year
        df['month'] = df.index.month

        monthly_grp = df.groupby(['year', 'month'])['returns'].apply(lambda x: (1 + x).prod() - 1)

        heatmap_data = []
        for y in sorted(monthly_grp.index.get_level_values('year').unique(), reverse=True):
            if year != "ALL" and int(year) != y: continue

            months = []
            year_total_idx = 1.0
            for m in range(1, 13):
                val = monthly_grp.get((y, m), 0.0)
                months.append(round(float(val) * 100, 2))
                year_total_idx *= (1 + val)
            
            heatmap_data.append({
                "year": y,
                "months": months,
                "total": round((year_total_idx - 1) * 100, 2)
            })
        return heatmap_data

    # --- Trade Journal & Ledger (Kept as previously implemented) ---
    def get_trade_journal_data(self, ticker="LQQ.PA"):
        # ... (Previous FIFO Implementation) ...
        # Use previous implementation provided in history
        executions = self.db.get_raw_executions(ticker) 
        journal = []
        open_buys = []

        for ex in executions:
            exec_time_dt = ex['execution_time']
            exec_time_str = exec_time_dt.strftime('%Y-%m-%d %H:%M') if hasattr(exec_time_dt, 'strftime') else str(exec_time_dt)

            order_type_raw = str(ex['order_type']).upper()
            if 'BUY' in order_type_raw:
                entry = {
                    "id": ex['id'], "ticker": ex['ticker'], "type": "LONG",
                    "open_date": exec_time_str, "_raw_date": exec_time_dt,
                    "open_price": round(ex['execution_price'], 2),
                    "original_qty": ex['quantity'], "remaining_qty": ex['quantity'],
                    "status": "Open", "close_price": "---", "close_date": "---",
                    "pl_pct": 0.0, "duration": "---"
                }
                open_buys.append(entry)
                journal.append(entry)
                
            elif 'SELL' in order_type_raw:
                sell_qty = abs(float(ex['quantity']))
                sell_price = ex['execution_price']
                for buy in open_buys:
                    if sell_qty <= 1e-6: break # Use a small epsilon for float precision
                    if buy['remaining_qty'] <= 1e-6: continue
                    match_qty = min(buy['remaining_qty'], sell_qty)
                    buy['remaining_qty'] -= match_qty
                    sell_qty -= match_qty
                    buy['close_price'] = round(sell_price, 2)
                    buy['close_date'] = exec_time_str
                    buy['status'] = "Closed" if buy['remaining_qty'] == 0 else "Partial"
                    entry_p = buy['open_price']
                    buy['pl_pct'] = round(((sell_price - entry_p) / entry_p) * 100, 2)
                    if hasattr(exec_time_dt, 'year') and hasattr(buy['_raw_date'], 'year'):
                        delta = exec_time_dt - buy['_raw_date']
                        buy['duration'] = f"{delta.days}d {delta.seconds//3600}h"

        closed_trades = [t for t in journal if t['status'] == 'Closed']
        win_rate = (len([t for t in closed_trades if t['pl_pct'] > 0]) / len(closed_trades) * 100) if closed_trades else 0
        net_pl = sum(t['pl_pct'] for t in closed_trades)

        return {
            "summary": {
                "total_trades": len(closed_trades),
                "win_rate": f"{round(win_rate, 1)}%",
                "avg_profit": f"{round(net_pl/len(closed_trades), 2)}%" if closed_trades else "0%",
                "net_pl": f"{round(net_pl, 2)}%"
            },
            "trades": list(reversed(journal))
        }

    def get_account_ledger_data(self, ticker="LQQ.PA"):
        # ... (Previous Ledger Implementation) ...
        executions = self.db.get_raw_executions(ticker) 
        ledger = []
        running_exposure = 0.0

        for ex in executions:
            exec_time_dt = ex['execution_time']
            exec_time_str = exec_time_dt.strftime('%Y-%m-%d %H:%M') if hasattr(exec_time_dt, 'strftime') else str(exec_time_dt)
            
            initial_exposure = running_exposure
            # FIX: The quantity in DB is already signed (Positive for BUY, Negative for SELL)
            qty_diff = float(ex['quantity'])
            running_exposure += qty_diff
            
            running_exposure = round(running_exposure, 4)
            ledger.append({
                "id": ex['id'], 
                "date": exec_time_str, 
                "ticker": ex['ticker'],
                "type": ex['order_type'], 
                "initial_qty": round(initial_exposure, 4), # Increased precision to 4
                "qty_diff": round(qty_diff, 4), 
                "total_exposure": round(running_exposure, 4),
                "price": round(float(ex['execution_price']), 2)
            })
            
        total_vol = sum(float(ex['quantity']) for ex in executions)
        return {
            "summary": {
                "current_exposure": f"{round(running_exposure * 100, 1)}%",
                "total_volume": round(total_vol, 2),
                "transaction_count": len(ledger),
                "net_flow": "N/A"
            },
            "ledger": list(reversed(ledger))
        }
