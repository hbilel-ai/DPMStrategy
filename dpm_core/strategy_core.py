# dpm_core/strategy_core.py

import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import yaml
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Dict, Any
import os
import logging
import shutil
import itertools

# Set logging to INFO
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# ==============================================================================
# --- BROKER ABSTRACTION LAYER (CRITICAL FOR MODULARITY) ---
# ==============================================================================

class BrokerClient(ABC):
    """Abstract Base Class for Broker/Execution layer."""

    @abstractmethod
    def get_current_position(self, trade_ticker: str) -> float:
        """Returns the current allocation/quantity of the asset held."""
        pass

    @abstractmethod
    def get_current_cash(self) -> float:
        """Returns the current available cash."""
        pass

    @abstractmethod
    def execute_order(self, ticker: str, target_allocation: float, current_price: float) -> Any:
        """
        Executes a trade to reach the target allocation.
        For a mock client, this logs the intended trade.
        """
        pass

class MockBrokerClient(BrokerClient):
    """
    A concrete implementation of BrokerClient that simulates execution.
    This tracks mock cash and positions for logging purposes.
    """

    def __init__(self, initial_cash: float = 100000.0):
        # Simplistic portfolio tracking:
        self._portfolio = {'cash': initial_cash, 'positions': {}}
        logging.info("MockBrokerClient initialized with $100,000.0 initial cash.")

    def get_current_position(self, trade_ticker: str) -> float:
        """Returns the current quantity held (0 for a new asset)."""
        return self._portfolio['positions'].get(trade_ticker, 0)

    def get_current_cash(self) -> float:
        """Returns the current available mock cash."""
        return self._portfolio['cash']

    def execute_order(self, ticker: str, target_allocation: float, current_price: float) -> Any:
        """
        Calculates the required order (BUY/SELL) to meet the target allocation
        and logs the intention without actual broker communication.
        """
        # Determine current equity for target calculation
        current_qty = self.get_current_position(ticker)
        current_value = current_qty * current_price

        # NOTE: This uses the initial cash if no other trades were mocked yet.
        # For a full simulation, this would track all past mock trades.
        # For this single run, it reflects the cash/position *before* the trade.
        total_equity = self._portfolio['cash'] + sum(qty * current_price for t, qty in self._portfolio['positions'].items())

        # Calculate target quantity based on total equity
        if current_price <= 0:
            logging.error(f"Cannot execute order for {ticker}: Current price is zero or negative.")
            return None

        target_value = total_equity * target_allocation
        target_qty = target_value / current_price

        # Order needed (positive for BUY, negative for SELL)
        order_qty = target_qty - current_qty

        # Ignore tiny orders (e.g., less than $10 change)
        if abs(order_qty) * current_price < 10.0:
            logging.info(f"[{ticker}] - No material change in allocation ({target_allocation:.1%}). No order sent.")
            return None

        action = "BUY" if order_qty > 0 else "SELL"

        # Log the mock order
        logging.info(
            f"--- MOCK ORDER EXECUTED ---"
            f"\n  Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
            f"\n  Asset: {ticker}"
            f"\n  Action: {action} (from {current_qty:.2f} qty to {target_qty:.2f} qty)"
            f"\n  Order Qty: {abs(order_qty):.2f}"
            f"\n  At Price: {current_price:.2f}"
            f"\n  Target Alloc: {target_allocation:.1%}"
            f"\n---------------------------"
        )

        # Update mock portfolio state
        self._portfolio['positions'][ticker] = target_qty

        # Calculate cash change and update
        cash_change = -order_qty * current_price
        self._portfolio['cash'] += cash_change

        return {'status': 'MOCKED', 'action': action, 'quantity': abs(order_qty)}

# ==============================================================================
# --- EXISTING CORE CLASSES ---
# ==============================================================================

class DataFetcher:
    def __init__(self, cache_dir: str = './data_cache', clear_cache: bool = False):
        self.cache_dir = cache_dir
        if clear_cache and os.path.exists(cache_dir):
            shutil.rmtree(cache_dir)
            logging.info(f"Cleared cache directory: {cache_dir}")
        os.makedirs(self.cache_dir, exist_ok=True)

    def fetch_data(self, asset: str, start_date: str, end_date: str) -> pd.DataFrame:
        cache_path = os.path.join(self.cache_dir, f"{asset}_{start_date}_{end_date}.csv")
        if os.path.exists(cache_path):
            logging.info(f"Loading cached data for {asset}")
            return pd.read_csv(cache_path, index_col='Date', parse_dates=True)

        logging.info(f"Fetching data for {asset} from yfinance")
        data = yf.download(asset, start=start_date, end=end_date, auto_adjust=True, ignore_tz=True, progress=False)
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)
        data.index.name = 'Date'
        data.to_csv(cache_path)
        if len(data) == 0:
            logging.warning(f"No data returned for {asset} — possibly invalid ticker or date range")
        return data

    def fetch_risk_free(self, start_date: str, end_date: str) -> pd.Series:
        data = self.fetch_data('^IRX', start_date, end_date)
        return data['Close'] / 100 if 'Close' in data.columns else pd.Series(dtype=float)

class SignalGenerator(ABC):
    @abstractmethod
    def compute_signal(self, prices: pd.Series, **kwargs) -> pd.Series:
        pass

class TMOMSignal(SignalGenerator):
    # lookback is in months
    def compute_signal(self, prices: pd.Series, rf_rates: pd.Series, lookback: int, **kwargs) -> pd.Series:
        monthly_prices = prices.resample('ME').last()
        daily_rf = rf_rates / 252
        window = min(int(21.5 * lookback), len(daily_rf))
        cum_rf_12m = ((1 + daily_rf).rolling(window=window).apply(np.prod, raw=True) - 1).resample('ME').last()
        price_ret = monthly_prices.pct_change(periods=lookback)
        excess = price_ret - cum_rf_12m
        signal = (excess > 0).astype(int)
        return signal.reindex(prices.index, method='ffill').fillna(0)

class SMASignal(SignalGenerator):
    # sma_period is in days
    def compute_signal(self, prices: pd.Series, sma_period: int, **kwargs) -> pd.Series:
        sma = prices.rolling(window=sma_period).mean()
        signal = (prices > sma).astype(int)
        return signal.fillna(0)

class DPMAllocator:
    def __init__(self, signals: Dict[str, SignalGenerator]):
        self.signals = signals

    def allocate(self, signal_prices: pd.DataFrame, rf_rates: pd.Series, tmom_lb: int, sma_p: int, method: str) -> tuple[pd.Series, pd.Series, pd.Series]:
        """
        Determines the daily allocation based on TMOM and SMA signals.
        """
        # 1. Compute Signals
        tmom_sig = TMOMSignal().compute_signal(signal_prices, rf_rates=rf_rates, lookback=tmom_lb)
        sma_sig  = SMASignal().compute_signal(signal_prices, sma_period=sma_p)

        # Re-index and align signals to the daily pricing index
        daily_tmom_sig = tmom_sig.reindex(signal_prices.index, method='ffill').fillna(0)
        daily_sma_sig  = sma_sig.reindex(signal_prices.index, method='ffill').fillna(0)

        # 2. Initialization and Setup
        allocations = pd.Series(0.0, index=signal_prices.index)
        allocations.iloc[0] = 0.0 # Start with 0 allocation
        
        # Mapping from allocation float to state key
        alloc_to_state = {1.0: 'Invested', 0.5: 'Partial', 0.0: 'Cash'}

        if method == 'Linear':
            # Simple linear allocation: (M + S) / 2
            allocations = (daily_tmom_sig + daily_sma_sig) / 2.0

        elif method == 'Conditional':
            # --- CONDITIONAL STATE MACHINE IMPLEMENTATION ---
            state_machine = {
                ('Cash', 0, 0): 0.0, ('Cash', 0, 1): 0.0,
                ('Cash', 1, 1): 1.0, ('Cash', 1, 0): 0.5,
                ('Partial', 0, 0): 0.0, ('Partial', 0, 1): 0.0,
                ('Partial', 1, 1): 1.0, ('Partial', 1, 0): 0.5,
                ('Invested', 0, 0): 0.0, ('Invested', 0, 1): 0.0,
                ('Invested', 1, 1): 1.0, ('Invested', 1, 0): 0.5,
            }

            for i in range(1, len(allocations)):
                p_prev = allocations.iloc[i-1]
                p_prev_state = alloc_to_state.get(round(p_prev, 1))
                s_sig = daily_sma_sig.iloc[i]
                m_sig = daily_tmom_sig.iloc[i]
                transition_key = (p_prev_state, s_sig, m_sig)
                allocations.iloc[i] = state_machine.get(transition_key, 0.0)

        else:
            raise ValueError(f"Unknown allocation method: {method}. Must be 'Linear' or 'Conditional'.")

        return allocations.dropna(), daily_tmom_sig.dropna(), daily_sma_sig.dropna() # <-- RETURN SIGNALS

class PortfolioSimulator:
    def __init__(self, transaction_cost: float = 0.001):
        self.transaction_cost = transaction_cost

    def simulate(self, allocations: pd.Series, asset_returns: pd.Series, rf_rates: pd.Series):
        idx = allocations.index.intersection(asset_returns.index).intersection(rf_rates.index)
        if len(idx) == 0:
            logging.warning("No overlapping dates — skipping simulation")
            return pd.DataFrame(index=pd.DatetimeIndex([]))
        alloc = allocations.reindex(idx).fillna(0)
        ret = asset_returns.reindex(idx).fillna(0)
        rf = rf_rates.reindex(idx).fillna(0) / 252

        costs = alloc.diff().abs().fillna(0) * self.transaction_cost
        port_ret = alloc * ret + (1 - alloc) * rf - costs
        equity = (1 + port_ret).cumprod()
        if len(equity) > 0 and not pd.isna(equity.iloc[0]):
            equity = equity / equity.iloc[0]

        pos_numeric = (alloc * 2).astype(int)
        return pd.DataFrame({
            'equity_curve': equity,
            'position_numeric': pos_numeric,
            'allocations': alloc
        }, index=idx)

class TradeTracker:
    def __init__(self, asset_name: str):
        self.asset_name = asset_name

    def get_trade_log(self, allocations: pd.Series, asset_prices: pd.Series, equity_curve: pd.Series) -> pd.DataFrame:
        data = pd.DataFrame({
            'allocation': allocations.fillna(0),
            'price': asset_prices,
            'equity': equity_curve
        }).dropna()

        data['is_in_position'] = data['allocation'] > 0
        trade_log = []
        current_trade = None

        for date, row in data.iterrows():
            if row['is_in_position'] and current_trade is None:
                current_trade = {
                    'Open Date': date,
                    'Entry Price': row['price'],
                    'Entry Equity': row['equity'],
                    'Max Equity In Trade': row['equity'],
                    'Min Equity In Trade': row['equity'],
                    'Allocation Level': row['allocation']
                }
            elif row['is_in_position'] and current_trade is not None:
                current_trade['Max Equity In Trade'] = max(current_trade['Max Equity In Trade'], row['equity'])
                current_trade['Min Equity In Trade'] = min(current_trade['Min Equity In Trade'], row['equity'])
            if not row['is_in_position'] and current_trade is not None:
                close_date = date
                pl_percent = (row['price'] / current_trade['Entry Price']) - 1.0
                max_dd_in_trade = (current_trade['Min Equity In Trade'] / current_trade['Entry Equity']) - 1.0
                trade_record = {
                    'Asset': self.asset_name,
                    'Open Date': current_trade['Open Date'],
                    'Close Date': close_date,
                    'Direction': 'Long',
                    'Allocation Level': current_trade['Allocation Level'],
                    'P/L (%)': pl_percent,
                    'MaxDD In Trade (%)': max_dd_in_trade,
                    'Days Held': (close_date - current_trade['Open Date']).days
                }
                trade_log.append(trade_record)
                current_trade = None

        trade_df = pd.DataFrame(trade_log)
        if not trade_df.empty:
            trade_df['P/L (float)'] = trade_df['P/L (%)']
            trade_df['MaxDD In Trade (float)'] = trade_df['MaxDD In Trade (%)']
            trade_df['P/L (%)'] = trade_df['P/L (%)'].apply(lambda x: f"{x:.2%}")
            trade_df['MaxDD In Trade (%)'] = trade_df['MaxDD In Trade (%)'].apply(lambda x: f"{x:.2%}")

        return trade_df.set_index('Open Date')

class PerformanceAnalyzer:
    def analyze(self, equity: pd.Series) -> Dict:
        if len(equity) < 2:
            return {'CAGR': np.nan, 'MaxDD': np.nan, 'Sharpe': np.nan, 'Vol': np.nan, 'Longest DD (Days)': 0}
        years = (equity.index[-1] - equity.index[0]).days / 365.25
        cagr = (equity.iloc[-1] / equity.iloc[0]) ** (1/years) - 1
        dd = (equity / equity.cummax() - 1).min()
        ret = equity.pct_change().dropna()
        volatility = ret.std() * np.sqrt(252)
        sharpe = ret.mean() / ret.std() * np.sqrt(252) if ret.std() != 0 else np.nan
        peaks = equity.cummax()
        peak_dates = equity[equity == peaks].index
        if len(peak_dates) > 1:
            dd_durations = peak_dates.to_series().diff().dt.days
            longest_dd_days = int(dd_durations.max()) if not dd_durations.empty else 0
        else:
            longest_dd_days = 0

        return {
            'CAGR': cagr, 'MaxDD': dd, 'Sharpe': sharpe, 'Vol': volatility, 'Longest DD (Days)': longest_dd_days
        }

    def plot_professional_report(self, equity: pd.Series, benchmark: pd.Series, pos_numeric: pd.Series, alloc: pd.Series,
                                 signal_prices: pd.Series, drawdown: pd.Series, rf: pd.Series, tmom_sig: pd.Series, sma_sig: pd.Series,
                                 asset_name: str, output_path: str, tmom_lookback: int, sma_period: int, algo_mode: str,
                                 metrics: Dict[str, float], monthly_returns_df: pd.DataFrame):
        import pandas as pd
        import matplotlib.pyplot as plt
        import numpy as np
        import os

        fig = plt.figure(figsize=(11.5, 16.5))

        # Grid Layout
        ax_dashboard = plt.subplot2grid((15, 4), (0, 0), rowspan=2, colspan=4)
        ax_monthly = plt.subplot2grid((15, 4), (2, 0), rowspan=4, colspan=4)
        ax_equity = plt.subplot2grid((15, 4), (6, 0), rowspan=3, colspan=4)
        ax_drawdown = plt.subplot2grid((15, 4), (9, 0), rowspan=2, colspan=4)
        ax_signals = plt.subplot2grid((15, 4), (11, 0), rowspan=2, colspan=4)
        ax_alloc = plt.subplot2grid((15, 4), (13, 0), rowspan=2, colspan=4)

        # --- DASHBOARD ---
        ax_dashboard.axis('off')
        rect = plt.Rectangle((0, 0), 1, 1, transform=ax_dashboard.transAxes, color='#f7f7f7', ec='lightgray', lw=1, zorder=-1)
        ax_dashboard.add_patch(rect)

        # Status Logic
        last_alloc = alloc.iloc[-1]
        pos_status = "FULLY INVESTED" if last_alloc == 1.0 else ("PARTIAL POSITION" if last_alloc > 0.0 else "CASH / OUT")
        pos_color = '#4CAF50' if last_alloc == 1.0 else ('#FFC107' if last_alloc > 0.0 else '#9E9E9E')
        pos_text_color = 'white' if last_alloc == 1.0 or last_alloc == 0.0 else 'black'

        tmom_status = 'POSITIVE' if tmom_sig.iloc[-1] > 0 else 'NEGATIVE'
        tmom_color = '#4CAF50' if tmom_sig.iloc[-1] > 0 else '#F44336'
        sma_status = 'POSITIVE' if sma_sig.iloc[-1] > 0 else 'NEGATIVE'
        sma_color = '#4CAF50' if sma_sig.iloc[-1] > 0 else '#F44336'

        # Dashboard Text
        ax_dashboard.text(0.5, 0.95, "Strategy Status & Core Metrics", fontsize=12, weight='bold', ha='center', va='top', transform=ax_dashboard.transAxes)

        ax_dashboard.text(0.25, 0.75, "Current Position", fontsize=10, weight='bold', ha='center', transform=ax_dashboard.transAxes)
        ax_dashboard.text(0.25, 0.55, pos_status, fontsize=14, color=pos_text_color, ha='center', va='center',
                          bbox=dict(facecolor=pos_color, edgecolor='none', boxstyle='round,pad=0.5'), transform=ax_dashboard.transAxes)

        ax_dashboard.text(0.05, 0.35, "TMOM Signal:", fontsize=9, ha='left', transform=ax_dashboard.transAxes)
        ax_dashboard.text(0.45, 0.35, tmom_status, fontsize=9, ha='right', color='white', weight='bold',
                          bbox=dict(facecolor=tmom_color, edgecolor='none', boxstyle='round,pad=0.3'), transform=ax_dashboard.transAxes)

        ax_dashboard.text(0.05, 0.20, "SMA Signal:", fontsize=9, ha='left', transform=ax_dashboard.transAxes)
        ax_dashboard.text(0.45, 0.20, sma_status, fontsize=9, ha='right', color='white', weight='bold',
                          bbox=dict(facecolor=sma_color, edgecolor='none', boxstyle='round,pad=0.3'), transform=ax_dashboard.transAxes)

        ax_dashboard.axvline(x=0.48, ymin=0.1, ymax=0.9, color='lightgray', linestyle='--', linewidth=1.5)

        # Performance Data
        returns_monthly = equity.resample('ME').last().pct_change().dropna()
        def safe_perf(series, periods):
            if len(series) < periods: return 0.0
            return (1 + series.iloc[-periods:].fillna(0)).prod() - 1

        mtd_ret = returns_monthly.iloc[-1] if not returns_monthly.empty else 0.0
        l3m_ret = safe_perf(returns_monthly, 3)
        l6m_ret = safe_perf(returns_monthly, 6)
        l12m_ret = safe_perf(returns_monthly, 12)

        perf_data = [("Current Month (MTD)", mtd_ret), ("Last 3 Months (L3M)", l3m_ret),
                     ("Last 6 Months (L6M)", l6m_ret), ("Last 12 Months (L12M)", l12m_ret)]

        kpi_data = [("CAGR (%)", f"{metrics.get('CAGR', 0) * 100:.2f}%"),
                    ("Sharpe Ratio", f"{metrics.get('Sharpe', 0):.2f}"),
                    ("Max Drawdown (%)", f"{metrics.get('MaxDD', 0) * 100:.2f}%"),
                    ("Num Trades", int(metrics.get('Num Trades', 0))),
                    ("Win Trades (%)", f"{metrics.get('Win Trades', 0):.1f}%"),
                    ("Profit Factor", f"{metrics.get('Profit Factor', 0):.2f}"),
                    ("Avg Trade Return", f"{metrics.get('Avg Trade Return', 0) * 100:.2f}%")]

        # --- LATEST PERF TABLE ---
        perf_rows = []
        perf_colors = []
        for label, ret in perf_data:
            color = 'green' if ret > 0.0001 else ('red' if ret < -0.0001 else 'black')
            ret_text = f"{ret * 100:.2f}%"
            indicator = '▲' if ret > 0.0001 else ('▼' if ret < -0.0001 else '—')
            perf_rows.append([label, f"{indicator} {ret_text}"])
            perf_colors.append(color)

        perf_table = ax_dashboard.table(cellText=[row[1:] for row in perf_rows], rowLabels=[row[0] for row in perf_rows],
            colLabels=["Latest Performance"], cellLoc='right', rowLoc='left', loc='center',
            bbox=[0.52, 0.55, 0.45, 0.35])

        # FIX: Index is (i + 1, 0) -> Row i+1 (skip header), Col 0 (data column)
        for i, color in enumerate(perf_colors):
            perf_table.get_celld()[(i + 1, 0)].set_text_props(color=color, weight='bold')
            perf_table.get_celld()[(i + 1, 0)].set_fontsize(9)

        # --- KPI TABLE ---
        kpi_table_rows = [[row[0], row[1]] for row in kpi_data]
        # FIX: Ensure cellText is a list of lists [[val], [val]] so it creates a proper column
        kpi_table = ax_dashboard.table(cellText=[[row[1]] for row in kpi_table_rows], rowLabels=[row[0] for row in kpi_table_rows],
            colLabels=["Core Metrics"], cellLoc='right', rowLoc='left', loc='center',
            bbox=[0.52, 0.05, 0.45, 0.40])

        # FIX: MaxDD is index 2 in data -> Row 3 (2+1) in table, Col 0
        kpi_table.get_celld()[(3, 0)].set_text_props(color='#F44336', weight='bold')

        # --- MONTHLY RETURNS ---
        monthly_data = monthly_returns_df.values
        color_matrix = np.full(monthly_data.shape, 'white', dtype=object)
        for i in range(monthly_data.shape[0]):
            for j in range(monthly_data.shape[1]):
                val = monthly_data[i, j]
                try:
                    num_val = float(str(val).replace('%', '').replace(' ', '').strip()) / 100.0 if isinstance(val, str) else val
                    if num_val > 0.0001: color_matrix[i, j] = '#e6ffe6'
                    elif num_val < -0.0001: color_matrix[i, j] = '#ffe6e6'
                except: pass
        table_monthly = ax_monthly.table(cellText=monthly_data, colLabels=monthly_returns_df.columns, rowLabels=monthly_returns_df.index,
            cellColours=color_matrix, cellLoc='center', loc='center', bbox=[0, 0, 1, 1])
        for (i, j), cell in table_monthly.get_celld().items():
            if j == monthly_data.shape[1] - 1: cell.set_text_props(weight='bold')
        ax_monthly.axis('off')
        ax_monthly.set_title("Annual & Monthly Returns Matrix", fontsize=12, pad=10)

        # --- CHARTS ---
        ax_equity.plot(equity, label=f'{asset_name} DPM Strategy', color='dodgerblue', linewidth=2)
        ax_equity.plot(benchmark.reindex(equity.index), label=f'{asset_name} Benchmark', color='gray', linestyle='--', linewidth=1)
        ax_equity.set_title(f"Cumulative Returns - {asset_name} ({algo_mode}: TMOM {tmom_lookback}m, SMA {sma_period}d)", fontsize=14)
        ax_equity.legend(loc='upper left')
        ax_equity.grid(True, linestyle=':', alpha=0.6)
        ax_equity.tick_params(axis='x', rotation=45)

        ax_drawdown.fill_between(drawdown.index, drawdown, 0, color='indianred', alpha=0.7)
        ax_drawdown.set_title("Max Drawdown Over Time", fontsize=12)
        ax_drawdown.grid(True, linestyle=':', alpha=0.6)
        y_ticks = ax_drawdown.get_yticks()
        ax_drawdown.set_yticks(y_ticks)
        ax_drawdown.set_yticklabels([f'{int(x*100)}%' for x in y_ticks])

        ax_signals.plot(signal_prices, label=f'{asset_name} Signal Price', color='darkorange', alpha=0.8, linewidth=1)
        tmom_on_days = tmom_sig[tmom_sig > 0].index
        ax_signals.plot(signal_prices.loc[tmom_on_days], 'o', markersize=2, color='green', alpha=0.5, label='TMOM Signal ON')
        sma_on_days = sma_sig[sma_sig > 0].index
        ax_signals.plot(signal_prices.loc[sma_on_days], 'x', markersize=2, color='purple', alpha=0.5, label='SMA Signal ON')
        ax_signals.set_title(f"Signal Price and Component Signals", fontsize=12)
        ax_signals.legend(loc='upper left', fontsize=8)
        ax_signals.grid(True, linestyle=':', alpha=0.6)

        ax_alloc.fill_between(alloc.index, alloc, 0, color='forestgreen', alpha=0.6, step='post')
        ax_alloc.set_title("Strategy Allocation (0.0 to 1.0)", fontsize=12)
        ax_alloc.set_ylim(-0.05, 1.05)
        ax_alloc.grid(True, linestyle=':', alpha=0.6)

        fig.suptitle(f'Dynamic Portfolio Manager (DPM) Strategy Report\nAsset: {asset_name} | Mode: {algo_mode}', fontsize=16, fontweight='bold', y=0.98)
        fig.tight_layout(rect=[0, 0.01, 1, 0.97])
        plt.close(fig)
        logging.info(f"Generated professional report (File output suppressed in this environment).")


    def calculate_trade_metrics(self, trade_log_df: pd.DataFrame) -> Dict[str, float]:
        if trade_log_df.empty:
            return {'Num Trades': 0.0, 'Win Trades': 0.0, 'Profit Factor': 0.0, 'Avg Trade Return': 0.0}
        pl_data = trade_log_df['P/L (float)']
        num_trades = len(trade_log_df)
        win_trades_count = len(pl_data[pl_data > 0])
        win_rate = (win_trades_count / num_trades) * 100.0 if num_trades > 0 else 0.0
        gross_profit = pl_data[pl_data > 0].sum()
        gross_loss = pl_data[pl_data < 0].sum()
        profit_factor = gross_profit / abs(gross_loss) if gross_loss != 0 else np.nan
        avg_trade_return = pl_data.mean()
        return {'Num Trades': float(num_trades), 'Win Trades': win_rate, 'Profit Factor': profit_factor, 'Avg Trade Return': avg_trade_return}

    def calculate_monthly_returns(self, equity_curve: pd.Series) -> pd.DataFrame:
        daily_returns = equity_curve.pct_change().fillna(0)
        monthly_returns = daily_returns.resample('ME').apply(lambda x: (1 + x).prod() - 1)
        monthly_returns_df = monthly_returns.to_frame(name='Monthly Return')
        monthly_returns_df['Year'] = monthly_returns_df.index.year
        monthly_returns_df['Month'] = monthly_returns_df.index.strftime('%b')
        month_order = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        monthly_returns_pivot = monthly_returns_df.pivot(index='Year', columns='Month', values='Monthly Return').reindex(columns=month_order)
        ytd_returns = monthly_returns.groupby(monthly_returns.index.year).apply(lambda x: (1 + x).prod() - 1)
        monthly_returns_pivot['YTD'] = ytd_returns
        def format_return(ret): return '' if pd.isna(ret) else f"{ret:.2%}"
        return monthly_returns_pivot.map(format_return)
