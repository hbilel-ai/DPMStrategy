### DPM Strategy — Configurable Algo (Linear vs Conditional)

import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import yaml
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Dict
import os
import logging
import shutil
import itertools 

# Set logging to INFO, but switch to ERROR during sweep for cleaner output
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

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
    def allocate(self, signal_prices: pd.DataFrame, rf_rates: pd.Series, tmom_lb: int, sma_p: int, method: str) -> pd.Series:
        """
        Determines the daily allocation based on TMOM and SMA signals.
        
        This method uses forward-filled signals to ensure daily re-evaluation
        and calculates the new allocation based on the previous day's allocation
        and the signals, enforcing the chosen logic (Linear or Conditional).
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
        
        # Store previous allocation (1 day lag)
        prev_alloc = allocations.shift(1).fillna(0.0)
        
        # Mapping from allocation float to state key
        alloc_to_state = {1.0: 'Invested', 0.5: 'Partial', 0.0: 'Cash'}
        
        if method == 'Linear':
            # Simple linear allocation: (M + S) / 2
            allocations = (daily_tmom_sig + daily_sma_sig) / 2.0
            
        elif method == 'Conditional':
            
            # --- CONDITIONAL STATE MACHINE IMPLEMENTATION ---
            
            # Define the 12-state transition map based on your logic:
            # Key: (Previous_State, SMA_Signal, TMOM_Signal) -> New_Allocation_Value
            state_machine = {
                # From Cash (0.0): S=0 blocks re-entry (Persistence Filter)
                ('Cash', 0, 0): 0.0, ('Cash', 0, 1): 0.0, 
                ('Cash', 1, 1): 1.0, ('Cash', 1, 0): 0.5,
                
                # From Partial (0.5): S=0 forces exit (Asymmetric Exit)
                ('Partial', 0, 0): 0.0, ('Partial', 0, 1): 0.0,
                ('Partial', 1, 1): 1.0, ('Partial', 1, 0): 0.5,
                
                # From Invested (1.0): S=0 forces exit (Asymmetric Exit)
                ('Invested', 0, 0): 0.0, ('Invested', 0, 1): 0.0,
                ('Invested', 1, 1): 1.0, ('Invested', 1, 0): 0.5,
            }

            # Apply the state machine daily starting from the second day
            for i in range(1, len(allocations)):
                # Get the previous day's final allocation (which is the current state)
                p_prev = allocations.iloc[i-1] 
                
                # Convert the previous allocation value to the state string
                # We need to handle potential floating point errors by rounding
                p_prev_state = alloc_to_state.get(round(p_prev, 1))

                # Get current signals
                s_sig = daily_sma_sig.iloc[i]
                m_sig = daily_tmom_sig.iloc[i]
                
                # Look up the new allocation in the state machine
                transition_key = (p_prev_state, s_sig, m_sig)
                
                # Default safety: if key is not found (should not happen), keep 0.0
                allocations.iloc[i] = state_machine.get(transition_key, 0.0)

        else:
            raise ValueError(f"Unknown allocation method: {method}. Must be 'Linear' or 'Conditional'.")

        return allocations.dropna()

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
    """
    Processes the daily simulation results (allocations, prices, equity) to generate
    a trade-by-trade log for closed trades.

    A trade is considered 'closed' when the allocation moves from > 0 to 0.
    """
    def __init__(self, asset_name: str):
        self.asset_name = asset_name

    def get_trade_log(self, allocations: pd.Series, asset_prices: pd.Series, equity_curve: pd.Series) -> pd.DataFrame:

        # Align all series by date index
        # Note: We rely on `asset_prices` being the closing price of the traded asset.
        data = pd.DataFrame({
            'allocation': allocations.fillna(0),
            'price': asset_prices,
            'equity': equity_curve
        }).dropna()

        # Identify when a position is open
        data['is_in_position'] = data['allocation'] > 0

        trade_log = []
        current_trade = None

        for date, row in data.iterrows():

            # --- 1. OPEN TRADE DETECTION ---
            # Position moves from 0 to > 0
            if row['is_in_position'] and current_trade is None:
                current_trade = {
                    'Open Date': date,
                    'Entry Price': row['price'],
                    'Entry Equity': row['equity'],
                    'Max Equity In Trade': row['equity'],
                    'Min Equity In Trade': row['equity'],
                    'Allocation Level': row['allocation']
                }

            # --- 2. TRACKING DURING TRADE ---
            elif row['is_in_position'] and current_trade is not None:
                # Track max/min equity during the holding period
                current_trade['Max Equity In Trade'] = max(current_trade['Max Equity In Trade'], row['equity'])
                current_trade['Min Equity In Trade'] = min(current_trade['Min Equity In Trade'], row['equity'])

            # --- 3. CLOSE TRADE DETECTION ---
            # Position moves from > 0 to 0
            if not row['is_in_position'] and current_trade is not None:

                # Calculate metrics for the closed trade
                close_date = date
                exit_price = row['price']
                exit_equity = row['equity']

                # Calculate metrics
                # P/L based on asset price movement (more standard for trade log)
                pl_percent = (exit_price / current_trade['Entry Price']) - 1.0

                # Max Drawdown IN the trade (relative to entry equity)
                # MaxDD is the maximum drop from the entry point during the trade
                max_dd_in_trade = (current_trade['Min Equity In Trade'] / current_trade['Entry Equity']) - 1.0

                trade_record = {
                    'Asset': self.asset_name,
                    'Open Date': current_trade['Open Date'],
                    'Close Date': close_date,
                    'Direction': 'Long', # DPM only takes long positions
                    'Allocation Level': current_trade['Allocation Level'],
                    'P/L (%)': pl_percent,
                    'MaxDD In Trade (%)': max_dd_in_trade,
                    'Days Held': (close_date - current_trade['Open Date']).days
                }
                trade_log.append(trade_record)

                # Reset state
                current_trade = None

        trade_df = pd.DataFrame(trade_log)

        # --- REVISED FORMATTING FOR ANALYSIS ---
        if not trade_df.empty:
            # RETAIN RAW FLOAT FOR CALCULATIONS (New column)
            trade_df['P/L (float)'] = trade_df['P/L (%)']
            trade_df['MaxDD In Trade (float)'] = trade_df['MaxDD In Trade (%)']

            # FORMAT THE OLD COLUMNS TO STRINGS FOR PRINTING/DISPLAY (Existing code, kept for compatibility)
            trade_df['P/L (%)'] = trade_df['P/L (%)'].apply(lambda x: f"{x:.2%}")
            trade_df['MaxDD In Trade (%)'] = trade_df['MaxDD In Trade (%)'].apply(lambda x: f"{x:.2%}")

        return trade_df.set_index('Open Date')

class PerformanceAnalyzer:
    def analyze(self, equity: pd.Series) -> Dict:
        if len(equity) < 2:
            return {'CAGR': np.nan, 'MaxDD': np.nan, 'Sharpe': np.nan, 'Vol': np.nan, 'Longest DD (Days)': 0}
        years = (equity.index[-1] - equity.index[0]).days / 365.25
        cagr = (equity.iloc[-1] / equity.iloc[0]) ** (1/years) - 1
        # Max Drawdown
        dd = (equity / equity.cummax() - 1).min()
        # Returns for Volatility and Sharpe
        ret = equity.pct_change().dropna()
        # Annualized Volatility
        volatility = ret.std() * np.sqrt(252)
        # Sharpe Ratio
        sharpe = ret.mean() / ret.std() * np.sqrt(252) if ret.std() != 0 else np.nan
        # Longest Drawdown Cycle in Days (Peak to Peak)
        peaks = equity.cummax()
        peak_dates = equity[equity == peaks].index
        if len(peak_dates) > 1:
            # Duration between consecutive peaks (length of drawdown cycle)
            dd_durations = peak_dates.to_series().diff().dt.days
            longest_dd_days = int(dd_durations.max()) if not dd_durations.empty else 0
        else:
            longest_dd_days = 0

        return {
            'CAGR': cagr,
            'MaxDD': dd,
            'Sharpe': sharpe,
            'Vol': volatility,
            'Longest DD (Days)': longest_dd_days
        }

    def plot_professional_report(self, equity: pd.Series, benchmark: pd.Series, pos_numeric: pd.Series, alloc: pd.Series,
                                 signal_prices: pd.Series, drawdown: pd.Series, rf: pd.Series, tmom_sig: pd.Series, sma_sig: pd.Series,
                                 asset_name: str, output_path: str, tmom_lookback: int, sma_period: int, algo_mode: str,
                                 metrics: Dict[str, float], monthly_returns_df: pd.DataFrame):
        """
        Generates a professional-style report PDF with multiple panels and tables.
        Includes a Dashboard and color-coded Monthly Returns.
        """
        import pandas as pd
        import matplotlib.pyplot as plt
        import numpy as np
        import os
        import logging
        from typing import Dict

        # Set up the figure
        fig = plt.figure(figsize=(11.5, 16.5))

        # --- REDEFINED GRID (11 rows, 4 columns) for new layout ---
        ax_monthly = plt.subplot2grid((11, 4), (0, 0), rowspan=2, colspan=3)
        ax_dashboard = plt.subplot2grid((11, 4), (0, 3), rowspan=2, colspan=1)

        ax_equity = plt.subplot2grid((11, 4), (2, 0), rowspan=3, colspan=4)

        ax_drawdown = plt.subplot2grid((11, 4), (5, 0), rowspan=2, colspan=4)
        ax_signals = plt.subplot2grid((11, 4), (7, 0), rowspan=2, colspan=4)
        ax_alloc = plt.subplot2grid((11, 4), (9, 0), rowspan=2, colspan=4)

        # ----------------------------------------------------------------------
        # ROW 0-1 LEFT: MONTHLY RETURNS TABLE (Ax_monthly)
        # ----------------------------------------------------------------------

        monthly_data = monthly_returns_df.values
        col_labels = monthly_returns_df.columns
        row_labels = monthly_returns_df.index

        # Color Monthly Returns (Green for positive, Red for negative)
        color_matrix = np.full(monthly_data.shape, 'white', dtype=object)
        for i in range(monthly_data.shape[0]):
            for j in range(monthly_data.shape[1]):
                value = monthly_data[i, j]
                val = 0.0
                if isinstance(value, str):
                    try:
                        # Attempt to parse as float (remove %)
                        if '%' in value:
                            # We use replace twice for safety just in case of double-spaces
                            val = float(value.replace('%', '').replace(' ', '').strip()) / 100.0
                        else:
                            val = float(value)
                    except ValueError:
                        pass
                else:
                    val = value

                if val > 0.0001:
                    color_matrix[i, j] = '#e6ffe6'
                elif val < -0.0001:
                    color_matrix[i, j] = '#ffe6e6'
                else:
                    color_matrix[i, j] = 'white'

        # Create the table
        table_monthly = ax_monthly.table(
            cellText=monthly_data,
            colLabels=col_labels,
            rowLabels=row_labels,
            cellColours=color_matrix,
            cellLoc='center',
            loc='center',
            bbox=[0, 0, 1, 1]
        )
        for (i, j), cell in table_monthly.get_celld().items():
            if j == monthly_data.shape[1] - 1:
                cell.set_text_props(weight='bold')

        ax_monthly.axis('off')
        ax_monthly.set_title("Annual & Monthly Returns Matrix", fontsize=12, pad=10)

        # ----------------------------------------------------------------------
        # ROW 0-1 RIGHT: NEW DASHBOARD / SHRUNK KPIS (Ax_dashboard)
        # ----------------------------------------------------------------------
        ax_dashboard.axis('off')

        # Position Status
        last_alloc = alloc.iloc[-1]
        if last_alloc == 1.0:
            pos_status = "Fully Invested"
            pos_color = 'forestgreen'
        elif last_alloc > 0.0:
            pos_status = "Partial Position"
            pos_color = 'gold'
        else:
            pos_status = "Cash / Out"
            pos_color = 'lightgray'

        # Signal Status
        tmom_status = 'Positive' if tmom_sig.iloc[-1] > 0 else 'Negative'
        tmom_color = 'green' if tmom_sig.iloc[-1] > 0 else 'red'

        sma_status = 'Positive' if sma_sig.iloc[-1] > 0 else 'Negative'
        sma_color = 'green' if sma_sig.iloc[-1] > 0 else 'red'


        # Draw Dashboard Text
        ax_dashboard.text(0.5, 0.92, "Current Position", fontsize=10, weight='bold', ha='center', transform=ax_dashboard.transAxes)
        ax_dashboard.text(0.5, 0.84, pos_status, fontsize=12, color='black', ha='center', va='center',
                          bbox=dict(facecolor=pos_color, edgecolor='black', boxstyle='round,pad=0.5'),
                          transform=ax_dashboard.transAxes)

        ax_dashboard.text(0.05, 0.73, "TMOM Signal:", fontsize=9, ha='left', transform=ax_dashboard.transAxes)
        ax_dashboard.text(0.95, 0.73, tmom_status, fontsize=9, ha='right', color='white',
                          bbox=dict(facecolor=tmom_color, edgecolor='none', boxstyle='round,pad=0.2'),
                          transform=ax_dashboard.transAxes)

        ax_dashboard.text(0.05, 0.65, "SMA Signal:", fontsize=9, ha='left', transform=ax_dashboard.transAxes)
        ax_dashboard.text(0.95, 0.65, sma_status, fontsize=9, ha='right', color='white',
                          bbox=dict(facecolor=sma_color, edgecolor='none', boxstyle='round,pad=0.2'),
                          transform=ax_dashboard.transAxes)

        # --- Section B: Latest Perfs & Core KPIs ---

        # Calculate latest perfs (FIX: use 'ME' for month-end frequency)
        returns_monthly = equity.resample('ME').last().pct_change().dropna()

        def safe_perf(series, periods):
            if len(series) < periods:
                return 0.0
            return (1 + series.iloc[-periods:].fillna(0)).prod() - 1

        mtd_ret = returns_monthly.iloc[-1] if not returns_monthly.empty else 0.0
        l3m_ret = safe_perf(returns_monthly, 3)
        l6m_ret = safe_perf(returns_monthly, 6)
        l12m_ret = safe_perf(returns_monthly, 12)

        perf_data = [
            ("Current Month (MTD)", f"{mtd_ret * 100:.2f}%"),
            ("Last 3 Months (L3M)", f"{l3m_ret * 100:.2f}%"),
            ("Last 6 Months (L6M)", f"{l6m_ret * 100:.2f}%"),
            ("Last 12 Months (L12M)", f"{l12m_ret * 100:.2f}%")
        ]

        # Core KPIs (Request 2 implemented: removed Vol and Longest DD)
        kpi_data = [
            ("CAGR (%)", f"{metrics.get('CAGR', 0) * 100:.2f}%"),
            ("Sharpe Ratio", f"{metrics.get('Sharpe', 0):.2f}"),
            ("Max Drawdown (%)", f"{metrics.get('MaxDD', 0) * 100:.2f}%"),
            # Re-adding these trade metrics, assuming they were fixed in the other function
            ("Num Trades", int(metrics.get('Num Trades', 0))),
            ("Win Trades (%)", f"{metrics.get('Win Trades', 0):.1f}%"),
            ("Profit Factor", f"{metrics.get('Profit Factor', 0):.2f}"),
            ("Avg Trade Return", f"{metrics.get('Avg Trade Return', 0) * 100:.2f}%"),
        ]

        # Combine KPIs and latest perfs
        table_rows = [row[0] for row in kpi_data] + ["---"] + [row[0] for row in perf_data]
        cell_values = [[str(row[1])] for row in kpi_data] + [["---"]] + [[row[1]] for row in perf_data]

        ax_dashboard.table(
            cellText=cell_values,
            rowLabels=table_rows,
            colLabels=["Core Metrics / Performance"],
            cellLoc='right',
            rowLoc='left',
            loc='bottom',
            bbox=[0.05, 0.0, 0.9, 0.55]
        )
        ax_dashboard.set_title("Strategy Status & Latest Performance", fontsize=12, pad=10)


        # ----------------------------------------------------------------------
        # ROW 2: EQUITY CURVE
        # ----------------------------------------------------------------------

        ax_equity.plot(equity, label=f'{asset_name} DPM Strategy', color='dodgerblue', linewidth=2)
        ax_equity.plot(benchmark.reindex(equity.index), label=f'{asset_name} Benchmark', color='gray', linestyle='--', linewidth=1)
        ax_equity.set_title(f"Cumulative Returns - {asset_name} ({algo_mode}: TMOM {tmom_lookback}m, SMA {sma_period}d)", fontsize=14)
        ax_equity.set_ylabel("Equity Curve (Base 1.0)", fontsize=10)
        ax_equity.legend(loc='upper left')
        ax_equity.grid(True, linestyle=':', alpha=0.6)
        ax_equity.tick_params(axis='x', rotation=45)

        # ----------------------------------------------------------------------
        # ROW 3: DRAWDOWN (With UserWarning fix)
        # ----------------------------------------------------------------------

        ax_drawdown.fill_between(drawdown.index, drawdown, 0, color='indianred', alpha=0.7)
        ax_drawdown.set_title("Max Drawdown Over Time", fontsize=12)
        ax_drawdown.set_ylabel("Drawdown (%)", fontsize=10)
        ax_drawdown.tick_params(axis='x', rotation=45)
        ax_drawdown.grid(True, linestyle=':', alpha=0.6)

        y_ticks = ax_drawdown.get_yticks()
        ax_drawdown.set_yticks(y_ticks)
        ax_drawdown.set_yticklabels([f'{int(x*100)}%' for x in y_ticks])

        # ----------------------------------------------------------------------
        # ROW 4, 5: SIGNALS & ALLOCATION
        # ----------------------------------------------------------------------

        ax_signals.plot(signal_prices, label=f'{asset_name} Signal Price', color='darkorange', alpha=0.8, linewidth=1)
        tmom_on_days = tmom_sig[tmom_sig > 0].index
        ax_signals.plot(signal_prices.loc[tmom_on_days], 'o', markersize=2, color='green', alpha=0.5, label='TMOM Signal ON')
        sma_on_days = sma_sig[sma_sig > 0].index
        ax_signals.plot(signal_prices.loc[sma_on_days], 'x', markersize=2, color='purple', alpha=0.5, label='SMA Signal ON')
        ax_signals.set_title(f"Signal Price and Component Signals", fontsize=12)
        ax_signals.set_ylabel("Price ($)", fontsize=10)
        ax_signals.legend(loc='upper left', fontsize=8)
        ax_signals.grid(True, linestyle=':', alpha=0.6)
        ax_signals.tick_params(axis='x', rotation=45)

        ax_alloc.fill_between(alloc.index, alloc, 0, color='forestgreen', alpha=0.6, step='post')
        ax_alloc.set_title("Strategy Allocation (0.0 to 1.0)", fontsize=12)
        ax_alloc.set_ylabel("Allocation", fontsize=10)
        ax_alloc.set_ylim(-0.05, 1.05)
        ax_alloc.grid(True, linestyle=':', alpha=0.6)
        ax_alloc.tick_params(axis='x', rotation=45)


        # Final adjustments
        fig.suptitle(f'Dynamic Portfolio Manager (DPM) Strategy Report\nAsset: {asset_name} | Mode: {algo_mode}', fontsize=16, fontweight='bold', y=0.98)
        fig.tight_layout(rect=[0, 0.03, 1, 0.96])

        # Save as a professional PDF
        file_name = f'{asset_name}_{algo_mode}_{tmom_lookback}m_{sma_period}d_professional_report.pdf'
        plt.savefig(os.path.join(output_path, file_name), format='pdf')
        plt.close(fig)
        logging.info(f"Generated professional report: {file_name}")

    def calculate_trade_metrics(self, trade_log_df: pd.DataFrame) -> Dict[str, float]:
        """
        Calculates key trading statistics from the closed trade log.
        Requires 'P/L (float)' column containing unformatted P/L values.
        """
        if trade_log_df.empty:
            return {
                'Num Trades': 0.0,
                'Win Trades': 0.0, # Now stores Win Rate as float
                'Profit Factor': 0.0,
                'Avg Trade Return': 0.0
            }

        pl_data = trade_log_df['P/L (float)']

        # 1. Total Trades
        num_trades = len(trade_log_df)

        # 2. Win Rate (as float percentage)
        win_trades_count = len(pl_data[pl_data > 0])
        win_rate = (win_trades_count / num_trades) * 100.0 if num_trades > 0 else 0.0

        # 3. Profit Factor: Gross Profit / Gross Loss (absolute value)
        gross_profit = pl_data[pl_data > 0].sum()
        gross_loss = pl_data[pl_data < 0].sum()

        profit_factor = gross_profit / abs(gross_loss) if gross_loss != 0 else np.nan

        # 4. Average Trade Return (raw float)
        avg_trade_return = pl_data.mean()

        return {
            'Num Trades': float(num_trades),
            'Win Trades': win_rate, # Stores Win Rate (%)
            'Profit Factor': profit_factor,
            'Avg Trade Return': avg_trade_return # Stores raw float
        }

    def calculate_monthly_returns(self, equity_curve: pd.Series) -> pd.DataFrame:
        """Calculates monthly returns pivot table, including YTD."""

        # 1. Calculate Daily Returns and Resample to Monthly
        daily_returns = equity_curve.pct_change().fillna(0)
        monthly_returns = daily_returns.resample('ME').apply(lambda x: (1 + x).prod() - 1)

        # 2. Create Pivot Table (Year x Month)
        monthly_returns_df = monthly_returns.to_frame(name='Monthly Return')
        monthly_returns_df['Year'] = monthly_returns_df.index.year
        monthly_returns_df['Month'] = monthly_returns_df.index.strftime('%b') # Abbreviated month name

        month_order = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        monthly_returns_pivot = monthly_returns_df.pivot(index='Year', columns='Month', values='Monthly Return').reindex(columns=month_order)

        # 3. Calculate Year-To-Date (YTD) Return
        ytd_returns = monthly_returns.groupby(monthly_returns.index.year).apply(lambda x: (1 + x).prod() - 1)
        monthly_returns_pivot['YTD'] = ytd_returns

        # 4. Formatting (Percentage strings)
        def format_return(ret):
            if pd.isna(ret):
                return ''
            return f"{ret:.2%}"

        # Apply formatting map (converts float values to strings)
        monthly_returns_pivot_formatted = monthly_returns_pivot.map(format_return)

        return monthly_returns_pivot_formatted

def main(config_path: str = 'config.yaml'):
    global rf 
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
    except FileNotFoundError:
        logging.warning(f"Config file not found at {config_path}. Using default settings.")
        config = {}

    run_sweep = config.get('run_sweep', False)
    algo_mode = config.get('algo_mode', 'Conditional') # Default to Conditional if missing
    
    tmom_lookbacks = config.get('tmom_lookbacks', [6, 9, 12, 15])
    sma_periods = config.get('sma_periods', [100, 150, 200, 250])
    
    single_tmom_lb = config.get('lookback', 15)
    single_sma_p = config.get('sma_period', 100)
    
    default_assets = [
        {'signal_ticker': 'QQQ', 'trade_ticker': 'LQQ.PA', 'benchmark_ticker': 'QQQ', 'name': 'Nasdaq-100 2x Leveraged'},
        {'signal_ticker': 'QQQ', 'trade_ticker': 'QQQ', 'benchmark_ticker': 'QQQ', 'name': 'QQQ'},
        {'signal_ticker': 'SPY', 'trade_ticker': 'SPY', 'benchmark_ticker': 'SPY', 'name': 'SPY'},
    ]
    assets = config.get('assets', default_assets)
    
    start = config.get('start_date', '2010-07-01')
    end = config.get('end_date', datetime.now().strftime('%Y-%m-%d'))
    tc = config.get('transaction_cost', 0.001)
    clear_cache = config.get('clear_cache', False)
    out_dir = config.get('output_dir', './results')
    os.makedirs(out_dir, exist_ok=True)

    fetcher = DataFetcher(clear_cache=clear_cache)
    allocator = DPMAllocator({'TMOM': TMOMSignal(), 'SMA': SMASignal()})
    simulator = PortfolioSimulator(transaction_cost=tc)
    analyzer = PerformanceAnalyzer()
    
    all_tickers = set([a['signal_ticker'] for a in assets] + [a['trade_ticker'] for a in assets] + [a['benchmark_ticker'] for a in assets])
    for ticker in all_tickers:
        fetcher.fetch_data(ticker, start, end)
    rf = fetcher.fetch_risk_free(start, end) 

    if run_sweep:
        logging.getLogger().setLevel(logging.ERROR)
        print("\n" + "="*80)
        print(f"DPM STRATEGY PARAMETER SWEEP INITIALIZED")
        print(f"ALGO MODE: {algo_mode}")
        print(f"TMOM Lookback Range (Months): {tmom_lookbacks}")
        print(f"SMA Period Range (Days): {sma_periods}")
        print("="*80)
        
        all_results = []
        print("\nStarting Parameter Sweep...")
        
        for asset_config in assets:
            signal_ticker = asset_config['signal_ticker']
            trade_ticker = asset_config['trade_ticker']
            benchmark_ticker = asset_config.get('benchmark_ticker', trade_ticker)
            name = asset_config['name']
            
            print(f"\n--- Running Sweep for Asset: {name} ({trade_ticker}) ---")

            signal_prices = fetcher.fetch_data(signal_ticker, start, end)['Close']
            trade_prices = fetcher.fetch_data(trade_ticker, start, end)['Close']
            asset_ret = trade_prices.pct_change().fillna(0)
            
            for tmom_lb, sma_p in itertools.product(tmom_lookbacks, sma_periods):
                # Pass algo_mode here
                allocations = allocator.allocate(signal_prices, rf, tmom_lb, sma_p, method=algo_mode)

                results = simulator.simulate(allocations, asset_ret, rf)
                metrics = analyzer.analyze(results['equity_curve'])

                all_results.append({
                    'Asset': name,
                    'TMOM_LB (m)': tmom_lb,
                    'SMA_P (d)': sma_p,
                    'Sharpe': metrics['Sharpe'],
                    'CAGR (%)': metrics['CAGR'] * 100,
                    'MaxDD (%)': metrics['MaxDD'] * 100,
                })
                
        if not all_results:
            print("\nNo results collected.")
            return

        results_df = pd.DataFrame(all_results)
        results_df = results_df.sort_values(by=['Asset', 'Sharpe'], ascending=[True, False]).reset_index(drop=True)

        print("\n" + "="*100)
        print(f"✨ DPM STRATEGY SWEEP RESULTS ({algo_mode}) ✨")
        print("="*100)
        
        for asset_name in results_df['Asset'].unique():
            asset_table = results_df[results_df['Asset'] == asset_name]
            best_row = asset_table.iloc[0]
            print(f"\n--- Best Parameters for {asset_name} ---")
            print(f"  Sharpe: {best_row['Sharpe']:.2f} | CAGR: {best_row['CAGR (%)']:.2f}% | MaxDD: {best_row['MaxDD (%)']:.2f}%")
            print(f"  Best Combo: TMOM={best_row['TMOM_LB (m)']}m, SMA={best_row['SMA_P (d)']}d")
            print("-" * 35)
            print(asset_table[['TMOM_LB (m)', 'SMA_P (d)', 'Sharpe', 'CAGR (%)', 'MaxDD (%)']].to_string(index=False, float_format=lambda x: f'{x:.2f}'))
            
    else:
        logging.getLogger().setLevel(logging.INFO)
        tmom_lb = single_tmom_lb
        sma_p = single_sma_p
        
        print("\n" + "="*80)
        print(f"RUNNING SINGLE OPTIMIZED REPORT MODE")
        print(f"ALGO MODE: {algo_mode}")
        print(f"PARAMETERS: TMOM={tmom_lb} months, SMA={sma_p} days")
        print("="*80)

        for asset_config in assets:
            signal_ticker = asset_config['signal_ticker']
            trade_ticker = asset_config['trade_ticker']
            benchmark_ticker = asset_config.get('benchmark_ticker', trade_ticker)
            name = asset_config['name']
            
            print(f"\n--- Processing Asset: {name} ({trade_ticker}) ---")

            signal_prices = fetcher.fetch_data(signal_ticker, start, end)['Close']
            trade_prices = fetcher.fetch_data(trade_ticker, start, end)['Close']
            bench_data = fetcher.fetch_data(benchmark_ticker, start, end)
            asset_ret = trade_prices.pct_change().fillna(0)
            bench_ret = bench_data['Close'].pct_change().fillna(0)
            benchmark = (1 + bench_ret).cumprod()

            # Pass algo_mode here
            allocations = allocator.allocate(signal_prices, rf, tmom_lb, sma_p, method=algo_mode)
            
            tmom_sig = TMOMSignal().compute_signal(signal_prices, rf_rates=rf, lookback=tmom_lb)
            sma_sig  = SMASignal().compute_signal(signal_prices, sma_period=sma_p)

            # Combine signals and allocations into a diagnostic DataFrame
            diagnostic_df = pd.DataFrame({
                'allocation': allocations,
                # Align signals to the allocation index
                'tmom_sig': tmom_sig.reindex(allocations.index, method='ffill').fillna(0),
                'sma_sig': sma_sig.reindex(allocations.index, method='ffill').fillna(0)
            }).dropna()

            # Export the diagnostic data
            diagnostic_file_name = f'{name}_{algo_mode}_{tmom_lb}m_{sma_p}d_diagnostic_signals.csv'
            diagnostic_df.to_csv(os.path.join(out_dir, diagnostic_file_name))
            
            logging.info(f"Saved diagnostic signals to: {diagnostic_file_name}")

            results = simulator.simulate(allocations, asset_ret, rf)
            equity = results['equity_curve']
            drawdown = equity / equity.cummax() - 1 

            metrics = analyzer.analyze(equity)
            logging.info(f"{name} → CAGR {metrics['CAGR']:.2%} | MaxDD {metrics['MaxDD']:.1%} | Sharpe {metrics['Sharpe']:.2f}")

            # --- NEW TIER 1 -> TIER 2 INTEGRATION: Trade Tracking ---

            # 1. Get the price data for the traded asset
            # NOTE: Using 'Close' column, adjust if necessary based on your fetched data structure
            trade_data = fetcher.fetch_data(trade_ticker, start, end)
            trade_prices = trade_data['Close']

            # 2. Instantiate and run the TradeTracker
            tracker = TradeTracker(asset_name=name)
            trade_log_df = tracker.get_trade_log(
                allocations=results['allocations'],
                asset_prices=trade_prices,
                equity_curve=equity
            )

            # --- START TIER 3 INTEGRATION: Data Reporting & Analysis (Phase 2, Step 2) ---

            # The base metrics (Sharpe, CAGR, MaxDD) must be calculated first
            # We assume metrics = analyzer.analyze(equity) was called earlier.

            # 1. Calculate the New Trade Metrics (Section 1 KPIs)
            trade_metrics = analyzer.calculate_trade_metrics(trade_log_df)
            metrics.update(trade_metrics) # Merge new trade metrics into the existing 'metrics' dict

            # 2. Calculate the Monthly Returns Matrix (Section 2 Data)
            # This is stored in a separate DataFrame
            monthly_returns_df = analyzer.calculate_monthly_returns(equity)

            # 3. Log the new core metrics (for quick inspection)
            logging.info(f"Generated Trade Metrics: Num Trades={metrics['Num Trades']}, Win Trades={metrics['Win Trades']}, Profit Factor={metrics['Profit Factor']:.2f}")
            logging.info(f"Generated Monthly Returns: {monthly_returns_df.index[0]}-{monthly_returns_df.index[-1]}")

            # --- END TIER 3 INTEGRATION ---

            # --- START TIER 4 INTEGRATION: Data Export (Phase 2, Step 3) ---

            # 1. Export the Trade Log
            trade_log_file_name = f'{name}_{algo_mode}_{tmom_lb}m_{sma_p}d_trade_log.csv'
            trade_log_df.to_csv(os.path.join(out_dir, trade_log_file_name))
            logging.info(f"Saved Trade Log to: {trade_log_file_name}")

            # 2. Export the Monthly Returns Matrix
            monthly_returns_file_name = f'{name}_{algo_mode}_{tmom_lb}m_{sma_p}d_monthly_returns.csv'
            monthly_returns_df.to_csv(os.path.join(out_dir, monthly_returns_file_name))
            logging.info(f"Saved Monthly Returns Matrix to: {monthly_returns_file_name}")

            # --- END TIER 4 INTEGRATION ---

            analyzer.plot_professional_report(
                equity=equity,
                benchmark=benchmark,
                # --- NEW ARGUMENTS ---
                metrics=metrics,                      # Pass the full metrics dictionary
                monthly_returns_df=monthly_returns_df,  # Pass the monthly returns table
                # --- EXISTING ARGUMENTS ---
                pos_numeric=results['position_numeric'],
                alloc=results['allocations'],
                signal_prices=signal_prices,
                drawdown=drawdown,
                rf=rf,
                tmom_sig=tmom_sig,
                sma_sig=sma_sig,
                asset_name=name,
                output_path=out_dir,
                tmom_lookback=tmom_lb,
                sma_period=sma_p,
                algo_mode=algo_mode # Pass algo_mode for title
            )

            results[['equity_curve', 'position_numeric', 'allocations']].to_csv(
                os.path.join(out_dir, f'{name}_{algo_mode}_full_data.csv')
            )

if __name__ == '__main__':
    main()
