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
from dpm_core.clients_notify import NotificationManager
from config.settings import settings


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
    # 1. FIX THE SIGNATURE to accept live_ticker
    def __init__(self, initial_cash: float = 100000.0, db_manager=None, live_ticker: str = None):
        self._portfolio: Dict[str, Any] = {
            'cash': initial_cash,
            'positions': {}, # Key: Ticker, Value: Quantity
            'last_trade_time': None
        }
        self.db_manager = db_manager
        self.initial_cash = initial_cash
        self.live_ticker = live_ticker # <--- STORE THE TICKER HERE

        if self.db_manager:
            self._load_state_from_db(initial_cash)

    def _load_state_from_db(self, initial_cash: float):
        """Retrieves the latest portfolio state (cash and position) from the DB."""
        if not self.db_manager:
            return

        # CHANGE: Use the stored ticker instead of global settings
        live_ticker = self.live_ticker
        if not live_ticker:
             logging.warning("BROKER_DEBUG: Cannot load position state, live_ticker is missing.")
             return
        # --- DEBUG TRACE ADDITION (2a of 2) ---
        #logging.info(f"BROKER_DEBUG: Live Ticker used for lookup: {live_ticker}")
        # --- END DEBUG TRACE ---

        # --- 1. Load Cash State from Snapshot (STILL REQUIRED) ---
        latest_snapshot = self.db_manager.get_latest_portfolio_snapshot()

        if latest_snapshot:
            self._portfolio['cash'] = latest_snapshot['cash_balance']
            #logging.info(f"Broker state loaded cash: ${self._portfolio['cash']:,.2f}")
        else:
            self._portfolio['cash'] = initial_cash
            #logging.info(f"Broker starting with initial cash: ${initial_cash:,.2f}")

        # --- 2. Reconstruct Positions from CUMULATIVE Trades (THE FIX) ---
        # The internal position tracking dictionary is reset on every run:
        self._portfolio['positions'] = {}

        # Call the new cumulative aggregation function
        net_qty = self.db_manager.get_net_position_quantity(live_ticker)

        # --- NEW DEBUG CALL ---
        if net_qty == 0.0:
            logging.warning("Broker failed to load position. Dumping trade orders for diagnosis.")
            self.db_manager.debug_dump_all_state(live_ticker)
        # --- END NEW DEBUG CALL ---

        # --- DEBUG TRACE ADDITION (2b of 2) ---
        #logging.info(f"BROKER_DEBUG: Net Qty returned by DBManager: {net_qty}")
        # --- END DEBUG TRACE ---

        if net_qty > 0.0:
            self._portfolio['positions'][live_ticker] = net_qty
            logging.info(f"Broker state loaded position: {net_qty:.2f} Qty of {live_ticker} (Cumulative).")
        else:
            self._portfolio['positions'][live_ticker] = 0.0
            logging.info("Broker confirmed zero net position.")

        # --- Final Portfolio State Check ---
        #logging.info(f"BROKER_DEBUG: Final internal cash: {self._portfolio['cash']}")
        #logging.info(f"BROKER_DEBUG: Final internal position: {self._portfolio['positions']}")

        # --- 3. Now the broker has the correct cash and position state ---

    def get_current_position(self, trade_ticker: str) -> float:
        """Returns the current quantity held (0 for a new asset)."""
        return self._portfolio['positions'].get(trade_ticker, 0)

    def get_current_cash(self) -> float:
        """Returns the current available mock cash."""
        return self._portfolio['cash']

    def execute_order(self, ticker: str, target_allocation: float, current_price: float) -> Any:
        """
        Calculates the required order (BUY/SELL) to meet the target allocation,
        updates the mock state, and persists the trade to the database.
        """
        # Determine current equity for target calculation
        current_qty = self.get_current_position(ticker)
        current_value = current_qty * current_price

        # NOTE: This uses the initial cash if no other trades were mocked yet.
        total_equity = self._portfolio['cash'] + sum(qty * current_price for t, qty in self._portfolio['positions'].items())

        # Calculate target quantity based on total equity
        if current_price <= 0:
            logging.error(f"Cannot execute order for {ticker}: Current price is zero or negative.")
            return None

        # --- FIX FOR NAME ERROR: Ensure these variables are defined ---
        target_value = total_equity * target_allocation
        target_qty = target_value / current_price
        # --- END FIX ---

        # Order needed (positive for BUY, negative for SELL)
        order_qty = target_qty - current_qty

        # Ignore tiny orders (e.g., less than $10 change)
        if abs(order_qty) * current_price < 10.0:
            logging.info(f"[{ticker}] - No material change in allocation ({target_allocation:.1%}). No order sent.")
            return None

        action = "BUY" if order_qty > 0 else "SELL"
        trade_time = datetime.now() # Capture the time for logging and DB write

        # Log the mock order
        logging.info(
            f"--- MOCK ORDER EXECUTED ---"
            f"\n  Date: {trade_time.strftime('%Y-%m-%d %H:%M:%S')}"
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

        # ====================================================================
        # --- CRITICAL FIX: PERSIST TRADE TO DATABASE & DUMP ---
        # ====================================================================

        # 1. Prepare trade data for DB persistence
        order_data = {
            'execution_time': trade_time,
            'ticker': ticker,
            'quantity': abs(order_qty),
            'price': current_price,
            'order_type': f"{action} (from {current_qty:.2f} qty to {target_qty:.2f} qty)",
            'status': 'FILLED'
        }

        # 2. Save the executed trade record
        if self.db_manager:
            self.db_manager.save_trade_order(order_data)

            # 3. DEBUG TRACE: Immediate Post-Save Dump to Validate Write
            #logging.info("TRADE_SAVE_CONFIRMATION: Dumping DB state immediately after save_trade_order.")
            #self.db_manager.debug_dump_all_state(ticker)
            # --- END DEBUG TRACE ---

        return {'status': 'MOCKED', 'action': action, 'quantity': abs(order_qty)}

class BoursoramaBrokerClient(MockBrokerClient):
    """
    A concrete implementation for Boursorama (notification-only).
    Inherits mock functionality but overrides execute_order to send a notification
    after the mock trade is processed.
    """
    # FIX 3: Update Boursorama signature to accept db_manager and pass it to super
    def __init__(self, initial_cash: float = 100000.0, notify_manager: NotificationManager = None, db_manager=None, live_ticker: str = None):
        # Pass live_ticker to the parent class
        super().__init__(initial_cash=initial_cash, db_manager=db_manager, live_ticker=live_ticker)
        self.notify_manager = notify_manager

        # If DB was used, the parent class already logged the starting cash.
        if not self.db_manager or not self.db_manager.get_latest_portfolio_snapshot():
            logging.info(f"BoursoramaBrokerClient initialized with ${initial_cash:,.2f} initial cash.")

    def execute_order(self, ticker: str, target_allocation: float, current_price: float) -> dict:
        # Step 1: Execute the trade logic (same as Mock, updates internal state)
        order_result = super().execute_order(ticker, target_allocation, current_price)

        if order_result is None:
            # If the base class returned None, it means NO_TRADE was executed.
            # We must return a dictionary to satisfy the calling logic (live_engine.py).
            return {
                'status': 'NO_TRADE',
                'action': 'HOLD',
                'order_qty': 0.0,
                'price': current_price,
                'message': 'No material change in target allocation.'
            }

        status = order_result.get('status')

        # Step 2: Send Notification if a trade occurred (Logic moved from live_engine.py)
        #if status != 'NO_TRADE' and self.notify_manager:

        #    # Recalculate portfolio state based on updated internal state
        #    new_qty = self.get_current_position(ticker)
        #    new_cash = self.get_current_cash()
        #    new_value = new_qty * current_price
        #    new_total_equity = new_cash + new_value

        #    subject = f"Boursorama Trade Alert: {ticker} ({status})"
        #    body = (
        #        f"Trade initiated for {ticker}.\n"
        #        f"Target Allocation: {target_allocation:.1%}\n"
        #        f"Execution Price: {current_price:.2f}\n"
        #        f"New Portfolio State:\n"
        #        f"  Cash: ${new_cash:,.2f}\n"
        #        f"  Position ({new_qty:.2f} Qty): ${new_value:,.2f}\n"
        #        f"  Total Equity: ${new_total_equity:,.2f}"
        #    )
        #    self.notify_manager.notify(subject, body)

        return order_result

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
        """All signals must accept 'prices' and flexible keyword arguments."""
        pass

class TMOMSignal(SignalGenerator):
    def compute_signal(self, prices: pd.Series, **kwargs) -> pd.Series:
        rf_rates = kwargs.get('rf_rates')
        lookback_months = kwargs.get('lookback_months', 12)
        
        if rf_rates is None:
            raise ValueError("TMOMSignal requires 'rf_rates' in kwargs.")

        lookback_days = lookback_months * 21
        cum_ret = prices.pct_change(lookback_days)

        # FIX: Align the RF rates to the price index to avoid comparison errors
        # Reindex RF to match prices, then fill gaps
        rf_aligned = rf_rates.reindex(prices.index).ffill().fillna(0)
        
        # Calculate cumulative risk-free return over the window
        # (Assuming rf_rates is annualized percentage like ^IRX)
        cum_rf = rf_aligned.rolling(window=lookback_days).sum() / 100.0 / 252 * 21 * lookback_months
        effective_rf = cum_rf.clip(lower=0) 

        # Now indices match perfectly
        return (cum_ret > effective_rf).astype(int).fillna(0)
    
class SMASignal(SignalGenerator):
    def __init__(self):
        # On initialise une variable pour stocker la série
        self.sma_series = None    
        
    def compute_signal(self, prices: pd.Series, **kwargs) -> pd.Series:
        # Extract specific parameters from kwargs
        period = kwargs.get('period', 200)
        
        #logging.info(f"DEBUG SMA: Received {len(prices)} points for period {period}")
        
        self.sma_series = prices.rolling(window=period).mean()
        last_values = self.sma_series.tail(5).values
        #logging.info(f"DEBUG SMA: Last 5 SMA values: {last_values}")
        
        #sma = prices.rolling(window=period).mean()
        #return (prices > sma).astype(int).fillna(0)
        return (prices > self.sma_series).astype(int).fillna(0)
    
class VIXSignal(SignalGenerator):
    def compute_signal(self, vix_prices: pd.Series, threshold: int = 22) -> pd.Series:
        # 1 = Quiet/Bullish (VIX < Threshold), 0 = Danger/Bearish (VIX >= Threshold)
        return (vix_prices < threshold).astype(int).fillna(0)
    
class DPMAllocator:
    def __init__(self, signals: Dict[str, SignalGenerator]):
        self.signals = signals

    def allocate(self, macro_prices: pd.Series, sma_prices: pd.Series, execution_prices: pd.Series, 
                 rf_rates: pd.Series, vix_prices: pd.Series, 
                 tmom_lb: int, sma_p: int, vix_t: int = 22, 
                 method: str = 'Conditional', use_vix: bool = True) -> tuple:
        
        # 1. Generate signals using the Signal Objects
        # TMOM is computed on Macro (e.g., QQQ)
        tmom_sig = self.signals['TMOM'].compute_signal(
            macro_prices, 
            rf_rates=rf_rates, 
            lookback_months=tmom_lb
        )
        
        # SMA is computed on the Benchmark (e.g., PUST) as requested
        sma_sig = self.signals['SMA'].compute_signal(
            sma_prices, 
            period=sma_p
        )

        # Ensure vix_data is aligned with the execution index
        vix_data = vix_prices.reindex(execution_prices.index).ffill()
        
        # 2. VIX Logic
        vix_sig = (vix_data < vix_t).astype(int) if use_vix else pd.Series(1, index=execution_prices.index)

        tmom_aligned = tmom_sig.reindex(execution_prices.index).ffill()
        vix_aligned = vix_sig.reindex(execution_prices.index).ffill()
        sma_aligned = sma_sig.reindex(execution_prices.index).ffill()
        
        # --- NEW DEBUG TRACE: CHECK ALIGNMENT ---
        #logging.info("=" * 30)
        #logging.info("DEBUG_ALIGNMENT: Inspecting Signal Indices")
        #logging.info(f"SMA Signal (EU) Tail:\n{sma_sig.tail(3)}")
        #logging.info(f"TMOM Signal (US) Tail:\n{tmom_sig.tail(3)}")
        #logging.info(f"VIX Signal (US) Tail:\n{vix_sig.tail(3)}")
        
        # Check the raw merge BEFORE fillna(0) hides the NaNs
        #raw_merge = pd.DataFrame({'sma': sma_sig, 'tmom': tmom_sig, 'vix': vix_sig})
        #logging.info(f"Merged DataFrame Tail (Before Fill):\n{raw_merge.tail(3)}")
        #logging.info("=" * 30)
        # --- END DEBUG TRACE ---
      
        # 3. State Machine DataFrame
        signals_df = pd.DataFrame({
            'sma': sma_aligned,
            'tmom': tmom_aligned,
            'vix': vix_aligned
        }).fillna(0)

        def get_allocation_logic(row):
            s, t, v = row['sma'], row['tmom'], row['vix']
            if not use_vix:
                if s == 1 and t == 1: return 1.0
                if s == 1 and t == 0: return 0.5
                return 0.0
            else:
                if v == 0:  # HIGH VIX
                    if s == 1 and t == 1: return 0.70
                    if s == 1 and t == 0: return 0.40
                    if s == 0 and t == 1: return 0.10
                    return 0.0
                else:       # LOW VIX
                    if s == 1 and t == 1: return 1.00
                    if s == 1 and t == 0: return 0.80
                    if s == 0 and t == 1: return 0.60
                    return 0.20

        # 5. Final Calculation
        if method == 'Conditional':
            allocations = signals_df.apply(get_allocation_logic, axis=1)
        elif method == 'Linear':
            allocations = (tmom_sig + sma_sig) / 2.0
        else:
            raise ValueError(f"Unknown method: {method}")

        return allocations.dropna(), tmom_sig.dropna(), sma_sig.dropna(), vix_sig.dropna()

    @property
    def last_sma_value(self):
        # Access the signal via the 'SMA' key in the signals dictionary
        sma_obj = self.signals.get('SMA') 
        if sma_obj and hasattr(sma_obj, 'sma_series') and sma_obj.sma_series is not None:
            return float(sma_obj.sma_series.iloc[-1])
        return 0.0    
    
class PortfolioSimulator:
    def __init__(self, transaction_cost: float = 0.001, slippage: float = 0.0005):
        self.transaction_cost = transaction_cost
        self.slippage = slippage

    def simulate(self, allocations: pd.Series, returns: pd.Series, rf: pd.Series, lag: int = 0) -> pd.DataFrame:
        """
        Simulates performance. 
        lag=0: Trade at 5:20 PM (Same Day).
        lag=1: Trade at Next Day Open.
        """
        # For your 5:20 PM Europe goal, lag MUST stay 0
        shifted_alloc = allocations.shift(lag).fillna(0)
        
        # Calculate strategy returns
        # (Allocation * Asset Return) + (Cash * Risk Free Rate)
        portfolio_rets = (shifted_alloc * returns) + ((1 - shifted_alloc) * (rf / 252))
        
        # Subtract transaction costs only when allocation changes
        trades = shifted_alloc.diff().abs().fillna(0)
        costs = trades * (self.transaction_cost + self.slippage)
        final_rets = portfolio_rets - costs
        
        equity_curve = (1 + final_rets).cumprod()
        
        return pd.DataFrame({
            'equity_curve': equity_curve,
            'returns': final_rets,
            'allocations': shifted_alloc,
            'position_numeric': shifted_alloc
        })
    
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
        ret = equity.pct_change(fill_method=None).dropna()
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
        daily_returns = equity_curve.pct_change(fill_method=None).fillna(0)
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
