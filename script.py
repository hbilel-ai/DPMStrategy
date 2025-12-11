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
            return {'CAGR': np.nan, 'MaxDD': np.nan, 'Sharpe': np.nan}
        years = (equity.index[-1] - equity.index[0]).days / 365.25
        cagr = (equity.iloc[-1] / equity.iloc[0]) ** (1/years) - 1
        dd = (equity / equity.cummax() - 1).min()
        ret = equity.pct_change().dropna()
        sharpe = ret.mean() / ret.std() * np.sqrt(252) if ret.std() != 0 else np.nan
        return {'CAGR': cagr, 'MaxDD': dd, 'Sharpe': sharpe}

    def plot_professional_report(self, equity: pd.Series, benchmark: pd.Series,
                                pos_numeric: pd.Series, alloc: pd.Series,
                                signal_prices: pd.Series,
                                drawdown: pd.Series, 
                                rf: pd.Series, 
                                tmom_sig: pd.Series, 
                                sma_sig: pd.Series,  
                                asset_name: str, output_path: str,
                                tmom_lookback: int, sma_period: int,
                                algo_mode: str): # Added algo_mode for Title
        
        if len(equity) == 0:
            logging.warning(f"No data for {asset_name} — skipping plot")
            return
        
        # --- SMA Calculation for Plotting (Daily SMA) ---
        daily_sma = signal_prices.rolling(window=sma_period).mean() 
        
        last_date = equity.index[-1]
        current_alloc = alloc.iloc[-1]
        
        # Re-calculate raw signals for display purposes (Status Box)
        raw_tmom_sig = TMOMSignal().compute_signal(signal_prices, rf_rates=rf, lookback=tmom_lookback).reindex(alloc.index, method='ffill').fillna(0)
        raw_sma_sig = SMASignal().compute_signal(signal_prices, sma_period=sma_period).reindex(alloc.index, method='ffill').fillna(0)
        
        current_tmom = int(raw_tmom_sig.iloc[-1])
        current_sma = int(raw_sma_sig.iloc[-1])
        
        if current_alloc == 1:
            current_pos = "INVESTED"
            color_pos = "lightgreen"
        elif current_alloc == 0.5:
            current_pos = "PARTIAL"
            color_pos = "orange"
        else:
            current_pos = "CASH"
            color_pos = "lightcoral"
            
        fig = plt.figure(figsize=(16, 12)) 
        gs = fig.add_gridspec(3, 1, height_ratios=[4, 1.5, 1.5], hspace=0.3) 

        # --- Subplot 1: Equity Curve ---
        ax1 = fig.add_subplot(gs[0])
        ax1.plot(equity.index, equity, label=f'DPM Strategy ({algo_mode})', color='tab:blue', lw=2.5)
        ax1.plot(benchmark.index, benchmark, label='Buy & Hold', color='black', lw=1.5, alpha=0.7)
        ax1_price = ax1.twinx()
        
        ax1_price.plot(signal_prices.index, signal_prices, label=f'Signal Price ({asset_name})', color='green', lw=1.0, alpha=0.5)
        ax1_price.plot(daily_sma.index, daily_sma, label=f'{sma_period}D SMA', color='red', linestyle='--', lw=1.5)
        ax1_price.set_ylabel(f'Signal Asset Price ({asset_name})', fontsize=12, color='black')
        
        lines, labels = ax1.get_legend_handles_labels()
        lines2, labels2 = ax1_price.get_legend_handles_labels()
        ax1.legend(lines + lines2, labels + labels2, fontsize=13, loc='upper left')
        
        ax1.set_yscale('log')
        ax1.set_title(f'{asset_name} (DPM Signals - {algo_mode} Logic)', fontsize=18, fontweight='bold', pad=20)
        ax1.set_ylabel('Portfolio Value (log scale)', fontsize=12)
        ax1.grid(alpha=0.3)
        plt.setp(ax1.get_xticklabels(), visible=False) 

        # --- Subplot 2: Max Drawdown ---
        ax2 = fig.add_subplot(gs[1], sharex=ax1) 
        ax2.fill_between(drawdown.index, 0, drawdown, color='tab:red', alpha=0.6)
        ax2.set_title('Max Drawdown Over Time', fontsize=13)
        ax2.set_ylabel('Drawdown', fontsize=12)
        ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.1%}'))
        ax2.grid(alpha=0.3)
        ax2.set_ylim(1.1 * drawdown.min(), 0.05)
        plt.setp(ax2.get_xticklabels(), visible=False) 
        
        # --- Subplot 3: Position Over Time ---
        ax3 = fig.add_subplot(gs[2], sharex=ax1) 
        ax3.fill_between(pos_numeric.index, 0, 1, where=pos_numeric==0, color='lightcoral', alpha=0.8)
        ax3.fill_between(pos_numeric.index, 1, 1.5, where=pos_numeric==1, color='orange', alpha=0.8)
        ax3.fill_between(pos_numeric.index, 1.5, 2, where=pos_numeric==2, color='lightgreen', alpha=0.8)
        ax3.step(pos_numeric.index, pos_numeric, where='post', color='black', lw=1.5)
        ax3.set_yticks([0, 1, 2])
        ax3.set_yticklabels(['CASH', 'PARTIAL', 'INVESTED'], fontweight='bold', fontsize=12)
        ax3.set_ylim(-0.1, 2.1)
        ax3.set_title('Position Over Time', fontsize=13)
        ax3.set_xlabel('Date', fontsize=12)
        ax3.grid(alpha=0.3)

        # Label Orange Rectangles
        df_signals = pd.DataFrame({
            'alloc': alloc,
            'T': tmom_sig.reindex(alloc.index, method='ffill').fillna(0),
            'S': sma_sig.reindex(alloc.index, method='ffill').fillna(0)
        }).dropna()
        
        for date, row in df_signals.resample('W-MON').first().dropna().iterrows():
            if row['alloc'] == 0.5: 
                if row['T'] == 1.0: 
                    label = 'M' 
                elif row['S'] == 1.0:
                    label = 'S' 
                else:
                    continue 
                ax3.text(date, 1.25, label, fontsize=9, fontweight='bold', color='black', ha='center', va='center') 

        # Red lines at trades 
        change_dates = alloc.diff().abs() > 0.01
        change_dates = change_dates[change_dates].index
        for date in change_dates:
            ax1.axvline(date, color='red', alpha=0.6, linestyle='--', linewidth=1.2)
            ax2.axvline(date, color='red', alpha=0.6, linestyle='--', linewidth=1.2)
            ax3.axvline(date, color='red', alpha=0.6, linestyle='--', linewidth=1.2)

        # KPI table
        metrics = self.analyze(equity)
        time_in_market = (alloc > 0).mean()
        kpi = [
            ["CAGR", f"{metrics['CAGR']:.2%}"],
            ["Max Drawdown", f"{metrics['MaxDD']:.1%}"],
            ["Sharpe Ratio", f"{metrics['Sharpe']:.2f}"],
            ["Time in Market", f"{time_in_market:.1%}"],
            ["Total Trades", f"{len(change_dates)}"],
        ]
        table = ax1.table(cellText=kpi, colWidths=[0.19, 0.14], cellLoc='center',
                          loc='upper left', bbox=[0.02, 0.62, 0.28, 0.28])
        table.auto_set_font_size(False)
        table.set_fontsize(11.5)
        for (r, c), cell in table.get_celld().items():
            cell.set_edgecolor('black')
            if r == 0:
                cell.set_facecolor('#204680')
                cell.set_text_props(color='white', weight='bold')

        # Status box
        status_text = (
            f"Current Position\n"
            f"{current_pos}\n\n"
            f"Logic: {algo_mode}\n"
            f"Raw TMOM Sig: {'Bullish' if current_tmom else 'Bearish'}\n"
            f"Raw SMA Sig : {'Above MA' if current_sma else 'Below MA'}\n"
            f"TMOM Lookback: {tmom_lookback} months\n"
            f"SMA Period: {sma_period} days\n"
            f"Last update: {last_date.strftime('%Y-%m-%d')}"
        )
        ax1.text(0.98, 0.02, status_text,
                 transform=ax1.transAxes,
                 fontsize=12,
                 verticalalignment='bottom',
                 horizontalalignment='right',
                 bbox=dict(boxstyle="round,pad=0.6", facecolor=color_pos, alpha=0.9),
                 linespacing=1.4)

        plt.suptitle(f'Downside Protection Model — {algo_mode} Logic (Labeled)',
                     fontsize=20, fontweight='bold', y=0.95)
        plt.subplots_adjust(top=0.92, bottom=0.06, left=0.06, right=0.96)
        plt.savefig(os.path.join(output_path, f'{asset_name}_DPM_{algo_mode}_{tmom_lookback}m_{sma_period}d_labeled.png'),
                    dpi=300, bbox_inches='tight', facecolor='white') 
        plt.close()

    def calculate_trade_metrics(self, trade_log_df: pd.DataFrame) -> Dict[str, float]:
        """
        Calculates key trading statistics from the closed trade log.
        Requires 'P/L (float)' column containing unformatted P/L values.
        """
        if trade_log_df.empty:
            return {
                'Num Trades': 0,
                'Win Trades': 0,
                'Profit Factor': 0.0,
                'Avg P/L (%)': 0.0
            }

        pl_data = trade_log_df['P/L (float)']

        # 1. Total Trades
        num_trades = len(trade_log_df)

        # 2. Win Trades
        win_trades = len(pl_data[pl_data > 0])

        # 3. Profit Factor: Gross Profit / Gross Loss (absolute value)
        gross_profit = pl_data[pl_data > 0].sum()
        gross_loss = pl_data[pl_data < 0].sum()

        # Avoid division by zero
        profit_factor = gross_profit / abs(gross_loss) if gross_loss != 0 else np.nan

        # 4. Average P/L (%)
        avg_pl = pl_data.mean() * 100 # Convert mean to percentage

        return {
            'Num Trades': num_trades,
            # We store win trades as an integer count, but the dictionary must conform to float type hint
            'Win Trades': float(win_trades),
            'Profit Factor': profit_factor,
            'Avg P/L (%)': avg_pl
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
