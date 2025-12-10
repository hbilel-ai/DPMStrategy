### DPM Strategy — Decoupled Logic (with Parameter Sweep & Config Control)

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

    def allocate(self, prices: pd.Series, rf_rates: pd.Series, lookback: int, sma_period: int) -> pd.Series:
        tmom = self.signals['TMOM'].compute_signal(prices, rf_rates=rf_rates, lookback=lookback)
        sma  = self.signals['SMA'].compute_signal(prices, sma_period=sma_period)
        return (tmom + sma) / 2.0

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
                                tmom_sig: pd.Series, sma_sig: pd.Series,
                                signal_prices: pd.Series,
                                drawdown: pd.Series, # <-- NEW PARAMETER: Drawdown series
                                asset_name: str, output_path: str,
                                tmom_lookback: int, sma_period: int):
        
        if len(equity) == 0:
            logging.warning(f"No data for {asset_name} — skipping plot")
            return
        
        # --- SMA Calculation for Plotting (Daily SMA) ---
        daily_sma = signal_prices.rolling(window=sma_period).mean() 
        
        last_date = equity.index[-1]
        current_alloc = alloc.iloc[-1]
        current_tmom = int(tmom_sig.iloc[-1])
        current_sma = int(sma_sig.iloc[-1])
        
        if current_alloc == 1:
            current_pos = "INVESTED"
            color_pos = "lightgreen"
        elif current_alloc == 0.5:
            current_pos = "PARTIAL"
            color_pos = "orange"
        else:
            current_pos = "CASH"
            color_pos = "lightcoral"

        # Change figure height to accommodate 3 plots
        fig = plt.figure(figsize=(16, 12)) 
        # Set up a 3-row grid: [Equity Curve (4), Drawdown (1.5), Position (1.5)]
        gs = fig.add_gridspec(3, 1, height_ratios=[4, 1.5, 1.5], hspace=0.3) 

        # --- Subplot 1: Equity Curve, Price, and SMA ---
        ax1 = fig.add_subplot(gs[0])
        ax1.plot(equity.index, equity, label='DPM Strategy', color='tab:blue', lw=2.5)
        ax1.plot(benchmark.index, benchmark, label='Buy & Hold', color='black', lw=1.5, alpha=0.7)
        ax1_price = ax1.twinx()
        
        # Plot the SMA and Price on the secondary Y-axis (ax1_price)
        ax1_price.plot(signal_prices.index, signal_prices, label=f'Signal Price ({asset_name})', color='green', lw=1.0, alpha=0.5)
        ax1_price.plot(daily_sma.index, daily_sma, label=f'{sma_period}D SMA', color='red', linestyle='--', lw=1.5)
        ax1_price.set_ylabel(f'Signal Asset Price ({asset_name})', fontsize=12, color='black')
        
        # Combine legends
        lines, labels = ax1.get_legend_handles_labels()
        lines2, labels2 = ax1_price.get_legend_handles_labels()
        ax1.legend(lines + lines2, labels + labels2, fontsize=13, loc='upper left')
        
        ax1.set_yscale('log')
        ax1.set_title(f'{asset_name} (DPM Signals)', fontsize=18, fontweight='bold', pad=20)
        ax1.set_ylabel('Portfolio Value (log scale)', fontsize=12)
        ax1.grid(alpha=0.3)
        plt.setp(ax1.get_xticklabels(), visible=False) # Remove X-axis labels

        # --- Subplot 2: Max Drawdown Over Time (NEW) ---
        ax2 = fig.add_subplot(gs[1], sharex=ax1) 
        ax2.fill_between(drawdown.index, 0, drawdown, color='tab:red', alpha=0.6)
        ax2.set_title('Max Drawdown Over Time', fontsize=13)
        ax2.set_ylabel('Drawdown', fontsize=12)
        ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.1%}'))
        ax2.grid(alpha=0.3)
        ax2.set_ylim(1.1 * drawdown.min(), 0.05) # Set Y-limit below 0
        plt.setp(ax2.get_xticklabels(), visible=False) # Remove X-axis labels
        
        # --- Subplot 3: Position Over Time (MODIFIED AX3) ---
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

        # Red lines at actual trades (now applied to all three subplots)
        change_dates = alloc.diff().abs() > 0.01
        change_dates = change_dates[change_dates].index
        for date in change_dates:
            ax1.axvline(date, color='red', alpha=0.6, linestyle='--', linewidth=1.2)
            ax2.axvline(date, color='red', alpha=0.6, linestyle='--', linewidth=1.2)
            ax3.axvline(date, color='red', alpha=0.6, linestyle='--', linewidth=1.2)

        # KPI table below legend
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

        # Current status box
        status_text = (
            f"Current Position\n"
            f"{current_pos}\n\n"
            f"TMOM Signal: {'Bullish' if current_tmom else 'Bearish'}\n"
            f"SMA Signal : {'Above MA' if current_sma else 'Below MA'}\n"
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

        plt.suptitle('Downside Protection Model — Clean & Professional Report',
                     fontsize=20, fontweight='bold', y=0.95)
        plt.subplots_adjust(top=0.92, bottom=0.06, left=0.06, right=0.96)
        plt.savefig(os.path.join(output_path, f'{asset_name}_DPM_clean_report_{tmom_lookback}m_{sma_period}d_with_dd.png'),
                    dpi=300, bbox_inches='tight', facecolor='white') # Added _with_dd for new file name
        plt.close()


def main(config_path: str = 'config.yaml'):
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
    except FileNotFoundError:
        logging.warning(f"Config file not found at {config_path}. Using default settings.")
        config = {}

    run_sweep = config.get('run_sweep', True)
    
    # Define the parameter ranges for sweep mode
    tmom_lookbacks = config.get('tmom_lookbacks', [6, 9, 12, 15])
    sma_periods = config.get('sma_periods', [100, 150, 200, 250])
    
    # Define single run parameters (used if run_sweep=False)
    single_tmom_lb = config.get('lookback', 12)
    single_sma_p = config.get('sma_period', 100)
    
    # Define assets to trade
    default_assets = [
        {'signal_ticker': 'QQQ', 'trade_ticker': 'LQQ.PA', 'benchmark_ticker': 'QQQ', 'name': 'Nasdaq-100 2x Leveraged'},
        {'signal_ticker': 'QQQ', 'trade_ticker': 'QQQ', 'benchmark_ticker': 'QQQ', 'name': 'QQQ'},
        {'signal_ticker': 'SPY', 'trade_ticker': 'SPY', 'benchmark_ticker': 'SPY', 'name': 'SPY'},
    ]
    assets = config.get('assets', default_assets)
    
    # Global backtest settings
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
    
    # --- Data Pre-fetching (Optimization) ---
    all_tickers = set([a['signal_ticker'] for a in assets] + [a['trade_ticker'] for a in assets] + [a['benchmark_ticker'] for a in assets])
    for ticker in all_tickers:
        fetcher.fetch_data(ticker, start, end)
    rf = fetcher.fetch_risk_free(start, end)

    if run_sweep:
        # (Sweep logic remains the same)
        # Switch logging to ERROR during sweep to minimize console output
        logging.getLogger().setLevel(logging.ERROR)
        
        print("\n" + "="*80)
        print(f"DPM STRATEGY PARAMETER SWEEP INITIALIZED")
        print(f"TMOM Lookback Range (Months): {tmom_lookbacks}")
        print(f"SMA Period Range (Days): {sma_periods}")
        print(f"Total Runs Per Asset: {len(tmom_lookbacks) * len(sma_periods)}")
        print(f"Total Assets: {len(assets)}")
        print("="*80)
        
        all_results = []
        print("\nStarting Parameter Sweep...")
        
        for asset_config in assets:
            signal_ticker = asset_config['signal_ticker']
            trade_ticker = asset_config['trade_ticker']
            benchmark_ticker = asset_config.get('benchmark_ticker', trade_ticker)
            name = asset_config['name']
            
            print(f"\n--- Running Sweep for Asset: {name} ({trade_ticker}) ---")

            # Load pre-fetched data
            signal_prices = fetcher.fetch_data(signal_ticker, start, end)['Close']
            trade_prices = fetcher.fetch_data(trade_ticker, start, end)['Close']
            bench_data = fetcher.fetch_data(benchmark_ticker, start, end)
            asset_ret = trade_prices.pct_change().fillna(0)
            
            bench_ret = bench_data['Close'].pct_change().fillna(0)
            benchmark = (1 + bench_ret).cumprod()
            
            # Iterate over all parameter combinations
            for tmom_lb, sma_p in itertools.product(tmom_lookbacks, sma_periods):
                # Generate signals
                tmom_sig = TMOMSignal().compute_signal(signal_prices, rf_rates=rf, lookback=tmom_lb)
                sma_sig  = SMASignal().compute_signal(signal_prices, sma_period=sma_p)
                allocations = (tmom_sig + sma_sig) / 2.0

                # Simulate
                results = simulator.simulate(allocations, asset_ret, rf)
                equity = results['equity_curve']
                
                # Analyze
                metrics = analyzer.analyze(equity)

                # Store results
                all_results.append({
                    'Asset': name,
                    'TMOM_LB (m)': tmom_lb,
                    'SMA_P (d)': sma_p,
                    'Sharpe': metrics['Sharpe'],
                    'CAGR (%)': metrics['CAGR'] * 100,
                    'MaxDD (%)': metrics['MaxDD'] * 100,
                })
                
        # --- FINAL REPORT ---
        if not all_results:
            print("\nNo results collected. Check data fetching and start/end dates.")
            return

        results_df = pd.DataFrame(all_results)
        
        # Sort by Sharpe Ratio to find the best-performing parameters
        results_df = results_df.sort_values(by=['Asset', 'Sharpe'], ascending=[True, False]).reset_index(drop=True)

        print("\n" + "="*100)
        print("✨ DPM STRATEGY PARAMETER SWEEP RESULTS (Sorted by Sharpe Ratio) ✨")
        print("="*100)
        
        # Print individual tables for each asset, highlighting the best one
        for asset_name in results_df['Asset'].unique():
            asset_table = results_df[results_df['Asset'] == asset_name]
            
            # Get best-performing row
            best_row = asset_table.iloc[0]
            
            print(f"\n--- Best Parameters for {asset_name} ---")
            print(f"  Sharpe: {best_row['Sharpe']:.2f} | CAGR: {best_row['CAGR (%)']:.2f}% | MaxDD: {best_row['MaxDD (%)']:.2f}%")
            print(f"  Best Combo: TMOM={best_row['TMOM_LB (m)']}m, SMA={best_row['SMA_P (d)']}d")
            print("-" * 35)

            print(asset_table[['TMOM_LB (m)', 'SMA_P (d)', 'Sharpe', 'CAGR (%)', 'MaxDD (%)']].to_string(
                index=False, float_format=lambda x: f'{x:.2f}'
            ))
            
    # --- SINGLE OPTIMIZED PLOT MODE (FIXED TO LOOP OVER ALL ASSETS) ---
    else:
        # Ensure logging is at INFO level for single run feedback
        logging.getLogger().setLevel(logging.INFO)
        
        # Use single run parameters from config
        tmom_lb = single_tmom_lb
        sma_p = single_sma_p
        
        print("\n" + "="*80)
        print(f"RUNNING SINGLE OPTIMIZED REPORT MODE")
        print(f"PARAMETERS (Applied to All Assets): TMOM={tmom_lb} months, SMA={sma_p} days")
        print("="*80)

        # FIX: Iterate through all assets defined in the config
        for asset_config in assets:
            
            signal_ticker = asset_config['signal_ticker']
            trade_ticker = asset_config['trade_ticker']
            benchmark_ticker = asset_config.get('benchmark_ticker', trade_ticker)
            name = asset_config['name']
            
            print(f"\n--- Processing Asset: {name} ({trade_ticker}) ---")

            # Load pre-fetched data
            signal_prices = fetcher.fetch_data(signal_ticker, start, end)['Close']
            trade_prices = fetcher.fetch_data(trade_ticker, start, end)['Close']
            bench_data = fetcher.fetch_data(benchmark_ticker, start, end)
            asset_ret = trade_prices.pct_change().fillna(0)
            
            bench_ret = bench_data['Close'].pct_change().fillna(0)
            benchmark = (1 + bench_ret).cumprod()

            # Generate signals
            tmom_sig = TMOMSignal().compute_signal(signal_prices, rf_rates=rf, lookback=tmom_lb)
            sma_sig  = SMASignal().compute_signal(signal_prices, sma_period=sma_p)
            allocations = (tmom_sig + sma_sig) / 2.0

            # Simulate
            results = simulator.simulate(allocations, asset_ret, rf)
            equity = results['equity_curve']
            
            # CALCULATE DRAWDOWN SERIES (NEW)
            drawdown = equity / equity.cummax() - 1 
            
            # Analyze and Log
            metrics = analyzer.analyze(equity)
            logging.info(f"{name} → CAGR {metrics['CAGR']:.2%} | MaxDD {metrics['MaxDD']:.1%} | Sharpe {metrics['Sharpe']:.2f}")

            # Plot the professional report
            analyzer.plot_professional_report(
                equity=equity,
                benchmark=benchmark,
                pos_numeric=results['position_numeric'],
                alloc=results['allocations'],
                tmom_sig=tmom_sig,
                sma_sig=sma_sig,
                signal_prices=signal_prices,
                drawdown=drawdown, # <-- Pass new drawdown series
                asset_name=name,
                output_path=out_dir,
                tmom_lookback=tmom_lb,
                sma_period=sma_p
            )
            
            # Export allocations to CSV
            results[['equity_curve', 'position_numeric', 'allocations']].to_csv(os.path.join(out_dir, f'{name}_full_simulation_data.csv'))


if __name__ == '__main__':
    main()
