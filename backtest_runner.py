# backtest_runner.py

import os
import yaml
import logging
import itertools
from datetime import datetime
from dpm_core.strategy_core import (
    DataFetcher, TMOMSignal, SMASignal, DPMAllocator,
    PortfolioSimulator, TradeTracker, PerformanceAnalyzer
)
import pandas as pd
import numpy as np

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
out_dir = './results'
os.makedirs(out_dir, exist_ok=True)

def main(config_path: str = 'config.yaml'):
    global rf
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
    except FileNotFoundError:
        logging.warning(f"Config file not found at {config_path}. Using default settings.")
        config = {}

    run_sweep = config.get('run_sweep', False)
    algo_mode = config.get('algo_mode', 'Conditional')

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
                #allocations = allocator.allocate(signal_prices, rf, tmom_lb, sma_p, method=algo_mode)
                allocations, daily_tmom_sig, daily_sma_sig = allocator.allocate(
                    signal_prices=signal_prices,
                    rf_rates=rf,
                    tmom_lb=tmom_lb,
                    sma_p=sma_p,
                    method=algo_mode
                )

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

            #allocations = allocator.allocate(signal_prices, rf, tmom_lb, sma_p, method=algo_mode)
            allocations, daily_tmom_sig, daily_sma_sig = allocator.allocate(
                signal_prices=signal_prices,
                rf_rates=rf,
                tmom_lb=tmom_lb,
                sma_p=sma_p,
                method=algo_mode
            )

            tmom_sig = TMOMSignal().compute_signal(signal_prices, rf_rates=rf, lookback=tmom_lb)
            sma_sig  = SMASignal().compute_signal(signal_prices, sma_period=sma_p)

            diagnostic_df = pd.DataFrame({
                'allocation': allocations,
                'tmom_sig': tmom_sig.reindex(allocations.index, method='ffill').fillna(0),
                'sma_sig': sma_sig.reindex(allocations.index, method='ffill').fillna(0)
            }).dropna()

            diagnostic_file_name = f'{name}_{algo_mode}_{tmom_lb}m_{sma_p}d_diagnostic_signals.csv'
            diagnostic_df.to_csv(os.path.join(out_dir, diagnostic_file_name))
            logging.info(f"Saved diagnostic signals to: {diagnostic_file_name}")

            results = simulator.simulate(allocations, asset_ret, rf)
            equity = results['equity_curve']
            drawdown = equity / equity.cummax() - 1

            metrics = analyzer.analyze(equity)
            logging.info(f"{name} → CAGR {metrics['CAGR']:.2%} | MaxDD {metrics['MaxDD']:.1%} | Sharpe {metrics['Sharpe']:.2f}")

            trade_data = fetcher.fetch_data(trade_ticker, start, end)
            trade_prices = trade_data['Close']

            tracker = TradeTracker(asset_name=name)
            trade_log_df = tracker.get_trade_log(
                allocations=results['allocations'],
                asset_prices=trade_prices,
                equity_curve=equity
            )

            trade_metrics = analyzer.calculate_trade_metrics(trade_log_df)
            metrics.update(trade_metrics)
            monthly_returns_df = analyzer.calculate_monthly_returns(equity)

            logging.info(f"Generated Trade Metrics: Num Trades={metrics['Num Trades']}, Win Trades={metrics['Win Trades']}, Profit Factor={metrics['Profit Factor']:.2f}")
            logging.info(f"Generated Monthly Returns: {monthly_returns_df.index[0]}-{monthly_returns_df.index[-1]}")

            trade_log_file_name = f'{name}_{algo_mode}_{tmom_lb}m_{sma_p}d_trade_log.csv'
            trade_log_df.to_csv(os.path.join(out_dir, trade_log_file_name))
            logging.info(f"Saved Trade Log to: {trade_log_file_name}")

            monthly_returns_file_name = f'{name}_{algo_mode}_{tmom_lb}m_{sma_p}d_monthly_returns.csv'
            monthly_returns_df.to_csv(os.path.join(out_dir, monthly_returns_file_name))
            logging.info(f"Saved Monthly Returns Matrix to: {monthly_returns_file_name}")

            analyzer.plot_professional_report(
                equity=equity,
                benchmark=benchmark,
                metrics=metrics,
                monthly_returns_df=monthly_returns_df,
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
                algo_mode=algo_mode
            )

            results[['equity_curve', 'position_numeric', 'allocations']].to_csv(
                os.path.join(out_dir, f'{name}_{algo_mode}_full_data.csv')
            )

if __name__ == '__main__':
    main()
