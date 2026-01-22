# backtest_runner.py

import os
import yaml
import logging
import itertools
from datetime import datetime
import pandas as pd
import numpy as np

from dpm_core.strategy_core import (
    DataFetcher, TMOMSignal, SMASignal, DPMAllocator,
    PortfolioSimulator, TradeTracker, PerformanceAnalyzer
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
out_dir = './results'
os.makedirs(out_dir, exist_ok=True)

def main(config_path: str = 'config.yaml'):
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
    except Exception as e:
        logging.error(f"Error loading config: {e}")
        return

    run_sweep = config.get('run_sweep', False)
    use_vix = config.get('use_vix', True)
    algo_mode = config.get('algo_mode', 'Conditional')
    
    # Parameters
    tmom_lookbacks = config.get('tmom_lookbacks', [6, 9, 12, 15])
    sma_periods = config.get('sma_periods', [100, 150, 200, 250])
    vix_thresholds = config.get('vix_thresholds', [20, 22, 25])
    
    single_tmom_lb = config.get('lookback', 15)
    single_sma_p = config.get('sma_period', 100)
    vix_threshold = config.get('vix_threshold', 22)

    # Hybrid Asset Config
    assets = config.get('assets', [])

    start = config.get('start_date', '2010-07-01')
    end = config.get('end_date', datetime.now().strftime('%Y-%m-%d'))
    tc = config.get('transaction_cost', 0.006)
    clear_cache = config.get('clear_cache', False)

    fetcher = DataFetcher(clear_cache=clear_cache)
    allocator = DPMAllocator({'TMOM': TMOMSignal(), 'SMA': SMASignal()})
    simulator = PortfolioSimulator(transaction_cost=tc)
    analyzer = PerformanceAnalyzer()

    # --- GLOBAL US DATA FETCHING WITH N-1 SHIFT ---
    vix_raw = fetcher.fetch_data('^VIX', start, end)['Close']
    vix_data_n1 = vix_raw.shift(1) # Shift VIX globally
    
    rf = fetcher.fetch_risk_free(start, end)

    if run_sweep:
        param_combinations = list(itertools.product(tmom_lookbacks, sma_periods, vix_thresholds))
        all_sweep_results = []
        
        print(f"\n--- STARTING TRIPLE-HYBRID SWEEP (N-1 US Logic / lag=0) ---")
        
        for asset_cfg in assets:
            name = asset_cfg['name']
            
            # 1. Fetch US Macro Price (QQQ) and apply N-1 shift
            m_prices_raw = fetcher.fetch_data(asset_cfg['macro_ticker'], start, end)['Close']
            macro_prices_n1 = m_prices_raw.shift(1)
            
            # 2. Fetch European Prices (SMA and Trade) - stay on Day N
            s_prices_raw = fetcher.fetch_data(asset_cfg['sma_ticker'], start, end)['Close']
            t_df_raw = fetcher.fetch_data(asset_cfg['trade_ticker'], start, end)
            
            # 3. Align everything to the European Asset Index
            common = macro_prices_n1.index.intersection(s_prices_raw.index).intersection(t_df_raw.index)
            macro_prices = macro_prices_n1.loc[common]
            vix_data_aligned = vix_data_n1.loc[common]
            sma_prices = s_prices_raw.loc[common]
            execution_prices = t_df_raw.loc[common]['Close']
            
            trade_returns = execution_prices.pct_change().fillna(0)

            for lb, p, vt in param_combinations:
                allocs, _, _, _ = allocator.allocate(
                    macro_prices=macro_prices, 
                    sma_prices=sma_prices,
                    execution_prices=execution_prices,
                    rf_rates=rf,
                    vix_prices=vix_data_aligned,
                    tmom_lb=lb, sma_p=p, vix_t=vt,
                    method=algo_mode, use_vix=use_vix
                )
                
                # Using lag=0 because we manually shifted US signals to N-1
                res = simulator.simulate(allocs, trade_returns, rf, lag=0)
                met = analyzer.analyze(res['equity_curve'])
                
                all_sweep_results.append({
                    'Asset': name, 
                    'TMOM_LB': lb, 
                    'SMA_P': p, 
                    'VIX_T': vt,
                    'Sharpe': met['Sharpe'], 
                    'CAGR': met['CAGR']*100, 
                    'MaxDD': met['MaxDD']*100
                })

        # --- DATA PROCESSING & PRINTING ---
        results_df = pd.DataFrame(all_sweep_results)
        results_df = results_df.rename(columns={
            'TMOM_LB': 'TMOM_LB (m)',
            'SMA_P': 'SMA_P (d)',
            'CAGR': 'CAGR (%)',
            'MaxDD': 'MaxDD (%)'
        })
        results_df = results_df.sort_values(['Asset', 'Sharpe'], ascending=[True, False])
        results_df.to_csv(os.path.join(out_dir, f"sweep_optimized_{algo_mode}.csv"), index=False)

        print("\n" + "="*100)
        print(f"✨ DPM STRATEGY SWEEP RESULTS ({algo_mode} | N-1 Logic) ✨")
        print("="*100)
        
        for asset_name in results_df['Asset'].unique():
            asset_table = results_df[results_df['Asset'] == asset_name]
            best_row = asset_table.iloc[0]
            
            print(f"\n--- Best Parameters for {asset_name} ---")
            print(f"  Sharpe: {best_row['Sharpe']:.2f} | CAGR: {best_row['CAGR (%)']:.2f}% | MaxDD: {best_row['MaxDD (%)']:.2f}%")
            print(f"  Best Combo: TMOM={best_row['TMOM_LB (m)']}m, SMA={best_row['SMA_P (d)']}d, VIX_T={best_row['VIX_T']}")
            print("-" * 50)
            
            cols_to_show = ['TMOM_LB (m)', 'SMA_P (d)', 'VIX_T', 'Sharpe', 'CAGR (%)', 'MaxDD (%)']
            print(asset_table[cols_to_show].head(10).to_string(index=False, float_format=lambda x: f'{x:.2f}'))
        
        print("\n" + "="*100)
        
    else:
        # --- SINGLE REPORT MODE ---
        print(f"\n--- GENERATING PROFESSIONAL REPORT (N-1 US Logic) ---")

        print("\n" + "="*90)
        print(f"RUNNING SINGLE OPTIMIZED REPORT MODE")
        print(f"ALGO MODE: {algo_mode}")
        print(f"PARAMETERS: TMOM={single_tmom_lb} months, SMA={single_sma_p} days, VIX={use_vix}")
        print("="*90)

        for asset_cfg in assets:
            name = asset_cfg['name']
            print(f"\n--- Processing Asset: {name} ({asset_cfg['trade_ticker']}) ---")

            m_prices_raw = fetcher.fetch_data(asset_cfg['macro_ticker'], start, end)['Close']
            macro_prices_n1 = m_prices_raw.shift(1)
            
            s_prices_raw = fetcher.fetch_data(asset_cfg['sma_ticker'], start, end)['Close']
            t_df_raw = fetcher.fetch_data(asset_cfg['trade_ticker'], start, end)
            
            common = macro_prices_n1.index.intersection(s_prices_raw.index).intersection(t_df_raw.index)
            macro_prices = macro_prices_n1.loc[common]
            vix_data_aligned = vix_data_n1.loc[common]
            sma_prices = s_prices_raw.loc[common]
            execution_prices = t_df_raw.loc[common]['Close']
            trade_returns = execution_prices.pct_change().fillna(0)

            allocations, tmom_sig, sma_sig, vix_sig = allocator.allocate(
                macro_prices=macro_prices,
                sma_prices=sma_prices,
                execution_prices=execution_prices,
                rf_rates=rf,
                vix_prices=vix_data_aligned,
                tmom_lb=single_tmom_lb, 
                sma_p=single_sma_p, 
                vix_t=vix_threshold,
                method=algo_mode, 
                use_vix=use_vix
            )

            # Execution logic using lag=0 because US data is pre-shifted
            results = simulator.simulate(allocations, trade_returns, rf, lag=0)
            equity = results['equity_curve']
            
            metrics = analyzer.analyze(equity)
            tracker = TradeTracker(asset_name=name)
            trade_log = tracker.get_trade_log(results['allocations'], execution_prices, equity)
            metrics.update(analyzer.calculate_trade_metrics(trade_log))
            
            logging.info(f"{name} → CAGR {metrics['CAGR']:.2%} | MaxDD {metrics['MaxDD']:.1%} | Sharpe {metrics['Sharpe']:.2f}")

            # --- RESTORED: PERSIST FULL DATA & DASHBOARD EXPORT ---
            # 1. Full data for deep analysis
            full_data_path = os.path.join(out_dir, f'{name}_{algo_mode}_full_data.csv')
            results[['equity_curve', 'position_numeric', 'allocations']].to_csv(full_data_path)

            # 2. Dashboard Export (Used for DB population)
            dashboard_export = pd.DataFrame({
                'close': execution_prices,
                'equity': equity,
                'tmom_sig': tmom_sig.reindex(equity.index).ffill().fillna(0),
                'sma_sig': sma_sig.reindex(equity.index).ffill().fillna(0),
                'vix_sig': vix_sig.reindex(equity.index).ffill().fillna(0),
                'position': results['position_numeric']
            }).ffill().dropna()

            export_name = f"{name}_dashboard_ready.csv"
            dashboard_export.to_csv(os.path.join(out_dir, export_name))
            logging.info(f"✨ Dashboard Export saved: {export_name}")

            # --- CONSOLIDATED SIGNAL & VALUE AUDIT ---
            last_dt = common[-1]
            prev_dt = common[-2]
            
            # 1. VIX Alignment (N-1)
            # vix_data_n1 is pre-shifted globally
            final_vix_val = vix_data_n1.loc[last_dt]
            
            # 2. SMA Alignment (N)
            final_eu_price = sma_prices.loc[last_dt]
            # Use raw data for the rolling mean to ensure a full window is available
            final_sma_val = s_prices_raw.rolling(window=single_sma_p).mean().loc[last_dt]
            
            # 3. Momentum Alignment (N-1)
            tmom_curr_p = macro_prices_n1.loc[last_dt]
            # macro_prices_n1 is already shifted(1). 
            # We shift by the lookback to find the reference price (N - 1 - lookback)
            tmom_lag_p = macro_prices_n1.shift(single_tmom_lb * 21).loc[last_dt]
            tmom_perf = (tmom_curr_p / tmom_lag_p - 1)

            logging.info(f"--- [FINAL BACKTEST AUDIT: TIMING & VALUES] ---")
            logging.info(f"    ● Target Execution Date (N):   {last_dt.date()}")
            logging.info(f"    ------------------------------------------------")
            logging.info(f"    ● US Momentum (N-1) | Ref: {prev_dt.date()} | Perf: {tmom_perf:.2%} (Price: {tmom_curr_p:.2f})")
            logging.info(f"    ● EU SMA Trend (N)  | Ref: {last_dt.date()} | Price: {final_eu_price:.2f} vs SMA: {final_sma_val:.2f}")
            logging.info(f"    ● VIX Risk (N-1)    | Ref: {prev_dt.date()} | Value: {final_vix_val:.2f}")
            logging.info(f"    ------------------------------------------------")
            
            # --- PLOT REPORT ---
            benchmark_ticker = asset_cfg.get('benchmark_ticker', asset_cfg['macro_ticker'])
            b_prices = fetcher.fetch_data(benchmark_ticker, start, end)['Close']
            b_prices = b_prices.loc[b_prices.index.intersection(equity.index)]
            
            analyzer.plot_professional_report(
                equity=equity,
                benchmark=b_prices / b_prices.iloc[0],
                pos_numeric=results['position_numeric'],
                alloc=results['allocations'],
                signal_prices=execution_prices,
                drawdown=equity / equity.cummax() - 1,
                rf=rf, tmom_sig=tmom_sig, sma_sig=sma_sig,
                asset_name=name, output_path=out_dir,
                tmom_lookback=single_tmom_lb, sma_period=single_sma_p,
                algo_mode=algo_mode, metrics=metrics,
                monthly_returns_df=analyzer.calculate_monthly_returns(equity)
            )
            
if __name__ == '__main__':
    main()
