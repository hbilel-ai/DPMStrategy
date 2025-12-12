# live_engine.py - Entry point for ON-LINE EXECUTION (daily scheduler script)

import os
import yaml
import logging
from datetime import datetime, timedelta
from dpm_core.strategy_core import (
    DataFetcher, TMOMSignal, SMASignal, DPMAllocator,
    MockBrokerClient
)
import pandas as pd

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
out_dir = './results'
os.makedirs(out_dir, exist_ok=True)

# ==============================================================================
# --- LIVE TRADING EXECUTION FUNCTION ---\
# ==============================================================================

def run_live_execution(config):
    """
    The main execution function run by a scheduler (e.g., daily cron job).
    It fetches data, calculates signals, determines target allocation, and executes a trade via the broker.
    """

    # 1. Load Parameters and Setup
    algo_mode = config.get('algo_mode', 'Conditional')
    tmom_lb = config.get('lookback', 12)
    sma_p = config.get('sma_period', 100)

    default_assets = [{'signal_ticker': 'QQQ', 'trade_ticker': 'LQQ.PA', 'name': 'Nasdaq-100 2x Leveraged'}]
    assets = config.get('assets', default_assets)

    if not assets:
        logging.error("No assets configured for live execution.")
        return

    asset_config = assets[0]
    signal_ticker = asset_config['signal_ticker']
    trade_ticker = asset_config['trade_ticker']
    asset_name = asset_config['name']

    initial_cash = config.get('initial_cash', 100000.0)

    logging.info(
        f"================================================================================"
        f"\nRUNNING ON-LINE MOCK EXECUTION (LIVE_ENGINE)"
        f"\nASSET: {asset_name} | ALGO MODE: {algo_mode} | TMOM: {tmom_lb}m, SMA: {sma_p}d"
        f"\n================================================================================"
    )

    fetcher = DataFetcher(clear_cache=False)
    # NOTE: In a real system, the broker state would be initialized/loaded here.
    broker = MockBrokerClient(initial_cash=initial_cash)
    allocator = DPMAllocator(signals={'TMOM': TMOMSignal(), 'SMA': SMASignal()})

    # 3. Data Fetching Range
    today = datetime.now()
    signal_end_date = today.strftime('%Y-%m-%d')
    start = config.get('start_date', '2010-07-01')

    logging.info(f"Fetching historical data for signal calculation ({start} to {signal_end_date})...")

    # Fetch signal price data (QQQ)
    signal_prices_df = fetcher.fetch_data(signal_ticker, start, signal_end_date)
    signal_prices = signal_prices_df['Close'] if 'Close' in signal_prices_df.columns else pd.Series(dtype=float)

    # Fetch risk-free rate
    rf_rates = fetcher.fetch_risk_free(start, signal_end_date)

    if signal_prices.empty or rf_rates.empty:
        logging.error("Signal or Risk-Free data is missing/empty. Cannot calculate signal.")
        return

    # 4. Signal Calculation
    logging.info("Calculating target allocation and signals...")
    # UNPACKING THE NEW RETURN TUPLE
    allocations_series, daily_tmom_sig, daily_sma_sig = allocator.allocate(
        signal_prices=signal_prices,
        rf_rates=rf_rates,
        tmom_lb=tmom_lb,
        sma_p=sma_p,
        method=algo_mode
    )

    # Extract final signal and allocation values
    target_allocation = allocations_series.iloc[-1]
    tmom_sig_today = int(daily_tmom_sig.iloc[-1])
    sma_sig_today = int(daily_sma_sig.iloc[-1])

    # 5. Get Current Market Price (Execution Price)
    execution_end_date = (today - timedelta(days=1)).strftime('%Y-%m-%d')
    execution_start_date = (today - timedelta(days=5)).strftime('%Y-%m-%d')

    try:
        logging.info(f"Attempting to fetch execution price for {trade_ticker}. Looking for close on or before {execution_end_date}...")

        trade_data_hist = fetcher.fetch_data(trade_ticker, execution_start_date, execution_end_date)

        if trade_data_hist.empty or 'Close' not in trade_data_hist.columns:
            logging.warning(f"No reliable market price data found for {trade_ticker} on or before {execution_end_date}. Skipping trade.")
            return

        current_price = trade_data_hist['Close'].iloc[-1]

        logging.info(f"Current Execution Price for {trade_ticker} (Close on {trade_data_hist.index[-1].strftime('%Y-%m-%d')}): {current_price:.2f}")

    except Exception as e:
        logging.error(f"Failed to fetch current price for execution: {e}")
        return

    # --- PRE-EXECUTION DECISION LOGGING (UNCHANGED) ---
    current_qty = broker.get_current_position(trade_ticker)
    current_cash = broker.get_current_cash()
    current_value = current_qty * current_price
    total_equity_pre_trade = current_cash + current_value

    previous_allocation = current_value / total_equity_pre_trade if total_equity_pre_trade > 0 else 0.0

    logging.info(
        f"\n--- PRE-TRADE ANALYSIS ({datetime.now().strftime('%Y-%m-%d')}) ---"
        f"\n[ASSET: {trade_ticker} @ {current_price:.2f}]"
        f"\n"
        f"\n  -- Today's Signals --"
        f"\n  TMOM ({tmom_lb}m): {tmom_sig_today} (1=IN, 0=OUT)"
        f"\n  SMA ({sma_p}d): {sma_sig_today} (1=IN, 0=OUT)"
        f"\n  Target Allocation ({algo_mode}): {target_allocation:.1%}"
        f"\n"
        f"\n  -- Current Portfolio State --"
        f"\n  Cash: ${current_cash:,.2f}"
        f"\n  Position ({current_qty:.2f} Qty): ${current_value:,.2f}"
        f"\n  Previous Allocation: {previous_allocation:.1%}"
        f"\n  Total Equity: ${total_equity_pre_trade:,.2f}"
        f"\n"
        f"\n  -- Required Action --"
        f"\n  Goal: Adjust position from {previous_allocation:.1%} to {target_allocation:.1%}"
        f"\n  Result: See Mock Order Log Below"
        f"\n----------------------------------------"
    )

    # 6. Execute Trade (via Abstracted Broker Client)
    order_result = broker.execute_order(
        ticker=trade_ticker,
        target_allocation=target_allocation,
        current_price=current_price
    )

    if order_result and order_result.get('status') == 'MOCKED':

        # --- NEW: POST-EXECUTION LOGGING ---
        new_qty = broker.get_current_position(trade_ticker)
        new_cash = broker.get_current_cash()
        new_value = new_qty * current_price
        new_total_equity = new_cash + new_value

        # New allocation should be the target allocation (100.0% in this case)
        new_allocation = new_value / new_total_equity if new_total_equity > 0 else 0.0

        logging.info(
            f"\n--- POST-TRADE STATE ---"
            f"\n  -- New Portfolio State --"
            f"\n  Cash: ${new_cash:,.2f}"
            f"\n  Position ({new_qty:.2f} Qty): ${new_value:,.2f}"
            f"\n  New Allocation: {new_allocation:.1%}"
            f"\n  Total Equity: ${new_total_equity:,.2f}"
            f"\n------------------------"
        )
        logging.info("Trade execution simulation complete.")
    else:
        logging.info("No order executed (no material change or cash/out decision).")


def main(config_path: str = 'config.yaml'):
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
    except FileNotFoundError:
        logging.error(f"Config file not found at {config_path}. Cannot run live engine.")
        return

    try:
        run_live_execution(config)
    except Exception as e:
        logging.error(f"An unexpected error occurred during live execution: {e}", exc_info=True)


if __name__ == '__main__':
    main()
