# live_engine.py - Entry point for ON-LINE EXECUTION (daily scheduler script)

import os
import yaml
import logging
from datetime import datetime, timedelta
from dpm_core.strategy_core import (
    DataFetcher, TMOMSignal, SMASignal, DPMAllocator,
    MockBrokerClient, BrokerClient, BoursoramaBrokerClient
)
from dpm_core.clients_notify import NotificationManager
import pandas as pd
from dpm_core.db_manager import DBManager

# Dynamic imports for IBKRClient
try:
    from dpm_core.clients_ibkr import IBKRClient # <-- CORRECT IMPORT LOCATION
except ImportError:
    logging.warning("IBKRClient is not available (ibapi not installed or clients_ibkr.py missing).")
    IBKRClient = None

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
out_dir = './results'
os.makedirs(out_dir, exist_ok=True)

# ==============================================================================
# --- LIVE TRADING EXECUTION FUNCTION ---\
# ==============================================================================
def run_live_execution(config, notify_manager: NotificationManager, broker: BrokerClient):
    """
    The main execution function run by a scheduler (e.g., daily cron job).
    It fetches data, calculates signals, determines target allocation, and executes a trade via the broker.

    The broker argument is the dynamically loaded client (Mock, Boursorama, or IBKR).
    """
    # 0. Initialize Database Manager
    db = DBManager(db_path='dpm_live.db')

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

    # REMOVED: initial_cash = config.get('initial_cash', 100000.0)
    # REMOVED: broker is now passed in, so we don't need to initialize it here.

    # ---------------------------------------------------------------------------------
    # CHANGE 1: Update logging to reflect dynamic broker type
    logging.info(
        f"================================================================================"
        f"\nRUNNING ON-LINE EXECUTION"
        f"\nCLIENT: {broker.__class__.__name__} | ASSET: {asset_name} | ALGO MODE: {algo_mode}"
        f"\n================================================================================"
    )
    # ---------------------------------------------------------------------------------

    fetcher = DataFetcher(clear_cache=False)
    # REMOVED: broker = MockBrokerClient(initial_cash=initial_cash)
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

    # Determine string representation of state (for DB readability)
    alloc_state_str = "Cash"
    if target_allocation == 1.0: alloc_state_str = "Invested"
    elif target_allocation == 0.5: alloc_state_str = "Partial"

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
        f"\n  Result: See Broker Execution Log Below"
        f"\n----------------------------------------"
    )

    # 6. Execute Trade (via Abstracted Broker Client)
    order_result = broker.execute_order(
        ticker=trade_ticker,
        target_allocation=target_allocation,
        current_price=current_price
    )

    # --- POST-EXECUTION DATA GATHERING ---

    # Get final numbers from broker (whether trade happened or not)
    final_qty = broker.get_current_position(trade_ticker)
    final_cash = broker.get_current_cash()
    final_equity = final_cash + (final_qty * current_price)

    # ---------------------------------------------------------------------------------
    # CHANGE 2: Simplified POST-EXECUTION LOGGING
    # The clients now handle their own specific post-trade actions (like sending notifications).
    # We only check if *any* action was taken (status is NOT 'NO_TRADE') and log the new state.
    # ---------------------------------------------------------------------------------

    status = order_result.get('status', 'NO_RESULT') if order_result else 'NO_RESULT'

    if status != 'NO_TRADE':

        # --- POST-EXECUTION LOGGING ---
        # Query the broker for the new state, which is updated regardless of
        # whether the execution was mock, notification, or live trade.
        new_qty = broker.get_current_position(trade_ticker)
        new_cash = broker.get_current_cash()
        new_value = new_qty * current_price
        new_total_equity = new_cash + new_value

        # New allocation calculation
        new_allocation = new_value / new_total_equity if new_total_equity > 0 else 0.0

        logging.info(
            f"\n--- POST-TRADE STATE ({status}) ---"
            f"\n  -- New Portfolio State --"
            f"\n  Cash: ${new_cash:,.2f}"
            f"\n  Position ({new_qty:.2f} Qty): ${new_value:,.2f}"
            f"\n  New Allocation: {new_allocation:.1%}"
            f"\n  Total Equity: ${new_total_equity:,.2f}"
            f"\n----------------------------------"
        )

        # REMOVED: Explicit SEND NOTIFICATION block. This logic is now inside
        # BoursoramaBrokerClient.execute_order().

        logging.info(f"Trade execution complete via {broker.__class__.__name__}.")
    else:
        logging.info(f"Execution Status: {status}. No order executed (no material change or cash/out decision).")

    # ==========================================================================
    # --- NEW: SAVE STATE TO DATABASE ---
    # ==========================================================================

    db_state = {
        'date': today.strftime('%Y-%m-%d'),
        'ticker': trade_ticker,
        'tmom_signal': tmom_sig_today,
        'sma_signal': sma_sig_today,
        'target_allocation': float(target_allocation),
        'allocation_state': alloc_state_str,
        'executed_qty': float(final_qty),
        'share_price': float(current_price),
        'total_equity': float(final_equity),
        'cash_balance': float(final_cash)
    }

    db.save_daily_state(db_state)
    logging.info("Daily state successfully persisted to dpm_live.db")

# =============================================================================
# --- MAIN EXECUTION ENTRY POINT ---
# ==============================================================================

def main(config_path: str = 'config.yaml'):
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
    except FileNotFoundError:
        logging.error(f"Config file not found at {config_path}. Cannot run live engine.")
        return

    # 1. Initialize Notification Manager
    notify_manager = NotificationManager(config.get('notification', {}))

    # 2. Broker Initialization is driven ENTIRELY by config.yaml
    broker = None
    # Assuming config.yaml has a section like: live_broker: {type: 'ibkr', ...}
    broker_config = config.get('live_broker', {})
    broker_type = broker_config.get('type', 'mock').lower()
    initial_cash = broker_config.get('initial_cash', 100000.0)

    if config.get('assets'):
        primary_asset = config['assets'][0]
        # We prioritize the trade_ticker from the config over the global setting
        live_ticker = primary_asset.get('trade_ticker', settings.LIVE_TICKER)
    else:
        live_ticker = settings.LIVE_TICKER # Fallback

    # Instantiate the correct client based on config.yaml
    if broker_type == 'ibkr' and IBKRClient is not None:
        try:
            # IBKR requires host, port, client_id from config
            broker = IBKRClient(
                host=broker_config.get('host', '127.0.0.1'),
                port=broker_config.get('port', 7497),
                client_id=broker_config.get('client_id', 1)
            )
            logging.info(f"Using Broker Client: {broker.__class__.__name__}")
        except Exception as e:
            # GRACEFUL FALLBACK if IBKR connection fails
            logging.critical(f"IBKRClient connection failed. Falling back to Mock. Error: {e}")
            broker = MockBrokerClient(initial_cash=initial_cash)
            logging.info(f"Using Broker Client: {broker.__class__.__name__}")

    elif broker_type == 'boursorama' and BoursoramaBrokerClient is not None:
        # Boursorama is a mock/notification-only client in this architecture
        broker = BoursoramaBrokerClient(
            initial_cash=initial_cash,
            notify_manager=notify_manager, # Pass the manager to Boursorama for internal notification
            db_manager=db_manager,
            live_ticker=live_ticker # <-- PASS THE CORRECT TICKER (LQQ.PA)
        )
        logging.info(f"Using Broker Client: {broker.__class__.__name__}")

    else:
        # Default or if type is explicitly 'mock' or client is unavailable
        broker = MockBrokerClient(
            initial_cash=initial_cash,
            db_manager=db_manager,
            live_ticker=live_ticker # <-- PASS THE CORRECT TICKER (LQQ.PA)
        )
        logging.info(f"Using Broker Client: {broker.__class__.__name__}")

    # 3. Run Execution
    if broker:
        try:
            # Pass the initialized broker client to the execution function
            run_live_execution(config, notify_manager, broker)
        except Exception as e:
            logging.error(f"Critical error during live execution run: {e}", exc_info=True)

if __name__ == '__main__':
    main()
