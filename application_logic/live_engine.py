# live_engine.py - Entry point for ON-LINE EXECUTION (Hybrid-Time Alignment)

import os
import sys
import yaml
import logging
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
from ib_insync import Contract

# Dynamically add the project root to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dpm_core.strategy_core import (
    DataFetcher, TMOMSignal, SMASignal, DPMAllocator,
    MockBrokerClient, BrokerClient, BoursoramaBrokerClient
)
from dpm_core.clients_notify import NotificationManager
from data_layer.db_manager import DBManager
from data_layer.models import MarketSignal, PortfolioSnapshot, TradeOrder
from config.settings import settings

# Dynamic imports for IBKRClient
try:
    from dpm_core.clients_ibkr import IBKRClient
    logging.info("Custom IBKRClient module loaded.")
except ImportError:
    logging.warning("dpm_core.clients_ibkr not found.")
    IBKRClient = None

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
# --- ADD THESE LINES TO SILENCE IBKR OBJECT DUMPS ---
logging.getLogger('ib_insync').setLevel(logging.WARNING)
logging.getLogger('IBKRClient').setLevel(logging.WARNING)
# ==============================================================================
# --- DATABASE HELPERS ---
# ==============================================================================

def _log_market_and_signal(db_manager, ticker, price, signal_data, position):
    """Logs the final state to the database."""
    try:
        signal = MarketSignal(
            timestamp=datetime.now(),
            ticker=ticker,
            close_price=price,
            signal_A_value=signal_data.get('tmom_signal'),
            signal_B_value=signal_data.get('sma_signal'),
            signal_VIX_value=signal_data.get('vix_signal'),
            action_alert=signal_data.get('action_alert'),
            position=position,
            is_live_signal=True
        )
        with db_manager.get_session() as session:
            session.add(signal)
            session.commit()
    except Exception as e:
        logging.error(f"DB Error (MarketSignal): {e}")

def _log_portfolio_snapshot(db_manager, cash, total_equity):
    """Logs the account value post-execution."""
    try:
        snapshot = PortfolioSnapshot(
            snapshot_date=datetime.now().replace(hour=0, minute=0, second=0, microsecond=0),
            total_value=total_equity,
            cash_balance=cash,
            cash_flow=0.0,
            mode='LIVE'
        )
        with db_manager.get_session() as session:
            session.add(snapshot)
            session.commit()
    except Exception as e:
        logging.error(f"DB Error (PortfolioSnapshot): {e}")
        
# ==============================================================================
# --- REPORTING HELPERS ---
# ==============================================================================

def _generate_strategy_report(data: dict) -> tuple[str, str]:
    today_str = data['date'].strftime('%Y-%m-%d %H:%M')
    vix_status = "ENABLED" if data['use_vix'] else "DISABLED"
    
    # 🟢 Professional Console Output (5 Standardized Sections)
    text_report = (
        f"\n{'='*60}\n"
        f"[1] ENVIRONMENT\n"
        f"{'='*60}\n"
        f"    Broker:       {data['broker_name']} ({data['account_id']})\n"
        f"    Mode:         {data['mode']} | VIX: {vix_status}\n"
        f"    Dry Run:      {data['dry_run']}\n\n"
        f"{'='*60}\n"
        f"[2] PORTFOLIO SNAPSHOT (PRE-ORDER)\n"
        f"{'='*60}\n"
        f"    Data Source:  LIVE API\n"
        f"    Asset:        {data['ticker']} ({data['pre_qty']:.0f} Shares @ {data['current_price']:,.2f} EUR)\n\n"
        f"    ● Value in Shares: {data['pre_val']:,.2f} EUR\n"
        f"    ● Value in Cash:   {data['pre_cash']:,.2f} EUR\n"
        f"    ● TOTAL EQUITY:    {data['pre_total']:,.2f} EUR\n\n"
        f"{'='*60}\n"
        f"[3] SIGNAL SCORECARD\n"
        f"{'='*60}\n"
        f"    ● US Momentum (N-1): {'BULLISH [+]' if data['tmom'] else 'BEARISH [-]'}\n"
        f"    ● EU Trend    (N):   {'BULLISH [+]' if data['sma'] else 'BEARISH [-]'} (Price {data['signal_price']:.2f} {'>' if data['sma'] else '<'} SMA {data['sma_val']:.2f})\n"
        f"    ● Risk Filter (N):   {'SAFE    [+]' if data['vix_sig'] else 'DANGER  [-]'} (Vix {data['vix_val']:.2f})\n\n"
        f"{'='*60}\n"
        f"[4] EXECUTION DETAILS\n"
        f"{'='*60}\n"
        f"    Target Alloc: {data['target_alloc']:.2%}\n"
        f"    Action:       {data['action']}\n"
        f"    Qty Ordered:  {data['qty_ordered']} shares\n"
        f"    Reasoning:    {data['reasoning']}\n\n"
        f"{'='*60}\n"
        f"[5] PORTFOLIO SNAPSHOT (POST-ORDER)\n"
        f"{'='*60}\n"
        f"    ● Final Shares:  {data['post_qty']:.0f}\n"
        f"    ● Final Cash:    {data['final_cash']:,.2f} EUR\n"
        f"    ● Est. Comm.:    {data['commissions']:,.2f} EUR\n"
        f"    ● TOTAL EQUITY:  {data['total_equity']:,.2f} EUR\n"
        f"{'='*60}"
    )

    # 📧 Professional HTML Email
    status_icon = "🟢" if data['action'] != 'SELL' else "🔴"
    if data['action'] == 'HOLD': status_icon = "🛡️"

    html_report = f"""
    <html>
    <body style="font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; color: #333; line-height: 1.6;">
        <div style="max-width: 600px; border: 1px solid #e0e0e0; padding: 20px; border-radius: 10px;">
            <h2 style="color: #2c3e50; border-bottom: 2px solid #f8f9fa; padding-bottom: 10px;">
                {status_icon} DPM Strategy: {data['action']}
            </h2>
            <p style="font-size: 0.9em; color: #666;"><b>Session:</b> {today_str} | <b>Broker:</b> {data['broker_name']}</p>

            <h3 style="color: #2980b9;">[1] ENVIRONMENT</h3>
            <p style="background:#f9f9f9; padding:10px; border-radius:5px;">
                Mode: {data['mode']} | VIX: {vix_status} | Dry Run: {data['dry_run']}
            </p>

            <h3 style="color: #2980b9;">[2] PORTFOLIO SNAPSHOT (Pre-Execution)</h3>
            <div style="background: #fdfefe; border: 1px solid #eee; padding: 15px; border-radius: 5px;">
                <b>Asset:</b> {data['ticker']} ({data['pre_qty']:.0f} Shares @ {data['current_price']:,.2f} EUR)<br>
                <b>Total Equity: {data['pre_total']:,.2f} EUR</b>
            </div>

            <h3 style="color: #2980b9;">[3] SIGNAL SCORECARD</h3>
            <table style="width: 100%; border-collapse: collapse;">
                <tr><td style="padding:8px; border:1px solid #ddd;">US Momentum</td><td>{'🟢 BULLISH' if data['tmom'] else '🔴 BEARISH'}</td></tr>
                <tr><td style="padding:8px; border:1px solid #ddd;">EU Trend</td><td>{'🟢 BULLISH' if data['sma'] else '🔴 BEARISH'}</td></tr>
                <tr><td style="padding:8px; border:1px solid #ddd;">Risk (VIX)</td><td>{'🛡️ SAFE' if data['vix_sig'] else '⚠️ DANGER'}</td></tr>
            </table>

            <h3 style="color: #2980b9;">[4] EXECUTION DETAILS</h3>
            <div style="background: #e8f4fd; border-left: 5px solid #3498db; padding: 15px;">
                <b>Action:</b> {data['action']} | <b>Target:</b> {data['target_alloc']:.2%}<br>
                <i>{data['reasoning']}</i>
            </div>

            <h3 style="color: #2980b9;">[5] PORTFOLIO SNAPSHOT (Post-Execution)</h3>
            <div style="background: #f4fdf4; border: 1px solid #dff0d8; padding: 15px; border-radius: 5px;">
                <b>Final Cash: {data['final_cash']:,.2f} EUR</b><br>
                <b>Total Equity: {data['total_equity']:,.2f} EUR</b>
            </div>
        </div>
    </body>
    </html>
    """
    return text_report, html_report

# ==============================================================================
# --- MAIN EXECUTION LOGIC ---
# ==============================================================================
def validate_and_align_data(m_df, s_df, t_df, v_df, config):
    today = pd.Timestamp(datetime.now().date())
    prev_bus_day = today - pd.tseries.offsets.BusinessDay(1)

    # --- TRACE 2: CALENDAR ALIGNMENT ---
    logging.info(f"DEBUG ALIGNMENT: Wall Clock: {today.date()} | Expected N-1: {prev_bus_day.date()}")
    
    # Get tickers from config
    asset_cfg = config['assets'][0]
    trade_ticker = asset_cfg.get('trade_ticker')
    sma_ticker = asset_cfg.get('sma_ticker')
    macro_ticker = asset_cfg.get('macro_ticker')

    # 1. US Momentum (Lagged N-1)
    # Ensure it ends on the previous business day
    if m_df.index[-1] < prev_bus_day:
        logging.warning(f"DATA GAP: {macro_ticker} is behind. Last: {m_df.index[-1].date()}")
    macro_out = m_df[m_df.index <= prev_bus_day]['Close']

    # 2. European / Trade Assets (Real-time N)
    # We expect these to have been patched with TODAY'S date
    for name, df in [(sma_ticker, s_df), (trade_ticker, t_df)]:
        if df.index[-1] < today:
            logging.warning(f"MISSING LIVE PATCH: {name} ends at {df.index[-1].date()}, expected {today.date()}")

    # --- TRACE 3: SLICING RESULTS ---
    logging.info(f"DEBUG SLICE: Macro input ends on {macro_out.index[-1].date() if not macro_out.empty else 'N/A'}")
    
    # If the patch is missing, we use what we have, but the warning alerts the user
    return macro_out, s_df['Close'], t_df['Close'], v_df['Close']

def run_live_execution(config, notify_manager: NotificationManager, broker: BrokerClient, db_manager: DBManager):
    
    # 0. CHECK FOR DRY RUN MODE
    dry_run = config.get('dry_run', False)
    if dry_run:
        logging.warning("!!! DRY RUN MODE ENABLED - NO REAL ORDERS WILL BE SENT !!!")
        
    # 1. LOAD TICKERS DYNAMICALLY (Matches Backtest config.yaml)
    asset_cfg = config['assets'][0]
    macro_t = asset_cfg.get('macro_ticker', 'QQQ')
    sma_t = asset_cfg.get('sma_ticker', 'PUST.PA')
    trade_t = asset_cfg.get('trade_ticker', 'LQQ.PA')
 
    # 1. PRE-FLIGHT CHECK: Is market open?
    # We ask the broker directly instead of checking local clock
    if hasattr(broker, 'is_market_open') and not broker.is_market_open(trade_t):
        logging.warning(f"MARKET CLOSED: Skipping execution for {trade_t} to prevent rejected orders.")
        return
    
    # --- TRACE DEBUG INITIALE : ÉTAT DE LA DB AVANT CALCULS ---
    #logging.info(f"🔍 DEBUG ENGINE: Vérification de l'état DB pour {trade_t}")
    #try:
    #    with db_manager.get_session() as session:
    #        all_trades = session.query(TradeOrder).filter(TradeOrder.ticker == trade_t).all()
    #        sum_qty = sum(t.quantity for t in all_trades)
    #        logging.info(f"🔍 DEBUG ENGINE: Nombre total de lignes TradeOrder en DB : {len(all_trades)}")
    #        logging.info(f"🔍 DEBUG ENGINE: Somme brute calculée ici : {sum_qty}")
     #       if len(all_trades) > 0:
     #           logging.info(f"🔍 DEBUG ENGINE: Dernier trade ID: {all_trades[-1].id} | Date: {all_trades[-1].execution_time}")
    #except Exception as e:
    #    logging.error(f"❌ DEBUG ENGINE: Erreur lors du check DB : {e}")
    # ---------------------------------------------------------
    
    algo_mode = config.get('algo_mode', 'Conditional')
    tmom_lb = config.get('lookback', 12)
    sma_p = config.get('sma_period', 200)
    vix_t = config.get('vix_threshold', 22)
    use_vix = config.get('use_vix', True)

    logging.info(f"STARTING LIVE EXECUTION | Mode: {algo_mode} | Hybrid: {macro_t}(N-1) + {sma_t}(N)")

    fetcher = DataFetcher(clear_cache=False)
    allocator = DPMAllocator(signals={'TMOM': TMOMSignal(), 'SMA': SMASignal()})

    # 2. DATA FETCHING (Period Padding for 200d SMA)
    today = datetime.now()
    today_str = today.strftime('%Y-%m-%d')
    signal_end_date = today.strftime('%Y-%m-%d')
    start = config.get('start_date', '2015-01-01')

    # Fetch Dataframes
    m_df = fetcher.fetch_data(macro_t, start, signal_end_date)
    s_df = fetcher.fetch_data(sma_t, start, signal_end_date)
    t_df = fetcher.fetch_data(trade_t, start, today_str)
    v_df = fetcher.fetch_data('^VIX', start, signal_end_date)
    rf_rates = fetcher.fetch_risk_free(start, signal_end_date)

    if m_df.empty or s_df.empty or v_df.empty:
        logging.error("Incomplete data from YFinance. Aborting execution.")
        return

    # --- TRACE 1: RAW FETCH VERIFICATION ---
    #for name, df in [(macro_t, m_df), (sma_t, s_df), (trade_t, t_df), ("VIX", v_df)]:
    #    if not df.empty:
    #        logging.info(f"DEBUG RAW FETCH: {name} | Start: {df.index[0].date()} | End: {df.index[-1].date()} | Count: {len(df)}")
    #    else:
    #        logging.error(f"DEBUG RAW FETCH: {name} is EMPTY")

    # --- IMPROVED LIVE PRICE INJECTION (Independent Tickers) ---
    for ticker, df_ref in [(sma_t, s_df), (trade_t, t_df)]:
        try:
            #logging.info(f"DEBUG PATCH: Fetching live price for {ticker}...")
            live_data = yf.download(ticker, period='1d', interval='1m', progress=False)
        
            if not live_data.empty:
                # FIX: Ensure we get a single float value, handling MultiIndex if necessary
                if isinstance(live_data['Close'], pd.DataFrame):
                    live_p = float(live_data['Close'].iloc[-1, 0]) # Get first column of last row
                else:
                    live_p = float(live_data['Close'].iloc[-1])
            
                # Create a clean single-value DataFrame for the patch
                new_row = pd.DataFrame({'Close': [live_p]}, index=[pd.Timestamp(today.date())])
                new_row.index.name = 'Date'

                if ticker == sma_t:
                    s_df = pd.concat([s_df, new_row])
                    # Clean up: ensure the index is unique and sorted
                    s_df = s_df[~s_df.index.duplicated(keep='last')].sort_index()
                    #logging.info(f"DEBUG PATCH: Injected {live_p:.2f} into SMA series ({ticker})")
                else:
                    t_df = pd.concat([t_df, new_row])
                    t_df = t_df[~t_df.index.duplicated(keep='last')].sort_index()
                    #logging.info(f"DEBUG PATCH: Injected {live_p:.2f} into TRADE series ({ticker})")
            
            else:
                logging.warning(f"DEBUG PATCH: No live data for {ticker}. Siganl/Trade will be STALE.")
            
        except Exception as e:
            logging.error(f"DEBUG PATCH ERROR for {ticker}: {str(e)}")
        
    macro_input, sma_input, trade_input, vix_input = validate_and_align_data(m_df, s_df, t_df, v_df, config)
    
    # --- HYBRID ALIGNMENT DEBUG TRACES ---
    #logging.info("====================================================")
    #logging.info("DEBUG: INPUT DATA ALIGNMENT VERIFICATION")

    # N-1 US Momentum Data
    #logging.info(f"SIGNAL (N-1) QQQ   | End Date: {macro_input.index[-1].date()} | Price: {macro_input.iloc[-1]:.2f}")

    # N European Trend Data
    #logging.info(f"SIGNAL (N) PUST.PA | End Date: {sma_input.index[-1].date()} | Price: {sma_input.iloc[-1]:.2f}")

    # N European Execution Data
    #logging.info(f"TRADE  (N) LQQ.PA  | End Date: {trade_input.index[-1].date()} | Price: {trade_input.iloc[-1]:.2f}")

    # VIX Data
    #logging.info(f"FILTER (N) VIX     | End Date: {vix_input.index[-1].date()} | Price: {vix_input.iloc[-1]:.2f}")
    #logging.info("====================================================")
    
    # We pass sma_series_n (EU Today) for both SMA and Execution Price logic
    allocations_series, daily_tmom_sig, daily_sma_sig, daily_vix_sig = allocator.allocate(
        macro_prices=macro_input,      # N-1 US Data
        sma_prices=sma_input,          # N European Data
        execution_prices=trade_input,    # N European Data (matches backtest execution)
        rf_rates=rf_rates,
        vix_prices=vix_input,         # Pass the full VIX series
        tmom_lb=tmom_lb,
        sma_p=sma_p,
        vix_t=vix_t,
        method=algo_mode,
        use_vix=use_vix
    )
 
    sma_obj = allocator.signals.get('SMA')
    if sma_obj:
        logging.info(f"DEBUG ENGINE: Last value in sma_series: {sma_obj.sma_series.iloc[-1] if sma_obj.sma_series is not None else 'None'}")
    
    # Extract final signal and allocation values
    target_allocation = float(allocations_series.iloc[-1])
    #target_allocation = 0 # Gardez votre override si nécessaire pour test
    
    current_trade_price = float(trade_input.iloc[-1]) # Force float Python
    tmom_sig_today = int(daily_tmom_sig.iloc[-1])
    sma_sig_today = int(daily_sma_sig.iloc[-1])
    vix_sig_today = int(daily_vix_sig.iloc[-1])
    vix_value_now = float(vix_input.iloc[-1]) # Force float Python
  
    # 6. EXECUTION FLOW
    current_qty = float(broker.get_current_position(trade_t))
    current_cash = float(broker.get_current_cash())
    current_val = current_qty * current_trade_price
    total_eq = current_cash + current_val
    
    prev_alloc = current_val / total_eq if total_eq > 0 else 0.0
    alloc_delta = target_allocation - prev_alloc

    # --- SOURCE DE VÉRITÉ UNIQUE POUR L'ORDRE ---
    # On définit une variable unique 'order_type_final' pour tout le reste du script
    if abs(alloc_delta) < 0.01:
        order_type_final = 'HOLD'
        value_delta = 0.0
    else:
        # On utilise des labels stricts 'BUY' / 'SELL' pour la DB et le journal FIFO
        order_type_final = 'BUY' if alloc_delta > 0 else 'SELL'
        value_delta = float(alloc_delta * total_eq)
    
    # Log Signal (Sécurisé avec order_type_final)
    signal_data = {
        'tmom_signal': tmom_sig_today,
        'sma_signal': sma_sig_today,
        'vix_signal': vix_sig_today,
        'vix_actual': vix_value_now,
        'action_alert': order_type_final
    }    
    sig_id = _log_market_and_signal(db_manager, trade_t, current_trade_price, signal_data, target_allocation)

    # 3. EXECUTION
 # ... (Calculations of allocations remain the same) ...

    # 3. PRE-CALCULATE REASONING
    qty_to_trade = 0
    reasoning = "Portfolio already aligned with strategy."
    
    if order_type_final != 'HOLD':
        # Potential quantity based on money
        potential_qty = int(abs(value_delta) / current_trade_price)
        
        if potential_qty < 1:
            reasoning = f"Insufficient cash ({current_cash:.2f}€) to buy 1 share ({current_trade_price:.2f}€)."
            qty_to_trade = 0
        else:
            reasoning = f"Adjusting allocation by {alloc_delta:+.2%}."
            qty_to_trade = potential_qty

    # 4. EXECUTION
    trade_report = None
    if qty_to_trade > 0:
        if not dry_run:
            # The broker now RETURNS the data instead of saving it
            trade_report = broker.execute_order(trade_t, target_allocation, current_trade_price)
            if trade_report:
                db_manager.save_trade_order(trade_report)
        else:
            logging.warning(f"[DRY RUN] Intercepted {order_type_final} for {qty_to_trade} shares.")
            # Record the DRY RUN in the database for history
            trade_report = {
                'ticker': trade_t,
                'order_type': order_type_final,
                'quantity': float(qty_to_trade),
                'execution_price': current_trade_price,
                'execution_time': datetime.now(),
                'status': 'DRY_RUN'
            }
            db_manager.save_trade_order(trade_report)
    
    # 5. FINAL RECONCILIATION
    final_qty = broker.get_current_position(trade_t)
    final_cash = broker.get_current_cash()
    final_eq = final_cash + (final_qty * current_trade_price)
    commissions = abs(total_eq - final_eq) if not dry_run else 0.0

    # --- CRITICAL: PERSIST SNAPSHOT TO DB ---
    # This records your equity curve every time the script runs
    _log_portfolio_snapshot(db_manager, final_cash, final_eq)
    logging.info(f"DB Update: Portfolio Snapshot saved. Total Equity: {final_eq:.2f}")
    
    # 6. ASSEMBLE ENRICHED REPORT DATA
    report_data = {         
        'date': today,
        'ticker': trade_t,
        'broker_name': type(broker).__name__,
        'account_id': getattr(broker, 'account_id', 'N/A'),
        'dry_run': "YES" if dry_run else "NO",
        'mode': algo_mode,
        'use_vix': use_vix,
        'tmom': tmom_sig_today,
        'sma': sma_sig_today,
        'vix_sig': vix_sig_today,
        'vix_val': vix_value_now,
        'current_price': current_trade_price,
        'signal_price': float(sma_input.iloc[-1]),
        'sma_val': float(sma_obj.sma_series.iloc[-1]),
        'target_alloc': target_allocation,
        'action': 'HOLD' if qty_to_trade == 0 else order_type_final,
        'qty_ordered': qty_to_trade,
        'reasoning': reasoning,
        'pre_qty': current_qty,
        'pre_val': current_val,
        'pre_cash': current_cash,
        'pre_total': total_eq,
        'post_qty': final_qty,
        'final_cash': final_cash,
        'commissions': commissions,
        'total_equity': final_eq
    } 
    
    # 7. NOTIFY & LOG (This now handles the full 5-section layout)
    text_rep, html_rep = _generate_strategy_report(report_data)
    logging.info(text_rep)    
    
    subject = f"{'🛡️' if qty_to_trade==0 else '🚀'} DPM {report_data['action']}: {trade_t}"
    notify_manager.notify(subject, html_rep)
    
# ==============================================================================
# --- MAIN ENTRY ---
# ==============================================================================

def main(config_path: str = 'config.yaml'):
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
    except FileNotFoundError:
        logging.error("Config not found.")
        return

    # --- TRACE DEBUG CONNEXION ---
    #logging.info(f"🔍 DEBUG MAIN: URL Base de données utilisée : {settings.DATABASE_URL}")
    db_manager = DBManager(settings.DATABASE_URL)
    # -----------------------------
    
    notify_manager = NotificationManager(config.get('notification', {}))
    
    # Broker setup (Reduced for brevity, identical to your original main logic)
    broker_cfg = config.get('live_broker', {})
    b_type = broker_cfg.get('type', 'mock').lower()
    init_cash = broker_cfg.get('initial_cash', 100000.0)
    live_ticker = config['assets'][0].get('trade_ticker', 'LQQ.PA')

    if b_type == 'ibkr' and IBKRClient:
       # Extract settings from config
        host = broker_cfg.get('host', '127.0.0.1')
        port = broker_cfg.get('port', 7497)
        c_id = broker_cfg.get('client_id', 1)
        
        # Instantiate the new wrapper class
        try:
            broker = IBKRClient(
                host=host, 
                port=port, 
                client_id=c_id, 
                live_ticker=live_ticker
            )
        except Exception as e:
            logging.error(f"Critical: Could not initialize IBKR Client: {e}")
            return
        
    elif b_type == 'boursorama':
        broker = BoursoramaBrokerClient(initial_cash=init_cash, notify_manager=notify_manager, db_manager=db_manager, live_ticker=live_ticker)
    else:
        #logging.info(f"🔍 DEBUG MAIN: Initialisation MockBrokerClient avec cash={init_cash}")
        broker = MockBrokerClient(initial_cash=init_cash, db_manager=db_manager, live_ticker=live_ticker)

    run_live_execution(config, notify_manager, broker, db_manager)

if __name__ == '__main__':
    main()
