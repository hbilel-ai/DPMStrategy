import pandas as pd
import logging
from data_layer.db_manager import DBManager
from data_layer.models import MarketSignal, PortfolioSnapshot, TradeOrder
from config.settings import settings

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(message)s')
db = DBManager(settings.DATABASE_URL)

def import_backtest_to_db():
    csv_path = "./results/Nasdaq-100 2x Leveraged_dashboard_ready.csv"
    ticker_name = "LQQ.PA"

    logging.info(f"🚀 Starting Price-Rebased import from {csv_path}...")

    try:
        df = pd.read_csv(csv_path)
        df['Date'] = pd.to_datetime(df['Date'])

        # --- STEP 1 LOGIC: GET STARTING PRICE ---
        # We use the first price to scale the equity from 1.0 to the actual price
        starting_price = float(df['close'].iloc[0])
        logging.info(f"📈 Re-basing Factor (Initial Price): {starting_price}")

        with db.get_session() as session:
            # 1. Clean existing data
            logging.info(f"🧹 Cleaning existing records...")
            session.query(MarketSignal).filter(MarketSignal.ticker == ticker_name).delete()
            session.query(PortfolioSnapshot).filter(PortfolioSnapshot.mode == 'BACKTEST').delete()
            session.query(TradeOrder).delete()

            logging.info(f"📥 Processing {len(df)} records with re-basing...")

            last_pos_scaled = 0.0
            max_pos_allowed = 2.0

            for _, row in df.iterrows():
                current_date = row['Date']
                current_price = row['close']

                # --- APPLY RE-BASING ---
                # Re-base current_equity: (1.0 index -> ~15.45 dollar value)
                rebased_equity = float(row['equity'] * starting_price)

                # Convert CSV (0, 1, 2) to standard scale (0.0, 0.5, 1.0)
                raw_csv_pos = float(row['position'])
                current_pos_scaled = raw_csv_pos / max_pos_allowed

                # --- TRADE DETECTION LOGIC ---
                if current_pos_scaled != last_pos_scaled:
                    side = "BUY" if current_pos_scaled > last_pos_scaled else "SELL"
                    # Trade Journal stays unit-based (0.0 to 1.0 scale)
                    qty = abs(current_pos_scaled - last_pos_scaled)

                    order = TradeOrder(
                        ticker=ticker_name,
                        order_type=side,
                        quantity=qty,
                        execution_price=current_price,
                        status='FILLED',
                        execution_time=current_date
                    )
                    session.add(order)

                # Update state
                last_pos_scaled = current_pos_scaled

                # --- SIGNALS LOGIC ---
                # Proposed Fix for line 70 in import_backtest.py
                if current_pos_scaled == 1.0:
                    alert = "INVESTED"
                elif current_pos_scaled == 0.5:
                    alert = "PARTIAL"
                else:
                    alert = "CASH"

                signal = MarketSignal(
                    timestamp=current_date,
                    ticker=ticker_name,
                    close_price=current_price,
                    position=current_pos_scaled,
                    signal_A_value=row['tmom_sig'],
                    signal_B_value=row['sma_sig'],
                    action_alert=alert,
                    is_live_signal=False
                )
                session.add(signal)

                # --- PORTFOLIO SNAPSHOT LOGIC (RE-BASED) ---
                # Correct Cash Calculation: Rebased Equity * (1 - % Invested)
                calculated_cash = rebased_equity * (1.0 - current_pos_scaled)

                snapshot = PortfolioSnapshot(
                    snapshot_date=current_date,
                    total_value=rebased_equity, # STORE THE THOUSANDS SCALE
                    cash_balance=calculated_cash,
                    mode='BACKTEST'
                )
                session.add(snapshot)

            session.commit()
            logging.info(f"✅ Import successful!")
            logging.info(f"Final Backtest Equity: {rebased_equity:,.2f}")

    except Exception as e:
        logging.error(f"❌ Failed to import data: {e}")

if __name__ == "__main__":
    import_backtest_to_db()
