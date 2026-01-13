import pandas as pd
import logging
import sqlalchemy as sa 
from datetime import datetime
from data_layer.db_manager import DBManager
from data_layer.models import MarketSignal, PortfolioSnapshot, TradeOrder
from config.settings import settings

logging.basicConfig(level=logging.INFO, format='%(message)s')
db = DBManager(settings.DATABASE_URL)

def import_backtest_to_db():
    csv_path = "./results/Nasdaq-100 2x Leveraged_dashboard_ready.csv"
    ticker_name = "LQQ.PA"

    try:
        df = pd.read_csv(csv_path)
        df['Date'] = pd.to_datetime(df['Date'])
        starting_price = float(df['close'].iloc[0])

        with db.get_session() as session:
            logging.info("🧹 Cleaning Database...")
            session.query(MarketSignal).filter(MarketSignal.ticker == ticker_name).delete()
            session.query(PortfolioSnapshot).delete()
            session.query(TradeOrder).delete()

            last_pos = 0.0
            running_qty = 0.0 
            
            logging.info(f"📥 Importing {len(df)} points...")

            for i, row in df.iterrows():
                current_date = row['Date']
                current_price = float(row['close'])
                current_pos = float(row['position'])
                current_rebased_eq = float(row['equity']) * starting_price
                
                # --- LOGIQUE DE REBALANCEMENT ---
                if current_pos != last_pos:
                    # On calcule la cible basée sur l'équité actuelle
                    target_total_qty = (current_rebased_eq * current_pos) / current_price
                    trade_qty = target_total_qty - running_qty
                    
                    if abs(trade_qty) > 1e-4:
                        session.add(TradeOrder(
                            execution_time=current_date,
                            ticker=ticker_name,
                            order_type="BUY" if trade_qty > 0 else "SELL",
                            quantity=trade_qty,
                            execution_price=current_price,
                            status="FILLED"
                        ))
                        # Mise à jour immédiate après l'ordre
                        running_qty = target_total_qty
                        logging.info(f"TRADING: {current_date.date()} | Signal {last_pos}->{current_pos} | Order: {trade_qty:+.4f} units")

                # --- CRUCIAL : SYNC QUOTIDIENNE ---
                # Même sans ordre, on enregistre les snapshots
                session.add(MarketSignal(timestamp=current_date, ticker=ticker_name, close_price=current_price, position=current_pos, is_live_signal=False, signal_A_value=row['tmom_sig'], signal_B_value=row['sma_sig'], signal_VIX_value=row['vix_sig']))
                session.add(PortfolioSnapshot(snapshot_date=current_date, total_value=current_rebased_eq, cash_balance=current_rebased_eq * (1.0 - current_pos), mode='BACKTEST'))

                last_pos = current_pos

            session.commit()
            
            # --- VERIFICATION FINALE ---
            # On récupère la somme (qui revient souvent sous forme de Decimal)
            actual_db_qty_raw = session.query(sa.func.sum(TradeOrder.quantity)).filter(TradeOrder.ticker == ticker_name).scalar() or 0.0
            
            # CORRECTION : Conversion forcée en float pour les calculs d'affichage
            actual_db_qty = float(actual_db_qty_raw) 
            
            logging.info(f"\n--- DATABASE AUDIT ---")
            logging.info(f"📊 SUM(quantity) in TradeOrder Table : {actual_db_qty:.4f}")
            
            # CORRECTION : Les deux opérandes sont maintenant des floats
            market_value_db = actual_db_qty * current_price
            
            logging.info(f"💰 Market Value in DB: ${market_value_db:.2f}")
            logging.info(f"📈 Expected CSV Equity: ${current_rebased_eq:.2f}")
            
            if abs(market_value_db - current_rebased_eq) < 1.0:
                logging.info("✅ PERFECT ALIGNMENT: Data is clean and ready.")
                
    except Exception as e:
        logging.error(f"❌ Error: {e}")

if __name__ == "__main__":
    import_backtest_to_db()
