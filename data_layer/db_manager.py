# data_layer/db_manager.py

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.sql import func
from typing import List, Dict, Any, Tuple
from datetime import datetime
import logging

# Import all models from the same directory
from .models import Base, PortfolioSnapshot, TradeOrder, MarketSignal

class DBManager:
    """
    Handles all interactions with the database using SQLAlchemy ORM.
    Responsible for session management and translating model data for the
    Application Layer.
    """
    def __init__(self, db_url: str):
        # 1. Initialize DB Engine
        self.engine = create_engine(db_url)

        # 2. Create tables if they don't exist
        Base.metadata.create_all(self.engine)

        # 3. Create a configured "Session" class
        self.SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=self.engine)

        print(f"DBManager initialized and connected to: {db_url}")

    def get_session(self) -> Session:
        """Utility method to get a new session for a transaction."""
        return self.SessionLocal()

    # --- Core CRUD Method for Live Engine ---
    def save_trade_order(self, order_data: Dict[str, Any]):
        """
        Stores a new executed order in the database, handling field name mapping.
        """
        from .models import TradeOrder # Ensure this import is present

        # --- DEBUG TRACE 1: Incoming Data (Already confirmed) ---
        logging.info(f"TRADE_SAVE_DEBUG: Incoming order data: {order_data}")
        # --- END DEBUG TRACE 1 ---

        try:
            # CRITICAL FIX: Use explicit bracket notation for mandatory fields
            # and robust retrieval for execution_price

            # Use bracket notation for mandatory fields confirmed to be in order_data
            execution_time = order_data['execution_time']
            ticker = order_data['ticker']
            order_type = order_data['order_type']
            quantity = order_data['quantity']

            # Robustly retrieve price, mapping 'price' key to 'execution_price' ORM field
            price_value = order_data.get('execution_price') or order_data.get('price') or order_data.get('at_price')

            if price_value is None:
                raise ValueError("Missing price field in trade order data.")

            order = TradeOrder(
                execution_time=execution_time, # Now guaranteed to be the datetime object
                ticker=ticker,                 # Now guaranteed to be 'LQQ.PA'
                order_type=order_type,         # Now guaranteed to be the action string
                quantity=quantity,
                execution_price=price_value,
                # engine_signal_id is likely NULLABLE or should be supplied if required
                engine_signal_id=order_data.get('engine_signal_id'),
                status=order_data.get('status', 'FILLED')
            )

            logging.info(f"TRADE_SAVE_DEBUG: TradeOrder object created: Type={order.order_type}, Price={order.execution_price}")

            with self.get_session() as session:
                session.add(order)
                session.commit()

            logging.info("TRADE_SAVE_DEBUG: Trade order successfully committed.")
            return True
        except Exception as e:
            # Re-running the dump after the final save attempt to show the failure
            logging.error(f"CRITICAL ERROR saving trade order for {order_data.get('ticker')}: {e}", exc_info=True)
            return False

    def get_daily_nav_snapshots(self, start_date: str) -> List[Dict[str, Any]]:
        """
        Retrieves the required time-series data for return calculations.
        Returns a list of dicts for easy DataFrame conversion in AnalyticsService.
        """
        with self.get_session() as session:
            snapshots = session.query(
                PortfolioSnapshot.snapshot_date,
                PortfolioSnapshot.total_value,
                PortfolioSnapshot.cash_flow
            ).filter(
                PortfolioSnapshot.snapshot_date >= datetime.strptime(start_date, '%Y-%m-%d')
            ).order_by(
                PortfolioSnapshot.snapshot_date.asc()
            ).all()

        # Convert tuples/ORM objects to list of dictionaries
        return [
            {
                "snapshot_date": row[0],
                "total_value": float(row[1]),
                "cash_flow": float(row[2])
            }
            for row in snapshots
        ]

    def get_executed_trades(self, start_date=None, end_date=None) -> List[Dict[str, Any]]:
        """
        Retrieves a list of executed trades/orders for the Audit Log API.
        """
        with self.get_session() as session:
            query = session.query(TradeOrder).order_by(TradeOrder.execution_time.desc())

            if start_date:
                query = query.filter(TradeOrder.execution_time >= datetime.strptime(start_date, '%Y-%m-%d'))
            if end_date:
                query = query.filter(TradeOrder.execution_time <= datetime.strptime(end_date, '%Y-%m-%d'))

            #trades = query.limit(100).all() # Limit for pagination placeholder
            trades = query.all()
            # --- DEBUG TRACE ---
            print(f"DEBUG_DB: Raw orders fetched from DB: {len(trades)}")
            if trades:
                for t in trades[:3]: # Log first 3 to verify structure
                    print(f"DEBUG_DB: Order ID: {t.id} | {t.ticker} | {t.order_type} | Qty: {t.quantity}")
            # -------------------

        # Convert ORM objects to dicts for API response
        return [
            {
                "id": t.id,
                "time": t.execution_time.isoformat(),
                "ticker": t.ticker,
                "type": t.order_type,
                "qty": t.quantity,
                "price": float(t.execution_price) if t.execution_price else 0.0,
                "status": t.status
            }
            for t in trades
        ]

    def get_price_and_signal_data(self, ticker: str, start_date: str) -> Tuple[Dict[str, List[Any]], str]:
        """
        Retrieves historical price and signal data for charting,
        now including the final position/allocation.
        """
        with self.get_session() as session:
            # 1. Fetch time series data - Added MarketSignal.position here
            signals = session.query(
                MarketSignal.timestamp,
                MarketSignal.close_price,
                MarketSignal.signal_A_value,
                MarketSignal.signal_B_value,
                MarketSignal.position  # <--- CRITICAL: ADD THIS
            ).filter(
                MarketSignal.ticker == ticker,
                MarketSignal.timestamp >= datetime.strptime(start_date, '%Y-%m-%d')
            ).order_by(MarketSignal.timestamp.asc()).all()

            # 2. Find the earliest 'live' signal date (unchanged)
            first_live_signal = session.query(
                func.min(MarketSignal.timestamp)
            ).filter(
                MarketSignal.ticker == ticker,
                MarketSignal.is_live_signal == True
            ).scalar()

        # Update the data dictionary to include the new position list
        data = {
            'date': [row[0].isoformat() for row in signals],
            'price': [float(row[1]) for row in signals],
            'signal_A': [float(row[2]) if row[2] is not None else None for row in signals],
            'signal_B': [float(row[3]) if row[3] is not None else None for row in signals],
            'position': [float(row[4]) if row[4] is not None else 0.0 for row in signals], # <--- AND THIS
        }
        live_start_date = first_live_signal.isoformat() if first_live_signal else None

        return data, live_start_date

    def get_latest_portfolio_snapshot(self) -> Dict[str, Any] | None:
        """Retrieves the latest portfolio snapshot state for broker initialization."""
        from .models import PortfolioSnapshot # Ensure PortfolioSnapshot is imported locally or globally

        with self.get_session() as session:
            # Get the single most recent snapshot by ID (assuming auto-incrementing ID is reliable)
            # OR by snapshot_date (desc) as demonstrated below:
            latest = session.query(PortfolioSnapshot).order_by(
                PortfolioSnapshot.snapshot_date.desc()
            ).first()

            if latest:
                return {
                    'snapshot_date': latest.snapshot_date.isoformat(),
                    'total_value': float(latest.total_value),
                    'cash_balance': float(latest.cash_balance),
                    'cash_flow': float(latest.cash_flow)
                }
            return None

    def get_net_position_quantity(self, ticker: str) -> float:
        from .models import TradeOrder
        from sqlalchemy import case, func

        with self.get_session() as session:
            # --- CRITICAL FIX: Use LIKE for robust string matching ---
            net_quantity_case = case(
                (TradeOrder.order_type.like('BUY%'), TradeOrder.quantity),
                (TradeOrder.order_type.like('SELL%'), -TradeOrder.quantity),
                (TradeOrder.order_type.like('EXIT%'), -TradeOrder.quantity),
                else_=0
            )
            # --- END CRITICAL FIX ---

            net_position = session.query(
                func.sum(net_quantity_case)
            ).filter(
                TradeOrder.ticker == ticker
            ).scalar()

            logging.info(f"DB_DEBUG: Aggregating net position for ticker '{ticker}'. Result (scalar): {net_position}")

            return float(net_position) if net_position is not None else 0.0

    def debug_dump_all_state(self, ticker: str):
        """DEBUG: Dumps critical broker state tables."""
        from .models import TradeOrder, PortfolioSnapshot
        import logging

        logging.info("--- START DB STATE DUMP ---")

        # 1. Dump Portfolio Snapshots (Should show latest cash balance)
        with self.get_session() as session:
            snapshots = session.query(PortfolioSnapshot).order_by(PortfolioSnapshot.snapshot_date.desc()).all()
            logging.info(f"DB_DUMP: Found {len(snapshots)} Portfolio Snapshots.")
            for s in snapshots:
                logging.info(f"DB_DUMP: SNAPSHOT | Date: {s.snapshot_date.isoformat()}, Cash: {s.cash_balance:.2f}, Total: {s.total_value:.2f}")

        # 2. Dump Trade Orders (Should show the BUY trade)
        with self.get_session() as session:
            trades = session.query(TradeOrder).filter(
                TradeOrder.ticker == ticker
            ).order_by(
                TradeOrder.execution_time.desc()
            ).all()

            logging.info(f"DB_DUMP: Found {len(trades)} Trade Orders for {ticker}.")
            for t in trades:
                logging.info(f"DB_DUMP: TRADE | ID: {t.id}, Time: {t.execution_time.isoformat()}, Ticker: {t.ticker}, Type: '{t.order_type}', Qty: {float(t.quantity):.2f}")

        logging.info("--- END DB STATE DUMP ---")

    # --- State Consistency Check (Task 3 Implementation) ---
    def check_state_consistency(self) -> Dict[str, Any]:
        """
        Runs integrity checks across the database invariants.
        """
        inconsistencies = []
        is_consistent = True

        with self.get_session() as session:
            # 1. CHECK: Orphaned TradeOrders (TradeOrder must link to an existing MarketSignal)
            # Find TradeOrder records where engine_signal_id exists but no corresponding MarketSignal exists
            orphaned_trades = session.query(TradeOrder).filter(
                TradeOrder.engine_signal_id.isnot(None),
                ~TradeOrder.engine_signal_id.in_(session.query(MarketSignal.id))
            ).limit(10).all() # Limit to sample for report

            if orphaned_trades:
                is_consistent = False
                inconsistencies.append({
                    "rule": "Orphaned Trade Orders",
                    "count": len(orphaned_trades),
                    "severity": "HIGH",
                    "sample": [t.id for t in orphaned_trades]
                })

            # 2. CHECK: Negative Portfolio Snapshots (Total Value must be non-negative)
            negative_navs = session.query(PortfolioSnapshot).filter(
                PortfolioSnapshot.total_value < 0
            ).limit(1).all()

            if negative_navs:
                is_consistent = False
                inconsistencies.append({
                    "rule": "Negative NAV (Total Value)",
                    "count": session.query(PortfolioSnapshot).filter(PortfolioSnapshot.total_value < 0).count(),
                    "severity": "CRITICAL",
                    "sample": [f"ID {n.id} on {n.snapshot_date.date()}" for n in negative_navs]
                })

            session.close()

        return {
            "is_consistent": is_consistent,
            "details": "No issues found." if is_consistent else inconsistencies
        }

    def get_latest_market_signal(self, ticker: str = None) -> Dict[str, Any] | None:
        """
        Fetches the single latest MarketSignal record from the database.
        Optionally filters by ticker.
        """
        from .models import MarketSignal # Ensure model is imported locally

        with self.get_session() as session:
            query = session.query(MarketSignal).order_by(MarketSignal.timestamp.desc())

            if ticker:
                query = query.filter(MarketSignal.ticker == ticker)

            latest_signal = query.first()

            if latest_signal:
                return {
                    "timestamp": latest_signal.timestamp.isoformat(),
                    "ticker": latest_signal.ticker,
                    "close_price": float(latest_signal.close_price),
                    "signal_A_value": float(latest_signal.signal_A_value) if latest_signal.signal_A_value is not None else None,
                    "signal_B_value": float(latest_signal.signal_B_value) if latest_signal.signal_B_value is not None else None,
                    "action_alert": latest_signal.action_alert,
                    "is_live_signal": latest_signal.is_live_signal,
                }
            return None

    def get_trade_journal(self, ticker: str = "LQQ.PA") -> List[Dict[str, Any]]:
        """
        Pairs BUY and SELL/EXIT orders to create a Journal view.
        Uses FIFO matching to calculate duration and P&L for each trade cycle.
        """
        from .models import TradeOrder

        with self.get_session() as session:
            # Fetch all orders for this ticker, oldest first to process chronologically
            orders = session.query(TradeOrder).filter(
                TradeOrder.ticker == ticker
            ).order_by(TradeOrder.execution_time.asc()).all()
            # TRACE 1: Check raw DB count
            print(f"DEBUG_DB: Total orders found for {ticker}: {len(orders)}")
            if orders:
               print(f"DEBUG_DB: Sample Order Type from DB: '{orders[0].order_type}'")

        journal = []
        buy_stack = []  # Internal buffer for open 'lots'

        for order in orders:
            o_type = order.order_type.upper()
            o_price = float(order.execution_price)
            o_time = order.execution_time
            o_qty = float(order.quantity)

            # CASE A: ENTRY (BUY)
            if "BUY" in o_type:
                # Add to stack as an open lot
                buy_stack.append({
                    "id": order.id,
                    "time": o_time,
                    "price": o_price,
                    "qty": o_qty
                })
                print(f"DEBUG_JOURNAL: Added to buy_stack. Current size: {len(buy_stack)}")

            # CASE B: EXIT (SELL or EXIT)
            elif "SELL" in o_type or "EXIT" in o_type:
                print(f"DEBUG_JOURNAL: Processing EXIT. Current buy_stack size: {len(buy_stack)}")
                sell_qty = float(order.quantity)
                sell_price = float(order.execution_price)
                sell_time = order.execution_time

                # Match against the oldest buys (FIFO)
                while sell_qty > 0 and buy_stack:
                    entry = buy_stack[0]

                    # Calculate matched quantity (handles partial exits)
                    matched_qty = min(entry['qty'], sell_qty)

                    # Calculate Duration
                    duration = sell_time - entry['time']
                    days = duration.days
                    hours = duration.seconds // 3600
                    duration_str = f"{days}d {hours}h" if days > 0 else f"{hours}h"

                    # Calculate P&L %
                    pl_pct = ((sell_price - entry['price']) / entry['price']) * 100

                    journal.append({
                        "ticker": ticker,
                        "type": "LONG", # Strategy is currently Long-biased
                        "open_date": entry['time'].strftime('%Y-%m-%d %H:%M'),
                        "open_price": round(entry['price'], 2),
                        "close_date": sell_time.strftime('%Y-%m-%d %H:%M'),
                        "close_price": round(sell_price, 2),
                        "duration": duration_str,
                        "pl_pct": round(pl_pct, 2),
                        "status": "Closed",
                        "qty": matched_qty
                    })

                    # Update stack/quantities
                    sell_qty -= matched_qty
                    entry['qty'] -= matched_qty

                    if entry['qty'] <= 0:
                        buy_stack.pop(0)

        # CASE C: REMAINDERS (OPEN POSITIONS)
        for open_lot in buy_stack:
            if open_lot['qty'] > 0:
                journal.append({
                    "ticker": ticker,
                    "type": "LONG",
                    "open_date": open_lot['time'].strftime('%Y-%m-%d %H:%M'),
                    "open_price": round(open_lot['price'], 2),
                    "close_date": "---",
                    "close_price": "---",
                    "duration": "Live",
                    "pl_pct": None,
                    "status": "Open",
                    "qty": open_lot['qty']
                })

        # Return reversed so latest trades are at the top of the UI
        print(f"DEBUG_JOURNAL: Final journal count: {len(journal)}")
        return list(reversed(journal))
