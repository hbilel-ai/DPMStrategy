# data_layer/models.py
from sqlalchemy import Column, Integer, String, Numeric, DateTime, Boolean, create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

# Base class for declarative class definitions
Base = declarative_base()

# --- 1. Portfolio Snapshot Model ---
class PortfolioSnapshot(Base):
    __tablename__ = 'portfolio_snapshot'
    
    id = Column(Integer, primary_key=True)
    snapshot_date = Column(DateTime, nullable=False, index=True)
    total_value = Column(Numeric, nullable=False)
    cash_balance = Column(Numeric, nullable=False)
    cash_flow = Column(Numeric, default=0.0)
    mode = Column(String, default='LIVE') 

# --- 2. Executed Trade Order Model ---
class TradeOrder(Base):
    __tablename__ = 'trade_order'
    
    id = Column(Integer, primary_key=True)
    execution_time = Column(DateTime, nullable=False, index=True)
    ticker = Column(String, nullable=False)
    order_type = Column(String, nullable=False) # BUY or SELL
    quantity = Column(Numeric(precision=18, scale=8), nullable=False)
    execution_price = Column(Numeric, nullable=False)
    
    # Simple reference to the signal that triggered this (can be null if manual trade)
    engine_signal_id = Column(Integer, index=True) 
    status = Column(String, default='FILLED')

# --- 3. Market Signal/Data Model ---
class MarketSignal(Base):
    __tablename__ = 'market_signal' # Ensure this matches your table name

    id = Column(Integer, primary_key=True)
    timestamp = Column(DateTime, nullable=False, index=True)
    ticker = Column(String, nullable=False)
    close_price = Column(Numeric, nullable=False)

    # Component Signal values
    signal_A_value = Column(Numeric)
    signal_B_value = Column(Numeric)
    signal_VIX_value = Column(Numeric)

    # --- ADD THIS LINE ---
    position = Column(Numeric, default=0.0)

    action_alert = Column(String)
    is_live_signal = Column(Boolean, default=False)
    
# Example setup function (for demonstration/initialization)
def setup_database(db_url):
    engine = create_engine(db_url)
    Base.metadata.create_all(engine)
    return sessionmaker(bind=engine)

# Note: The actual DBManager class will use these models to perform its queries.
