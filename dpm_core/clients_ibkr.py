# dpm_core/clients_ibkr.py
import logging
from ib_insync import IB, Contract, MarketOrder
from dpm_core.strategy_core import BrokerClient
from data_layer.models import TradeOrder
from datetime import datetime
import pytz

class IBKRClient(BrokerClient):
    def __init__(self, host, port, client_id, live_ticker):
        self.host = host
        self.port = port
        self.client_id = client_id
        self.live_ticker = live_ticker
        self.account_id = "Not Connected"
        self.ib = IB()
        self.connect()

    def connect(self):
        try:
            if not self.ib.isConnected():
                self.ib.connect(self.host, self.port, clientId=self.client_id)
                managed_accounts = self.ib.managedAccounts()
                if managed_accounts:
                    self.account_id = managed_accounts[0]
                logging.info(f"IBKR: Connected to {self.host}:{self.port} (Account: {self.account_id})")
        except Exception as e:
            logging.error(f"IBKR: Connection failed: {e}")
            raise

    def _get_contract(self, ticker: str):
        if ticker == "LQQ.PA":
            return Contract(
                symbol='LQQ', secType='STK', exchange='SMART', 
                primaryExchange='SBF', currency='EUR'
            )
        # Default for USD assets
        return Contract(symbol=ticker.split('.')[0], secType='STK', exchange='SMART', currency='USD')

    def is_market_open(self, ticker: str) -> bool:
        """Checks IBKR contract details to see if market is currently open."""
        contract = self._get_contract(ticker)
        details = self.ib.reqContractDetails(contract)
        if not details:
            return False
        
        # liquidHours format: '20260109:0900-1730;...'
        # ib_insync provides a utility to check if we are in liquid hours
        # For simplicity and reliability, we check the exchange timezone
        tz = pytz.timezone(details[0].timeZoneId)
        now_exchange = datetime.now(tz)
        
        # Basic check for Euronext: 09:00 - 17:30
        # For a truly robust check, you can parse details[0].liquidHours
        is_open = 9 <= now_exchange.hour < 17 or (now_exchange.hour == 17 and now_exchange.minute <= 30)
        
        if not is_open:
            logging.warning(f"IBKR: Market for {ticker} is CLOSED (Exchange Time: {now_exchange.strftime('%H:%M')})")
        return is_open

    def get_current_position(self, ticker: str) -> float:
        symbol = ticker.split('.')[0]
        positions = self.ib.positions()
        for p in positions:
            if p.contract.symbol == symbol:
                return float(p.position)
        return 0.0

    def get_current_cash(self) -> float:
        summary = self.ib.accountSummary()
        cash_value = 0.0
        for item in summary:
            if item.tag == 'TotalCashValue' and item.currency == 'EUR':
                cash_value = float(item.value)
        
        # Safety Buffer: 0.5%
        return cash_value * 0.995
    
    def execute_order(self, ticker: str, target_allocation: float, current_price: float):
        # 1. Market Availability Check
        if not self.is_market_open(ticker):
            logging.error(f"IBKR: Execution aborted. {ticker} is not currently tradable.")
            return

        contract = self._get_contract(ticker)
        self.ib.qualifyContracts(contract)

        # 2. Calculation
        current_qty = self.get_current_position(ticker)
        cash = self.get_current_cash()
        total_equity = cash + (current_qty * current_price)
        
        target_qty = int((total_equity * target_allocation) / current_price)
        delta_qty = target_qty - current_qty

        if abs(delta_qty) < 1:
            logging.info(f"IBKR: Target reached ({target_qty} shares). No trade.")
            return

        # 3. Order Submission
        action = 'BUY' if delta_qty > 0 else 'SELL'
        # Added tif='DAY' to fix Error 10349
        order = MarketOrder(action, abs(delta_qty), tif='DAY')
        
        logging.info(f"IBKR: Sending {action} order for {abs(delta_qty)} shares...")
        trade = self.ib.placeOrder(contract, order)

        # 4. Wait for Confirmation (Max 10 seconds)
        timeout = 10
        start_time = datetime.now()
        while not trade.isDone():
            self.ib.waitOnUpdate(timeout=0.5)
            if (datetime.now() - start_time).seconds > timeout:
                logging.warning("IBKR: Order timeout reached. Check TWS for status.")
                break

        # Prepare standardized result object
        status = 'FILLED' if trade.orderStatus.status == 'Filled' else 'FAILED'
        fill_price = trade.orderStatus.avgFillPrice if status == 'FILLED' else current_price

        return {
            'ticker': ticker,
            'order_type': action,
            'quantity': float(abs(delta_qty)),
            'execution_price': float(fill_price),
            'execution_time': datetime.now(),
            'status': status
        }



    def disconnect(self):
        if self.ib.isConnected():
            self.ib.disconnect()
