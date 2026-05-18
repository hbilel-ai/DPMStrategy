from ib_insync import IB
import logging

logging.basicConfig(level=logging.INFO)
ib = IB()

try:
    # This connects to your laptop port 7496, which Docker sends to 4004, 
    # which Socat sends to 4002, where IBKR is waiting.
    ib.connect('127.0.0.1', 7496, clientId=99)
    print("------------------------------------------")
    print(f"SUCCESS! Connected to Account: {ib.managedAccounts()}")
    print("------------------------------------------")
    ib.disconnect()
except Exception as e:
    print(f"CONNECTION FAILED: {e}")
