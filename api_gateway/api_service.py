# api_service.py
from fastapi import FastAPI, HTTPException  # Added HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any
from starlette.middleware.cors import CORSMiddleware
from data_layer.db_manager import DBManager
from application_logic.analytics_service import AnalyticsService
from config.settings import settings
from datetime import datetime, timedelta

class EquityChartResponse(BaseModel):
    strategy: List[Dict[str, Any]]
    benchmark: List[Dict[str, Any]]
    benchmark_label: str

class CumulativeReturnPoint(BaseModel):
    date: str
    equity: float

# Initialize Services
DB_URL = settings.DATABASE_URL
db_manager = DBManager(DB_URL)
analytics_service = AnalyticsService(db_manager)

app = FastAPI(
    title="Trading Portal API",
    version="v1",
    description="Backend service for live trading data and analytics."
)

# CORS Middleware (Crucial for the Port 6060 -> 8000 communication)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/api/v1/performance/summary")
def get_performance_summary(start_date: str = '2020-01-01'):
    periodic_r = analytics_service.calculate_periodic_returns(start_date)
    risk_m = analytics_service.calculate_risk_metrics(start_date)
    return {**periodic_r, **risk_m}

@app.get("/api/v1/charts/returns")
def get_cumulative_returns_chart_data(days: int = 365):
    """
    EVOLVED: Fetches timeframe-based cumulative returns for both 
    the Strategy and the Benchmark defined in config.yaml.
    """
    # 1. Calculate the start date based on the 'days' parameter
    start_date = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')
    
    # 2. Extract benchmark ticker from settings (Asset 0)
    # This ensures it matches your config.yaml: assets[0].benchmark_ticker
    benchmark_ticker = settings.assets[0].benchmark_ticker if settings.assets else "LQQ.PA"
    
    try:
        # 3. Call the evolved service method
        # Returns: {"strategy": [...], "benchmark": [...], "benchmark_label": "..."}
        data = analytics_service.get_cumulative_returns(
            start_date=start_date, 
            benchmark_ticker=benchmark_ticker
        )
        return data
    except Exception as e:
        print(f"API Error in charts/returns: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/v1/kpis/current")
def get_current_kpis(ticker: str = "LQQ.PA"):
    latest_signal = db_manager.get_latest_market_signal(ticker=ticker)
    if not latest_signal:
        return {"error": f"No market signal found for ticker {ticker}."}

    val_a = latest_signal.get('signal_A_value', 0) or 0
    val_b = latest_signal.get('signal_B_value', 0) or 0
    exposure = (val_a + val_b) / 2

    return {
        "timestamp": latest_signal['timestamp'],
        "ticker": latest_signal['ticker'],
        "current_price": latest_signal['close_price'],
        "tmom_signal_value": latest_signal['signal_A_value'],
        "sma_signal_value": latest_signal['signal_B_value'],
        "exposure": exposure,
        "action_alert": latest_signal['action_alert'],
        "is_live_signal": latest_signal['is_live_signal']
    }

@app.get("/api/v1/charts/signal_price")
def get_signal_price_chart_data(ticker: str = "LQQ.PA", start_date: str = '2020-01-01'):

    data = analytics_service.get_signal_chart_data(ticker, start_date)
    return data

# Unified Trade Log
@app.get("/api/v1/trades")
def get_trade_log():
    trades = db_manager.get_executed_trades()
    return {"trades": trades, "count": len(trades)}

@app.get("/api/v1/analytics/cumulative_returns", response_model=EquityChartResponse)
def get_cumulative_returns_data(days: int = 365):
    start_date = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')
    # This now returns the Dict {'strategy':..., 'benchmark':...}
    return analytics_service.get_cumulative_returns(start_date=start_date)

@app.get("/api/v1/performance/history")
def get_performance_history(year: str = "ALL"):
    start_date = f"{year}-01-01" if year != "ALL" else "2000-01-01"
    equity_data = analytics_service.calculate_cumulative_returns(start_date)
    risk_metrics = analytics_service.calculate_risk_metrics(start_date)
    heatmap_data = analytics_service.get_monthly_returns_heatmap(year)

    return {
        "stats": {
            "pf": risk_metrics.get('pf', "0.0"),
            "mdd": risk_metrics.get('mdd', "0%"),
            "rf": risk_metrics.get('rf', "0.0"),
            "wr": risk_metrics.get('wr', "0%"),
            "mdd_dates": risk_metrics.get('mdd_dates', "N/A to N/A")
        },
        "monthly_returns": heatmap_data or [],
        "equity_curve": {
            "timestamps": equity_data.get('timestamps', []),
            "values": equity_data.get('cumulative_returns', []),
            "drawdowns": equity_data.get('drawdown_series', [])
        }
    }

# --- FIXED JOURNAL ENDPOINT (Removed Duplicate) ---
@app.get("/api/v1/trades/journal")
async def get_trade_journal(ticker: str = "LQQ.PA"):
    print(f"DEBUG_API: Received request for Trade Journal for {ticker}")
    try:
        data = analytics_service.get_trade_journal_data(ticker)
        print(f"DEBUG_API: Returning {len(data.get('trades', []))} trade records")
        return data
    except Exception as e:
        print(f"DEBUG_API: Error in Journal API: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# --- FIXED AUDIT ENDPOINT ---
@app.get("/api/v1/trades/audit")
async def get_audit_log():
    print("DEBUG_API: Received request for Audit Log")
    try:
        trades = db_manager.get_executed_trades()
        return {"trades": trades, "count": len(trades)}
    except Exception as e:
        print(f"DEBUG_API: Error in Audit API: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
