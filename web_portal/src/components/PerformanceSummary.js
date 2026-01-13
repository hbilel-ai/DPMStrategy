// web_portal/src/components/PerformanceSummary.js

const API_BASE = `http://${window.location.hostname}:8000`;
const API_ENDPOINT = `${API_BASE}/api/v1/performance/summary`;

export function PerformanceSummary() {
    return `
        <div class="card performance-summary-card" id="performance-summary">
            <h3>📈 Performance Analytics</h3>
            <p>Fetching ledger metrics...</p> 
        </div>
    `;
}

/**
 * Helper to colorize and format percentage values
 */
function formatValue(value, isPercent = true, threshold = 0) {
    if (value === null || value === undefined || isNaN(value)) return 'N/A';
    
    let displayValue = isPercent ? (value * 100).toFixed(2) + '%' : value.toFixed(2);
    let color = 'white';
    
    if (value > threshold) color = '#4caf50'; // Green
    else if (value < threshold) color = '#ff4d4d'; // Red
    
    return `<span style="color: ${color}; font-weight: bold;">${displayValue}</span>`;
}

async function fetchData() {
    const card = document.getElementById('performance-summary');
    if (!card) return;

    try {
        const response = await fetch(`${API_ENDPOINT}?start_date=2000-01-01`);
        if (!response.ok) throw new Error('Failed to fetch performance summary.');
        const data = await response.json();

        card.innerHTML = `
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 15px;">
                <h3 style="margin:0;">📈 Performance Analytics</h3>
                <small style="color: #888;">Source: Strategy Ledger (Verified)</small>
            </div>
            <table class="performance-table">
                <thead>
                    <tr style="border-bottom: 2px solid #444;">
                        <th style="text-align: left; padding-bottom: 10px;">Metric</th>
                        <th style="text-align: right; padding-bottom: 10px;">Value</th>
                    </tr>
                </thead>
                <tbody>
                    <tr><td>Sharpe Ratio</td><td style="text-align: right;">${formatValue(data.sharpe_ratio, false, 1.0)}</td></tr>
                    <tr><td>Max Drawdown</td><td style="text-align: right;"><span style="color: #ff4d4d;">${data.mdd}</span></td></tr>
                    <tr><td>Volatility (Ann.)</td><td style="text-align: right;">${(data.volatility_annualized * 100).toFixed(2)}%</td></tr>
                    
                    <tr style="background: rgba(255,255,255,0.03)"><td colspan="2" style="font-size: 0.7rem; color: #888; padding-top: 10px;">STRATEGY EDGE</td></tr>
                    <tr><td>Win Rate (Daily)</td><td style="text-align: right;">${formatValue(parseFloat(data.wr)/100, true, 0.5)}</td></tr>
                    <tr><td>Profit Factor</td><td style="text-align: right;">${formatValue(parseFloat(data.pf), false, 1.0)}</td></tr>
                    <tr><td>Recovery Factor</td><td style="text-align: right;">${formatValue(parseFloat(data.rf), false, 1.5)}</td></tr>

                    <tr style="background: rgba(255,255,255,0.03)"><td colspan="2" style="font-size: 0.7rem; color: #888; padding-top: 10px;">TRAILING RETURNS</td></tr>
                    <tr><td>1 Month Return</td><td style="text-align: right;">${formatValue(data['1M'])}</td></tr>
                    <tr><td>3 Month Return</td><td style="text-align: right;">${formatValue(data['3M'])}</td></tr>
                    <tr><td>YTD Return</td><td style="text-align: right;">${formatValue(data['YTD'])}</td></tr>
                </tbody>
            </table>
            <div style="margin-top: 10px; font-size: 0.75rem; color: #666;">
                * MDD Period: ${data.mdd_dates || 'N/A'}
            </div>
        `;

        // Apply internal table styling
        const table = card.querySelector('table');
        table.style.width = '100%';
        table.style.borderCollapse = 'collapse';
        card.querySelectorAll('td').forEach(td => {
            td.style.padding = '10px 0';
            td.style.borderBottom = '1px solid #333';
        });

    } catch (error) {
        console.error("Error loading Performance Summary:", error);
        card.innerHTML = `<p style="color: #ff4d4d;">⚠️ Error loading metrics. Check API connection.</p>`;
    }
}

// Initial fetch
setTimeout(fetchData, 100);
