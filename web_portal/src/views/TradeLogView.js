// web_portal/src/views/TradeLogView.js
const API_BASE = `http://${window.location.hostname}:8000`;

export function TradeLogView() {
    return `
        <div class="trade-log-container">
            <h2>📋 Trade Journal & Audit</h2>

            <div id="journal-summary-ribbon" class="summary-ribbon">
                <div class="card shadow-sm">
                    <small>Total Closed Trades</small>
                    <h3 id="stat-total-trades">--</h3>
                </div>
                <div class="card shadow-sm">
                    <small>Win Rate</small>
                    <h3 id="stat-win-rate">--</h3>
                </div>
                <div class="card shadow-sm">
                    <small>Avg. Profit/Trade</small>
                    <h3 id="stat-avg-profit">--</h3>
                </div>
                <div class="card shadow-sm">
                    <small>Net P&L (Sum)</small>
                    <h3 id="stat-net-pl">--</h3>
                </div>
            </div>

            <div class="filter-bar card">
                <label>Filter Status:</label>
                <select id="journal-status-filter">
                    <option value="ALL">Show All</option>
                    <option value="Open">Open Positions Only</option>
                    <option value="Closed">Closed Trades Only</option>
                </select>
                <button id="btn-refresh-journal" class="btn-primary">Refresh Data</button>
            </div>

            <div class="card table-responsive">
                <table class="journal-table">
                    <thead>
                        <tr>
                            <th>Ticker</th>
                            <th>Type</th>
                            <th>Open Date / Price</th>
                            <th>Close Date / Price</th>
                            <th>Duration</th>
                            <th>P&L %</th>
                            <th>Status</th>
                        </tr>
                    </thead>
                    <tbody id="journal-table-body">
                        <tr><td colspan="7" style="text-align:center;">Loading journal data...</td></tr>
                    </tbody>
                </table>
            </div>
        </div>
    `;
}

// Global variable to store fetched data for filtering without re-fetching
let cachedJournalData = [];

export async function initTradeLogView() {
    const refreshBtn = document.getElementById('btn-refresh-journal');
    const filterSelect = document.getElementById('journal-status-filter');

    const loadData = async () => {
        try {
            // Update to your actual API URL
            const response = await fetch(`${API_BASE}/api/v1/trades/journal?ticker=LQQ.PA`);
            const data = await response.json();
            
            cachedJournalData = data.trades;
            renderSummary(data.summary);
            applyFilter(); 
        } catch (error) {
            console.error("Journal Fetch Error:", error);
        }
    };

    const applyFilter = () => {
        const filterValue = filterSelect.value;
        const filtered = filterValue === 'ALL' 
            ? cachedJournalData 
            : cachedJournalData.filter(t => t.status === filterValue);
        
        renderTable(filtered);
    };

    // Event Listeners
    refreshBtn.onclick = loadData;
    filterSelect.onchange = applyFilter;

    // Initial Load
    await loadData();
}

function renderSummary(summary) {
    if (!summary) return;

    // Check for values, if they are null/undefined, show the '--' placeholder
    document.getElementById('stat-total-trades').innerText = summary.total_trades ?? '0';
    document.getElementById('stat-win-rate').innerText = summary.win_rate ?? '0%';
    document.getElementById('stat-avg-profit').innerText = summary.avg_profit ?? '--';
    document.getElementById('stat-net-pl').innerText = summary.net_pl ?? '--';
}

function renderTable(trades) {
    const tbody = document.getElementById('journal-table-body');

    if (trades.length === 0) {
        tbody.innerHTML = `<tr><td colspan="7" style="text-align:center;">No trades found for this filter.</td></tr>`;
        return;
    }

    tbody.innerHTML = trades.map(t => {
        const plValue = t.pl_pct !== null ? t.pl_pct : 0;
        const plClass = plValue > 0 ? 'ret-pos' : (plValue < 0 ? 'ret-neg' : '');
        const statusClass = t.status === 'Open' ? 'status-open' : 'status-closed';

        // NEW: Logic for coloring the Type badge
        const typeClass = t.type.toUpperCase() === 'LONG' ? 'type-long' : 'type-exit';

        return `
            <tr>
                <td><strong>${t.ticker}</strong></td>
                <td><span class="type-badge ${typeClass}">${t.type}</span></td>
                <td>${t.open_date}<br><small class="text-muted">$${t.open_price}</small></td>
                <td>${t.close_date}<br><small class="text-muted">${t.close_price !== '---' ? '$' + t.close_price : ''}</small></td>
                <td>${t.duration}</td>
                <td class="${plClass} font-weight-bold">${t.pl_pct !== null ? t.pl_pct.toFixed(2) + '%' : '--'}</td>
                <td><span class="badge ${statusClass}">${t.status}</span></td>
            </tr>
        `;
    }).join('');
}
