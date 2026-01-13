// web_portal/src/views/TradeLogView.js
const API_BASE = `http://${window.location.hostname}:8000`;

// Global State
let currentView = 'journal'; // 'journal' or 'ledger'
let cachedData = [];

export function TradeLogView() {
    return `
        <div class="trade-log-container">
            <div class="header-with-toggle">
                <h2>📋 Trade Journal & Audit</h2>
                <div class="view-switcher">
                    <button id="switch-journal" class="switch-btn active">FIFO Journal</button>
                    <button id="switch-ledger" class="switch-btn">Account Statement</button>
                </div>
            </div>

            <div id="journal-summary-ribbon" class="summary-ribbon">
                <div class="card shadow-sm">
                    <small id="label-stat-1">Total Closed Trades</small>
                    <h3 id="stat-total-trades">--</h3>
                </div>
                <div class="card shadow-sm">
                    <small id="label-stat-2">Win Rate</small>
                    <h3 id="stat-win-rate">--</h3>
                </div>
                <div class="card shadow-sm">
                    <small id="label-stat-3">Avg. Profit/Trade</small>
                    <h3 id="stat-avg-profit">--</h3>
                </div>
                <div class="card shadow-sm">
                    <small id="label-stat-4">Net P&L (Sum)</small>
                    <h3 id="stat-net-pl">--</h3>
                </div>
            </div>

            <div class="filter-bar card">
                <div id="journal-filters" class="filter-group">
                    <label>Filter Status:</label>
                    <select id="journal-status-filter">
                        <option value="ALL">Show All</option>
                        <option value="Open">Open Only</option>
                        <option value="Partial">Partial Only</option>
                        <option value="Closed">Closed Only</option>
                    </select>
                </div>
                <button id="btn-refresh-journal" class="btn-primary">Refresh Data</button>
            </div>

            <div class="table-responsive card">
                <table>
                    <thead id="journal-table-head">
                        </thead>
                    <tbody id="journal-table-body">
                        </tbody>
                </table>
            </div>
        </div>
    `;
}

export async function initTradeLogView() {
    const refreshBtn = document.getElementById('btn-refresh-journal');
    const filterSelect = document.getElementById('journal-status-filter');
    const journalBtn = document.getElementById('switch-journal');
    const ledgerBtn = document.getElementById('switch-ledger');

    const loadData = async () => {
        try {
            const endpoint = currentView === 'journal' ? 'journal' : 'ledger';
            const response = await fetch(`${API_BASE}/api/v1/trades/${endpoint}?ticker=LQQ.PA`);
            const data = await response.json();
            
            console.log(`[DEBUG] Mode: ${currentView} | Raw Data:`, data); // <--- AJOUT
            cachedData = (currentView === 'journal') ? data.trades : data.ledger;

            cachedData = (currentView === 'journal') ? data.trades : data.ledger;
            updateSummaryRibbon(data.summary);
            applyFilter(); 
        } catch (error) {
            console.error("Fetch Error:", error);
        }
    };

    const applyFilter = () => {
        const filterValue = filterSelect.value;
        // Ledger view doesn't use status filtering, only Journal
        const filtered = (currentView === 'journal' && filterValue !== 'ALL')
            ? cachedData.filter(t => t.status === filterValue)
            : cachedData;
        
        renderTable(filtered);
    };

    const toggleView = (view) => {
        currentView = view;
        journalBtn.classList.toggle('active', view === 'journal');
        ledgerBtn.classList.toggle('active', view === 'ledger');
        // Show/Hide status filter for Ledger
        document.getElementById('journal-filters').style.display = view === 'journal' ? 'block' : 'none';
        loadData();
    };

    refreshBtn.onclick = loadData;
    filterSelect.onchange = applyFilter;
    journalBtn.onclick = () => toggleView('journal');
    ledgerBtn.onclick = () => toggleView('ledger');

    await loadData();
}

function updateSummaryRibbon(summary) {
    if (!summary) return;
    
    const labels = {
        journal: ["Total Closed Trades", "Win Rate", "Avg. Profit/Trade", "Net P&L (Sum)"],
        ledger: ["Current Exposure", "Total Volume", "Transaction Count", "Net Flow"]
    };

    const currentLabels = labels[currentView];
    document.getElementById('label-stat-1').innerText = currentLabels[0];
    document.getElementById('label-stat-2').innerText = currentLabels[1];
    document.getElementById('label-stat-3').innerText = currentLabels[2];
    document.getElementById('label-stat-4').innerText = currentLabels[3];

    if (currentView === 'journal') {
        document.getElementById('stat-total-trades').innerText = summary.total_trades ?? '--';
        document.getElementById('stat-win-rate').innerText = summary.win_rate ?? '--';
        document.getElementById('stat-avg-profit').innerText = summary.avg_profit ?? '--';
        document.getElementById('stat-net-pl').innerText = summary.net_pl ?? '--';
    } else {
        document.getElementById('stat-total-trades').innerText = summary.current_exposure ?? '--';
        document.getElementById('stat-win-rate').innerText = summary.total_volume ?? '--';
        document.getElementById('stat-avg-profit').innerText = summary.transaction_count ?? '--';
        document.getElementById('stat-net-pl').innerText = summary.net_flow ?? '--';
    }
}

function renderTable(data) {
    const thead = document.getElementById('journal-table-head');
    const tbody = document.getElementById('journal-table-body');
    if (!tbody || !thead) return;

    if (currentView === 'journal') {
        thead.innerHTML = `
            <tr>
                <th>Ticker</th><th>Type</th><th>Qty (Orig/Rem)</th><th>Entry</th><th>Exit</th><th>Duration</th><th>P&L %</th><th>Status</th>
            </tr>`;
        
        tbody.innerHTML = data.map(t => {
            const plClass = t.pl_pct > 0 ? 'ret-pos' : (t.pl_pct < 0 ? 'ret-neg' : '');
            const statusClass = `status-${t.status.toLowerCase()}`;
            return `
                <tr>
                    <td><strong>${t.ticker}</strong></td>
                    <td><span class="type-badge type-long">${t.type}</span></td>
                    <td>${t.original_qty} / <strong style="color: #007bff;">${t.remaining_qty}</strong></td>
                    <td>${t.open_date}<br><small class="text-muted">$${t.open_price}</small></td>
                    <td>${t.close_date}<br><small class="text-muted">${t.close_price !== '---' ? '$' + t.close_price : ''}</small></td>
                    <td>${t.duration}</td>
                    <td class="${plClass} font-weight-bold">${t.pl_pct}%</td>
                    <td><span class="badge ${statusClass}">${t.status}</span></td>
                </tr>`;
        }).join('');
    } else {
        thead.innerHTML = `
            <tr>
                <th>Date</th><th>Type</th><th>Initial Exp.</th><th>Qty Diff</th><th>Total Exp.</th><th>Price</th>
            </tr>`;

        tbody.innerHTML = data.map(t => {
            const diffClass = t.qty_diff > 0 ? 'ret-pos' : (t.qty_diff < 0 ? 'ret-neg' : '');
            const typeClass = t.type === 'BUY' ? 'type-long' : 'type-exit';
            return `
                <tr>
                    <td>${t.date}</td>
                    <td><span class="type-badge ${typeClass}">${t.type}</span></td>
                    <td>${t.initial_qty}</td>
                    <td class="${diffClass} font-weight-bold">${t.qty_diff > 0 ? '+' : ''}${t.qty_diff}</td>
                    <td><strong>${t.total_exposure}</strong></td>
                    <td>$${t.price}</td>
                </tr>`;
        }).join('');
    }
}
