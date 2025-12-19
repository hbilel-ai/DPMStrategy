// web_portal/src/components/PerformanceSummary.js

const API_ENDPOINT = 'http://192.168.1.17:8000/api/v1/performance/summary';

export function PerformanceSummary() {
    return `
        <div class="card performance-summary-card" id="performance-summary">
            <h3>4. Performance Table (API: /performance/summary)</h3>
            <p>Loading performance metrics (Sharpe, Drawdown, Trailing Returns)...</p>
        </div>
    `;
}

// Function to format return percentages and color them
function formatReturn(value) {
    if (value === null || value === undefined) return 'N/A';
    
    const percent = (value * 100).toFixed(2) + '%';
    let color = 'white';
    if (value > 0) color = 'lightgreen';
    else if (value < 0) color = 'red';
    
    return `<span style="color: ${color};">${percent}</span>`;
}

// Function to fetch data and update the DOM
async function fetchData() {
    const card = document.getElementById('performance-summary');
    if (!card) return;

    try {
        const response = await fetch(API_ENDPOINT);
        if (!response.ok) {
            throw new Error('Failed to fetch performance summary.');
        }
        const data = await response.json();

        card.innerHTML = `
            <h3>4. Performance Table</h3>
            <table>
                <thead>
                    <tr><th>Metric</th><th>Value</th></tr>
                </thead>
                <tbody>
                    <tr><td>Sharpe Ratio</td><td>${data.sharpe_ratio.toFixed(2)}</td></tr>
                    <tr><td>Max Drawdown</td><td>${(data.max_drawdown * 100).toFixed(2)}%</td></tr>
                    <tr><td>Volatility (Ann.)</td><td>${(data.volatility_annualized * 100).toFixed(2)}%</td></tr>
                    <tr><td>1 Week Return</td><td>${formatReturn(data['1W'])}</td></tr>
                    <tr><td>1 Month Return</td><td>${formatReturn(data['1M'])}</td></tr>
                    <tr><td>3 Month Return</td><td>${formatReturn(data['3M'])}</td></tr>
                    <tr><td>YTD Return</td><td>${formatReturn(data['YTD'])}</td></tr>
                </tbody>
            </table>
        `;
        // Add basic table styling
        card.querySelector('table').style.width = '100%';
        card.querySelectorAll('th, td').forEach(el => {
            el.style.textAlign = 'left';
            el.style.padding = '8px 0';
            el.style.borderBottom = '1px solid #3a3a3a';
        });
        card.querySelector('tbody tr:last-child td').style.borderBottom = 'none';

    } catch (error) {
        console.error("Error loading Performance Summary:", error);
        card.innerHTML = `
            <h3>4. Performance Table (Error)</h3>
            <p style="color: red;">Could not connect to API: ${API_ENDPOINT}</p>
        `;
    }
}

setTimeout(fetchData, 100);
