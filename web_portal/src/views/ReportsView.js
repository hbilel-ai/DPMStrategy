// web_portal/src/views/ReportsView.js

export function ReportsView() {
    return `
        <div style="padding: 10px;">
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <h2>📊 Historical Performance Reports</h2>
                <div class="report-controls" style="display: flex; gap: 10px;">
                    <select id="report-year-select" style="padding: 5px; background: #252526; color: white; border: 1px solid #333;">
                        <option value="ALL">Full History</option>
                        <option value="2024">2024</option>
                        <option value="2023">2023</option>
                    </select>
                    <button id="btn-refresh-reports" class="btn-timeframe">Generate Report</button>
                </div>
            </div>

            <div class="card" style="margin-top: 20px;">
                <h3>Cumulative Returns & Drawdown</h3>
                <div id="historical-chart-container" style="height: 400px; width: 100%;">
                    <canvas id="historical-equity-canvas"></canvas>
                </div>
            </div>

            <div class="card-grid" style="margin-top: 20px; grid-template-columns: 2fr 1fr;">
                <div class="card">
                    <h3>Monthly Returns Heatmap (%)</h3>
                    <div id="monthly-heatmap-container">
                        <p style="color: #666;">Select a period and click generate...</p>
                    </div>
                </div>

                <div class="card">
                    <h3>Risk Statistics</h3>
                    <div id="historical-stats-container">
                        <ul style="list-style: none; padding: 0; line-height: 2.2;">
                            <li>Profit Factor: <span id="stat-pf" style="font-weight: bold;">-</span></li>
                            <li>Win Rate: <span id="stat-wr" style="font-weight: bold;">-</span></li>
                            <hr style="border: 0; border-top: 1px solid #333; margin: 10px 0;">
                            <li>Max Drawdown: <span id="stat-mdd" style="color: #ff4d4d; font-weight: bold;">-</span></li>
                            <li style="font-size: 0.85em; color: #aaa;">Period: <span id="stat-mdd-period">-</span></li>
                            <li>Recovery Factor: <span id="stat-rf" style="font-weight: bold;">-</span></li>
                        </ul>
                    </div>
                </div>
            </div>
        </div>
    `;
}
