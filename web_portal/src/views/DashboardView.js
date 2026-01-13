import { CurrentStatus } from '../components/CurrentStatus.js';
import { PerformanceSummary } from '../components/PerformanceSummary.js';
import { CumulativeReturnsChart } from '../components/CumulativeReturnsChart.js';
import { SignalPriceChart } from '../components/SignalPriceChart.js';

export function DashboardView() {
    return `
        <div style="padding: 10px;">
            <h2>🚀 Live Strategy Monitor <span style="font-size: 0.8rem; color: #4caf50; font-weight: normal; border: 1px solid #4caf50; padding: 2px 8px; border-radius: 12px; margin-left: 10px;">Ledger Verified</span></h2>

            <div class="card-grid">
                ${CurrentStatus()}
                ${PerformanceSummary()}
            </div>

            <div class="charts-stack" style="margin-top: 20px; display: flex; flex-direction: column; gap: 20px;">
                <div class="card">
                    <h3>Strategy Verification (Price vs Signal)</h3>
                    ${SignalPriceChart()}
                </div>
                <div class="card">
                    <h3>Equity Growth</h3>
                    ${CumulativeReturnsChart()}
                </div>
            </div>
        </div>
    `;
}
