// web_portal/src/components/CumulativeReturnsChart.js

const API_BASE = `http://${window.location.hostname}:8000`;
const API_ENDPOINT = `${API_BASE}/api/v1/analytics/cumulative_returns`;

let equityChart = null;

// Broadcast zoom/pan to other charts (SignalPriceChart)
const broadcastSync = (chart) => {
    window.dispatchEvent(new CustomEvent('sync-charts', {
        detail: { min: chart.scales.x.min, max: chart.scales.x.max, sender: 'equity' }
    }));
};

// Listen for sync events from other charts
window.addEventListener('sync-charts', (e) => {
    if (e.detail.sender !== 'equity' && equityChart) {
        equityChart.options.scales.x.min = e.detail.min;
        equityChart.options.scales.x.max = e.detail.max;
        equityChart.update('none');
    }
});

export function CumulativeReturnsChart() {
    return `
        <div class="chart-controls" style="margin-bottom: 10px; display: flex; gap: 10px; align-items: center;">
            <button class="btn-equity-timeframe" data-days="30">1M</button>
            <button class="btn-equity-timeframe" data-days="180">6M</button>
            <button class="btn-equity-timeframe" data-days="365">1Y</button>
            <button class="btn-equity-timeframe" data-days="3650">ALL</button>
            <span id="benchmark-label" style="font-size: 0.8rem; color: #888; margin-left: auto;">Benchmark: --</span>
        </div>
        <div style="height: 400px; width: 100%;">
            <canvas id="equityGrowthChart"></canvas>
        </div>
    `;
}

export async function fetchAndRenderEquity(days = 365) {
    const ctx = document.getElementById('equityGrowthChart');
    if (!ctx) return;

    // Calculate start_date based on days
    const startDate = new Date();
    startDate.setDate(startDate.getDate() - days);
    const dateStr = startDate.toISOString().split('T')[0];

    try {
        const response = await fetch(`${API_ENDPOINT}?days=${days}`);
        const data = await response.json();

        if (equityChart) {
            equityChart.destroy();
        }

        document.getElementById('benchmark-label').innerText = `Benchmark: ${data.benchmark_label || 'LQQ.PA'}`;

        equityChart = new Chart(ctx, {
            type: 'line',
            data: {
                datasets: [
                    {
                        label: 'Strategy (Verified Ledger)',
                        data: data.strategy.map(p => ({ x: p.date, y: p.equity * 100 })),
                        borderColor: '#007bff',
                        backgroundColor: 'rgba(0, 123, 255, 0.1)',
                        borderWidth: 2,
                        pointRadius: 0,
                        fill: false,
                        tension: 0.1
                    },
                    {
                        label: `Benchmark (${data.benchmark_label})`,
                        data: data.benchmark.map(p => ({ x: p.date, y: p.equity * 100 })),
                        borderColor: '#6c757d',
                        borderDash: [5, 5], // Dashed line for benchmark
                        borderWidth: 1.5,
                        pointRadius: 0,
                        fill: false,
                        tension: 0.1
                    }
                ]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                interaction: {
                    mode: 'index',
                    intersect: false,
                },
                plugins: {
                    legend: {
                        display: true,
                        labels: { color: '#ccc', font: { size: 11 } }
                    },
                    tooltip: {
                        enabled: true,
                        callbacks: {
                            label: (context) => `${context.dataset.label}: ${context.parsed.y.toFixed(2)}%`
                        }
                    },
                    zoom: {
                        zoom: {
                            wheel: { enabled: true },
                            pinch: { enabled: true },
                            mode: 'x',
                            onZoom: ({chart}) => broadcastSync(chart)
                        },
                        pan: {
                            enabled: true,
                            mode: 'x',
                            onPan: ({chart}) => broadcastSync(chart)
                        }
                    }
                },
                scales: {
                    x: {
                        type: 'time',
                        time: { unit: 'month' },
                        grid: { color: '#333' },
                        ticks: { color: '#888' }
                    },
                    y: {
                        grid: { color: '#333' },
                        ticks: { 
                            color: '#888',
                            callback: (value) => value + '%' 
                        },
                        title: { display: true, text: 'Cumulative Return (%)', color: '#ccc' }
                    }
                }
            }
        });
    } catch (e) {
        console.error("Equity Chart Render Error:", e);
    }
}

// Global timeframe listener
document.addEventListener('click', (e) => {
    if (e.target.classList.contains('btn-equity-timeframe')) {
        const days = parseInt(e.target.getAttribute('data-days'));
        fetchAndRenderEquity(days);
    }
});

// Auto-init for first load
setTimeout(() => fetchAndRenderEquity(365), 200);
