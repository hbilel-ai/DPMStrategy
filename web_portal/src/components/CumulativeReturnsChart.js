// Change this to match your actual local API IP if different
const API_BASE = `http://${window.location.hostname}:8000`;
const API_ENDPOINT = `${API_BASE}/api/v1/analytics/cumulative_returns`;

let equityChart = null;

export function CumulativeReturnsChart() {
    return `
        <div class="chart-controls" style="margin-bottom: 10px; display: flex; gap: 10px; flex-wrap: wrap;">
            <button class="btn-equity-timeframe" data-days="30">1M</button>
            <button class="btn-equity-timeframe" data-days="180">6M</button>
            <button class="btn-equity-timeframe" data-days="365">1Y</button>
            <button class="btn-equity-timeframe" data-days="3650">ALL</button>
            <span style="font-size: 0.8em; color: #aaa; align-self: center; margin-left: 10px;">
                📈 Equity Curve | 🔍 Scroll to Zoom
            </span>
        </div>
        <div class="chart-wrapper" style="position: relative; height: 350px; width: 100%;">
            <div class="chart-container" style="position: absolute; top: 0; left: 0; right: 0; bottom: 0; background: #1a1a1a; padding: 10px; border-radius: 8px; overflow: hidden;">
                <canvas id="cumulative-returns-chart-canvas"></canvas>
            </div>
        </div>
    `;
}

export async function fetchAndRenderEquity(days = 365) {
    const chartId = 'cumulative-returns-chart-canvas';
    const canvas = document.getElementById(chartId);
    if (!canvas) return;

    try {
        const response = await fetch(`${API_ENDPOINT}?days=${days}`);
        const data = await response.json(); 
        // Data is now: { strategy: [...], benchmark: [...], benchmark_label: "..." }

        if (equityChart) {
            equityChart.destroy();
        }

        const ctx = canvas.getContext('2d');
        equityChart = new Chart(ctx, {
            type: 'line',
            data: {
                datasets: [
                    {
                        label: 'DPM Strategy',
                        data: data.strategy.map(d => ({ x: d.date, y: d.equity * 100 })),
                        borderColor: '#4bc0c0',
                        backgroundColor: 'rgba(75, 192, 192, 0.1)',
                        fill: true,
                        tension: 0.1,
                        pointRadius: 0
                    },
                    {
                        label: `Benchmark (${data.benchmark_label || 'Target'})`,
                        data: data.benchmark.map(d => ({ x: d.date, y: d.equity * 100 })),
                        borderColor: '#ff6384',
                        borderDash: [5, 5], // Dashed line for visual distinction
                        fill: false,
                        tension: 0.1,
                        pointRadius: 0
                    }
                ]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                interaction: { mode: 'index', intersect: false },
                plugins: {
                    legend: { display: true, labels: { color: '#ccc' } },
                    tooltip: {
                        callbacks: {
                            label: (context) => `${context.dataset.label}: ${context.parsed.y.toFixed(2)}%`
                        }
                    },
                    zoom: {
                        zoom: {
                            wheel: { enabled: true },
                            pinch: { enabled: true },
                            mode: 'x',
                        },
                        pan: {
                            enabled: true,
                            mode: 'x',
                            threshold: 10,
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
                            callback: (value) => value + '%' // Show as percentage
                        },
                        title: { display: true, text: 'Cumulative Return (%)', color: '#ccc' }
                    }
                }
            }
        });
    } catch (e) {
        console.error("Equity Chart Error:", e);
    }
}

// Event Listener for timeframe buttons
document.addEventListener('click', (e) => {
    if (e.target.classList.contains('btn-equity-timeframe')) {
        const days = parseInt(e.target.getAttribute('data-days'));
        fetchAndRenderEquity(days);
    }
});
