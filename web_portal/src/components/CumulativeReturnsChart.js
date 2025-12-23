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
    const canvas = document.getElementById('cumulative-returns-chart-canvas');
    if (!canvas) return;

    try {
        // Fetch from the evolved backend endpoint with the 'days' parameter
        const response = await fetch(`${API_ENDPOINT}?days=${days}`);
        const data = await response.json();

        if (equityChart) equityChart.destroy();

        equityChart = new Chart(canvas.getContext('2d'), {
            type: 'line',
            data: {
                labels: data.map(d => d.date),
                datasets: [{
                    label: 'Portfolio Value (€)',
                    data: data.map(d => d.equity),
                    borderColor: '#4bc0c0',
                    backgroundColor: 'rgba(75, 192, 192, 0.1)',
                    fill: true,
                    tension: 0.1,
                    pointRadius: 0,
                    borderWidth: 2
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                resizeDelay: 100,
                interaction: { mode: 'index', intersect: false },
                plugins: {
                    legend: { display: true, labels: { color: '#ccc' } },
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
                        ticks: { color: '#888' },
                        bounds: 'data'
                    },
                    y: {
                        grid: { color: '#333' },
                        ticks: { color: '#888' },
                        title: { display: true, text: 'Value (€)', color: '#ccc' }
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
