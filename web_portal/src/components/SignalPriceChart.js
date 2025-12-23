const API_BASE = `http://${window.location.hostname}:8000`;
const API_ENDPOINT = `${API_BASE}/api/v1/charts/signal_price?ticker=LQQ.PA&start_date=2024-01-01`;

let currentChart = null;

export function SignalPriceChart() {
    return `
        <div class="chart-controls" style="margin-bottom: 10px; display: flex; gap: 10px; flex-wrap: wrap;">
            <button class="btn-timeframe" data-days="30">1M</button>
            <button class="btn-timeframe" data-days="180">6M</button>
            <button class="btn-timeframe" data-days="365">1Y</button>
            <button class="btn-timeframe" data-days="3650">ALL</button>
        </div>
        <div class="chart-wrapper" style="position: relative; height: 350px; width: 100%;">
            <div class="chart-container" style="position: absolute; top: 0; left: 0; right: 0; bottom: 0; background: #1a1a1a; padding: 10px; border-radius: 8px; overflow: hidden;">
                <canvas id="signal-price-chart-canvas"></canvas>
            </div>
        </div>
    `;
}

async function fetchAndRender(days = 365) {
    const chartId = 'signal-price-chart-canvas';
    const canvas = document.getElementById(chartId);
    if (!canvas) return;

    try {
        const startDate = new Date();
        startDate.setDate(startDate.getDate() - days);
        const dateStr = startDate.toISOString().split('T')[0];

        const url = `${API_BASE}/api/v1/charts/signal_price?ticker=LQQ.PA&start_date=${dateStr}`;
        const response = await fetch(url);
        const data = await response.json();

        if (currentChart) currentChart.destroy();

        const ctx = canvas.getContext('2d');

        // HELPER: Determine color based on position (exposure)
        // 1.0 = Green (Full), 0.5 = Orange (Partial), 0.0 = Red (Cash)
        const getSegmentColor = (ctx) => {
            const pos = data.position[ctx.p0DataIndex];
            if (pos >= 1.0) return '#4bc0c0'; // Green
            if (pos > 0 && pos < 1.0) return '#ff9f40'; // Orange
            return '#ff6384'; // Red
        };

        currentChart = new Chart(ctx, {
            type: 'line',
            data: {
                labels: data.timestamps,
                datasets: [
                    {
                        label: 'Price (€)',
                        data: data.price,
                        yAxisID: 'y',
                        pointRadius: 0,
                        borderWidth: 3,
                        // Apply dynamic coloring to the line segments
                        segment: {
                            borderColor: ctx => getSegmentColor(ctx)
                        }
                    },
                    {
                        label: 'TMOM Signal',
                        data: data.component_signals.find(s => s.name === 'TMOM')?.data || [],
                        borderColor: '#36a2eb',
                        borderWidth: 2,
                        stepped: true,
                        yAxisID: 'y1',
                        pointRadius: 0,
                        fill: false
                    },
                    {
                        label: 'SMA Signal',
                        data: data.component_signals.find(s => s.name === 'SMA')?.data || [],
                        borderColor: '#9966ff',
                        borderWidth: 2,
                        borderDash: [5, 5], // Dashed to see it when overlapping with TMOM
                        stepped: true,
                        yAxisID: 'y1',
                        pointRadius: 0,
                        fill: false
                    }
                ]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                interaction: { mode: 'index', intersect: false },
                plugins: {
                    legend: { display: true, labels: { color: '#ccc' } },
                    zoom: {
                        zoom: { wheel: { enabled: true }, pinch: { enabled: true }, mode: 'x' },
                        pan: { enabled: true, mode: 'x', threshold: 10 }
                    }
                },
                scales: {
                    x: {
                        type: 'time',
                        time: { unit: 'month' },
                        grid: { color: '#333' },
                        bounds: 'data'
                    },
                    y: {
                        position: 'left',
                        title: { display: true, text: 'Price (€)', color: '#ccc' }
                    },
                    y1: {
                        position: 'right',
                        min: 0,
                        max: 5, // Set max to 5 so the 0/1 signals stay in the bottom 20%
                        display: false,
                        grid: { drawOnChartArea: false }
                    }
                }
            }
        });

    } catch (e) { console.error("Signal Chart Error:", e); }
}

document.addEventListener('click', (e) => {
    if (e.target.classList.contains('btn-timeframe')) {
        const days = parseInt(e.target.getAttribute('data-days'));
        fetchAndRender(days);
    }
});

setTimeout(() => fetchAndRender(365), 800);
