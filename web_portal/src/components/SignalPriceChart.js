const API_BASE = `http://${window.location.hostname}:8000`;
const API_ENDPOINT = `${API_BASE}/api/v1/charts/signal_price?ticker=LQQ.PA&start_date=2024-01-01`;

let currentChart = null;
let debugChart = null; 

const broadcastSync = (chart, senderName = 'signal') => {
    window.dispatchEvent(new CustomEvent('sync-charts', {
        detail: { min: chart.scales.x.min, max: chart.scales.x.max, sender: senderName }
    }));
};

// 2. New Listener
window.addEventListener('sync-charts', (e) => {
    // 1. Sync the Price Chart (if not the sender)
    if (e.detail.sender !== 'signal' && currentChart) {
        currentChart.options.scales.x.min = e.detail.min;
        currentChart.options.scales.x.max = e.detail.max;
        currentChart.update('none');
    }
    if (e.detail.sender !== 'debug' && debugChart) { // Add this check
        debugChart.options.scales.x.min = e.detail.min;
        debugChart.options.scales.x.max = e.detail.max;
        debugChart.update('none');
    }    
});

/**
 * Helper to interpolate colors between Red (0), Yellow (0.5), and Green (1)
 * @param {number} value - Position value between 0 and 1
 */
function getPositionColor(value) {
    // Ensure value is between 0 and 1
    const v = Math.max(0, Math.min(1, value));
    
    let r, g, b;
    
    if (v < 0.5) {
        // Fade from Red (220, 53, 69) to Yellow (255, 193, 7)
        const ratio = v * 2; // scale 0-0.5 to 0-1
        r = 220 + (255 - 220) * ratio;
        g = 53 + (193 - 53) * ratio;
        b = 69 + (7 - 69) * ratio;
    } else {
        // Fade from Yellow (255, 193, 7) to Green (40, 167, 69)
        const ratio = (v - 0.5) * 2; // scale 0.5-1 to 0-1
        r = 255 + (40 - 255) * ratio;
        g = 193 + (167 - 193) * ratio;
        b = 7 + (69 - 7) * ratio;
    }
    
    return `rgb(${Math.round(r)}, ${Math.round(g)}, ${Math.round(b)})`;
}

export function SignalPriceChart() {
    return `
        <div class="chart-controls" style="margin-bottom: 10px; display: flex; gap: 10px; flex-wrap: wrap;">
            <button class="btn-timeframe" data-days="30">1M</button>
            <button class="btn-timeframe" data-days="180">6M</button>
            <button class="btn-timeframe" data-days="365">1Y</button>
            <button class="btn-timeframe" data-days="3650">ALL</button>
        </div>
        <div class="chart-wrapper" style="height: 350px; position: relative; background: #1a1a1a; padding: 10px; border-radius: 8px;">
             <canvas id="signal-price-chart-canvas"></canvas>
        </div>
        <div class="chart-wrapper" style="height: 120px; position: relative; background: #1a1a1a; padding: 10px; border-radius: 8px; margin-top: 10px;">
             <canvas id="position-debug-chart-canvas"></canvas>
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
                            borderColor: ctx => {
                                // ctx.p1DataIndex gives us the index of the second point of the segment
                                const index = ctx.p1DataIndex;
                                const positionValue = data.position[index]; 
                                return getPositionColor(positionValue);
                            }
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
                    },
                    {
                        label: 'VIX Signal',
                        data: data.component_signals.find(s => s.name === 'VIX')?.data || [],
                        borderColor: '#ff9f40', // Orange color for contrast
                        borderWidth: 2,
                        borderDash: [2, 2],    // Dotted line to differentiate from TMOM/SMA
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
                        zoom: { 
                            wheel: { enabled: true }, 
                            pinch: { enabled: true }, 
                            mode: 'x', 
                            enabled: true, 
                            onZoom: ({chart}) => broadcastSync(chart) 
                            
                        },
                        pan: { 
                            enabled: true, 
                            mode: 'x', 
                            enabled: true, 
                            onPan: ({chart}) => broadcastSync(chart), 
                            threshold: 10                             
                        }
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
        
        const debugCtx = document.getElementById('position-debug-chart-canvas').getContext('2d');
        if (debugChart) debugChart.destroy();

        // Create a linear gradient for the line
        const gradient = debugCtx.createLinearGradient(0, 0, 0, 120); // y0 to y1 height
        gradient.addColorStop(0, 'rgba(40, 167, 69, 0.4)');   // Top: Green
        gradient.addColorStop(0.5, 'rgba(255, 193, 7, 0.4)'); // Mid: Yellow
        gradient.addColorStop(1, 'rgba(220, 53, 69, 0.4)');   // Bottom: Red

        debugChart = new Chart(debugCtx, {
            type: 'line',
            data: {
                labels: data.timestamps,
                datasets: [{
                    label: 'Actual Position %',
                    data: data.position.map(v => v * 100),
                    // Style settings for a solid debug line
                    borderColor: '#555', // Neutral dark line
                    backgroundColor: gradient, // Fills area with your custom gradient
                    fill: true,
                    borderWidth: 2,
                    pointRadius: 0, // Keeps it clean like the price chart
                    stepped: true,  // Important: This shows the "all-or-nothing" signal jumps better
                    tension: 0                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                interaction: {
                    mode: 'index',
                    intersect: false,
                },
                plugins: { 
                    legend: { display: false },
                    zoom: {
                        zoom: { wheel: { enabled: true }, mode: 'x', onZoom: ({chart}) => broadcastSync(chart, 'debug') },
                        pan: { enabled: true, mode: 'x', onPan: ({chart}) => broadcastSync(chart, 'debug') }
                    }
                },
                scales: {
                    x: { 
                        type: 'time', 
                        display: false,
                        bounds: 'data'
                    },
                    y: { 
                        min: 0, 
                        max: 105, // Slightly higher than 100 to avoid clipping the line
                        ticks: { color: '#888', stepSize: 50 }, 
                        title: { display: true, text: 'POS %', color: '#888'} 
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
