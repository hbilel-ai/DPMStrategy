// web_portal/src/components/CurrentStatus.js
const API_BASE = `http://${window.location.hostname}:8000`;
const API_ENDPOINT = `${API_BASE}/api/v1/kpis/current`;

export function CurrentStatus() {
    return `<div class="card current-status-card" id="current-status"><h3>1. Current Status</h3><p>Loading...</p></div>`;
}

export function renderStateMachine(exposure) {
    const isCash = exposure === 0;
    const isPartial = exposure === 0.5;
    const isInvested = exposure === 1.0;

    return `
        <div class="state-machine-container">
            <div class="state-node ${isCash ? 'active-cash' : ''}">CASH</div>
            <div class="state-connector"></div>
            <div class="state-node ${isPartial ? 'active-partial' : ''}">PARTIAL</div>
            <div class="state-connector"></div>
            <div class="state-node ${isInvested ? 'active-invested' : ''}">INVESTED</div>
        </div>
    `;
}

export function renderGauge(exposure) {
    // Map 0.0-1.0 to -90deg to 90deg
    const rotation = (exposure * 180) - 90;
    return `
        <div class="gauge-container">
            <div class="gauge-arc"></div>
            <div class="gauge-needle" style="transform: translateX(-50%) rotate(${rotation}deg)"></div>
        </div>
    `;
}

async function fetchData() {
    const card = document.getElementById('current-status');
    if (!card) return;

    try {
        const response = await fetch(API_ENDPOINT);
        const data = await response.json();

        card.innerHTML = `
            <h3>1. Current Status</h3>
            
            <div class="signal-indicators">
                <div class="sig-btn ${data.tmom_on ? 'sig-on' : 'sig-off'}">TMOM</div>
                <div class="sig-btn ${data.sma_on ? 'sig-on' : 'sig-off'}">SMA</div>
                <div class="sig-btn ${data.vix_safe ? 'sig-on' : 'sig-off'}">VIX</div>
            </div>

            ${renderGauge(data.exposure)}

            <div style="text-align: center; margin-top: -5px; margin-bottom: 15px;">
                <span style="font-size: 1.5em; font-weight: bold;">${(data.exposure * 100).toFixed(0)}%</span>
                <p style="font-size: 0.75em; color: #888; margin: 0;">TARGET EXPOSURE</p>
            </div>

            <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 8px; font-size: 0.8em; background: #1e1e1e; padding: 10px; border-radius: 4px;">
                <div><span style="color: #666;">Ticker:</span> ${data.ticker}</div>
                <div><span style="color: #666;">Price:</span> ${(data.current_price || 0).toFixed(2)}</div> 
                <div style="grid-column: span 2; border-top: 1px solid #333; padding-top: 5px; color: var(--primary-color);">
                    <b>Alert:</b> ${data.action_alert} 
                </div>
            </div>
        `;
    } catch (error) {
        console.error("DEBUG_JS_ERROR:", error); // This will show the exact error in F12
        card.innerHTML = `<p style="color:red;">Error loading status</p>`;
    }
}

setTimeout(fetchData, 500);
