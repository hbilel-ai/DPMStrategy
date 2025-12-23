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

async function fetchData() {
    const card = document.getElementById('current-status');
    if (!card) return;

    try {
        const response = await fetch(API_ENDPOINT);
        const data = await response.json();

        // FIX: Ensure we have the exposure value
        const exposure = data.exposure !== undefined ? data.exposure : (data.tmom_signal_value + data.sma_signal_value) / 2;

        card.innerHTML = `
            <h3>1. Current Status</h3>
            <div style="font-size: 1.1em; margin-bottom: 15px;">
                <p>
                    <strong>Current Signal:</strong>
                    <span style="color: ${data.action_alert === 'CASH' ? '#ff4d4d' : (data.action_alert === 'LONG' ? 'lightgreen' : 'orange')};">
                        ${data.action_alert}
                    </span>
                </p>
            </div>

            ${renderStateMachine(exposure)}

            <div style="margin-top: 15px; font-size: 0.85em; color: #888; text-align: center;">
                Signals: TMOM (${data.tmom_signal_value}) | SMA (${data.sma_signal_value})
            </div>
        `;
    } catch (error) {
        card.innerHTML = `<h3>1. Current Status</h3><p style="color:red;">Offline</p>`;
    }
}

setTimeout(fetchData, 500);
