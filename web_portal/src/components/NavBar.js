// web_portal/src/components/NavBar.js

export function NavBar(onNavigate, activeView) {
    return `
        <div class="navbar">
            <h2>Strategy Monitor</h2>
            <button onclick="onNavigate('Dashboard')" class="${activeView === 'Dashboard' ? 'active' : ''}">
                A. Live Dashboard
            </button>
            <button onclick="onNavigate('Reports')" class="${activeView === 'Reports' ? 'active' : ''}">
                B. Historical Reports
            </button>
            <button onclick="onNavigate('TradeLog')" class="${activeView === 'TradeLog' ? 'active' : ''}">
                C. Trade Log/Audit
            </button>
        </div>
    `;
}
