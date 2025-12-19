// web_portal/src/App.js

import { NavBar } from './components/NavBar.js';
import { DashboardView } from './views/DashboardView.js';
import { ReportsView } from './views/ReportsView.js';
import { TradeLogView } from './views/TradeLogView.js';
import { fetchAndRenderEquity } from './components/CumulativeReturnsChart.js';

const VIEWS = {
    Dashboard: DashboardView,
    Reports: ReportsView,
    TradeLog: TradeLogView
};

let currentView = 'Dashboard';

// Delay slightly to ensure DOM is ready
setTimeout(() => {
    fetchAndRenderEquity(365);
}, 1000);

function render(viewName = currentView) {
    const root = document.getElementById('root');
    const onNavigate = (name) => {
        currentView = name;
        render(); // Re-render the application when navigation occurs
    };

    // 1. Get the current view content
    const ViewComponent = VIEWS[viewName] || DashboardView;
    const viewContent = ViewComponent();

    // 2. Assemble the final HTML
    root.innerHTML = `
        ${NavBar(onNavigate, viewName)}
        <div class="main-content">
            ${viewContent}
        </div>
    `;

    if (viewName === 'Reports') {
        // We wait for the DOM to paint, then initialize the reports logic
        import('./components/ReportsEngine.js').then(module => {
            module.initReportsView();
        });
    }

    // NEW: Logic for C. Trade Log/Audit
    if (viewName === 'TradeLog') {
        // We import the view logic dynamically to keep the initial load light
        import('./views/TradeLogView.js').then(module => {
            module.initTradeLogView();
        });
    }

    // Logic for A. Live Dashboard (Existing manual trigger)
    if (viewName === 'Dashboard') {
        triggerDashboardRefresh();
    }

    // Make the navigation handler available globally for the buttons to work
    window.onNavigate = onNavigate;
}

function onNavigate(viewName) {
    const root = document.getElementById('root');

    if (viewName === 'Dashboard') {
        // 1. Inject the HTML skeleton
        root.innerHTML = DashboardView();

        // 2. MANUALLY trigger the data fetches for the new DOM elements
        // Do not rely on setTimeout inside the component files
        triggerDashboardRefresh();
    }
}

async function triggerDashboardRefresh() {
    // Import and call the fetchers explicitly
    // This ensures the charts and state machine render on every click
    await Promise.all([
        fetchCurrentStatus(),
        fetchPerformanceSummary(),
        renderSignalPriceChart(),
        renderEquityChart()
    ]);
}


// Initial render when the script loads
document.addEventListener('DOMContentLoaded', render);

// Export the render function (optional, but good practice)
export { render };
