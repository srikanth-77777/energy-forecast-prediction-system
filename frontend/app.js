/**
 * EnvisionAI — Multi-page Dashboard Application Logic
 * Pages: Forecaster, Model Comparison, Data Explorer, Analytics
 */
document.addEventListener("DOMContentLoaded", async () => {
    const API = window.location.origin;

    // ─── State ──────────────────────────────────────────────
    let currentRegion = '';
    let regions = [];
    let charts = {};

    // ─── DOM Refs ───────────────────────────────────────────
    const globalRegion = document.getElementById('global-region');
    const modelSelect = document.getElementById('model-select');
    const dateInput = document.getElementById('date-input');
    const hourInput = document.getElementById('hour-input');
    const hourDisplay = document.getElementById('hour-display');
    const predictBtn = document.getElementById('predict-btn');
    const compareBtn = document.getElementById('compare-btn');
    const compareDate = document.getElementById('compare-date');
    const compareHour = document.getElementById('compare-hour');
    const compareHourDisplay = document.getElementById('compare-hour-display');
    const explorerDays = document.getElementById('explorer-days');

    // ─── Chart.js Defaults ──────────────────────────────────
    Chart.defaults.color = '#7a7e8f';
    Chart.defaults.borderColor = 'rgba(255,255,255,0.04)';
    Chart.defaults.font.family = "'Inter', sans-serif";

    const COLORS = {
        blue: '#3b82f6', green: '#00e676', red: '#ef4444',
        orange: '#f59e0b', purple: '#8b5cf6', cyan: '#06b6d4',
        blueFill: 'rgba(59,130,246,0.1)', greenFill: 'rgba(0,230,118,0.1)',
    };
    const MODEL_COLORS = [COLORS.blue, COLORS.green, COLORS.orange, COLORS.purple, COLORS.cyan];

    // ─── Navigation ─────────────────────────────────────────
    document.querySelectorAll('.nav-link').forEach(link => {
        link.addEventListener('click', (e) => {
            e.preventDefault();
            const pageName = link.dataset.page;
            document.querySelectorAll('.nav-link').forEach(l => l.classList.remove('active'));
            link.classList.add('active');
            document.querySelectorAll('.page').forEach(p => p.classList.remove('active'));
            document.getElementById(`page-${pageName}`).classList.add('active');
            if (pageName === 'explorer') loadExplorer();
            if (pageName === 'analytics') loadAnalytics();
            if (pageName === 'summary') loadSummary();
        });
    });

    // ─── Initialization ────────────────────────────────────
    try {
        const res = await fetch(`${API}/api/regions`);
        regions = await res.json();
        globalRegion.innerHTML = regions.map(r =>
            `<option value="${r.key}">${r.key} (${r.mean_demand.toLocaleString()} MW avg)</option>`
        ).join('');
        if (regions.length > 0) {
            currentRegion = regions[0].key;
            await onRegionChange();
        }
    } catch (e) {
        console.error("Failed to load regions:", e);
        globalRegion.innerHTML = '<option>Error loading regions</option>';
    }

    globalRegion.addEventListener('change', async () => {
        currentRegion = globalRegion.value;
        await onRegionChange();
    });

    async function onRegionChange() {
        document.getElementById('forecast-region-badge').textContent = currentRegion;
        // Load models for this region
        try {
            const res = await fetch(`${API}/api/models/${currentRegion}`);
            const modelList = await res.json();
            modelSelect.innerHTML = modelList.map(m =>
                `<option value="${m.name}">${m.name}</option>`
            ).join('');
        } catch (e) {
            modelSelect.innerHTML = '<option>Error loading</option>';
        }
        // Set date range from region info
        const rInfo = regions.find(r => r.key === currentRegion);
        if (rInfo) {
            dateInput.min = rInfo.date_min;
            dateInput.max = rInfo.date_max;
            dateInput.value = rInfo.date_max;
            compareDate.min = rInfo.date_min;
            compareDate.max = rInfo.date_max;
            compareDate.value = rInfo.date_max;
        }
    }

    // ─── Hour Sliders ───────────────────────────────────────
    hourInput.oninput = () => { hourDisplay.textContent = `${hourInput.value.padStart(2,'0')}:00`; };
    compareHour.oninput = () => { compareHourDisplay.textContent = `${compareHour.value.padStart(2,'0')}:00`; };

    // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    // PAGE 1: FORECASTER
    // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    function initForecastChart() {
        const ctx = document.getElementById('forecastChart').getContext('2d');
        charts.forecast = new Chart(ctx, {
            type: 'line',
            data: {
                labels: [],
                datasets: [
                    {
                        label: 'Historical',
                        data: [],
                        borderColor: COLORS.blue,
                        backgroundColor: COLORS.blueFill,
                        fill: true,
                        tension: 0.3,
                        pointRadius: 1,
                        borderWidth: 2,
                    },
                    {
                        label: 'Forecast',
                        data: [],
                        borderColor: COLORS.green,
                        backgroundColor: COLORS.greenFill,
                        fill: true,
                        borderDash: [6, 3],
                        tension: 0.3,
                        pointRadius: 2,
                        borderWidth: 2,
                    },
                ]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                interaction: { intersect: false, mode: 'index' },
                plugins: { legend: { display: false } },
                scales: {
                    y: { title: { display: true, text: 'Demand (MW)' } },
                    x: { ticks: { maxTicksAutoSkip: true, maxRotation: 0 } },
                },
            }
        });
    }

    predictBtn.addEventListener('click', async () => {
        if (!dateInput.value || !currentRegion) return;
        predictBtn.disabled = true;
        predictBtn.textContent = 'Running...';
        try {
            const res = await fetch(`${API}/api/predict`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    region: currentRegion,
                    date: dateInput.value,
                    hour: parseInt(hourInput.value),
                    model: modelSelect.value,
                    is_holiday_override: document.getElementById('sim-holiday').checked,
                    simulate_anomaly: document.getElementById('sim-anomaly').checked
                }),
            });
            if (!res.ok) throw new Error((await res.json()).detail || 'Server error');
            const data = await res.json();
            updateForecaster(data);
        } catch (e) {
            alert(`Prediction failed: ${e.message}`);
        }
        predictBtn.disabled = false;
        predictBtn.textContent = 'Run Prediction';
    });

    function updateForecaster(data) {
        const m = data.model_metrics || {};
        document.getElementById('kpi-predicted').textContent = `${data.predicted.toLocaleString()} MW`;
        document.getElementById('kpi-actual').textContent = `${data.actual.toLocaleString()} MW`;
        document.getElementById('kpi-error').textContent = `${(data.predicted - data.actual).toLocaleString()} MW`;
        
        const insightCard = document.getElementById('insight-card');
        document.getElementById('insight-text').textContent = data.ai_insight || '';
        if (document.getElementById('sim-anomaly').checked || Math.abs((data.predicted - data.actual) / data.actual) > 0.25) {
            insightCard.classList.add('critical');
        } else {
            insightCard.classList.remove('critical');
        }
        document.getElementById('metric-rmse').textContent = `${(m.rmse || 0).toLocaleString()} MW`;
        document.getElementById('metric-mae').textContent = `${(m.mae || 0).toLocaleString()} MW`;
        document.getElementById('metric-mape').textContent = `${(m.mape || 0)}%`;

        // Save to Summary
        saveToSummary(data);

        const hist = data.history || [];
        const fore = data.forecast_24h || [];
        const allLabels = [...hist, ...fore].map(p => p.time.substring(11, 16));

        charts.forecast.data.labels = allLabels;
        charts.forecast.data.datasets[0].data = hist.map(p => p.val).concat(new Array(fore.length).fill(null));
        const pad = new Array(Math.max(0, hist.length - 1)).fill(null);
        const lastHist = hist.length > 0 ? hist[hist.length - 1].val : null;
        charts.forecast.data.datasets[1].data = pad.concat([lastHist, ...fore.map(p => p.val)]);
        charts.forecast.update('active');
    }

    // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    // PAGE 2: MODEL COMPARISON
    // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    function initComparisonCharts() {
        charts.r2 = new Chart(document.getElementById('r2Chart').getContext('2d'), {
            type: 'bar', data: { labels: [], datasets: [{ data: [], backgroundColor: MODEL_COLORS }] },
            options: { responsive: true, maintainAspectRatio: false, plugins: { legend: { display: false } }, scales: { y: { beginAtZero: false, title: { display: true, text: 'R-squared' } } } }
        });
        charts.rmse = new Chart(document.getElementById('rmseChart').getContext('2d'), {
            type: 'bar', data: { labels: [], datasets: [{ data: [], backgroundColor: MODEL_COLORS }] },
            options: { responsive: true, maintainAspectRatio: false, plugins: { legend: { display: false } }, scales: { y: { title: { display: true, text: 'RMSE (MW)' } } } }
        });
        charts.pred = new Chart(document.getElementById('predChart').getContext('2d'), {
            type: 'bar', data: { labels: [], datasets: [] },
            options: { responsive: true, maintainAspectRatio: false, plugins: { legend: { position: 'top' } } }
        });
    }

    compareBtn.addEventListener('click', async () => {
        if (!compareDate.value || !currentRegion) return;
        compareBtn.disabled = true;
        compareBtn.textContent = 'Comparing...';
        try {
            const res = await fetch(`${API}/api/compare`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    region: currentRegion,
                    date: compareDate.value,
                    hour: parseInt(compareHour.value),
                }),
            });
            if (!res.ok) throw new Error((await res.json()).detail || 'Server error');
            const data = await res.json();
            updateComparison(data);
        } catch (e) {
            alert(`Comparison failed: ${e.message}`);
        }
        compareBtn.disabled = false;
        compareBtn.textContent = 'Compare All Models';
    });

    function updateComparison(data) {
        const comps = data.comparisons || [];
        const names = comps.map(c => c.model);
        const bestR2 = Math.max(...comps.map(c => c.metrics.r2 || 0));

        // R2 chart
        charts.r2.data.labels = names;
        charts.r2.data.datasets[0].data = comps.map(c => c.metrics.r2 || 0);
        charts.r2.update();

        // RMSE chart
        charts.rmse.data.labels = names;
        charts.rmse.data.datasets[0].data = comps.map(c => c.metrics.rmse || 0);
        charts.rmse.update();

        // Prediction chart (actual vs predicted)
        charts.pred.data.labels = names;
        charts.pred.data.datasets = [
            { label: `Actual (${data.actual.toLocaleString()} MW)`, data: names.map(() => data.actual), backgroundColor: COLORS.blue },
            { label: 'Predicted', data: comps.map(c => c.predicted), backgroundColor: MODEL_COLORS },
        ];
        charts.pred.update();

        // Table
        const tbody = document.getElementById('comparison-tbody');
        tbody.innerHTML = comps.map(c => {
            const isBest = c.metrics.r2 === bestR2 ? 'best-row' : '';
            const m = c.metrics;
            return `<tr class="${isBest}">
                <td>${c.model}${isBest ? ' *' : ''}</td>
                <td>${c.predicted.toLocaleString()}</td>
                <td>${c.error.toLocaleString()}</td>
                <td>${c.error_pct}%</td>
                <td>${((m.r2||0)*100).toFixed(2)}%</td>
                <td>${(m.rmse||0).toLocaleString()}</td>
                <td>${(m.mae||0).toLocaleString()}</td>
                <td>${m.mape||0}%</td>
            </tr>`;
        }).join('');
    }

    // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    // PAGE 3: DATA EXPLORER
    // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    function initExplorerCharts() {
        charts.history = new Chart(document.getElementById('historyChart').getContext('2d'), {
            type: 'line',
            data: { labels: [], datasets: [{ label: 'Demand', data: [], borderColor: COLORS.blue, backgroundColor: COLORS.blueFill, fill: true, tension: 0.2, pointRadius: 0, borderWidth: 1.5 }] },
            options: { responsive: true, maintainAspectRatio: false, plugins: { legend: { display: false } }, scales: { y: { title: { display: true, text: 'MW' } } } }
        });
        charts.dist = new Chart(document.getElementById('distChart').getContext('2d'), {
            type: 'bar',
            data: { labels: [], datasets: [{ label: 'Frequency', data: [], backgroundColor: COLORS.purple }] },
            options: { responsive: true, maintainAspectRatio: false, plugins: { legend: { display: false } }, scales: { y: { title: { display: true, text: 'Count' } } } }
        });
    }

    explorerDays.addEventListener('change', loadExplorer);

    async function loadExplorer() {
        if (!currentRegion) return;
        try {
            const days = explorerDays.value;
            const res = await fetch(`${API}/api/history/${currentRegion}?days=${days}`);
            const data = await res.json();
            const pts = data.data || [];
            const stats = data.stats || {};

            document.getElementById('stat-rows').textContent = (stats.total_rows || 0).toLocaleString();
            document.getElementById('stat-mean').textContent = `${(stats.mean || 0).toLocaleString()} MW`;
            document.getElementById('stat-max').textContent = `${(stats.max || 0).toLocaleString()} MW`;
            document.getElementById('stat-min').textContent = `${(stats.min || 0).toLocaleString()} MW`;

            charts.history.data.labels = pts.map(p => p.time.substring(0, 10));
            charts.history.data.datasets[0].data = pts.map(p => p.val);
            charts.history.update();

            // Distribution histogram
            const vals = pts.map(p => p.val);
            const min = Math.min(...vals), max = Math.max(...vals);
            const bins = 20;
            const binWidth = (max - min) / bins;
            const counts = new Array(bins).fill(0);
            vals.forEach(v => { const i = Math.min(Math.floor((v - min) / binWidth), bins - 1); counts[i]++; });
            const binLabels = counts.map((_, i) => `${Math.round(min + i * binWidth)}`);

            charts.dist.data.labels = binLabels;
            charts.dist.data.datasets[0].data = counts;
            charts.dist.update();
        } catch (e) {
            console.error("Explorer error:", e);
        }
    }

    // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    // PAGE 4: ANALYTICS
    // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    function initAnalyticsCharts() {
        const barOpts = { responsive: true, maintainAspectRatio: false, plugins: { legend: { display: false } } };
        charts.hourly = new Chart(document.getElementById('hourlyChart').getContext('2d'), {
            type: 'bar', data: { labels: [], datasets: [{ data: [], backgroundColor: COLORS.blue }] }, options: barOpts
        });
        charts.dow = new Chart(document.getElementById('dowChart').getContext('2d'), {
            type: 'bar', data: { labels: [], datasets: [{ data: [], backgroundColor: COLORS.purple }] }, options: barOpts
        });
        charts.monthly = new Chart(document.getElementById('monthlyChart').getContext('2d'), {
            type: 'bar', data: { labels: [], datasets: [{ data: [], backgroundColor: COLORS.orange }] }, options: barOpts
        });
        charts.yearly = new Chart(document.getElementById('yearlyChart').getContext('2d'), {
            type: 'line', data: { labels: [], datasets: [{ data: [], borderColor: COLORS.cyan, tension: 0.3, pointRadius: 3, borderWidth: 2 }] }, options: barOpts
        });
    }

    async function loadAnalytics() {
        if (!currentRegion) return;
        try {
            const res = await fetch(`${API}/api/stats/${currentRegion}`);
            const data = await res.json();

            // Hourly
            const hours = Object.keys(data.hourly_avg).sort((a, b) => +a - +b);
            charts.hourly.data.labels = hours.map(h => `${h}:00`);
            charts.hourly.data.datasets[0].data = hours.map(h => data.hourly_avg[h]);
            charts.hourly.update();

            // Day of week
            const dows = Object.keys(data.dow_avg);
            charts.dow.data.labels = dows;
            charts.dow.data.datasets[0].data = dows.map(d => data.dow_avg[d]);
            charts.dow.update();

            // Monthly
            const months = Object.keys(data.monthly_avg).sort((a, b) => +a - +b);
            const monthNames = ['', 'Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'];
            charts.monthly.data.labels = months.map(m => monthNames[+m]);
            charts.monthly.data.datasets[0].data = months.map(m => data.monthly_avg[m]);
            charts.monthly.update();

            // Yearly
            const years = Object.keys(data.yearly_avg).sort();
            charts.yearly.data.labels = years;
            charts.yearly.data.datasets[0].data = years.map(y => data.yearly_avg[y]);
            charts.yearly.update();

            // KPIs
            document.getElementById('ana-weekday').textContent = `${data.weekday_avg.toLocaleString()} MW`;
            document.getElementById('ana-weekend').textContent = `${data.weekend_avg.toLocaleString()} MW`;
            document.getElementById('ana-peak').textContent = `${data.peak_avg.toLocaleString()} MW`;
            document.getElementById('ana-offpeak').textContent = `${data.offpeak_avg.toLocaleString()} MW`;
        } catch (e) {
            console.error("Analytics error:", e);
        }
    }

    // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    // PAGE 5: SAVED SUMMARIES
    // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    
    function saveToSummary(data) {
        let history = JSON.parse(localStorage.getItem('forecastHistory') || '[]');
        const isHoliday = document.getElementById('sim-holiday').checked;
        const isAnomaly = document.getElementById('sim-anomaly').checked;
        
        let simLabels = [];
        if (isHoliday) simLabels.push("🌲 Holiday");
        if (isAnomaly) simLabels.push("⚠️ Outage");

        const record = {
            execTime: new Date().toLocaleTimeString(),
            region: data.region,
            targetTime: data.target_time.substring(0, 16),
            model: data.model,
            predicted: data.predicted,
            actual: data.actual,
            errorPct: data.actual ? ((Math.abs(data.predicted - data.actual) / data.actual) * 100).toFixed(2) : 0,
            simulators: simLabels.length ? simLabels.join(', ') : "None"
        };
        
        history.unshift(record);
        if (history.length > 50) history.pop(); // Keep max 50
        
        localStorage.setItem('forecastHistory', JSON.stringify(history));
    }

    function loadSummary() {
        const history = JSON.parse(localStorage.getItem('forecastHistory') || '[]');
        document.getElementById('summary-count-badge').textContent = `${history.length} saved`;
        const tbody = document.getElementById('summary-tbody');
        
        if (history.length === 0) {
            tbody.innerHTML = '<tr><td colspan="8" style="text-align:center;">No summaries saved yet. Run a prediction to save it here.</td></tr>';
            return;
        }

        tbody.innerHTML = history.map(h => `
            <tr>
                <td>${h.execTime}</td>
                <td class="accent">${h.region}</td>
                <td>${h.targetTime}</td>
                <td>${h.model}</td>
                <td>${h.predicted.toLocaleString()}</td>
                <td>${h.actual.toLocaleString()}</td>
                <td class="${h.errorPct > 10 ? 'error' : ''}">${h.errorPct}%</td>
                <td>${h.simulators}</td>
            </tr>
        `).join('');
    }

    document.getElementById('clear-summary-btn').addEventListener('click', () => {
        localStorage.removeItem('forecastHistory');
        loadSummary();
    });

    // ─── Init All Charts ────────────────────────────────────
    initForecastChart();
    initComparisonCharts();
    initExplorerCharts();
    initAnalyticsCharts();
});
